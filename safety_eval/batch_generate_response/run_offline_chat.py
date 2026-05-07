#!/usr/bin/env python3
"""Offline response generation via ``vllm.LLM.chat`` (no JSONL batch files)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from safety_eval.batch_utils import get_responses_save_path  # noqa: E402
from safety_eval.dataset_utils import load_ours_dataset  # noqa: E402
from safety_eval.utils import load_json_list, save_json_list  # noqa: E402
from safety_eval.vllm_offline_chat import (  # noqa: E402
    build_llm,
    chat_template_kwargs_generate,
    configure_stdio,
    parallel_map_ordered,
    prepare_generate_messages_row,
    resolve_chat_batch_size,
    resolve_prepare_max_workers,
    run_chat_all,
    sampling_params_generate,
)

configure_stdio()


@hydra.main(
    config_path="../../configs",
    config_name="safety_eval_generate_response.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    save_path = get_responses_save_path(cfg, REPO_ROOT)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_ours_dataset(cfg)
    model_path = str(cfg.model.model_name_or_path)

    if save_path.exists() and not bool(cfg.overwrite):
        results = load_json_list(save_path)
        completed_ids = {str(r.get("persona_id")) for r in results}
        print(
            f"Resume mode: {len(completed_ids)} persona_ids already in {save_path}",
            flush=True,
        )
    else:
        results = []
        completed_ids = set()

    mw = resolve_prepare_max_workers(cfg)

    def select_row(i: int) -> tuple[str, Any]:
        sample = dict(ds[i])
        persona_id = str(sample.get("persona_id") or "").strip()
        if not persona_id:
            return ("no_pid", None)
        if persona_id in completed_ids:
            return ("completed", None)
        return ("ok", dict(sample))

    select_out = parallel_map_ordered(
        select_row,
        list(range(len(ds))),
        max_workers=mw,
        progress_desc="select_samples",
    )
    candidates: list[dict[str, Any]] = []
    n_missing_pid = 0
    for tag, payload in select_out:
        if tag == "ok":
            candidates.append(payload)
        elif tag == "no_pid":
            n_missing_pid += 1

    if n_missing_pid:
        print(
            f"Skipping {n_missing_pid} samples with missing persona_id.",
            flush=True,
        )
    if not candidates:
        print("Nothing to run (all samples already completed or empty dataset).", flush=True)
        save_json_list(save_path, results)
        return

    chat_bs = resolve_chat_batch_size(cfg)
    chunk_size = chat_bs if chat_bs is not None else len(candidates)
    print(
        f"Running {len(candidates)} samples in chat chunks of {chunk_size} "
        f"(prepare_max_workers={mw!r}, chat_batch_size={chat_bs!r}) …",
        flush=True,
    )

    use_tqdm = bool(cfg.offline_chat.use_tqdm)
    llm = build_llm(cfg, model_path=model_path)
    sp = sampling_params_generate(cfg)
    ctkw = chat_template_kwargs_generate(cfg)
    inner_tqdm = use_tqdm and len(candidates) <= chunk_size

    for start in range(0, len(candidates), chunk_size):
        cand_chunk = candidates[start : start + chunk_size]
        pending = [
            r
            for r in parallel_map_ordered(
                prepare_generate_messages_row,
                cand_chunk,
                max_workers=mw,
                progress_desc="prepare_messages",
            )
            if r is not None
        ]
        if not pending:
            continue
        work = [(pid, msg) for pid, msg, _ in pending]
        pairs = run_chat_all(
            llm,
            work,
            sampling_params=sp,
            use_tqdm=inner_tqdm,
            chat_template_kwargs=ctkw,
        )
        for (pid, text), (_, _, q) in zip(pairs, pending, strict=True):
            if text is None:
                print(
                    f"[persona_id={pid}] Missing assistant output; skip.",
                    flush=True,
                )
                continue
            results.append({"persona_id": pid, "query": q, "response": text})
        save_json_list(save_path, results)

    print(f"Done. Wrote {len(results)} rows to {save_path}", flush=True)


if __name__ == "__main__":
    main()
