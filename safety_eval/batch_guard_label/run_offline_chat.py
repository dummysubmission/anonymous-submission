#!/usr/bin/env python3
"""Offline guard labeling via ``vllm.LLM.chat`` (no JSONL batch files)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from safety_eval.batch_utils import (  # noqa: E402
    get_guard_save_path,
    get_response_path,
)
from safety_eval.dataset_utils import load_ours_dataset  # noqa: E402
from safety_eval.utils import (  # noqa: E402
    load_json_list,
    sanitize_model_key,
    save_json_list,
)
from safety_eval.vllm_offline_chat import (  # noqa: E402
    build_llm,
    configure_stdio,
    parallel_map_ordered,
    prepare_guard_messages_row,
    resolve_chat_batch_size,
    resolve_prepare_max_workers,
    run_chat_all,
    sampling_params_guard,
)

configure_stdio()


def _parse_guard_label(raw: str) -> str:
    if not raw:
        return "unparsed"
    for line in raw.splitlines():
        s = line.strip().lower()
        if not s:
            continue
        if s == "safe":
            return "safe"
        if s == "unsafe":
            return "unsafe"
    return "unparsed"


def _guard_max_model_len(cfg: DictConfig) -> int | None:
    raw = getattr(cfg.guard_model, "max_model_len", None)
    if raw in (None, "", "null", "None", "auto"):
        return None
    return int(raw)


@hydra.main(
    config_path="../../configs",
    config_name="safety_eval_guard_label.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    save_path = get_guard_save_path(cfg, REPO_ROOT)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    guard_key = sanitize_model_key(str(cfg.guard_model.short_name))
    guard_raw_key = f"{guard_key}_guard_raw"
    guard_label_key = f"{guard_key}_guard"

    response_path = get_response_path(cfg, REPO_ROOT)
    if not response_path.is_file():
        raise FileNotFoundError(
            f"Responses file not found: {response_path}\n"
            "Run batch_generate_response offline chat first, or set response_path= explicitly."
        )
    responses = load_json_list(response_path)
    response_by_id: dict[str, str] = {
        str(r.get("persona_id", "")): str(r.get("response", ""))
        for r in responses
        if r.get("persona_id")
    }
    responses_by_id: dict[str, dict[str, Any]] = {
        str(r.get("persona_id", "")): dict(r)
        for r in responses
        if r.get("persona_id")
    }

    ds = load_ours_dataset(cfg)
    guard_path = str(cfg.guard_model.model_name_or_path)

    if save_path.exists() and not bool(cfg.overwrite):
        results = load_json_list(save_path)
        completed_ids = {
            str(item.get("persona_id"))
            for item in results
            if item.get(guard_raw_key)
        }
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
        response_text = response_by_id.get(persona_id)
        if response_text is None:
            return ("no_response", None)
        return ("ok", (sample, response_text))

    select_out = parallel_map_ordered(
        select_row,
        list(range(len(ds))),
        max_workers=mw,
        progress_desc="select_samples",
    )
    guard_rows: list[tuple[dict[str, Any], str]] = []
    n_skipped_no_response = 0
    n_missing_pid = 0
    for tag, payload in select_out:
        if tag == "ok":
            guard_rows.append(payload)
        elif tag == "no_response":
            n_skipped_no_response += 1
        elif tag == "no_pid":
            n_missing_pid += 1

    if n_missing_pid:
        print(
            f"Skipping {n_missing_pid} samples with missing persona_id.",
            flush=True,
        )
    if not guard_rows:
        print("Nothing to run (all samples already completed or empty dataset).", flush=True)
        save_json_list(save_path, results)
        if n_skipped_no_response:
            print(
                f"Warning: {n_skipped_no_response} samples skipped (no response in JSON).",
                flush=True,
            )
        return

    chat_bs = resolve_chat_batch_size(cfg)
    chunk_size = chat_bs if chat_bs is not None else len(guard_rows)
    print(
        f"Running {len(guard_rows)} samples in chat chunks of {chunk_size} "
        f"(prepare_max_workers={mw!r}, chat_batch_size={chat_bs!r}) …",
        flush=True,
    )

    use_tqdm = bool(cfg.offline_chat.use_tqdm)
    llm = build_llm(cfg, model_path=guard_path, max_model_len=_guard_max_model_len(cfg))
    sp = sampling_params_guard(cfg)
    inner_tqdm = use_tqdm and len(guard_rows) <= chunk_size

    for start in range(0, len(guard_rows), chunk_size):
        row_chunk = guard_rows[start : start + chunk_size]
        pending = list(
            parallel_map_ordered(
                prepare_guard_messages_row,
                row_chunk,
                max_workers=mw,
                progress_desc="prepare_messages",
            )
        )
        if not pending:
            continue
        work = [(pid, msg) for pid, msg, _ in pending]
        pairs = run_chat_all(
            llm,
            work,
            sampling_params=sp,
            use_tqdm=inner_tqdm,
            chat_template_kwargs=None,
        )
        for (pid, text), (_, _, _q) in zip(pairs, pending, strict=True):
            if text is None:
                print(
                    f"[persona_id={pid}] Missing guard output; skip.",
                    flush=True,
                )
                continue
            base = dict(responses_by_id.get(pid, {"persona_id": pid}))
            base[guard_raw_key] = text
            base[guard_label_key] = _parse_guard_label(text)
            results.append(base)
        save_json_list(save_path, results)
    if n_skipped_no_response:
        print(
            f"Warning: {n_skipped_no_response} samples skipped (no response in JSON).",
            flush=True,
        )
    print(f"Done. Wrote {len(results)} rows to {save_path}", flush=True)


if __name__ == "__main__":
    main()
