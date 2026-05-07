#!/usr/bin/env python3
"""Offline judge labeling via ``vllm.LLM.chat`` (no JSONL batch files)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from safety_eval.batch_utils import (  # noqa: E402
    get_judge_save_path,
    get_response_path,
)
from safety_eval.dataset_utils import load_ours_dataset  # noqa: E402
from safety_eval.utils import (  # noqa: E402
    JUDGE_STRONGREJECT_MAX_VALID_ATTEMPTS,
    is_strongreject_judge_parsed_conformant,
    judge_key_prefix,
    load_json_list,
    merge_judge_rows_by_persona_id,
    normalize_judge_rubric,
    parse_judge_output_for_rubric,
    print_judge_summary,
    save_json_list,
)
from safety_eval.vllm_offline_chat import (  # noqa: E402
    build_llm,
    configure_stdio,
    parallel_map_ordered,
    prepare_judge_messages_row,
    resolve_chat_batch_size,
    resolve_prepare_max_workers,
    run_chat_all,
    sampling_params_judge,
    chat_template_kwargs_generate,
)

configure_stdio()


def _judge_max_model_len(cfg: DictConfig) -> int | None:
    raw = getattr(cfg.vllm, "judge_max_model_len", None)
    if raw in (None, "", "null", "None", "auto"):
        return None
    return int(raw)


@hydra.main(
    config_path="../../configs",
    config_name="safety_eval_judge_label.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    judge_model_path = getattr(cfg.judge, "model_name_or_path", None)
    if judge_model_path in (None, "", "null", "None"):
        raise ValueError(
            "judge.model_name_or_path must be set for offline chat inference. "
            "API-only judges are not supported by this script."
        )

    save_path = get_judge_save_path(cfg, REPO_ROOT)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    judge_key = judge_key_prefix(cfg.judge)
    judge_raw_key = f"{judge_key}_judge_raw"
    judge_parsed_key = f"{judge_key}_judge"
    rubric = normalize_judge_rubric(getattr(cfg.judge, "rubric", "strongreject"))

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
    raw_sys = getattr(cfg.judge, "system_prompt", None)
    if raw_sys in (None, "", "null", "None"):
        system_prompt_val: str | None = None
    else:
        system_prompt_val = str(raw_sys)

    if save_path.exists() and not bool(cfg.overwrite):
        loaded = merge_judge_rows_by_persona_id(
            load_json_list(save_path), judge_parsed_key
        )
        judged_by_pid: dict[str, dict[str, Any]] = {
            str(r.get("persona_id")): r
            for r in loaded
            if r.get("persona_id")
        }
        completed_ids = {
            pid
            for pid, row in judged_by_pid.items()
            if is_strongreject_judge_parsed_conformant(row.get(judge_parsed_key))
        }
        print(
            f"Resume mode: {len(completed_ids)}/{len(judged_by_pid)} persona_id(s) "
            f"with conformant {judge_parsed_key} in {save_path} "
            f"({len(loaded)} row(s) after deduplication by persona_id)",
            flush=True,
        )
    else:
        judged_by_pid = {}
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
        return ("ok", (sample, response_text, system_prompt_val, rubric))

    select_out = parallel_map_ordered(
        select_row,
        list(range(len(ds))),
        max_workers=mw,
        progress_desc="select_samples",
    )
    judge_rows: list[tuple[dict[str, Any], str, str | None, str]] = []
    n_skipped_no_response = 0
    n_missing_pid = 0
    for tag, payload in select_out:
        if tag == "ok":
            judge_rows.append(payload)
        elif tag == "no_response":
            n_skipped_no_response += 1
        elif tag == "no_pid":
            n_missing_pid += 1
    print(f"n_skipped_no_response: {n_skipped_no_response}")
    print(f"n_missing_pid: {n_missing_pid}")
    print(f"judge_rows: {len(judge_rows)}")

    if n_missing_pid:
        print(
            f"Skipping {n_missing_pid} samples with missing persona_id.",
            flush=True,
        )
    if not judge_rows:
        print("Nothing to run (all samples already completed or empty dataset).", flush=True)
        save_json_list(save_path, list(judged_by_pid.values()))
        if n_skipped_no_response:
            print(
                f"Warning: {n_skipped_no_response} samples skipped (no response in JSON).",
                flush=True,
            )
        print_judge_summary(
            list(judged_by_pid.values()),
            title="Judge summary (batch_judge_label/run_offline_chat)",
        )
        return

    chat_bs = resolve_chat_batch_size(cfg)
    chunk_size = chat_bs if chat_bs is not None else len(judge_rows)
    print(
        f"Running {len(judge_rows)} samples in chat chunks of {chunk_size} "
        f"(prepare_max_workers={mw!r}, chat_batch_size={chat_bs!r}) …",
        flush=True,
    )

    use_tqdm = bool(cfg.offline_chat.use_tqdm)
    llm = build_llm(
        cfg,
        model_path=str(judge_model_path),
        max_model_len=_judge_max_model_len(cfg),
    )
    sp = sampling_params_judge(cfg)
    ctkw = chat_template_kwargs_generate(cfg)
    inner_tqdm = use_tqdm and len(judge_rows) <= chunk_size

    max_valid_attempts = max(
        1,
        int(
            getattr(
                cfg.judge,
                "max_valid_parse_attempts",
                JUDGE_STRONGREJECT_MAX_VALID_ATTEMPTS,
            )
        ),
    )

    for start in range(0, len(judge_rows), chunk_size):
        row_chunk = judge_rows[start : start + chunk_size]
        pending = list(
            parallel_map_ordered(
                prepare_judge_messages_row,
                row_chunk,
                max_workers=mw,
                progress_desc="prepare_messages",
            )
        )
        if not pending:
            continue
        all_chunk_pids = {str(p) for p, _, _ in pending}
        done_pids: set[str] = set()
        last_out: dict[str, tuple[str | None, Any]] = {}
        inflight: list[tuple[str, list[dict[str, Any]], str]] = list(pending)
        for round_idx in range(max_valid_attempts):
            if not inflight:
                break
            work = [(p_id, msg) for p_id, msg, _q in inflight]
            # Match original: tqdm only for the first batched vLLM call in this sub-chunk.
            use_tqdm_for_batch = bool(inner_tqdm) and (round_idx == 0)
            pairs = run_chat_all(
                llm,
                work,
                sampling_params=sp,
                use_tqdm=use_tqdm_for_batch,
                chat_template_kwargs=ctkw,
            )
            new_inflight: list[tuple[str, list[dict[str, Any]], str]] = []
            for (pid, text), pend in zip(pairs, inflight, strict=True):
                stext = (text is not None) and (str(text).strip() != "")
                if stext:
                    parsed = parse_judge_output_for_rubric(str(text), rubric)
                    last_out[pid] = (text, parsed)
                else:
                    parsed = None
                if stext and is_strongreject_judge_parsed_conformant(parsed):
                    base = dict(responses_by_id.get(pid, {"persona_id": pid}))
                    base[judge_raw_key] = text
                    base[judge_parsed_key] = parsed
                    judged_by_pid[pid] = base
                    done_pids.add(pid)
                elif round_idx + 1 < max_valid_attempts:
                    new_inflight.append(pend)
            inflight = [x for x in new_inflight if x[0] not in done_pids]
        for pid in all_chunk_pids - done_pids:
            if pid not in last_out:
                print(
                    f"[persona_id={pid}] Missing judge output after {max_valid_attempts} valid attempt(s); skip.",
                    flush=True,
                )
                continue
            text, parsed = last_out[pid]
            if text is None or str(text).strip() == "":
                print(
                    f"[persona_id={pid}] Missing judge output; skip.",
                    flush=True,
                )
                continue
            base = dict(responses_by_id.get(pid, {"persona_id": pid}))
            base[judge_raw_key] = text
            base[judge_parsed_key] = parsed
            judged_by_pid[pid] = base
            if not is_strongreject_judge_parsed_conformant(parsed):
                print(
                    f"Warning: [persona_id={pid}] non-conformant parse after {max_valid_attempts} valid attempt(s); "
                    f"keeping last model output for this row.",
                    flush=True,
                )
        save_json_list(save_path, list(judged_by_pid.values()))
    if n_skipped_no_response:
        print(
            f"Warning: {n_skipped_no_response} samples skipped (no response in JSON).",
            flush=True,
        )
    n_out = len(judged_by_pid)
    print(f"Done. Wrote {n_out} row(s) to {save_path}", flush=True)
    print_judge_summary(
        list(judged_by_pid.values()),
        title="Judge summary (batch_judge_label/run_offline_chat)",
    )


if __name__ == "__main__":
    main()
