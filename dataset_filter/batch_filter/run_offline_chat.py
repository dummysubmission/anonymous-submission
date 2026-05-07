#!/usr/bin/env python3
"""Offline dataset filter via ``vllm.LLM.chat`` (mirrors batch dataset audit pattern)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset_filter.batch_utils import get_batch_filter_save_path  # noqa: E402
from dataset_filter.client_utils import (  # noqa: E402
    extract_non_thinking_section,
    load_json_list,
    resolve_image_path,
    save_json_list,
)
from dataset_filter.dataset_utils import load_m3risk_dataset, samples_json_path_m3risk  # noqa: E402
from dataset_filter.filter import (  # noqa: E402
    build_filter_messages,
    filter_fields_from_payload,
    parse_intention_fidelity_payload,
    require_m3risk_benchmark,
)
from safety_eval.vllm_offline_chat import (  # noqa: E402
    build_llm,
    chat_template_kwargs_generate,
    configure_stdio,
    parallel_map_ordered,
    resolve_chat_batch_size,
    resolve_prepare_max_workers,
    run_chat_all,
    sampling_params_generate,
)

configure_stdio()


def _filter_indices(cfg: DictConfig, n: int) -> list[int]:
    start_index = int(getattr(cfg, "start_index", 0) or 0)
    max_samples = getattr(cfg, "max_samples", None)
    if max_samples not in (None, "null", "None"):
        end_index = min(n, start_index + int(max_samples))
        return list(range(start_index, end_index))
    if start_index > 0:
        return list(range(start_index, n))
    return list(range(n))


def _prepare_filter_row(
    row: tuple[int, dict[str, Any], Path],
) -> dict[str, Any] | None:
    idx, sample, repo_root = row
    sample_id = str(sample.get("id") or "").strip() or f"index_{idx}"
    mi = (sample.get("malicious_intent") or "").strip()
    if not mi:
        return None
    img_path = resolve_image_path(sample, repo_root=repo_root)
    if img_path is None:
        return None
    try:
        from PIL import Image

        img = Image.open(img_path)
        img.load()
    except Exception:
        return None
    try:
        messages = build_filter_messages(sample, img)
    except ValueError:
        return None
    return {"sample_id": sample_id, "messages": messages, "idx": idx}


@hydra.main(
    config_path="../../configs",
    config_name="dataset_filter_batch",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    benchmark = require_m3risk_benchmark(cfg)
    samples_path = samples_json_path_m3risk(repo_root=REPO_ROOT)
    if not samples_path.is_file():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    save_path = get_batch_filter_save_path(cfg, REPO_ROOT)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    overwrite = bool(cfg.overwrite)
    if save_path.exists() and not overwrite:
        samples = load_json_list(save_path)
        print(
            f"Resume mode: loaded {len(samples)} rows from {save_path}",
            flush=True,
        )
    else:
        samples = load_m3risk_dataset(benchmark=benchmark, repo_root=REPO_ROOT)
        print(
            f"Loaded {len(samples)} samples for benchmark={benchmark!r} from {samples_path}",
            flush=True,
        )

    indices = _filter_indices(cfg, len(samples))
    candidates: list[tuple[int, dict[str, Any], Path]] = []
    for idx in indices:
        sample = samples[idx]
        if "filter_malicious_intention_fidelity" in sample and not overwrite:
            continue
        candidates.append((idx, dict(sample), REPO_ROOT))

    if not candidates:
        print(
            "Nothing to run (all selected samples already filtered or empty selection).",
            flush=True,
        )
        save_json_list(save_path, samples)
        return

    mw = resolve_prepare_max_workers(cfg)
    prepared_raw = parallel_map_ordered(
        _prepare_filter_row,
        candidates,
        max_workers=mw,
        progress_desc="prepare_filter_messages",
    )
    prepared = [p for p in prepared_raw if p is not None]
    if len(prepared) < len(candidates):
        print(
            f"Skipping {len(candidates) - len(prepared)} samples (missing image, query, or intent).",
            flush=True,
        )
    if not prepared:
        print("No valid rows after image/query checks.", flush=True)
        save_json_list(save_path, samples)
        return

    model_path = str(cfg.model.model_name_or_path)
    chat_bs = resolve_chat_batch_size(cfg)
    chunk_size = chat_bs if chat_bs is not None else len(prepared)
    print(
        f"Running {len(prepared)} samples in chat chunks of {chunk_size} "
        f"(prepare_max_workers={mw!r}, chat_batch_size={chat_bs!r}) …",
        flush=True,
    )

    use_tqdm = bool(cfg.offline_chat.use_tqdm)
    llm = build_llm(cfg, model_path=model_path)
    sp = sampling_params_generate(cfg)
    ctkw = chat_template_kwargs_generate(cfg)
    inner_tqdm = use_tqdm and len(prepared) <= chunk_size

    for start in range(0, len(prepared), chunk_size):
        chunk = prepared[start : start + chunk_size]
        work = [(p["sample_id"], p["messages"]) for p in chunk]
        pairs = run_chat_all(
            llm,
            work,
            sampling_params=sp,
            use_tqdm=inner_tqdm,
            chat_template_kwargs=ctkw,
        )
        for (sample_id, text), meta in zip(pairs, chunk, strict=True):
            idx = int(meta["idx"])
            if text is None:
                print(
                    f"[id={sample_id}] Missing assistant output; skip.",
                    flush=True,
                )
                continue
            cleaned = extract_non_thinking_section(text)
            parsed = parse_intention_fidelity_payload(cleaned)
            if parsed is None:
                print(
                    f"[id={sample_id}] Failed to parse filter JSON; leaving sample unchanged.",
                    flush=True,
                )
                continue
            row = samples[idx]
            samples[idx] = {**row, **filter_fields_from_payload(parsed)}
            if bool(cfg.verbose):
                print(
                    f"[id={sample_id}] filter_keep={samples[idx].get('filter_keep')}",
                    flush=True,
                )
        save_json_list(save_path, samples)

    print(f"Done. Wrote {len(samples)} rows to {save_path}", flush=True)


if __name__ == "__main__":
    main()
