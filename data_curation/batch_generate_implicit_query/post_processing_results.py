#!/usr/bin/env python3
"""Merge vLLM OpenAI-batch JSONL into the same JSON format as generate_implicit_query.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO_ROOT))

from data_curation.generate_implicit_query import (  # noqa: E402
    extract_non_thinking_section,
    is_null_like,
    load_json_list,
    parse_implicit_query_payload,
    resolve_path,
    save_json_list,
)


def configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


configure_stdio()


def load_batch_results_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
    return rows


def assistant_content_from_batch_row(row: dict[str, Any]) -> str | None:
    err = row.get("error")
    if err not in (None, {}):
        return None
    response = row.get("response")
    if not isinstance(response, dict):
        return None
    body = response.get("body")
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if content is None:
        return None
    return str(content)


def load_samples_for_post(
    cfg: DictConfig, source_path: Path, save_path: Path
) -> list[dict[str, Any]]:
    """Match ``generate_implicit_query.main`` sample loading / init semantics."""
    if save_path.exists() and save_path != source_path and not bool(cfg.overwrite):
        samples = load_json_list(save_path)
        print(f"Resuming from {len(samples)} existing results at {save_path}", flush=True)
        return samples
    samples = load_json_list(source_path)
    print(f"Loaded {len(samples)} samples from {source_path}", flush=True)
    if save_path != source_path:
        save_json_list(save_path, samples)
        print(f"Initialized output at {save_path}", flush=True)
    return samples


@hydra.main(
    config_path="../../configs",
    config_name="generate_implicit_query.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    raw_path = getattr(cfg, "batch_results_path", None)
    if raw_path in (None, "", "null", "None"):
        raise ValueError(
            "Set batch_results_path=/path/to/vllm_batch_output.jsonl "
            "(Hydra: batch_results_path=... or +batch_results_path=...)."
        )
    batch_results_path = Path(str(raw_path))
    if not batch_results_path.is_file():
        raise FileNotFoundError(f"batch results not found: {batch_results_path}")

    source_path = resolve_path(
        cfg.samples_source, must_exist=True, label="samples_source"
    )
    if is_null_like(cfg.save_path):
        save_path = source_path
    else:
        save_path = resolve_path(cfg.save_path, label="save_path")

    samples = load_samples_for_post(cfg, source_path, save_path)
    overwrite = bool(cfg.overwrite)

    rows = load_batch_results_jsonl(batch_results_path)
    updated = 0
    skipped = 0
    failed = 0

    for row in tqdm(rows, total=len(rows), desc="post_process"):
        raw_id = str(row.get("custom_id") or "").strip()
        if not raw_id:
            failed += 1
            continue
        try:
            idx = int(raw_id)
        except ValueError:
            print(f"[custom_id={raw_id}] Expected integer index; skip.", flush=True)
            failed += 1
            continue

        if idx < 0 or idx >= len(samples):
            print(f"[index={idx}] Out of range for {len(samples)} samples; skip.", flush=True)
            failed += 1
            continue

        sample = samples[idx]
        persona_id = str(sample.get("persona_id") or "").strip() or f"index_{idx}"

        if sample.get("implicit_text_query") and not overwrite:
            skipped += 1
            continue

        raw_text = assistant_content_from_batch_row(row)
        if raw_text is None:
            print(
                f"[{idx} persona_id={persona_id}] Missing assistant content or batch error; skip.",
                flush=True,
            )
            failed += 1
            continue

        cleaned = extract_non_thinking_section(raw_text)
        parsed = parse_implicit_query_payload(cleaned)
        if parsed is None:
            print(
                f"[{idx} persona_id={persona_id}] Failed to parse JSON payload; skip.",
                flush=True,
            )
            failed += 1
            continue

        samples[idx] = {
            **sample,
            "implicit_text_query": parsed.implicit_text_query,
            "implicit_query_explanation": parsed.explanation,
        }
        updated += 1

    save_json_list(save_path, samples)
    print(
        f"Done. Saved {len(samples)} samples to {save_path} "
        f"(updated={updated}, skipped_existing={skipped}, failed={failed}).",
        flush=True,
    )


if __name__ == "__main__":
    main()
