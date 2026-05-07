#!/usr/bin/env python3
"""Merge vLLM OpenAI-batch JSONL results into the same JSON format as generate_intention_bank.py."""

import json
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO_ROOT))

from data_curation.generate_intention_bank import (  # noqa: E402
    HarmfulIntentBankEntry,
    build_persona_input,
    extract_non_thinking_section,
    get_save_path,
    load_json_list,
    load_persona_dataset,
    parse_bank_response,
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
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {path}"
                ) from exc
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


@hydra.main(
    config_path="../../configs",
    config_name="generate_intention_bank.yaml",
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
        raise FileNotFoundError(f"Batch results not found: {batch_results_path}")

    bank_size = int(cfg.bank_size)
    save_path = get_save_path(cfg)
    dataset = load_persona_dataset(cfg)

    samples_by_id: dict[str, dict[str, Any]] = {}
    for sample in dataset:
        pid = str(sample.get("uuid") or "").strip()
        if pid:
            samples_by_id[pid] = sample

    if save_path.exists() and not bool(cfg.overwrite):
        results = load_json_list(save_path)
        print(f"Loaded {len(results)} existing bank entries from {save_path}")
    else:
        results = []

    completed_persona_ids = {str(item.get("persona_id")) for item in results}
    rows = load_batch_results_jsonl(batch_results_path)

    added = 0
    skipped_existing = 0
    failed = 0

    for row in tqdm(rows, total=len(rows), desc="post_process"):
        persona_id = str(row.get("custom_id") or "").strip()
        if not persona_id:
            failed += 1
            continue
        if persona_id in completed_persona_ids:
            skipped_existing += 1
            continue

        raw_text = assistant_content_from_batch_row(row)
        if raw_text is None:
            print(
                f"[persona_id={persona_id}] Missing assistant content or batch error; skip.",
                flush=True,
            )
            failed += 1
            continue

        cleaned = extract_non_thinking_section(raw_text)
        bank = parse_bank_response(cleaned, bank_size)
        if bank is None:
            print(
                f"[persona_id={persona_id}] Failed to parse bank of {bank_size} valid "
                "entries; skip.",
                flush=True,
            )
            failed += 1
            continue

        sample = samples_by_id.get(persona_id)
        if sample is None:
            print(
                f"[persona_id={persona_id}] Not found in current persona slice; "
                "persona_input will be empty.",
                flush=True,
            )
            persona_input = ""
        else:
            persona_input = build_persona_input(sample)

        new_entries = [
            {
                **HarmfulIntentBankEntry(
                    persona_id=persona_id,
                    bank_index=idx,
                    **entry.model_dump(),
                ).model_dump(),
                "persona_input": persona_input,
                "raw_generation_response": cleaned,
            }
            for idx, entry in enumerate(bank)
        ]

        results.extend(new_entries)
        completed_persona_ids.add(persona_id)
        added += 1

    save_json_list(save_path, results)
    print(
        f"Done. Wrote {len(results)} total bank entries to {save_path} "
        f"(+{added} personas from batch, skipped_existing={skipped_existing}, "
        f"failed={failed})."
    )


if __name__ == "__main__":
    main()
