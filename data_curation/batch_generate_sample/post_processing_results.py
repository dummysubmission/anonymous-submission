#!/usr/bin/env python3
"""Merge vLLM OpenAI-batch JSONL into the same JSON format as generate_sample.py."""

import json
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO_ROOT))

from data_curation.generate_sample import (  # noqa: E402
    SamplePayload,
    extract_non_thinking_section,
    get_save_path,
    load_intentions,
    load_json_list,
    parse_generation_payload,
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


@hydra.main(
    config_path="../../configs",
    config_name="generate_sample.yaml",
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

    save_path = get_save_path(cfg)
    intentions = load_intentions(cfg)
    intentions_by_id: dict[str, dict[str, Any]] = {}
    for intention in intentions:
        pid = str(intention.get("persona_id") or "").strip()
        if pid:
            intentions_by_id[pid] = intention

    if save_path.exists() and not bool(cfg.overwrite):
        results = load_json_list(save_path)
        print(f"Loaded {len(results)} existing rows from {save_path}")
    else:
        results = []

    completed_ids = {str(item.get("persona_id")) for item in results}
    rows = load_batch_results_jsonl(batch_results_path)

    added = 0
    skipped_existing = 0
    failed = 0

    for row in tqdm(rows, total=len(rows), desc="post_process"):
        persona_id = str(row.get("custom_id") or "").strip()
        if not persona_id:
            failed += 1
            continue
        if persona_id in completed_ids:
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
        parsed = parse_generation_payload(cleaned)
        if parsed is None:
            print(f"[persona_id={persona_id}] Failed to parse JSON payload; skip.", flush=True)
            failed += 1
            continue

        intention = intentions_by_id.get(persona_id)
        if intention is None:
            print(
                f"[persona_id={persona_id}] Not found in current intention slice; "
                "scp_category / offender_motivation / harmful_intent / persona_input may be empty.",
                flush=True,
            )
            scp_category = ""
            offender_motivation = ""
            harmful_intent = ""
            persona_input = ""
            mllm_use_case = ""
            explanation = ""
        else:
            scp_category = str(intention.get("scp_category", "") or "")
            offender_motivation = str(intention.get("offender_motivation", "") or "")
            harmful_intent = str(intention.get("harmful_intent", "") or "")
            persona_input = str(intention.get("persona_input", "") or "")
            mllm_use_case = str(intention.get("mllm_use_case", "") or "")
            explanation = str(intention.get("explanation", "") or "")

        output = SamplePayload(
            persona_id=persona_id,
            **parsed.model_dump(),
        )
        results.append(
            {
                **output.model_dump(),
                "scp_category": scp_category,
                "offender_motivation": offender_motivation,
                "harmful_intent": harmful_intent,
                "persona_input": persona_input,
                "mllm_use_case": mllm_use_case,
                "raw_generation_response": cleaned,
            }
        )
        completed_ids.add(persona_id)
        added += 1

    save_json_list(save_path, results)
    print(
        f"Done. Wrote {len(results)} total rows to {save_path} "
        f"(+{added} from batch, skipped_existing={skipped_existing}, failed={failed})."
    )


if __name__ == "__main__":
    main()
