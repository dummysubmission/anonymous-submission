#!/usr/bin/env python3
"""Build OpenAI-compatible offline batch JSONL for Stage 2 sample generation."""

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
    build_generation_messages,
    get_save_path,
    load_intentions,
    load_json_list,
)


def configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


configure_stdio()


def default_batch_input_path(save_path: Path) -> Path:
    return save_path.parent / "openai_batch" / f"{save_path.stem}_batch.jsonl"


def resolve_batch_input_path(cfg: DictConfig, save_path: Path) -> Path:
    raw = getattr(cfg, "batch_input_path", None)
    if raw not in (None, "", "null", "None"):
        return Path(str(raw))
    return default_batch_input_path(save_path)


def build_chat_completion_line(
    *,
    custom_id: str,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


@hydra.main(
    config_path="../../configs",
    config_name="generate_sample.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    save_path = get_save_path(cfg)
    batch_path = resolve_batch_input_path(cfg, save_path)
    batch_path.parent.mkdir(parents=True, exist_ok=True)

    intentions = load_intentions(cfg)
    model = str(cfg.model.model_name_or_path)
    max_tokens = int(cfg.sampling_params.max_new_tokens)
    temperature = float(cfg.temperature)
    enable_thinking = bool(cfg.enable_thinking)

    completed_ids: set[str] = set()
    if save_path.exists() and not bool(cfg.overwrite):
        existing = load_json_list(save_path)
        completed_ids = {str(item.get("persona_id")) for item in existing}
        print(
            f"Resume mode: {len(completed_ids)} persona_ids already in {save_path}; "
            "they will be omitted from the batch file."
        )

    n_written = 0
    with batch_path.open("w", encoding="utf-8") as out_f:
        for intention in tqdm(intentions, total=len(intentions)):
            persona_id = str(intention.get("persona_id") or "").strip()
            if not persona_id:
                print("Skipping intention with missing persona_id.", flush=True)
                continue
            if persona_id in completed_ids:
                continue

            messages = build_generation_messages(intention)
            line = build_chat_completion_line(
                custom_id=persona_id,
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                enable_thinking=enable_thinking,
            )
            out_f.write(json.dumps(line, ensure_ascii=False) + "\n")
            n_written += 1

    print(
        f"Wrote {n_written} batch requests to {batch_path} "
        f"(model={model}, save_path target={save_path})."
    )


if __name__ == "__main__":
    main()
