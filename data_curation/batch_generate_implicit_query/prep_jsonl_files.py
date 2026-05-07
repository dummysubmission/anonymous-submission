#!/usr/bin/env python3
"""Build OpenAI-compatible offline batch JSONL for Stage 5 implicit query generation."""

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
    build_generation_messages,
    is_null_like,
    load_json_list,
    pil_image_to_data_url,
    resolve_image_path,
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


def default_batch_input_path(save_path: Path) -> Path:
    return save_path.parent / "openai_batch" / f"{save_path.stem}_batch.jsonl"


def resolve_batch_input_path(cfg: DictConfig, save_path: Path) -> Path:
    raw = getattr(cfg, "batch_input_path", None)
    if raw not in (None, "", "null", "None"):
        return Path(str(raw))
    return default_batch_input_path(save_path)


def load_samples_for_prep(
    cfg: DictConfig, source_path: Path, save_path: Path
) -> list[dict[str, Any]]:
    """Match ``generate_implicit_query.main`` sample loading / init semantics."""
    if save_path.exists() and save_path != source_path and not bool(cfg.overwrite):
        return load_json_list(save_path)
    samples = load_json_list(source_path)
    if save_path != source_path:
        save_json_list(save_path, samples)
        print(f"Initialized output at {save_path}", flush=True)
    return samples


def compute_indices(n_samples: int, cfg: DictConfig) -> list[int]:
    start_index = int(getattr(cfg, "start_index", 0) or 0)
    max_samples = getattr(cfg, "max_samples", None)
    if max_samples not in (None, "null", "None"):
        end_index = min(n_samples, start_index + int(max_samples))
        return list(range(start_index, end_index))
    if start_index > 0:
        return list(range(start_index, n_samples))
    return list(range(n_samples))


def build_chat_completion_line(
    *,
    custom_id: str,
    model: str,
    messages: list[dict[str, Any]],
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
    config_name="generate_implicit_query.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    from PIL import Image

    source_path = resolve_path(
        cfg.samples_source, must_exist=True, label="samples_source"
    )
    if is_null_like(cfg.save_path):
        save_path = source_path
    else:
        save_path = resolve_path(cfg.save_path, label="save_path")

    samples = load_samples_for_prep(cfg, source_path, save_path)
    print(f"Loaded {len(samples)} samples from {source_path}")
    batch_path = resolve_batch_input_path(cfg, save_path)
    batch_path.parent.mkdir(parents=True, exist_ok=True)

    model = str(cfg.model.model_name_or_path)
    max_tokens = int(cfg.sampling_params.max_new_tokens)
    temperature = float(cfg.temperature)
    enable_thinking = bool(cfg.enable_thinking)
    overwrite = bool(cfg.overwrite)
    require_stages = bool(cfg.require_prior_stages)

    indices = compute_indices(len(samples), cfg)
    n_written = 0
    with batch_path.open("w", encoding="utf-8") as out_f:
        for idx in tqdm(indices, total=len(indices), desc="prep_batch"):
            sample = samples[idx]
            persona_id = str(sample.get("persona_id") or "").strip()

            if sample.get("implicit_text_query") and not overwrite:
                continue

            image_path = resolve_image_path(sample)
            if image_path is None:
                print(
                    f"[{idx} persona_id={persona_id}] No image found; skipping batch line.",
                    flush=True,
                )
                continue

            try:
                img = Image.open(image_path)
            except Exception as exc:
                print(
                    f"[{idx} persona_id={persona_id}] Failed to load image "
                    f"{image_path}: {exc}",
                    flush=True,
                )
                continue

            data_url = pil_image_to_data_url(img)
            messages = build_generation_messages(sample, data_url)
            line = build_chat_completion_line(
                custom_id=str(idx),
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
        f"(model={model}, save_path={save_path}).",
        flush=True,
    )


if __name__ == "__main__":
    main()
