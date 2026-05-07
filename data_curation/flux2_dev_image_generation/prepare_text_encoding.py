#!/usr/bin/env python3
"""Compute and save FLUX.2-dev text encodings for a sample set.

Loads samples the same way as ``data_curation/generate_image.py``, encodes
each prompt with **only** the text-encoder (Mistral3) portion of
``Flux2Pipeline`` — the transformer and VAE are never loaded — and writes a
``.pt`` file containing ``prompt_embeds`` and ``text_ids`` for every processed
sample.

Saved file layout (dict)::

    {
        "sample_indices": list[int],           # original indices in samples.json
        "prompt_embeds":  Tensor[N, L, D],     # float16/bfloat16, on CPU
        "text_ids":       Tensor[N, L, 4],     # float32, on CPU
        "run_name":       str,
        "samples_load_path": str,
    }

The companion script ``generate_image.py`` in this directory loads the file
and runs only the transformer + VAE, so the two heavy components never
co-reside in GPU memory.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from diffusers import Flux2Pipeline
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Stdio / helpers (mirrors generate_image.py)
# ---------------------------------------------------------------------------

def configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


configure_stdio()


def slugify(name: str) -> str:
    s = re.sub(r"\s+", "_", name.strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def ensure_sample_fields(sample: dict[str, Any]) -> dict[str, Any]:
    for field in (
        "persona_id",
        "explanation",
        "scp_category",
        "visual_input_description",
        "text_query",
        "offender_motivation",
        "harmful_intent",
        "persona_input",
        "raw_generation_response",
        "image_path",
    ):
        sample[field] = str(sample.get(field) or "")
    return sample


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, found {type(data).__name__}.")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected every entry in {path} to be a JSON object.")
    return data


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    key = (name or "").lower()
    if key not in mapping:
        raise ValueError(f"Unknown dtype {name!r}; expected one of {sorted(mapping)}.")
    return mapping[key]


def image_file_exists(repo_relative: str) -> bool:
    repo_relative = (repo_relative or "").strip()
    if not repo_relative:
        return False
    p = Path(repo_relative)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.is_file()


def _cfg_optional_path_str(cfg: DictConfig, key: str) -> str | None:
    if key not in cfg or OmegaConf.is_missing(cfg, key):
        return None
    raw = cfg[key]
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("null", "none"):
        return None
    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="../../configs/flux2_dev_image_generation",
    config_name="prepare_text_encoding",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    run_name = str(cfg.source_model_name).strip()
    if not run_name:
        raise ValueError("source_model_name must be a non-empty string.")

    # ------------------------------------------------------------------ #
    # Resolve sample paths
    # ------------------------------------------------------------------ #
    samples_source_s = _cfg_optional_path_str(cfg, "samples_source")
    if samples_source_s is None:
        raise ValueError("samples_source must be provided.")

    raw_path = Path(samples_source_s)
    if not raw_path.is_absolute():
        raw_path = REPO_ROOT / raw_path

    samples_save_s = _cfg_optional_path_str(cfg, "samples_save")
    if samples_save_s is not None:
        save_path = Path(samples_save_s)
        if not save_path.is_absolute():
            save_path = REPO_ROOT / save_path
    load_path = raw_path

    if not load_path.exists():
        raise FileNotFoundError(f"No samples file at {load_path}.")

    samples = [ensure_sample_fields(dict(s)) for s in load_json_list(load_path)]
    start = int(cfg.start_index or 0)
    end = len(samples)
    if cfg.max_samples not in (None, "null", "None"):
        end = min(end, start + int(cfg.max_samples))
    indices = range(start, end)

    # ------------------------------------------------------------------ #
    # Build work list (same skip_existing logic as generate_image.py)
    # ------------------------------------------------------------------ #
    skip_existing = bool(cfg.skip_existing)
    work: list[tuple[int, str]] = []
    for i in indices:
        sample = samples[i]
        prompt = (sample.get("visual_input_description") or "").strip()
        if not prompt:
            print(f"[{i}] Empty visual_input_description; skipping.", flush=True)
            continue
        existing_rel = str(sample.get("image_path") or "").strip()
        if skip_existing and existing_rel and image_file_exists(existing_rel):
            continue
        work.append((i, prompt))

    if not work:
        print("No samples to encode.", flush=True)
        return

    # ------------------------------------------------------------------ #
    # Resolve encodings save path
    # ------------------------------------------------------------------ #
    encodings_save_s = _cfg_optional_path_str(cfg, "encodings_save")
    if encodings_save_s is not None:
        encodings_save_path = Path(encodings_save_s)
        if not encodings_save_path.is_absolute():
            encodings_save_path = REPO_ROOT / encodings_save_path
    else:
        samples_stem = load_path.stem
        encodings_save_path = (
            REPO_ROOT
            / "output"
            / "text_encodings"
            / slugify(run_name)
            / f"{samples_stem}.pt"
        )

    print(f"Will save {len(work)} encodings to: {encodings_save_path}", flush=True)

    # ------------------------------------------------------------------ #
    # Load ONLY text_encoder + tokenizer — transformer and VAE are skipped
    # ------------------------------------------------------------------ #
    dtype = resolve_dtype(str(cfg.dtype))
    device = str(cfg.device)

    print(
        "Loading text_encoder + tokenizer "
        "(transformer=None, vae=None — skipped to save memory) …",
        flush=True,
    )
    pipe = Flux2Pipeline.from_pretrained(
        str(cfg.image_gen_model_path),
        torch_dtype=dtype,
        transformer=None,
        vae=None,
    )
    pipe.text_encoder = pipe.text_encoder.to(device)

    # ------------------------------------------------------------------ #
    # Encode in batches
    # ------------------------------------------------------------------ #
    batch_size = max(1, int(cfg.batch_size))
    max_sequence_length = int(getattr(cfg, "max_sequence_length", 512))

    all_prompt_embeds: list[torch.Tensor] = []
    all_text_ids: list[torch.Tensor] = []
    all_indices: list[int] = []

    with tqdm(total=len(work), desc="encode_text") as pbar:
        for b in range(0, len(work), batch_size):
            batch = work[b : b + batch_size]
            batch_indices = [idx for idx, _ in batch]
            batch_prompts = [p for _, p in batch]

            with torch.no_grad():
                prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=batch_prompts,
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=max_sequence_length,
                )
            # prompt_embeds: [B, seq_len, dim]  text_ids: [B, seq_len, 4]

            for j, idx in enumerate(batch_indices):
                all_prompt_embeds.append(prompt_embeds[j].cpu())
                all_text_ids.append(text_ids[j].cpu())
                all_indices.append(idx)

            pbar.update(len(batch))

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    encodings_save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_indices": all_indices,
        "prompt_embeds": torch.stack(all_prompt_embeds),  # [N, seq_len, dim]
        "text_ids": torch.stack(all_text_ids),            # [N, seq_len, 4]
        "run_name": run_name,
        "samples_load_path": str(load_path),
    }
    torch.save(payload, encodings_save_path)
    print(
        f"Saved {len(all_indices)} encodings → {encodings_save_path}",
        flush=True,
    )

    # empty the cache
    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
