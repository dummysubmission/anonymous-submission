#!/usr/bin/env python3
"""Generate images using FLUX.2-dev with pre-computed text encodings.

Loads text encodings produced by ``prepare_text_encoding.py`` and runs only the
transformer + VAE portion of ``Flux2Pipeline``.  The text encoder (Mistral3)
is never loaded, so the 7 B text encoder and the ~12 B transformer never
co-reside in GPU memory.

Output format and file layout are **identical** to
``data_curation/generate_image.py``: PNGs under
``data/sample_images/ours_iter_3/<run>/``, curated JSON at
``data/ours_iter_3/<run>/samples.json``.

Usage::

    python data_curation/flux2_dev_image_generation/generate_image.py \\
        source_model_name=<run> \\
        encodings_path=output/text_encodings/<run>/samples.pt \\
        samples_source=output/generate_sample/<run>/samples.json
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
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Stdio / helpers (mirrors generate_image.py exactly)
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


def image_output_path(images_dir: Path, sample: dict[str, Any], sample_index: int) -> Path:
    sample_id = str(sample.get("persona_id") or sample_index)
    return images_dir / f"{slugify(sample_id)}.png"


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


def process_and_save_image(image: Image.Image, output_path: Path) -> None:
    """Same as ``_process_and_save_image`` in ``data_curation_interface/server.py``."""
    mode = "RGBA" if ("A" in image.mode) else "RGB"
    image = image.convert(mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, found {type(data).__name__}.")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected every entry in {path} to be a JSON object.")
    return data


def write_json(path: Path, obj: Any) -> None:
    """Match ``_write_json`` in the curation server (iteration-3 saves)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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
# Diffusion helpers
# ---------------------------------------------------------------------------

# _WorkItem = (sample_index, sample_dict, prompt_embeds[1,L,D], text_ids[1,L,4])
_WorkItem = tuple[int, dict[str, Any], torch.Tensor, torch.Tensor]


def _call_pipe(
    pipe: Flux2Pipeline,
    prompt_embeds: torch.Tensor,  # [B, L, D]
    generators: list[torch.Generator],
    *,
    height: int,
    width: int,
    guidance_scale: float,
    num_inference_steps: int,
) -> list[Image.Image]:
    out = pipe(
        prompt_embeds=prompt_embeds,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generators if len(generators) > 1 else generators[0],
    )
    return list(out.images)


def generate_images_for_chunk(
    pipe: Flux2Pipeline,
    chunk: list[_WorkItem],
    *,
    device: str,
    base_seed: int,
    per_index_seed: bool,
    height: int,
    width: int,
    guidance_scale: float,
    num_inference_steps: int,
    strict: bool,
) -> list[tuple[int, Image.Image | None, str | None]]:
    """Run diffusion for one chunk using pre-computed embeddings.

    Returns ``(index, image_or_None, error_or_None)`` per item — same
    contract as ``generate_images_for_chunk`` in ``data_curation/generate_image.py``.
    """
    generators = [
        torch.Generator(device=device).manual_seed(
            base_seed + (idx if per_index_seed else 0)
        )
        for idx, _, _, _ in chunk
    ]
    # Stack per-sample [1,L,D] → [B,L,D] (all padded to the same max_sequence_length)
    batch_prompt_embeds = torch.cat([pe for _, _, pe, _ in chunk], dim=0).to(device)

    if len(chunk) == 1:
        idx = chunk[0][0]
        try:
            images = _call_pipe(
                pipe,
                batch_prompt_embeds,
                generators,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            return [(idx, images[0], None)]
        except Exception as exc:
            if strict:
                raise
            return [(idx, None, str(exc))]

    try:
        images = _call_pipe(
            pipe,
            batch_prompt_embeds,
            generators,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        return [(chunk[i][0], images[i], None) for i in range(len(chunk))]
    except Exception as batch_exc:
        if strict:
            raise
        print(
            f"Batch of {len(chunk)} failed ({batch_exc}); retrying one-by-one.",
            flush=True,
        )
        results: list[tuple[int, Image.Image | None, str | None]] = []
        for (idx, _, pe, _), gen in zip(chunk, generators):
            try:
                out_imgs = _call_pipe(
                    pipe,
                    pe,
                    [gen],
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                )
                results.append((idx, out_imgs[0], None))
            except Exception as exc:
                results.append((idx, None, str(exc)))
        return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="../../configs/flux2_dev_image_generation",
    config_name="generate_image",
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
    else:
        save_path = raw_path

    load_path = raw_path

    if not load_path.exists():
        raise FileNotFoundError(f"No samples file at {load_path}.")

    images_dir_s = _cfg_optional_path_str(cfg, "images_dir")
    if images_dir_s is not None:
        images_dir = Path(images_dir_s)
        if not images_dir.is_absolute():
            images_dir = REPO_ROOT / images_dir
    else:
        images_dir = save_path.parent / "images"

    samples = [ensure_sample_fields(dict(s)) for s in load_json_list(load_path)]

    # A chunk save file (written by a previous run of this script) carries a
    # _source_index field on every sample so that enc_lookup keys (which are
    # indexed relative to the original source file) can be resolved correctly.
    # When loading such a file, process all items from position 0 instead of
    # applying start_index / max_samples (which would be out-of-range).
    is_chunk = bool(samples and "_source_index" in samples[0])
    if is_chunk:
        start, end = 0, len(samples)
    else:
        start = int(cfg.start_index or 0)
        end = len(samples)
        if cfg.max_samples not in (None, "null", "None"):
            end = min(end, start + int(cfg.max_samples))
    indices = range(start, end)

    # ------------------------------------------------------------------ #
    # Load pre-computed text encodings
    # ------------------------------------------------------------------ #
    encodings_path_s = _cfg_optional_path_str(cfg, "encodings_path")
    if encodings_path_s is None:
        raise ValueError(
            "encodings_path must be set. Run prepare_text_encoding.py first."
        )
    encodings_path = Path(encodings_path_s)
    if not encodings_path.is_absolute():
        encodings_path = REPO_ROOT / encodings_path
    if not encodings_path.exists():
        raise FileNotFoundError(f"Encodings file not found: {encodings_path}")

    print(f"Loading encodings from {encodings_path} …", flush=True)
    encodings = torch.load(encodings_path, map_location="cpu", weights_only=True)

    # Build lookup: sample_index → (prompt_embeds[1,L,D], text_ids[1,L,4])
    enc_lookup: dict[int, tuple[torch.Tensor, torch.Tensor]] = {
        int(idx): (
            encodings["prompt_embeds"][j].unsqueeze(0),  # [1, L, D]
            encodings["text_ids"][j].unsqueeze(0),       # [1, L, 4]
        )
        for j, idx in enumerate(encodings["sample_indices"])
    }
    print(f"Loaded {len(enc_lookup)} encodings.", flush=True)

    # ------------------------------------------------------------------ #
    # Build work list
    # ------------------------------------------------------------------ #
    dtype = resolve_dtype(str(cfg.dtype))
    device = str(cfg.device)
    strict = bool(cfg.strict)
    skip_existing = bool(cfg.skip_existing)
    per_index_seed = bool(cfg.per_index_seed)
    base_seed = int(cfg.seed)
    batch_size = max(1, int(cfg.batch_size))

    work: list[_WorkItem] = []
    for i in indices:
        sample = samples[i]
        prompt = (sample.get("visual_input_description") or "").strip()
        if not prompt:
            msg = f"[{i}] Empty visual_input_description; skipping."
            print(msg, flush=True)
            if strict:
                raise ValueError(msg)
            continue

        existing_rel = str(sample.get("image_path") or "").strip()
        if skip_existing and existing_rel and image_file_exists(existing_rel):
            continue

        # Chunk save files carry _source_index so we can map back to enc_lookup,
        # which is keyed by position in the original source file.
        enc_key = int(sample.get("_source_index", i))
        if enc_key not in enc_lookup:
            msg = f"[{i}] No encoding found (source_index={enc_key}) in {encodings_path.name}; skipping."
            print(msg, flush=True)
            if strict:
                raise ValueError(msg)
            continue

        if not bool(sample.get("include", True)):
            print(f"[{i}] Exclude flag is set; skipping.", flush=True)
            continue

        pe, ti = enc_lookup[enc_key]
        work.append((i, sample, pe.to(dtype), ti))

    height = int(cfg.height)
    width = int(cfg.width)
    guidance_scale = float(cfg.guidance_scale)
    num_inference_steps = int(cfg.num_inference_steps)

    # ------------------------------------------------------------------ #
    # Load pipeline WITHOUT text_encoder / tokenizer
    # ------------------------------------------------------------------ #
    print(
        "Loading transformer + VAE + scheduler "
        "(text_encoder=None, tokenizer=None — skipped to save memory) …",
        flush=True,
    )
    pipe = Flux2Pipeline.from_pretrained(
        str(cfg.image_gen_model_path),
        torch_dtype=dtype,
        text_encoder=None,
        tokenizer=None,
    )
    pipe.enable_model_cpu_offload()

    # ------------------------------------------------------------------ #
    # Generate
    # ------------------------------------------------------------------ #
    with tqdm(total=len(work), desc="generate_image") as pbar:
        for b in range(0, len(work), batch_size):
            chunk = work[b : b + batch_size]
            results = generate_images_for_chunk(
                pipe,
                chunk,
                device=device,
                base_seed=base_seed,
                per_index_seed=per_index_seed,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strict=strict,
            )

            for idx, image, err in results:
                if err is not None:
                    print(f"[{idx}] Flux inference failed: {err}", flush=True)
                    if strict:
                        raise RuntimeError(err)
                    pbar.update(1)
                    continue
                assert image is not None
                sample = samples[idx]
                img_path = image_output_path(images_dir, sample, idx)
                process_and_save_image(image, img_path)
                sample["image_path"] = str(img_path.relative_to(REPO_ROOT))
                samples[idx] = sample
                if save_path == raw_path:
                    # In-place update: write the full list unchanged.
                    write_json(save_path, samples)
                else:
                    # Chunk output: write only the processed slice, tagging each
                    # sample with its position in the original source file so
                    # enc_lookup can be resolved correctly on resume.
                    write_json(
                        save_path,
                        [
                            {**samples[j], "_source_index": int(samples[j].get("_source_index", j))}
                            for j in range(start, end)
                        ],
                    )
                print(f"[{idx}] Wrote image path to {save_path.relative_to(REPO_ROOT)}")
                print(f"[{idx}] Wrote {sample['image_path']}", flush=True)
                pbar.update(1)

    print(f"Done. Curated samples: {save_path.relative_to(REPO_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
