#!/usr/bin/env python3
"""Generate iteration-3 grounded images with a local diffusion model.

Loads Stage-2 samples from ``output/generate_sample/<source_model_name>/samples.json``,
writes the curated copy to ``data/ours_iter_3/<source_model_name>/samples.json``, and
saves PNGs under ``data/sample_images/ours_iter_3/<source_model_name>/``, matching the
data curation interface iteration-3 upload behavior.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm
import torch
from diffusers import Flux2Pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent

ITERATION_3_SOURCE_DIR_REL = Path("output") / "generate_sample"
ITERATION_3_SAVE_DIR_REL = Path("data") / "ours_iter_3"


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


def iter3_source_path(run_name: str) -> Path:
    return REPO_ROOT / ITERATION_3_SOURCE_DIR_REL / run_name / "samples.json"


def iter3_save_path(run_name: str) -> Path:
    return REPO_ROOT / ITERATION_3_SAVE_DIR_REL / run_name / "samples.json"


def iter3_image_output_path(run_name: str, sample: dict[str, Any], sample_index: int) -> Path:
    sample_id = str(sample.get("persona_id") or sample_index)
    return (
        REPO_ROOT
        / "data"
        / "sample_images"
        / "ours_iter_3"
        / slugify(run_name)
        / f"{slugify(sample_id)}.png"
    )


def iter3_ensure_fields(sample: dict[str, Any]) -> dict[str, Any]:
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


def resolve_dtype(name: str):
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


def _call_pipe_batch(
    pipe: Flux2Pipeline,
    prompts: list[str],
    generators: list[torch.Generator],
    *,
    height: int,
    width: int,
    guidance_scale: float,
    num_inference_steps: int,
) -> list[Image.Image]:
    out = pipe(
        prompt=prompts,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generators,
    )
    return list(out.images)


def _call_pipe_single(
    pipe: Flux2Pipeline,
    prompt: str,
    generator: torch.Generator,
    *,
    height: int,
    width: int,
    guidance_scale: float,
    num_inference_steps: int,
) -> Image.Image:
    out = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )
    return out.images[0]


def generate_images_for_chunk(
    pipe: Flux2Pipeline,
    chunk: list[tuple[int, dict[str, Any], str]],
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
    """Run diffusion for one chunk. Returns ``(index, image or None, error or None)`` per item."""
    prompts = [p for _, _, p in chunk]
    generators = [
        torch.Generator(device=device).manual_seed(
            base_seed + (idx if per_index_seed else 0)
        )
        for idx, _, _ in chunk
    ]
    height_i = int(height)
    width_i = int(width)
    gs = float(guidance_scale)
    steps = int(num_inference_steps)

    if len(chunk) == 1:
        idx, _, pr = chunk[0]
        try:
            img = _call_pipe_single(
                pipe,
                pr,
                generators[0],
                height=height_i,
                width=width_i,
                guidance_scale=gs,
                num_inference_steps=steps,
            )
            return [(idx, img, None)]
        except Exception as exc:
            if strict:
                raise
            return [(idx, None, str(exc))]

    try:
        images = _call_pipe_batch(
            pipe,
            prompts,
            generators,
            height=height_i,
            width=width_i,
            guidance_scale=gs,
            num_inference_steps=steps,
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
        for (idx, _, pr), gen in zip(chunk, generators):
            try:
                img = _call_pipe_single(
                    pipe,
                    pr,
                    gen,
                    height=height_i,
                    width=width_i,
                    guidance_scale=gs,
                    num_inference_steps=steps,
                )
                results.append((idx, img, None))
            except Exception as exc:
                results.append((idx, None, str(exc)))
        return results


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


@hydra.main(config_path="../configs", config_name="generate_image", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    run_name = str(cfg.source_model_name).strip()
    if not run_name:
        raise ValueError("source_model_name must be a non-empty string.")

    default_source = iter3_source_path(run_name)
    default_save = iter3_save_path(run_name)

    samples_source_s = _cfg_optional_path_str(cfg, "samples_source")
    samples_save_s = _cfg_optional_path_str(cfg, "samples_save")

    if samples_source_s is not None:
        raw_path = Path(samples_source_s)
        if not raw_path.is_absolute():
            raw_path = REPO_ROOT / raw_path
        if samples_save_s is not None:
            save_path = Path(samples_save_s)
            if not save_path.is_absolute():
                save_path = REPO_ROOT / save_path
        else:
            save_path = REPO_ROOT / "data" / "ours_iter_3" / run_name / raw_path.name
        load_path = save_path if save_path.exists() else raw_path
        expect_msg_path = raw_path
    else:
        save_path = default_save
        load_path = save_path if save_path.exists() else default_source
        expect_msg_path = default_source

    if not load_path.exists():
        raise FileNotFoundError(
            f"No samples file at {load_path}. Expected source at {expect_msg_path}."
        )

    samples = [iter3_ensure_fields(dict(s)) for s in load_json_list(load_path)]
    start = int(cfg.start_index or 0)
    end = len(samples)
    if cfg.max_samples not in (None, "null", "None"):
        end = min(end, start + int(cfg.max_samples))
    indices = range(start, end)

    dtype = resolve_dtype(str(cfg.dtype))
    device = str(cfg.device)

    pipe = Flux2Pipeline.from_pretrained(
        str(cfg.image_gen_model_path),
        torch_dtype=dtype,
    )
    pipe.enable_model_cpu_offload()

    strict = bool(cfg.strict)
    skip_existing = bool(cfg.skip_existing)
    per_index_seed = bool(cfg.per_index_seed)
    base_seed = int(cfg.seed)
    batch_size = max(1, int(cfg.batch_size))

    work: list[tuple[int, dict[str, Any], str]] = []
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

        work.append((i, sample, prompt))

    height = int(cfg.height)
    width = int(cfg.width)
    guidance_scale = float(cfg.guidance_scale)
    num_inference_steps = int(cfg.num_inference_steps)

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
                img_path = iter3_image_output_path(run_name, sample, idx)
                process_and_save_image(image, img_path)
                sample["image_path"] = str(img_path.relative_to(REPO_ROOT))
                samples[idx] = sample
                write_json(save_path, samples)
                print(f"[{idx}] Wrote {sample['image_path']}", flush=True)
                pbar.update(1)

    print(f"Done. Curated samples: {save_path.relative_to(REPO_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
