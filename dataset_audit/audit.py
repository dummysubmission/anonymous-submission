#!/usr/bin/env python3
"""Run multimodal dataset audit (MII / MER / CB) on benchmark-style samples.

Uses the codebook in ``prompts/audit_codebook.py`` and Portkey / vLLM clients via
``dataset_audit.client_utils`` (httpx SSL verification for Portkey).

Usage::

    # Portkey (set AI_SANDBOX_KEY); ``benchmark`` selects ``data/samples/<benchmark>.json``.
    python dataset_audit/audit.py benchmark=vlsbench start_index=0 max_samples=10

    # Local vLLM
    python dataset_audit/audit.py \\
        benchmark=vlsbench \\
        base_url=http://127.0.0.1:8000/v1 \\
        model_name=null \\
        enable_thinking=false
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from openai import OpenAI
from pydantic import ValidationError
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset_audit.client_utils import (
    chat_completions_create,
    extract_non_thinking_section,
    get_model_name,
    is_null_like,
    make_portkey_client,
    parse_json_payload,
    pil_image_to_data_url,
    resolve_image_path,
    save_json_list,
)
from dataset_audit.dataset_utils import load_safety_dataset, samples_json_path_for_benchmark
from prompts.audit_codebook import AuditPayload, render_audit_json_shape, system_instruction


def require_benchmark_name(cfg: DictConfig) -> str:
    bm = getattr(cfg, "benchmark", None)
    if is_null_like(bm) or not str(bm).strip():
        raise ValueError(
            "dataset_audit requires ``benchmark`` (e.g. benchmark=vlsbench). "
            "Samples are always loaded from data/samples/<benchmark>.json."
        )
    return str(bm).strip()


def _resolve_save_path(cfg: DictConfig, samples_path: Path, repo_root: Path) -> Path:
    raw = getattr(cfg, "save_path", None)
    if is_null_like(raw):
        return samples_path
    path = Path(str(raw).strip())
    if not path.is_absolute():
        path = repo_root / path
    return path


def parse_audit_payload(raw_text: str) -> AuditPayload | None:
    payload = parse_json_payload(raw_text)
    if not isinstance(payload, dict):
        return None
    candidate = payload.get("audit") or payload.get("result") or payload
    if not isinstance(candidate, dict):
        return None
    try:
        return AuditPayload.model_validate(candidate)
    except ValidationError:
        return None


def load_dataset_audit_config(repo_root: Path) -> Any:
    """Load ``configs/dataset_audit.yaml`` (same defaults as the CLI)."""
    from omegaconf import OmegaConf

    path = repo_root / "configs" / "dataset_audit.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"dataset audit config not found: {path}")
    return OmegaConf.load(path)


def make_audit_client_and_model(cfg: Any) -> tuple[Any, str, bool]:
    """Same Portkey / vLLM client selection as ``main()``."""
    base_url = getattr(cfg, "base_url", None)
    if base_url is not None and not is_null_like(base_url):
        client: Any = OpenAI(api_key="EMPTY", base_url=str(base_url))
        openai_compatible = True
    else:
        import os

        api_key = os.getenv("AI_SANDBOX_KEY")
        if not api_key:
            raise ValueError(
                "Missing AI_SANDBOX_KEY for Portkey, or set base_url for vLLM."
            )
        client = make_portkey_client(api_key)
        openai_compatible = False

    if cfg.model_name is not None and not is_null_like(cfg.model_name):
        model_name = str(cfg.model_name)
    else:
        model_name = get_model_name(client, None)
    return client, model_name, openai_compatible


def audit_fields_from_payload(parsed: AuditPayload) -> dict[str, Any]:
    keep = (
        parsed.malicious_intent_identifiability == 1
        and parsed.multimodal_exploit_requirement == 1
        and parsed.capability_bounded == 1
    )
    audit_explanation = "\n\n".join(
        [
            f"Malicious intent identifiability: {parsed.malicious_intent_identifiability_explanation}",
            f"Multimodal exploit requirement: {parsed.multimodal_exploit_requirement_explanation}",
            f"Capability bounded: {parsed.capability_bounded_explanation}",
        ]
    )
    return {
        "audit_malicious_intent_identifiability": parsed.malicious_intent_identifiability,
        "audit_malicious_intent_identifiability_explanation": parsed.malicious_intent_identifiability_explanation,
        "audit_multimodal_exploit_requirement": parsed.multimodal_exploit_requirement,
        "audit_multimodal_exploit_requirement_explanation": parsed.multimodal_exploit_requirement_explanation,
        "audit_capability_bounded": parsed.capability_bounded,
        "audit_capability_bounded_explanation": parsed.capability_bounded_explanation,
        "audit_keep": keep,
        "audit_explanation": audit_explanation,
    }


def build_audit_user_prompt(sample: dict[str, Any]) -> str:
    sid = sample.get("id", "")
    if "explicit_text_query" in sample:
        q = sample.get("explicit_text_query", "")
    else:
        q = sample.get("query", "")
    if not q:
        raise ValueError(f"No query found for sample {sid}")
    return (
        f"Sample id (for logging only): {sid}\n\n"
        f"User text query:\n{q}\n\n"
        "The image is attached alongside this message.\n\n"
        "Return exactly one valid JSON object with the following shape:\n"
        f"{render_audit_json_shape()}\n"
    )


def build_audit_messages(sample: dict[str, Any], pil_image: Any) -> list[dict[str, Any]]:
    """System + user (image + audit prompt) messages for one sample."""
    data_url = pil_image_to_data_url(pil_image)
    user_prompt = build_audit_user_prompt(sample)
    user_content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": user_prompt},
    ]
    return [
        {"role": "system", "content": system_instruction.strip()},
        {"role": "user", "content": user_content},
    ]


def evaluate_audit(
    client: Any,
    model_name: str,
    sample: dict[str, Any],
    image_source: Path | Any,
    *,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    openai_compatible: bool,
    max_attempts: int,
    retry_sleep: float,
    verbose: bool,
    idx: int,
    sample_id: str,
) -> AuditPayload | None:
    from PIL import Image

    if isinstance(image_source, Path):
        try:
            img = Image.open(image_source)
        except Exception as exc:
            print(f"[{idx} id={sample_id}] Failed to load image {image_source}: {exc}")
            return None
    elif isinstance(image_source, Image.Image):
        img = image_source
    else:
        print(
            f"[{idx} id={sample_id}] Unsupported image type {type(image_source)}; "
            "expected pathlib.Path or PIL.Image.Image."
        )
        return None

    messages = build_audit_messages(sample, img)

    for attempt in range(1, max_attempts + 1):
        print(f"[{idx} id={sample_id}] Audit attempt {attempt}/{max_attempts}")
        try:
            response = chat_completions_create(
                client,
                model_name,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                enable_thinking=enable_thinking,
                openai_compatible=openai_compatible,
            )
        except Exception as exc:
            print(f"[{idx} id={sample_id}] Model call failed: {exc}")
            time.sleep(retry_sleep * attempt)
            continue

        if not response.choices:
            print(f"[{idx} id={sample_id}] No choices returned; retrying.")
            continue

        raw_text = response.choices[0].message.content or ""
        raw_text = extract_non_thinking_section(raw_text)

        if verbose:
            print("-" * 80)
            print(f"[dataset_audit] {sample_id}")
            print(raw_text)
            print("-" * 80)

        parsed = parse_audit_payload(raw_text)
        if parsed is None:
            print(
                f"[{idx} id={sample_id}] Failed to parse audit response; retrying."
            )
            continue

        return parsed

    return None


def run_interactive_audit(
    text_query: str,
    pil_image: Any,
    *,
    client: Any,
    model_name: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    openai_compatible: bool,
    max_attempts: int,
    retry_sleep: float,
    verbose: bool,
    sample_id: str = "interactive",
) -> AuditPayload | None:
    """Run the same audit as ``evaluate_audit`` on an in-memory image and free-form query."""
    q = (text_query or "").strip()
    if not q:
        raise ValueError("text_query is empty")
    sample: dict[str, Any] = {"id": sample_id, "explicit_text_query": q}
    return evaluate_audit(
        client,
        model_name,
        sample,
        pil_image,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
        openai_compatible=openai_compatible,
        max_attempts=max_attempts,
        retry_sleep=retry_sleep,
        verbose=verbose,
        idx=0,
        sample_id=sample_id,
    )


@hydra.main(config_path="../configs", config_name="dataset_audit", version_base=None)
def main(cfg: DictConfig) -> None:
    load_dotenv()
    repo_root = REPO_ROOT

    benchmark = require_benchmark_name(cfg)
    samples_path = samples_json_path_for_benchmark(benchmark, repo_root=repo_root)
    if not samples_path.is_file():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")
    save_path = _resolve_save_path(cfg, samples_path, repo_root)

    samples = load_safety_dataset(benchmark=benchmark, repo_root=repo_root)

    print(f"Loaded {len(samples)} samples for benchmark={benchmark!r} from {samples_path}")
    if save_path != samples_path:
        print(f"Writing audit output to {save_path}")

    start_index = int(getattr(cfg, "start_index", 0) or 0)
    max_samples = getattr(cfg, "max_samples", None)
    if max_samples not in (None, "null", "None"):
        end_index = min(len(samples), start_index + int(max_samples))
        indices = list(range(start_index, end_index))
    elif start_index > 0:
        indices = list(range(start_index, len(samples)))
    else:
        indices = list(range(len(samples)))

    client, model_name, openai_compatible = make_audit_client_and_model(cfg)
    print(f"Using model: {model_name}")

    max_tokens = int(cfg.sampling_params.max_new_tokens)
    temperature = float(cfg.temperature)
    enable_thinking = bool(getattr(cfg, "enable_thinking", False))
    overwrite = bool(cfg.overwrite)
    max_attempts = int(cfg.max_attempts_per_sample)
    retry_sleep = float(cfg.retry_sleep_seconds)

    for idx in tqdm(indices, total=len(indices)):
        sample = samples[idx]
        sample_id = str(sample.get("id") or "").strip() or f"index_{idx}"

        if "audit_malicious_intent_identifiability" in sample and not overwrite:
            continue

        image_path = resolve_image_path(sample, repo_root=repo_root)
        if image_path is None:
            print(f"[{idx} id={sample_id}] No image found; skipping.")
            continue

        print(f"[{idx} id={sample_id}] Starting audit")

        parsed = evaluate_audit(
            client,
            model_name,
            sample,
            image_path,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            openai_compatible=openai_compatible,
            max_attempts=max_attempts,
            retry_sleep=retry_sleep,
            verbose=bool(cfg.verbose),
            idx=idx,
            sample_id=sample_id,
        )

        if parsed is None:
            print(
                f"[{idx} id={sample_id}] "
                "Audit failed after retries; leaving sample unchanged."
            )
            continue

        sample_update = {**sample, **audit_fields_from_payload(parsed)}
        samples[idx] = sample_update
        save_json_list(save_path, samples)

        print(
            f"[{idx} id={sample_id}] "
            f"malicious_intent_identifiability={parsed.malicious_intent_identifiability} "
            f"multimodal_exploit_requirement={parsed.multimodal_exploit_requirement} "
            f"capability_bounded={parsed.capability_bounded} "
            f"audit_keep={sample_update['audit_keep']} — saved to {save_path}"
        )

    print(f"Done. Updated samples in {save_path}")


if __name__ == "__main__":
    main()
