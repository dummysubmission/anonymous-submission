#!/usr/bin/env python3
"""Stage 5: rewrite explicit text queries into implicit, dual-use framings.

Reads samples from ``samples_source`` (samples that have passed Stages 3-4),
sends the image alongside the harmful intent and text query to a multimodal model,
and rewrites the text query to be implicit while preserving the harmful intent.
Results are written to ``save_path`` (defaults to the source path if unset).

Supports both a local vLLM server (via ``base_url``) and the Portkey API
(when ``base_url`` is null).
"""

from __future__ import annotations

import ast
import base64
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))

from prompts.stage_5 import system_instruction


def configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


configure_stdio()


class ImplicitQueryPayload(BaseModel):
    explanation: str = Field(
        min_length=1,
        description="Brief 1-2 sentence explanation of the rewritten text query.",
    )
    implicit_text_query: str = Field(
        min_length=1,
        description="The rewritten text query with implicit, dual-use framing.",
    )


def normalize_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def extract_non_thinking_section(text: str) -> str:
    think_idx = text.find("</think>")
    if think_idx == -1:
        return text.strip()
    return text[think_idx + len("</think>") :].strip()


def parse_json_payload(text: str) -> Any:
    if not text:
        return None

    def _literal_value(blob: str) -> Any:
        try:
            return ast.literal_eval(blob)
        except Exception:
            return None

    def _parse(blob: str) -> Any:
        blob = blob.strip()
        if not blob:
            return None
        if blob.startswith("```"):
            blob = re.sub(r"^```(?:json)?\s*", "", blob.strip(), flags=re.IGNORECASE)
            blob = re.sub(r"\s*```$", "", blob.strip())

        try:
            parsed = json.loads(blob)
        except Exception:
            parsed = None

        if isinstance(parsed, str):
            try:
                return json.loads(parsed)
            except Exception:
                literal_parsed = _literal_value(parsed)
                if literal_parsed is not None:
                    return literal_parsed

        if parsed is not None:
            return parsed

        literal_parsed = _literal_value(blob)
        if literal_parsed is not None:
            return literal_parsed

        if any(token in blob for token in ('\\"', "\\n", "\\t", "\\u")):
            try:
                unescaped = bytes(blob, "utf-8").decode("unicode_escape")
            except Exception:
                unescaped = None
            if unescaped is not None:
                try:
                    return json.loads(unescaped)
                except Exception:
                    return _literal_value(unescaped)

        return None

    parsed = _parse(text)
    if parsed is not None:
        return parsed

    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if not match:
        return None
    return _parse(match.group(0))


def parse_implicit_query_payload(raw_text: str) -> ImplicitQueryPayload | None:
    payload = parse_json_payload(raw_text)
    if not isinstance(payload, dict):
        return None

    candidate = payload.get("sample") or payload.get("result") or payload
    if not isinstance(candidate, dict):
        return None

    normalized = {
        "explanation": normalize_whitespace(candidate.get("explanation")),
        "implicit_text_query": normalize_whitespace(
            candidate.get("implicit_text_query")
        ),
    }
    try:
        return ImplicitQueryPayload.model_validate(normalized)
    except ValidationError:
        return None


def pil_image_to_data_url(img: Any) -> str:
    buf = io.BytesIO()
    try:
        img.save(buf, format="PNG")
    except Exception:
        img.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, found {type(data).__name__}.")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected every entry in {path} to be a JSON object.")
    return data


def save_json_list(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=4)
        f.write("\n")


def resolve_image_path(sample: dict[str, Any]) -> Path | None:
    raw = (sample.get("image_path") or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p if p.is_file() else None


def is_null_like(value: Any) -> bool:
    return value in (None, "", "null", "None")


def resolve_path(raw: Any, *, must_exist: bool = False, label: str = "path") -> Path:
    if is_null_like(raw):
        raise ValueError(f"{label} must be set.")
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def render_json_shape() -> str:
    return "\n".join(
        [
            "{",
            '  "explanation": "string",',
            '  "implicit_text_query": "string"',
            "}",
        ]
    )


def build_sample_block(sample: dict[str, Any]) -> str:
    return (
        f"Offender's motivation:\n{sample.get('offender_motivation', '')}\n\n"
        f"Harmful intent:\n{sample.get('harmful_intent', '')}\n\n"
        f"Intended use of the MLLM:\n{sample.get('mllm_use_case', '')}\n\n"
        f"Text query:\n{sample.get('text_query', '')}"
    ).strip()


def build_generation_messages(
    sample: dict[str, Any], data_url: str
) -> list[dict[str, Any]]:
    user_text = (
        "Rewrite the text query for the following sample. "
        "The image is attached alongside this message.\n\n"
        f"{build_sample_block(sample)}\n\n"
        "Return exactly one valid JSON object with the following shape:\n"
        f"{render_json_shape()}\n"
    )
    user_content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": user_text},
    ]
    return [
        {"role": "system", "content": system_instruction.strip()},
        {"role": "user", "content": user_content},
    ]


def call_model(
    client: Any,
    model_name: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    use_vllm: bool,
    verbose: bool = False,
    response_label: str | None = None,
) -> str:
    if use_vllm:
        request_kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            response = client.chat.completions.create(
                **request_kwargs,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": enable_thinking}
                },
            )
        except TypeError:
            response = client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            if "extra_body" not in str(exc) and "chat_template_kwargs" not in str(exc):
                raise
            response = client.chat.completions.create(**request_kwargs)
    else:
        extra_args: dict[str, Any]
        if model_name == "gpt-5":
            extra_args = {"max_completion_tokens": max_tokens}
        else:
            extra_args = {"temperature": temperature, "max_tokens": max_tokens}
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **extra_args,
        )

    if not response.choices:
        raise RuntimeError("Model returned no choices.")

    raw_text = response.choices[0].message.content or ""
    if verbose:
        print("-" * 100, flush=True)
        if response_label:
            print(response_label, flush=True)
        print(raw_text, flush=True)
        print("-" * 100, flush=True)
    return extract_non_thinking_section(raw_text)


def get_model_name_from_server(client: Any, explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model
    models = client.models.list().data
    if not models:
        raise RuntimeError("No models available from the inference server.")
    return models[0].id


@hydra.main(
    config_path="../configs",
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

    if save_path.exists() and save_path != source_path and not bool(cfg.overwrite):
        samples = load_json_list(save_path)
        print(f"Resuming from {len(samples)} existing results at {save_path}")
    else:
        samples = load_json_list(source_path)
        print(f"Loaded {len(samples)} samples from {source_path}")
        if save_path != source_path:
            save_json_list(save_path, samples)
            print(f"Initialized output at {save_path}")

    start_index = int(getattr(cfg, "start_index", 0) or 0)
    max_samples = getattr(cfg, "max_samples", None)
    if max_samples not in (None, "null", "None"):
        end_index = min(len(samples), start_index + int(max_samples))
        indices = list(range(start_index, end_index))
    elif start_index > 0:
        indices = list(range(start_index, len(samples)))
    else:
        indices = list(range(len(samples)))

    use_vllm = not is_null_like(cfg.base_url)

    if use_vllm:
        from openai import OpenAI

        client = OpenAI(api_key="EMPTY", base_url=str(cfg.base_url))
    else:
        from dotenv import load_dotenv

        load_dotenv()
        import os

        from portkey_ai import Portkey

        api_key = os.getenv("AI_SANDBOX_KEY")
        if not api_key:
            raise ValueError(
                "Missing API key. Set AI_SANDBOX_KEY in your environment to enable "
                "API generation via Portkey."
            )
        client = Portkey(api_key=api_key)

    if cfg.model_name is not None:
        model_name = str(cfg.model_name)
    elif use_vllm:
        model_name = get_model_name_from_server(client, None)
    else:
        raise ValueError(
            "model_name must be set when using the Portkey API (base_url=null)."
        )
    print(f"Using model: {model_name}")

    max_tokens = int(cfg.sampling_params.max_new_tokens)
    temperature = float(cfg.temperature)
    overwrite = bool(cfg.overwrite)
    max_attempts = int(cfg.max_attempts_per_sample)
    retry_sleep = float(cfg.retry_sleep_seconds)
    enable_thinking = bool(cfg.enable_thinking) if use_vllm else False
    require_stages = bool(cfg.require_prior_stages)

    for idx in tqdm(indices, total=len(indices)):
        sample = samples[idx]
        persona_id = str(sample.get("persona_id") or "").strip() or f"index_{idx}"

        if require_stages:
            if not sample.get("include", False):
                continue
            if not sample.get("image_helpfulness", False):
                continue

        if sample.get("implicit_text_query") and not overwrite:
            continue

        image_path = resolve_image_path(sample)
        if image_path is None:
            print(f"[{idx} persona_id={persona_id}] No image found; skipping.")
            continue

        try:
            img = Image.open(image_path)
        except Exception as exc:
            print(
                f"[{idx} persona_id={persona_id}] Failed to load image "
                f"{image_path}: {exc}"
            )
            continue

        data_url = pil_image_to_data_url(img)
        messages = build_generation_messages(sample, data_url)

        success = False
        for attempt in range(1, max_attempts + 1):
            print(
                f"[{idx} persona_id={persona_id}] "
                f"Stage 5 attempt {attempt}/{max_attempts}"
            )
            try:
                raw_text = call_model(
                    client,
                    model_name,
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                    use_vllm=use_vllm,
                    verbose=bool(cfg.verbose),
                    response_label=f"[Stage 5] {persona_id}",
                )
            except Exception as exc:
                print(f"[{idx} persona_id={persona_id}] Model call failed: {exc}")
                time.sleep(retry_sleep * attempt)
                continue

            parsed = parse_implicit_query_payload(raw_text)
            if parsed is None:
                print(
                    f"[{idx} persona_id={persona_id}] "
                    "Failed to parse response; retrying."
                )
                continue

            samples[idx] = {
                **sample,
                "implicit_text_query": parsed.implicit_text_query,
                "implicit_query_explanation": parsed.explanation,
            }
            save_json_list(save_path, samples)
            print(
                f"[{idx} persona_id={persona_id}] "
                f"Saved implicit_text_query to {save_path}"
            )
            success = True
            break

        if not success:
            print(
                f"Warning: failed to generate implicit query for "
                f"index={idx} persona_id={persona_id} "
                f"after {max_attempts} attempts."
            )

    print(f"Done. Saved {len(samples)} samples to {save_path}")


if __name__ == "__main__":
    main()
