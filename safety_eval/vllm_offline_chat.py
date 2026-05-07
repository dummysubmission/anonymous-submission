"""Shared vLLM offline batched ``LLM.chat`` helpers for safety_eval."""

from __future__ import annotations

import sys
import os
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from omegaconf import DictConfig, OmegaConf

from safety_eval.utils import pil_image_to_data_url, text_from_vllm_request_output

T = TypeVar("T")
R = TypeVar("R")


def configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


def _nullish(v: Any) -> bool:
    return v in (None, "", "null", "None", "auto")


def resolve_prepare_max_workers(cfg: DictConfig) -> int | None:
    """Thread count for parallel message prep; ``None`` = ``ThreadPoolExecutor`` default."""
    raw = getattr(cfg.offline_chat, "prepare_max_workers", None)
    if raw in (None, "", "null", "None"):
        return None
    w = int(raw)
    if w < 1:
        return None
    return w


def resolve_chat_batch_size(cfg: DictConfig) -> int | None:
    """Max prompts per ``llm.chat`` call.

    Multimodal offline chat materializes large per-request payloads during the
    "Rendering conversations" phase; sending the full dataset in one call can
    OOM RAM even when the on-disk dataset fits comfortably.

    Returns:
        Positive int: chunk size.
        ``None``: no chunking (single ``llm.chat`` over all work items).
    """
    raw = getattr(cfg.offline_chat, "chat_batch_size", None)
    if raw in (None, "", "null", "None"):
        return 64
    n = int(raw)
    if n <= 0:
        return None
    return n


def parallel_map_ordered(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    max_workers: int | None = None,
    progress_desc: str | None = None,
) -> list[R]:
    """Run ``fn`` over ``items`` in parallel with a thread pool; output order matches ``items``."""
    if not items:
        return []
    try:
        from tqdm import tqdm as _tqdm
    except Exception:  # pragma: no cover
        _tqdm = None

    def _maybe_tqdm(it: Any, *, total: int) -> Any:
        if progress_desc and _tqdm is not None:
            return _tqdm(it, total=total, desc=progress_desc)
        return it

    if len(items) == 1:
        return [fn(items[0])]
    if max_workers == 1:
        return [fn(x) for x in _maybe_tqdm(items, total=len(items))]
    from concurrent.futures import ThreadPoolExecutor

    workers = max_workers
    if workers is None:
        workers = min(32, (os.cpu_count() or 4) * 4)
    chunksize = max(1, len(items) // (workers * 8))
    chunksize = min(64, chunksize)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        it = ex.map(fn, items, chunksize=chunksize)
        return list(_maybe_tqdm(it, total=len(items)))


def extra_llm_kwargs(cfg: DictConfig) -> dict[str, Any]:
    oc = getattr(cfg.offline_chat, "llm_kwargs", None)
    if oc is None:
        return {}
    raw = OmegaConf.to_container(oc, resolve=True)
    return raw if isinstance(raw, dict) else {}


def build_llm(
    cfg: DictConfig,
    *,
    model_path: str,
    max_model_len: int | None = None,
) -> Any:
    """Construct ``vllm.LLM`` from Hydra config (lazy-import vLLM)."""
    from vllm import LLM

    vllm_cfg = cfg.vllm
    kwargs: dict[str, Any] = {
        "model": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": int(vllm_cfg.tensor_parallel_size),
        "limit_mm_per_prompt": {
            "image": int(vllm_cfg.limit_mm_per_prompt_image),
            "video": int(vllm_cfg.limit_mm_per_prompt_video),
        },
    }

    mlen = max_model_len
    if mlen is None:
        raw_mlen = getattr(vllm_cfg, "max_model_len", None)
        if not _nullish(raw_mlen):
            mlen = int(raw_mlen)
    if mlen is not None:
        kwargs["max_model_len"] = int(mlen)

    gdn = getattr(cfg.offline_chat, "gdn_prefill_backend", None)
    if not _nullish(gdn):
        kwargs["gdn_prefill_backend"] = str(gdn)

    moe_backend: str | None = None
    model_cfg = OmegaConf.select(cfg, "model")
    for raw in (
        getattr(vllm_cfg, "moe_backend", None),
        getattr(model_cfg, "moe_backend", None) if model_cfg is not None else None,
    ):
        if not _nullish(raw):
            moe_backend = str(raw).strip().lower().replace("-", "_")
            break
    if moe_backend is not None:
        kwargs["moe_backend"] = moe_backend

    extra = extra_llm_kwargs(cfg)
    kwargs.update({k: v for k, v in extra.items() if not _nullish(v)})
    return LLM(**kwargs)


def _offline_sampling_params(cfg: DictConfig, *, stage: str) -> Any:
    """Build vLLM ``SamplingParams`` from top-level ``cfg.sampling_params`` only (no defaults)."""
    from vllm import SamplingParams

    if stage == "generation":
        assert (
            "temperature" in cfg.sampling_params
            and cfg.sampling_params.temperature == 0.0
        ), f"temperature must be 0.0 for generation"

    if stage == "judging":
        assert "seed" in cfg.sampling_params, f"seed must be set for judging"

    if not OmegaConf.is_config(cfg) or "sampling_params" not in cfg:
        raise ValueError(
            f"cfg.sampling_params is required for offline {stage}; "
            "set temperature and max_new_tokens in the stage YAML."
        )
    sp = cfg.sampling_params
    for k in ("temperature", "max_new_tokens", "seed"):
        if k not in sp:
            raise ValueError(
                f"sampling_params.{k} is required for offline {stage} (no defaults in code)."
            )
    return SamplingParams(
        temperature=float(sp.temperature),
        max_tokens=int(sp.max_new_tokens),
        seed=int(sp.seed),
    )


def sampling_params_generate(cfg: DictConfig) -> Any:
    return _offline_sampling_params(cfg, stage="generation")


def sampling_params_guard(cfg: DictConfig) -> Any:
    return _offline_sampling_params(cfg, stage="guard labeling")


def sampling_params_judge(cfg: DictConfig) -> Any:
    return _offline_sampling_params(cfg, stage="judging")


def conversation_user_image_text(query: str, image: Any) -> list[dict[str, Any]]:
    """Single user turn with optional image + text (OpenAI-compatible parts)."""
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": pil_image_to_data_url(image)},
            }
        )
    content.append({"type": "text", "text": query})
    return [{"role": "user", "content": content}]


def conversation_judge(
    query: str,
    image: Any,
    response_text: str,
    malicious_intent: str | None,
    *,
    system_prompt: str | None,
    rubric: str = "strongreject",
) -> list[dict[str, Any]]:
    from pathlib import Path
    import json

    from prompts.strongreject_rubric_enforce_format import (
        format_strongreject_user_prompt,
    )

    repo_root = Path(__file__).resolve().parent.parent
    judge_template_path = repo_root / "data" / "judge_template.json"
    with open(judge_template_path, "r", encoding="utf-8") as f:
        judge_templates = json.load(f)
    prompt = format_strongreject_user_prompt(
        rubric, query, response_text, malicious_intent=malicious_intent
    )
    user_content: list[dict[str, Any]] = []
    if image is not None:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": pil_image_to_data_url(image)},
            }
        )
    user_content.append({"type": "text", "text": prompt})
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    else:
        messages.append(
            {"role": "system", "content": judge_templates["strongreject_rubric_system"]}
        )
    messages.append({"role": "user", "content": user_content})
    return messages


def conversation_guard(
    query: str,
    image: Any,
    assistant_response: str,
) -> list[dict[str, Any]]:
    """User (image + query) then assistant (prior response), for guard models."""
    user_turn = conversation_user_image_text(query, image)[0]
    return [
        user_turn,
        {"role": "assistant", "content": assistant_response},
    ]


def chat_template_kwargs_generate(cfg: DictConfig) -> dict[str, Any] | None:
    if bool(getattr(cfg, "enable_thinking", False)):
        return {"enable_thinking": True}
    else:
        return {"enable_thinking": False}


def prepare_generate_messages_row(
    sample: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], str] | None:
    """Build one generate-response conversation; safe for parallel thread execution."""
    persona_id = str(sample.get("persona_id") or "").strip()
    if not persona_id:
        return None
    query = str(sample.get("query", ""))
    img = sample.get("image")
    return (persona_id, conversation_user_image_text(query, img), query)


def prepare_guard_messages_row(
    row: tuple[dict[str, Any], str],
) -> tuple[str, list[dict[str, Any]], str]:
    """Build one guard conversation from a dataset row and the stored model response."""
    sample, response_text = row
    persona_id = str(sample.get("persona_id") or "").strip()
    query = str(sample.get("query", ""))
    img = sample.get("image")
    return (persona_id, conversation_guard(query, img, response_text), query)


def prepare_judge_messages_row(
    row: tuple[dict[str, Any], str, str | None, str],
) -> tuple[str, list[dict[str, Any]], str]:
    """Build one judge conversation from a dataset row, response, optional system prompt, and rubric."""
    sample, response_text, system_prompt, rubric = row
    persona_id = str(sample.get("persona_id") or "").strip()
    query = str(sample.get("explicit_text_query", ""))
    img = sample.get("image")
    return (
        persona_id,
        conversation_judge(
            query,
            img,
            response_text,
            malicious_intent=sample.get("malicious_intent", None),
            system_prompt=system_prompt,
            rubric=rubric,
        ),
        query,
    )


def run_chat_all(
    llm: Any,
    work: Sequence[tuple[str, list[dict[str, Any]]]],
    *,
    sampling_params: Any,
    use_tqdm: bool,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[tuple[str, str | None]]:
    """Single ``llm.chat`` over all conversations; vLLM continuous-batches internally.

    Returns ``(persona_id, text_or_none)`` in the same order as ``work``.
    """
    if not work:
        return []
    batch_ids = [pid for pid, _ in work]
    batch_msgs = [msgs for _, msgs in work]
    kwargs: dict[str, Any] = {
        "messages": batch_msgs,
        "sampling_params": sampling_params,
        "use_tqdm": use_tqdm,
    }
    if chat_template_kwargs:
        kwargs["chat_template_kwargs"] = chat_template_kwargs
    outputs = llm.chat(**kwargs)
    return [
        (pid, text_from_vllm_request_output(o))
        for pid, o in zip(batch_ids, outputs, strict=True)
    ]
