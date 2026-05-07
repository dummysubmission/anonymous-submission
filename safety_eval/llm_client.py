"""OpenAI-compatible chat clients for safety_eval server scripts (vLLM, Portkey, Azure stub)."""

from __future__ import annotations

import os
from typing import Any
from dotenv import load_dotenv

from omegaconf import DictConfig, OmegaConf

load_dotenv(".env")


def _provider_from_cfg(model_cfg: Any) -> str:
    p = getattr(model_cfg, "provider", None)
    if p is None or str(p).strip() in ("", "null", "None"):
        raise ValueError(
            "Model config must set provider: vllm | portkey | azure "
            "(see configs/model/*.yaml)."
        )
    s = str(p).strip().lower()
    if s not in ("vllm", "portkey", "azure"):
        raise ValueError(f"Invalid provider {s!r}; expected vllm, portkey, or azure")
    return s


def uses_gpt5_style_completion_api(model_id: str) -> bool:
    """
    True when the routed model name should use the GPT-5-style chat completion API.

    Those models reject ``temperature`` / ``max_tokens``; use ``max_completion_tokens``
    only (from ``sampling_params.max_new_tokens``).
    """
    return "gpt-5" in str(model_id).lower()


def build_generation_chat_client(cfg: DictConfig, base_url: str) -> tuple[Any, str]:
    """
    Return ``(client, model_id)`` for ``client.chat.completions.create(model=..., ...)``.

    For ``provider=vllm``, ``model_id`` is the served model id (first from the server)
    unless ``cfg.lora_path`` is set, in which case it is ``ft_adapter``.
    """
    model_cfg = OmegaConf.select(cfg, "model")
    if model_cfg is None:
        raise ValueError(
            "Generation requires cfg.model (Hydra defaults: - model: <id> in the stage YAML)."
        )
    provider = _provider_from_cfg(model_cfg)
    if provider == "azure":
        if model_cfg.azure_model_name == "claude-sonnet-4-6":
            from anthropic import AnthropicFoundry

            api_key = os.getenv("AZURE_CLAUDE_API_KEY")
            endpoint = os.getenv("AZURE_CLAUDE_ENDPOINT")
            if not api_key or not endpoint:
                raise ValueError(
                    "Missing Azure Claude API key or endpoint. Set AZURE_CLAUDE_API_KEY and AZURE_CLAUDE_ENDPOINT in your environment to enable Azure Claude API generation."
                )
            client = AnthropicFoundry(api_key=api_key, base_url=endpoint)
            return client, str(model_cfg.azure_model_name).strip()
        elif model_cfg.azure_model_name == "grok-4-1-fast-non-reasoning":
            from openai import OpenAI

            model_name = getattr(model_cfg, "azure_model_name", None)
            endpoint = os.getenv("AZURE_GROK_ENDPOINT")
            if not endpoint:
                raise ValueError(
                    "Missing Azure Grok endpoint. Set AZURE_GROK_ENDPOINT in your environment to enable Azure API generation."
                )
            subscription_key = os.getenv("AZURE_GROK_API_KEY")
            if not subscription_key:
                raise ValueError(
                    "Missing Azure Grok API key. Set AZURE_GROK_API_KEY in your environment to enable Azure API generation."
                )
            client = OpenAI(
                base_url=f"{endpoint}/models",
                api_key=subscription_key,
                default_headers={"api-version": "2024-05-01-preview"},
            )
            return client, str(model_name).strip()
        else:
            from openai import AzureOpenAI

            model_name = getattr(model_cfg, "azure_model_name", None)
            if model_name is None or str(model_name).strip() in ("", "null", "None"):
                raise ValueError(
                    "provider=azure requires model.azure_model_name in model config"
                )
            endpoint = os.getenv("AZURE_ENDPOINT")
            if not endpoint:
                raise ValueError(
                    "Missing Azure endpoint. Set AZURE_ENDPOINT in your environment to enable Azure API generation."
                )
            subscription_key = os.getenv("AZURE_API_KEY")
            if not subscription_key:
                raise ValueError(
                    "Missing Azure API key. Set AZURE_API_KEY in your environment to enable Azure API generation."
                )
            api_version = "2024-12-01-preview"

            client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=subscription_key,
            )
            return client, str(model_name).strip()

    if provider == "portkey":
        model_name = getattr(model_cfg, "portkey_model_name", None)
        if model_name is None or str(model_name).strip() in ("", "null", "None"):
            raise ValueError(
                "provider=portkey requires portkey_model_name in model config"
            )
        try:
            from portkey_ai import Portkey  # type: ignore
        except Exception as e:
            raise ImportError(
                "portkey_ai is required for Portkey models. Install it in your environment."
            ) from e
        api_key = os.getenv("AI_SANDBOX_KEY")
        if not api_key:
            raise ValueError(
                "Missing API key. Set AI_SANDBOX_KEY in your environment to enable Portkey."
            )
        client = Portkey(api_key=api_key)
        return client, str(model_name).strip()

    base_s = (base_url or "").strip()
    if not base_s:
        raise ValueError(
            "provider=vllm requires a non-empty base_url in the Hydra config"
        )
    from openai import OpenAI

    client = OpenAI(api_key="EMPTY", base_url=base_s)
    lora_path = OmegaConf.select(cfg, "lora_path")
    if lora_path is not None and str(lora_path) not in ("null", "None", ""):
        return client, "ft_adapter"
    listed = client.models.list().data
    if not listed:
        raise RuntimeError(
            f"No models reported by OpenAI-compatible server at {base_s!r}"
        )
    return client, str(listed[0].id)


def _judge_provider(judge_cfg: Any) -> str:
    p = getattr(judge_cfg, "provider", None)
    if p is None or str(p).strip() in ("", "null", "None"):
        raise ValueError(
            "judge config must set provider: vllm | portkey | azure "
            "(see configs/judge/*.yaml)."
        )
    s = str(p).strip().lower()
    if s not in ("vllm", "portkey", "azure"):
        raise ValueError(
            f"Invalid judge.provider {s!r}; expected vllm, portkey, or azure"
        )
    return s


def build_judge_chat_client(cfg: DictConfig, base_url: str) -> tuple[Any, str]:
    """Return ``(client, model_id)`` for API judging."""
    j = cfg.judge
    provider = _judge_provider(j)
    if provider == "azure":
        from openai import AzureOpenAI

        model_name = getattr(j, "azure_model_name", None)
        if model_name is None or str(model_name).strip() in ("", "null", "None"):
            raise ValueError(
                "provider=azure requires model.azure_model_name in model config"
            )
        endpoint = os.getenv("AZURE_ENDPOINT")
        if not endpoint:
            raise ValueError(
                "Missing Azure endpoint. Set AZURE_ENDPOINT in your environment to enable Azure API generation."
            )
        subscription_key = os.getenv("AZURE_API_KEY")
        if not subscription_key:
            raise ValueError(
                "Missing Azure API key. Set AZURE_API_KEY in your environment to enable Azure API generation."
            )
        api_version = "2024-12-01-preview"

        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )
        return client, str(model_name).strip()
    elif provider == "portkey":
        jid = getattr(j, "portkey_model_name", None)
        if jid is None or str(jid).strip() in ("", "null", "None"):
            raise ValueError("provider=portkey requires judge.portkey_model_name")
        from portkey_ai import Portkey

        api_key = os.getenv("AI_SANDBOX_KEY")
        if not api_key:
            raise ValueError(
                "Missing API key. Set AI_SANDBOX_KEY in your environment to enable Portkey."
            )
        client = Portkey(api_key=api_key)
        return client, str(jid).strip()
    elif provider == "vllm":
        base_s = (base_url or "").strip()
        if not base_s:
            raise ValueError(
                "provider=vllm requires a non-empty base_url for the judge server"
            )
        from openai import OpenAI

        client = OpenAI(api_key="EMPTY", base_url=base_s)
        listed = client.models.list().data
        if not listed:
            raise RuntimeError(
                f"No models reported by OpenAI-compatible judge server at {base_s!r}"
            )
        return client, str(listed[0].id)
    else:
        print(f"Invalid judge provider: {provider}")
        raise ValueError(f"Invalid judge provider: {provider}")


def generation_completion_extra_args(
    cfg: DictConfig, *, model_id: str
) -> dict[str, Any]:
    """
    Build kwargs for ``chat.completions.create`` from ``cfg.sampling_params`` only.

    If ``model_id`` is a GPT-5-series route (substring ``gpt-5``), only
    ``max_completion_tokens`` is sent (from ``sampling_params.max_new_tokens``);
    ``temperature`` / ``top_p`` / ``max_tokens`` must not appear in config.

    Otherwise ``temperature``, ``top_p``, and ``max_new_tokens`` (→ ``max_tokens``) are required.
    """
    if not OmegaConf.is_config(cfg) or "sampling_params" not in cfg:
        raise ValueError("cfg.sampling_params is required for generation")
    sp = cfg.sampling_params
    if uses_gpt5_style_completion_api(model_id):
        for k in "temperature":
            if k in sp and OmegaConf.select(sp, k) is not None:
                print(
                    f"Skipping {k} for GPT-5-series model {model_id!r} in cfg.sampling_params"
                )
        if "max_new_tokens" not in sp:
            raise ValueError("sampling_params.max_new_tokens is required")
        return {"max_completion_tokens": int(sp.max_new_tokens)}
    for k in ("temperature", "max_new_tokens"):
        if k not in sp:
            raise ValueError(
                f"sampling_params.{k} is required for non-GPT-5 completion (model_id={model_id!r})"
            )
    return {
        "temperature": float(sp.temperature),
        "max_tokens": int(sp.max_new_tokens),
    }


def judge_completion_extra_args(cfg: DictConfig, *, model_id: str) -> dict[str, Any]:
    """
    Same rules as ``generation_completion_extra_args``, keyed off the judge ``model_id``
    passed to ``chat.completions.create``.

    Non-GPT-5 judges require ``temperature``, ``top_p``, and ``max_new_tokens`` (→ ``max_tokens``).
    """
    if not OmegaConf.is_config(cfg) or "sampling_params" not in cfg:
        raise ValueError("cfg.sampling_params is required for API judging")
    sp = cfg.sampling_params
    if uses_gpt5_style_completion_api(model_id):
        for k in "temperature":
            if k in sp and OmegaConf.select(sp, k) is not None:
                print(
                    f"Skipping {k} for GPT-5-series judge model {model_id!r} in cfg.sampling_params"
                )
        if "max_new_tokens" not in sp:
            raise ValueError("sampling_params.max_new_tokens is required")
        return {"max_completion_tokens": int(sp.max_new_tokens)}
    for k in ("temperature", "max_new_tokens", "seed"):
        if k not in sp:
            raise ValueError(
                f"sampling_params.{k} is required for judge model_id={model_id!r}"
            )
    return {
        "max_tokens": int(sp.max_new_tokens),
        "temperature": float(sp.temperature),
        "seed": int(sp.seed),
    }
