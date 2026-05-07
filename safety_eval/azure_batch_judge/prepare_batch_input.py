"""Build JSONL batch input for Azure OpenAI /v1/chat/completions (StrongREJECT judge)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from prompts.strongreject_rubric_enforce_format import (  # noqa: E402
    JUDGE_RUBRIC_IA_STRONGREJECT,
)
from safety_eval.batch_utils import get_response_path  # noqa: E402
from safety_eval.dataset_utils import _to_pil, load_ours_dataset  # noqa: E402
from safety_eval.utils import load_json_list, normalize_judge_rubric  # noqa: E402
from safety_eval.vllm_offline_chat import (  # noqa: E402
    configure_stdio,
    conversation_judge,
    parallel_map_ordered,
    resolve_prepare_max_workers,
)

configure_stdio()

MODEL_PLACEHOLDER = "gpt-4o-mini"


def _resolve_batch_model_name(cfg: DictConfig) -> str:
    for key in "azure_model_name":
        v = OmegaConf.select(cfg, f"judge.{key}")
        if v not in (None, "", "null", "None"):
            return str(v)
    return MODEL_PLACEHOLDER


def _sampling_chat_kwargs(cfg: DictConfig) -> dict[str, Any]:
    """Optional chat.completions fields from ``cfg.sampling_params``."""
    sp = getattr(cfg, "sampling_params", None)
    if sp is None:
        return {}
    out: dict[str, Any] = {}
    for hydra_key, body_key in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("frequency_penalty", "frequency_penalty"),
        ("presence_penalty", "presence_penalty"),
    ):
        if hydra_key in sp and sp[hydra_key] is not None:
            out[body_key] = sp[hydra_key]
    return out


def _line_for_chat_completion(
    persona_id: str,
    messages: list[dict[str, Any]],
    *,
    model: str,
    max_tokens: int,
    extra_sampling: dict[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    body.update(extra_sampling)
    return {
        "custom_id": str(persona_id),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


@hydra.main(
    config_path="../../configs",
    config_name="safety_eval_judge_label.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    out_sel = OmegaConf.select(cfg, "azure_batch_jsonl")
    if out_sel:
        output_path = Path(str(out_sel))
    else:
        output_path = (
            Path(__file__).resolve().parent
            / "batch_input_files"
            / cfg.target_model_name
            / f"batch_{cfg.dataset.dataset_split}.jsonl"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    response_path = get_response_path(cfg, REPO_ROOT)
    if not response_path.is_file():
        raise FileNotFoundError(
            f"Responses file not found: {response_path}\n"
            "Set response_path= or run response generation first."
        )
    responses = load_json_list(response_path)
    response_by_id: dict[str, str] = {
        str(r.get("persona_id", "")): str(r.get("response", ""))
        for r in responses
        if r.get("persona_id")
    }

    ds = load_ours_dataset(cfg)
    mw = resolve_prepare_max_workers(cfg)

    rubric = normalize_judge_rubric(getattr(cfg.judge, "rubric", "strongreject"))
    raw_sys = getattr(cfg.judge, "system_prompt", None)
    if raw_sys in (None, "", "null", "None"):
        system_prompt_val: str | None = None
    else:
        system_prompt_val = str(raw_sys)

    sp = getattr(cfg, "sampling_params", None)
    if sp is None or "max_new_tokens" not in sp:
        raise ValueError(
            "sampling_params.max_new_tokens is required for batch max_tokens"
        )
    max_tokens = int(sp.max_new_tokens)
    model = _resolve_batch_model_name(cfg)
    extra_sampling = _sampling_chat_kwargs(cfg)

    def prepare_row(i: int) -> tuple[str, dict[str, Any] | None]:
        sample = dict(ds[i])
        persona_id = str(sample.get("persona_id") or "").strip()
        if not persona_id:
            return ("no_pid", None)
        response_text = response_by_id.get(persona_id)
        if response_text is None:
            return ("no_response", None)
        if rubric == JUDGE_RUBRIC_IA_STRONGREJECT:
            mi = sample.get("malicious_intent")
            if mi is None or str(mi).strip() == "":
                return ("no_malicious_intent", None)

        query = str(sample.get("explicit_text_query", ""))
        image = _to_pil(sample.get("image"))
        messages = conversation_judge(
            query,
            image,
            response_text,
            malicious_intent=sample.get("malicious_intent", None),
            system_prompt=system_prompt_val,
            rubric=rubric,
        )
        line = _line_for_chat_completion(
            persona_id,
            messages,
            model=model,
            max_tokens=max_tokens,
            extra_sampling=extra_sampling,
        )
        return ("ok", line)

    ordered = parallel_map_ordered(
        prepare_row,
        list(range(len(ds))),
        max_workers=mw,
        progress_desc="prepare_azure_batch_jsonl",
    )

    counts: dict[str, int] = {}
    n_written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for tag, row in tqdm(ordered, desc="Writing batch input"):
            counts[tag] = counts.get(tag, 0) + 1
            if tag != "ok" or row is None:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    n_no_pid = counts.get("no_pid", 0)
    n_no_response = counts.get("no_response", 0)
    n_no_mi = counts.get("no_malicious_intent", 0)
    if n_no_pid or n_no_response or n_no_mi:
        print(
            f"Skipped: {n_no_pid} missing persona_id, "
            f"{n_no_response} missing response row, "
            f"{n_no_mi} missing malicious_intent (ia_strongreject).",
            flush=True,
        )

    print(f"Wrote {n_written} requests to {output_path}", flush=True)


if __name__ == "__main__":
    main()
