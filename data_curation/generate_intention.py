#!/usr/bin/env python3
"""Generate Stage 1 RAT-grounded harmful intentions from persona samples."""

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import hydra
from datasets import load_dataset
from omegaconf import DictConfig
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))

from prompts.stage_1 import example_output, example_persona, system_instruction


SCP_CATEGORIES = {
    "reduce the effort",
    "reduce the risk",
    "increase the rewards",
}


def configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


configure_stdio()


class GenerationPayload(BaseModel):
    explanation: str = Field(
        min_length=1,
        description=(
            "Brief 1-2 sentence explanation of the suitable target and how the MLLM "
            "would be used."
        ),
    )
    offender_motivation: str = Field(
        min_length=1,
        description=(
            'The specific driver (financial, social, etc.) making this persona "motivated."'
        ),
    )
    mllm_use_case: str = Field(
        min_length=1,
        description="How the MLLM will be used to achieve the harmful intent."
    )
    harmful_intent: str = Field(
        min_length=1,
        description="The harmful intent that is grounded by the RAT framework.",
    )
    scp_category: str = Field(
        min_length=1,
        description="The SCP category that the MLLM can enable the actor to bypass.",
    )


class HarmfulIntentPayload(GenerationPayload):
    persona_id: str = Field(
        min_length=1, description="Unique reference to the synthetic user profile."
    )


def is_null_like(value: Any) -> bool:
    return value in (None, "", "null", "None")


def normalize_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def render_json_shape(model_cls: type[BaseModel]) -> str:
    lines = ["{"]
    properties = model_cls.model_json_schema().get("properties", {})
    items = list(properties.items())
    for idx, (field_name, _field_schema) in enumerate(items):
        suffix = "," if idx < len(items) - 1 else ""
        lines.append(f'  "{field_name}": "string"{suffix}')
    lines.append("}")
    return "\n".join(lines)


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


def parse_generation_payload(raw_text: str) -> GenerationPayload | None:
    payload = parse_json_payload(raw_text)
    if not isinstance(payload, dict):
        return None

    candidate = payload.get("sample") or payload.get("result") or payload
    if not isinstance(candidate, dict):
        return None

    normalized = {
        "explanation": normalize_whitespace(candidate.get("explanation")),
        "offender_motivation": normalize_whitespace(
            candidate.get("offender_motivation")
        ),
        "harmful_intent": normalize_whitespace(candidate.get("harmful_intent")),
        "scp_category": normalize_whitespace(candidate.get("scp_category")),
        "mllm_use_case": normalize_whitespace(candidate.get("mllm_use_case")),
    }
    scp_category = normalize_whitespace(candidate.get("scp_category"))
    if scp_category.lower() not in SCP_CATEGORIES:
        return None
    try:
        return GenerationPayload.model_validate(normalized)
    except ValidationError:
        return None


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


def get_model_name(client: OpenAI, explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model
    models = client.models.list().data
    if not models:
        raise RuntimeError("No models available from the inference server.")
    return models[0].id


def build_persona_input(sample: dict[str, Any]) -> str:
    # fields = [
    #     ("persona_id", sample.get("uuid")),
    #     ("persona_summary", sample.get("persona")),
    #     ("cultural_background", sample.get("cultural_background")),
    #     ("skills_and_expertise", sample.get("skills_and_expertise")),
    #     ("skills_and_expertise_list", sample.get("skills_and_expertise_list")),
    #     ("hobbies_and_interests", sample.get("hobbies_and_interests")),
    #     ("hobbies_and_interests_list", sample.get("hobbies_and_interests_list")),
    #     ("career_goals_and_ambitions", sample.get("career_goals_and_ambitions")),
    #     ("sex", sample.get("sex")),
    #     ("age", sample.get("age")),
    #     ("marital_status", sample.get("marital_status")),
    #     ("education_level", sample.get("education_level")),
    #     ("bachelors_field", sample.get("bachelors_field")),
    #     ("occupation", sample.get("occupation")),
    #     ("city", sample.get("city")),
    #     ("state", sample.get("state")),
    #     ("zipcode", sample.get("zipcode")),
    #     ("country", sample.get("country")),
    # ]

    persona_input = ""
    if sample.get("professional_persona") is not None:
        persona_input += sample.get("professional_persona") + " "
    if sample.get("cultural_background") is not None:
        persona_input += sample.get("cultural_background") + " "
    if sample.get("age") is not None:
        if sample.get("sex") == "Male":
            persona_input += f"He is {sample.get('age')} years old. "
        else:
            persona_input += f"She is {sample.get('age')} years old. "
    return persona_input.strip()


def build_generation_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    persona_input = build_persona_input(sample)
    user_prompt = (
        "Example Persona Input:\n"
        f"{example_persona.strip()}\n\n"
        "Example Output:\n"
        f"{example_output.strip()}\n\n"
        "Persona Input:\n"
        f"{persona_input}\n\n"
        "Return exactly one valid JSON object with the following shape:\n"
        f"{render_json_shape(GenerationPayload)}\n\n"
    )
    return [
        {"role": "system", "content": system_instruction.strip()},
        {"role": "user", "content": user_prompt},
    ]


def call_model(
    client: OpenAI,
    model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    verbose: bool = False,
    response_label: str | None = None,
) -> str:
    request_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        response = client.chat.completions.create(
            **request_kwargs,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
    except TypeError:
        response = client.chat.completions.create(**request_kwargs)
    except Exception as exc:
        if "extra_body" not in str(exc) and "chat_template_kwargs" not in str(exc):
            raise
        response = client.chat.completions.create(**request_kwargs)

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


def get_save_path(cfg: DictConfig) -> Path:
    if cfg.save_path is not None:
        return Path(cfg.save_path)
    model_name = cfg.model_name or cfg.model.short_name
    return REPO_ROOT / "output" / "generate_intention" / model_name / "intentions.json"


def load_persona_dataset(cfg: DictConfig):
    if cfg.dataset.dataset_name != "persona":
        raise ValueError(
            "Stage 1 intention generation currently expects dataset.dataset_name=persona."
        )

    dataset = load_dataset(
        str(cfg.dataset.dataset_path),
        split=cfg.dataset.dataset_split,
    ).filter(lambda x: x.get("age") > 12, num_proc=int(getattr(cfg, "num_workers", 8)))
    start_index = int(getattr(cfg, "start_index", 0) or 0)
    max_samples = getattr(cfg, "max_samples", None)
    if hasattr(cfg, "random_sample") and bool(cfg.random_sample):
        dataset = dataset.shuffle(seed=42)
    if max_samples not in (None, "null", "None"):
        end_index = min(len(dataset), start_index + int(max_samples))
        dataset = dataset.select(range(start_index, end_index))
    elif start_index > 0:
        dataset = dataset.select(range(start_index, len(dataset)))

    print(f"Loaded {len(dataset)} persona samples from {cfg.dataset.dataset_path}")
    print(f"Unique persona IDs: {len(set([sample['uuid'] for sample in dataset]))}")
    return dataset


@hydra.main(
    config_path="../configs", config_name="generate_intention.yaml", version_base=None
)
def main(cfg: DictConfig) -> None:
    save_path = get_save_path(cfg)
    dataset = load_persona_dataset(cfg)

    if save_path.exists() and not bool(cfg.overwrite):
        results = load_json_list(save_path)
        print(f"Loaded {len(results)} existing Stage 1 generations from {save_path}")
    else:
        results = []

    completed_ids = {str(item.get("persona_id")) for item in results}

    if cfg.base_url is not None:
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
    elif cfg.lora_path is not None:
        model_name = "ft_adapter"
    else:
        model_name = get_model_name(client, cfg.model_name)
    print(f"Using model: {model_name}")

    max_tokens = int(cfg.sampling_params.max_new_tokens)

    for sample in tqdm(dataset, total=len(dataset)):
        persona_id = str(sample.get("uuid") or "").strip()
        if not persona_id:
            print("Skipping persona sample with missing uuid.")
            continue
        if persona_id in completed_ids:
            continue

        print(f"[persona_id={persona_id}] Generating Stage 1 intention")
        success = False
        for attempt in range(1, int(cfg.max_attempts_per_persona) + 1):
            print(
                f"[persona_id={persona_id}] Attempt "
                f"{attempt}/{int(cfg.max_attempts_per_persona)}"
            )
            try:
                raw_text = call_model(
                    client=client,
                    model_name=model_name,
                    messages=build_generation_messages(sample),
                    max_tokens=max_tokens,
                    temperature=float(cfg.temperature),
                    enable_thinking=bool(cfg.enable_thinking),
                    verbose=bool(cfg.verbose),
                    response_label=f"[Stage 1] {persona_id}",
                )
            except Exception as exc:
                print(f"[persona_id={persona_id}] Model call failed: {exc}")
                continue

            parsed = parse_generation_payload(raw_text)
            if parsed is None:
                print(f"[persona_id={persona_id}] Failed to parse response; retrying.")
                continue

            output = HarmfulIntentPayload(
                persona_id=persona_id,
                **parsed.model_dump(),
            )

            result = {
                **output.model_dump(),
                "persona_input": build_persona_input(sample),
                "raw_generation_response": raw_text,
            }
            results.append(result)
            completed_ids.add(persona_id)
            save_json_list(save_path, results)
            print(f"[persona_id={persona_id}] Saved Stage 1 output to {save_path}")
            success = True
            break

        if not success:
            print(
                f"Warning: failed to generate a valid Stage 1 output for "
                f"persona_id={persona_id} after {int(cfg.max_attempts_per_persona)} "
                "attempts."
            )

    print(f"Done. Saved {len(results)} Stage 1 outputs to {save_path}")


if __name__ == "__main__":
    main()
