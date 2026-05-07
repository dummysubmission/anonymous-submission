import json
import time
from pathlib import Path
from typing import Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from safety_eval.dataset_utils import get_ours_dataloader
from safety_eval.batch_utils import get_judge_save_path, get_response_path
from safety_eval.llm_client import build_judge_chat_client, judge_completion_extra_args
from prompts.strongreject_rubric_enforce_format import format_strongreject_user_prompt

from safety_eval.utils import (
    JUDGE_STRONGREJECT_MAX_VALID_ATTEMPTS,
    _merge_id_key,
    is_strongreject_judge_parsed_conformant,
    judge_key_prefix,
    load_json_list,
    load_response_rows,
    merge_judge_rows_by_persona_id,
    normalize_judge_rubric,
    parse_judge_output_for_rubric,
    pil_image_to_data_url,
    print_judge_summary,
    progress,
    save_json_list,
)

load_dotenv(".env")

judge_template_path = REPO_ROOT / "data" / "judge_template.json"
with open(judge_template_path, "r", encoding="utf-8") as f:
    JUDGE_TEMPLATES = json.load(f)


def _norm_id_part(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _response_join_key(row: dict[str, Any]) -> str:
    """Join key for aligning a response row to dataloader ``id`` (``id`` preferred, else ``persona_id``)."""
    if _norm_id_part(row.get("id")) != "":
        return _norm_id_part(row["id"])
    return _norm_id_part(row.get("persona_id"))


def _dataloader_join_key(sid: Any) -> str:
    return _norm_id_part(sid)


def _merge_keys_for_judged_row(j: dict[str, Any]) -> set[str]:
    """Keys under which a judged row should match ``results`` rows (batch rows often have only ``persona_id``)."""
    keys: set[str] = set()
    mid = _merge_id_key(j.get("id"))
    if mid is not None:
        keys.add(mid)
    pid = _norm_id_part(j.get("persona_id"))
    if pid:
        keys.add(pid)
    rj = _response_join_key(j)
    if rj:
        keys.add(rj)
    return keys


def _merge_response_list_with_judged(
    results: list[dict[str, Any]], judged: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep every response row; overlay judge fields from ``judged`` rows (matched by id / persona_id)."""
    if not judged:
        return list(results)
    by_key: dict[str, dict[str, Any]] = {}
    for j in judged:
        for k in _merge_keys_for_judged_row(j):
            by_key[k] = j
    out: list[dict[str, Any]] = []
    for r in results:
        mid = _merge_id_key(r.get("id"))
        rk = _response_join_key(r) or (mid if mid is not None else "")
        if rk and rk in by_key:
            row = dict(r)
            row.update(by_key[rk])
            out.append(row)
        else:
            out.append(dict(r))
    return out


def get_api_safety_label(
    dataloader,
    responses: list[dict[str, Any]],
    base_url: str,
    cfg: DictConfig,
    *,
    save_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Call a multimodal judge via Portkey, Azure (stub), or a local OpenAI-compatible (vLLM) server.

    Routing is controlled by ``cfg.judge.provider``. Sampling kwargs come only from
    top-level ``cfg.sampling_params`` (``configs/safety_eval_judge_label.yaml``).

    Stores:
      - <judge_model>_judge_raw: raw text
      - <judge_model>_judge: parsed JSON dict (if parseable)
    """
    if not responses:
        return responses

    system_prompt = cfg.judge.system_prompt
    client, judge_model_name = build_judge_chat_client(cfg, base_url)
    extra_args = judge_completion_extra_args(cfg, model_id=judge_model_name)
    if cfg.judge.provider == "vllm":
        extra_args["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": cfg.enable_thinking}
        }

    judge_key_prefix_val = judge_key_prefix(cfg.judge)
    judge_raw_key = f"{judge_key_prefix_val}_judge_raw"
    judge_parsed_key = f"{judge_key_prefix_val}_judge"
    rubric = normalize_judge_rubric(getattr(cfg.judge, "rubric", "strongreject"))

    responses_by_id: dict[str, dict[str, Any]] = {}
    for r in responses:
        k = _response_join_key(r)
        if k:
            responses_by_id[k] = r

    judged_results: list[dict[str, Any]] = []
    n_skipped_no_response = 0

    for batch_idx, batch in enumerate(progress(dataloader), start=1):
        batch_ids = batch["persona_id"]
        batch_explicit_queries = batch["explicit_text_query"]
        batch_implicit_queries = batch["implicit_text_query"]
        batch_queries = batch["query"]
        batch_images = batch["image"]
        batch_malicious_intents = batch["malicious_intent"]

        for sid, explicit_query, implicit_query, query, img, malicious_intent in zip(
            batch_ids, batch_explicit_queries, batch_implicit_queries, batch_queries, batch_images, batch_malicious_intents
        ):
            sk = _dataloader_join_key(sid)
            if sk not in responses_by_id:
                n_skipped_no_response += 1
                continue

            item = dict(responses_by_id[sk])
            item["persona_id"] = sid

            existing_raw = item.get(judge_raw_key)
            existing_raw_s = str(existing_raw or "").strip()
            if existing_raw_s != "" and not existing_raw_s.startswith("ERROR:"):
                if is_strongreject_judge_parsed_conformant(item.get(judge_parsed_key)):
                    judged_results.append(item)
                    continue

            item["explicit_text_query"] = explicit_query
            item["implicit_text_query"] = implicit_query
            item["query"] = query
            response_text = item.get("response", "")
            item["malicious_intent"] = malicious_intent

            prompt = format_strongreject_user_prompt(
                rubric,
                forbidden_prompt=explicit_query,  # when judging, always uses the explicit query
                response=response_text,
                malicious_intent=malicious_intent,
            )

            user_content: list[dict[str, Any]] = []
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": pil_image_to_data_url(img)},
                }
            )
            user_content.append({"type": "text", "text": prompt})

            max_transport_retries = int(cfg.judge.max_retries)
            retry_sleep = float(cfg.judge.retry_sleep_seconds)
            max_valid_attempts = max(
                1,
                int(
                    getattr(
                        cfg.judge,
                        "max_valid_parse_attempts",
                        JUDGE_STRONGREJECT_MAX_VALID_ATTEMPTS,
                    )
                ),
            )
            had_transport_failure = False
            for valid_idx in range(max_valid_attempts):
                last_err: Exception | None = None
                for _net_attempt in range(max_transport_retries):
                    try:
                        messages: list[dict[str, Any]] = []
                        if system_prompt:
                            messages.append(
                                {"role": "system", "content": system_prompt}
                            )
                        else:
                            messages.append(
                                {
                                    "role": "system",
                                    "content": JUDGE_TEMPLATES.get(
                                        "strongreject_rubric_system"
                                    ),
                                }
                            )
                        messages.append({"role": "user", "content": user_content})

                        resp = client.chat.completions.create(
                            model=judge_model_name,
                            messages=messages,
                            **extra_args,
                        )

                        raw = ""
                        try:
                            raw = resp.choices[0].message.content or ""
                        except Exception:
                            raw = str(resp)
                        item[judge_raw_key] = raw
                        item[judge_parsed_key] = parse_judge_output_for_rubric(
                            raw, rubric
                        )
                        print("-" * 100)
                        print(f"Raw: {raw}")
                        print(f"Parsed: {item[judge_parsed_key]}")
                        print("-" * 100)
                        last_err = None
                        break
                    except Exception as e:  # pragma: no cover
                        last_err = e
                        print(f"Error: {e}")
                        time.sleep(retry_sleep * (_net_attempt + 1))
                if last_err is not None:  # pragma: no cover
                    item[judge_raw_key] = (
                        f"ERROR: {type(last_err).__name__}: {last_err}"
                    )
                    item[judge_parsed_key] = None
                    had_transport_failure = True
                    break
                if is_strongreject_judge_parsed_conformant(item.get(judge_parsed_key)):
                    break
                if valid_idx + 1 < max_valid_attempts:
                    print(
                        f"Judge parse not conformant (attempt {valid_idx + 1}/{max_valid_attempts}), "
                        f"retrying after {retry_sleep * (valid_idx + 1):.1f}s …",
                        flush=True,
                    )
                    time.sleep(retry_sleep * (valid_idx + 1))
            if (
                not had_transport_failure
                and not is_strongreject_judge_parsed_conformant(
                    item.get(judge_parsed_key)
                )
            ):
                print(
                    f"Warning: non-conformant parse after {max_valid_attempts} valid attempt(s); "
                    f"keeping last model output for this row.",
                    flush=True,
                )

            judged_results.append(item)

        if save_path is not None:
            merged_partial = _merge_response_list_with_judged(responses, judged_results)
            save_json_list(save_path, merged_partial, quiet=True)
            print(
                f"[checkpoint] After batch {batch_idx}: wrote {len(merged_partial)} row(s) to {save_path}",
                flush=True,
            )

    if n_skipped_no_response:
        print(
            f"Note: skipped {n_skipped_no_response} dataset example(s) with no matching row "
            "in the response file (by id / persona_id).",
            flush=True,
        )
    print(
        f"Judged {len(judged_results)} example(s); response file had {len(responses)} row(s).",
        flush=True,
    )
    return judged_results


@hydra.main(
    config_path="../configs",
    config_name="safety_eval_judge_label.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    response_path = get_response_path(cfg, REPO_ROOT)
    if not response_path.is_file():
        raise FileNotFoundError(
            f"Responses file not found: {response_path}\n"
            "Run response generation (batch or legacy) so this path exists, or set response_path= explicitly."
        )
    print(f"Loading responses from {response_path.resolve()}", flush=True)
    results = load_response_rows(response_path)

    dataloader = get_ours_dataloader(cfg)
    n_ds = len(dataloader.dataset)
    n_resp = len(results)
    if n_resp != n_ds:
        print(
            f"Note: response file has {n_resp} row(s) but the dataset has {n_ds} example(s); "
            "judging only runs where a response row matches a dataset id (or persona_id).",
            flush=True,
        )

    save_path = get_judge_save_path(cfg, REPO_ROOT)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Judge output path (checkpoints + final): {save_path.resolve()}", flush=True)

    overwrite = bool(getattr(cfg, "overwrite", False))
    judge_key_prefix_val = judge_key_prefix(cfg.judge)
    judge_parsed_key = f"{judge_key_prefix_val}_judge"
    if save_path.is_file() and not overwrite:
        loaded = merge_judge_rows_by_persona_id(
            load_json_list(save_path), judge_parsed_key
        )
        judged_by_pid = {
            str(r.get("persona_id")): r
            for r in loaded
            if r.get("persona_id")
        }
        completed_ids = {
            pid
            for pid, row in judged_by_pid.items()
            if is_strongreject_judge_parsed_conformant(row.get(judge_parsed_key))
        }
        n_conf = len(completed_ids)
        print(
            f"Resume mode (overwrite=false): {n_conf}/{len(judged_by_pid)} persona_id(s) "
            f"with conformant {judge_parsed_key} in {save_path} "
            f"({len(loaded)} row(s) after deduplication by persona_id); merging into responses.",
            flush=True,
        )
        results = _merge_response_list_with_judged(results, loaded)

    judged = get_api_safety_label(
        dataloader=dataloader,
        responses=results,
        base_url=str(getattr(cfg, "base_url", "") or "").strip(),
        cfg=cfg,
        save_path=save_path,
    )
    merged = _merge_response_list_with_judged(results, judged)
    save_json_list(save_path, merged)
    print_judge_summary(merged, title="Judge summary (safety_eval/judge_label)")


if __name__ == "__main__":
    main()
