import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from safety_eval.batch_utils import get_guard_save_path, get_response_path
from safety_eval.dataset_utils import get_ours_dataloader
from safety_eval.llm_client import generation_completion_extra_args
from safety_eval.utils import (
    load_json_list,
    load_response_rows,
    pil_image_to_data_url,
    progress,
    sanitize_model_key,
    save_json_list,
)


def get_guard_safety_label(
    dataloader,
    responses: list[dict[str, Any]],
    cfg: DictConfig,
    base_url: str,
    *,
    out_rows: list[dict[str, Any]],
    completed_ids: set[str],
) -> None:
    """
    Call a guard model served via vLLM to label each (query, response) pair.

    Output keys are model-specific:
      - <guard_model>_guard_raw: raw text output
      - <guard_model>_guard: "safe" | "unsafe" | "unparsed"
    """
    if not responses:
        return

    from openai import OpenAI

    guard_key_prefix = sanitize_model_key(cfg.guard_model.short_name)
    guard_raw_key = f"{guard_key_prefix}_guard_raw"
    guard_label_key = f"{guard_key_prefix}_guard"

    if (
        guard_raw_key in responses[0]
        and guard_label_key in responses[0]
        and cfg.guard_safety_labels_use_cache
    ):
        out_rows.clear()
        out_rows.extend(dict(r) for r in responses)
        return

    client = OpenAI(api_key="EMPTY", base_url=base_url)
    model_name = client.models.list().data[0].id
    comp_kwargs = generation_completion_extra_args(cfg, model_id=model_name)

    def _parse_guard_label(raw: str) -> str:
        if not raw:
            return "unparsed"
        for line in raw.splitlines():
            s = line.strip().lower()
            if not s:
                continue
            if s == "safe":
                return "safe"
            if s == "unsafe":
                return "unsafe"
        return "unparsed"

    responses_by_id: dict[str, dict[str, Any]] = {}
    for r in responses:
        pid = str(r.get("persona_id", "")).strip()
        if pid:
            responses_by_id[pid] = dict(r)

    n_skipped_no_response = 0

    for batch in progress(dataloader):
        batch_ids = batch["persona_id"]
        batch_queries = batch["query"]
        batch_images = batch["image"]

        for sid, q, img in zip(batch_ids, batch_queries, batch_images):
            pid = str(sid).strip()
            if not pid:
                continue
            if pid in completed_ids:
                continue
            if pid not in responses_by_id:
                n_skipped_no_response += 1
                continue

            item = dict(responses_by_id[pid])

            item["query"] = q
            response_text = item.get("response", "")

            user_content: list[dict[str, Any]] = []
            if img is not None:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": pil_image_to_data_url(img)}}
                )
            user_content.append({"type": "text", "text": q})

            messages: list[dict[str, Any]] = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response_text},
            ]

            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **comp_kwargs,
            )
            raw = resp.choices[0].message.content or ""
            item[guard_raw_key] = raw
            item[guard_label_key] = _parse_guard_label(raw)
            out_rows.append(item)
            completed_ids.add(pid)

    if n_skipped_no_response:
        print(
            f"Note: skipped {n_skipped_no_response} dataset example(s) with no matching row "
            "in the response file (by persona_id).",
            flush=True,
        )


@hydra.main(
    config_path="../configs",
    config_name="safety_eval_guard_label.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    response_path = get_response_path(cfg, REPO_ROOT)
    if not response_path.is_file():
        raise FileNotFoundError(
            f"Responses file not found: {response_path}\n"
            "Run response generation (batch or server) so this path exists, or set response_path= explicitly."
        )
    print(f"Loading responses from {response_path.resolve()}", flush=True)
    responses = load_response_rows(response_path)

    save_path = get_guard_save_path(cfg, REPO_ROOT)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    guard_key_prefix = sanitize_model_key(str(cfg.guard_model.short_name))
    guard_raw_key = f"{guard_key_prefix}_guard_raw"

    overwrite = bool(getattr(cfg, "overwrite", False))
    if not overwrite and save_path.is_file():
        out_rows: list[dict[str, Any]] = load_json_list(save_path)
    else:
        out_rows = []

    completed_ids: set[str] = {
        str(r.get("persona_id", "")).strip()
        for r in out_rows
        if str(r.get("persona_id", "")).strip() and r.get(guard_raw_key)
    }
    if completed_ids:
        print(
            f"[resume] {len(completed_ids)} persona_id(s) already have {guard_raw_key} in {save_path}",
            flush=True,
        )

    dataloader = get_ours_dataloader(cfg)
    base_url = str(getattr(cfg, "base_url", "") or "").strip()

    get_guard_safety_label(
        dataloader=dataloader,
        responses=responses,
        cfg=cfg,
        base_url=base_url,
        out_rows=out_rows,
        completed_ids=completed_ids,
    )
    save_json_list(save_path, out_rows)
    print(f"Done. Wrote {len(out_rows)} row(s) to {save_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()

