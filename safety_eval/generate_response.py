import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
import time
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from safety_eval.batch_utils import get_responses_save_path
from safety_eval.dataset_utils import get_ours_dataloader
from safety_eval.llm_client import (
    build_generation_chat_client,
    generation_completion_extra_args,
)
from safety_eval.utils import (
    load_json_list,
    pil_image_to_base64,
    pil_image_to_data_url,
    progress,
    save_json_list,
)

SAVE_INTERVAL_BATCHES = 2

load_dotenv(".env")


def generate_response_with_server(
    dataloader,
    cfg: DictConfig,
    base_url: str,
    *,
    save_path: Path,
    all_rows: list[dict[str, object]],
) -> None:
    client, model_name = build_generation_chat_client(cfg, base_url)
    extra_args = generation_completion_extra_args(cfg, model_id=model_name)
    if cfg.model.provider == "vllm":
        extra_args["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": cfg.enable_thinking}
        }

    overwrite = bool(getattr(cfg, "overwrite", False))
    done_ids: set[str] = set()
    if not overwrite and all_rows:
        done_ids = {
            str(r.get("persona_id", "")).strip()
            for r in all_rows
            if str(r.get("persona_id", "")).strip()
        }
        if done_ids:
            print(
                f"[skip] {len(done_ids)} persona_id(s) already in {save_path}, skipping them",
                flush=True,
            )

    n_new = 0
    for batch_idx, batch in enumerate(progress(dataloader), start=1):
        batch_ids = batch["persona_id"]
        batch_queries = batch["query"]
        batch_images = batch["image"]

        for sid, q, img in zip(batch_ids, batch_queries, batch_images):
            pid = str(sid).strip()
            if not pid or pid in done_ids:
                continue

            content: list[dict[str, object]] = []
            if model_name == "claude-sonnet-4-6":
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": pil_image_to_base64(img),
                        },
                    }
                )
            else:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_image_to_data_url(img)},
                    }
                )
            content.append({"type": "text", "text": q})

            try:
                if model_name == "claude-sonnet-4-6":
                    resp = client.messages.create(
                        model=model_name,
                        messages=[{"role": "user", "content": content}],
                        **extra_args,
                    )
                    time.sleep(2)
                else:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": content}],
                        **extra_args,
                    )
            except Exception as e:
                print(f"[BadRequest] persona_id={sid}: {e}")
                text = "BadRequest"
                all_rows.append({"persona_id": pid, "query": q, "response": text})
                done_ids.add(pid)
                n_new += 1
                continue

            if hasattr(resp, "choices") and resp.choices:
                text = resp.choices[0].message.content or ""
                print("-" * 100)
                print(text)
                print("-" * 100)
            elif hasattr(resp, "content") and resp.content:
                # claude uses content instead of choices
                text = resp.content[0].text
                print("-" * 100)
                print(text)
                print("-" * 100)
            else:
                text = f"Error: {resp}"
                print(text)
            all_rows.append({"persona_id": pid, "query": q, "response": text})
            done_ids.add(pid)
            n_new += 1

        if batch_idx % SAVE_INTERVAL_BATCHES == 0:
            print(
                f"[checkpoint] Saving after batch {batch_idx} ({len(all_rows)} row(s) in file, "
                f"+{n_new} new this run)",
                flush=True,
            )
            save_json_list(save_path, all_rows, quiet=True)


@hydra.main(
    config_path="../configs",
    config_name="safety_eval_generate_response.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    save_path = get_responses_save_path(cfg, REPO_ROOT)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    overwrite = bool(getattr(cfg, "overwrite", False))
    if not overwrite and save_path.is_file():
        raw_all_rows: list[dict[str, object]] = load_json_list(save_path)
        all_rows = []
        for r in raw_all_rows:
            if (
                str(r.get("persona_id", "")).strip()
                and r.get("response", "")
                and r.get("response", "") != "BadRequest"
            ):
                all_rows.append(r)
        print(
            f"Loaded {len(all_rows)} valid row(s) from {save_path.resolve()}",
            flush=True,
        )
    else:
        all_rows = []

    dataloader = get_ours_dataloader(cfg)
    base_url = str(getattr(cfg, "base_url", "") or "").strip()

    generate_response_with_server(
        dataloader, cfg, base_url, save_path=save_path, all_rows=all_rows
    )
    save_json_list(save_path, all_rows)
    print(f"Done. Wrote {len(all_rows)} row(s) to {save_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
