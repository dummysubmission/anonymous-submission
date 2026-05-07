"""Dataset utilities for safety evaluation.

Supports only the 'ours' dataset with fields:
  - image: PIL image
  - explicit_text_query: explicit harmful query
  - implicit_text_query: implicit (benign-looking) harmful query
  - persona_id: unique identifier

The ``dataset_split`` config key controls which query field is mapped as the
active ``query`` column:
  - ``'explicit'``: use ``explicit_text_query``
  - ``'implicit'``: use ``implicit_text_query``

The HuggingFace dataset is always loaded with ``split='train'`` regardless of
the ``dataset_split`` setting.

Optional ``dataset.hf_data_dir`` (default ``full``) selects the HF data shard; keep it
aligned with the adversarial attack run when using ``use_adversarial_images``.

When ``use_adversarial_images`` is true, ``image`` is loaded from disk:
``<repo>/adversarial_output_dir/<sanitize_model_dir_name(cfg)>/<persona_id>.png``,
matching ``adversarial/run_attack.py`` (requires ``cfg.model`` in the Hydra compose).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any
import random
from datasets import Dataset
from datasets import load_dataset
from omegaconf import DictConfig
from PIL import Image as PILImage
from torch.utils.data import DataLoader

from safety_eval.utils import sanitize_model_dir_name

REPO_ROOT = Path(__file__).resolve().parent.parent


def _overlay_adversarial_image(
    example: dict[str, Any], *, image_root: Path
) -> dict[str, Any]:
    pid = str(example.get("persona_id", "")).strip()
    path = image_root / f"{pid}.png"
    if not path.is_file():
        raise FileNotFoundError(
            f"Adversarial image missing for persona_id={pid!r}: expected {path}"
        )
    return {"image": PILImage.open(path).convert("RGB")}


def load_ours_dataset(cfg: DictConfig) -> Dataset:
    """Load the 'ours' safety evaluation dataset from HuggingFace.

    Args:
        cfg: Hydra config with a ``dataset`` sub-config containing:
            - ``dataset_path``: local path or HF repo id
            - ``dataset_split``: ``'explicit'`` or ``'implicit'``
            - ``hf_data_dir`` (optional): HuggingFace ``data_dir`` (default ``full``)
            - ``use_adversarial_images`` (optional): load images from attack output tree
            - ``adversarial_output_dir`` (optional): same as ``attack.output_dir`` when ``use_adversarial_images``
            - ``max_samples`` (optional): truncate dataset for quick runs

    Returns:
        HuggingFace ``Dataset`` with columns including ``persona_id``,
        ``image``, ``query``, ``explicit_text_query``, ``implicit_text_query``.
    """
    dataset_cfg = getattr(cfg, "dataset", cfg)
    dataset_split = str(dataset_cfg.dataset_split)
    if dataset_split not in ("explicit", "implicit"):
        raise ValueError(
            f"dataset_split must be 'explicit' or 'implicit', got: {dataset_split!r}"
        )

    hf_data_dir = str(getattr(dataset_cfg, "hf_data_dir", "full"))
    ds: Dataset = load_dataset(
        str(dataset_cfg.dataset_path), data_dir=hf_data_dir, split="train"
    )

    query_field = (
        "implicit_text_query" if dataset_split == "implicit" else "explicit_text_query"
    )
    ds = ds.map(lambda ex: {"query": ex[query_field]})

    max_samples = getattr(dataset_cfg, "max_samples", None)
    if max_samples is not None:
        seed = 42
        random.seed(seed)
        idxs = random.sample(range(len(ds)), min(int(max_samples), len(ds)))
        ds = ds.select(idxs)

    if bool(getattr(dataset_cfg, "use_adversarial_images", False)):
        raw_out = getattr(dataset_cfg, "adversarial_output_dir", None)
        if raw_out in (None, "", "null", "None"):
            raise ValueError(
                "dataset.use_adversarial_images requires dataset.adversarial_output_dir "
                "(e.g. output/adversarial_attack)"
            )
        out_root = Path(str(raw_out))
        if not out_root.is_absolute():
            out_root = REPO_ROOT / out_root
        image_root = out_root / sanitize_model_dir_name(cfg)
        ds = ds.map(partial(_overlay_adversarial_image, image_root=image_root))
        print(
            f"Using adversarial images under {image_root} (matches cfg.model attack outputs)."
        )

    print(f"Loaded {len(ds)} samples from 'ours' (query_type={dataset_split})")
    return ds


def ours_row_join_key_from_example(example: dict[str, Any]) -> str:
    """Join key aligned with judge/responses JSON rows (see ``analysis.judge_summary.row_join_key``).

    Prefer ``id`` when non-empty, otherwise ``persona_id``.
    """
    i = str(example.get("id") or "").strip()
    if i:
        return i
    return str(example.get("persona_id") or "").strip()


def load_ours_mini_join_keys(cfg: DictConfig) -> frozenset[str]:
    """Stable join-key set for the HF ``ours`` shard loaded by ``load_ours_dataset(cfg)``.

    Use the same OmegaConf YAML as ``mini_export_response_and_judge --dataset-config``
    (including ``max_samples`` / seed behavior) so keys match judge row identifiers.
    """
    ds = load_ours_dataset(cfg)
    keys: set[str] = set()
    for ex in ds:
        k = ours_row_join_key_from_example(dict(ex))
        if k:
            keys.add(k)
    return frozenset(keys)


def _to_pil(img: Any) -> PILImage.Image | None:
    """Coerce an image value to PIL, handling HF decoded images, paths, and dicts."""
    if img is None:
        return None
    if isinstance(img, PILImage.Image):
        return img
    if isinstance(img, (str, Path)):
        return PILImage.open(img).convert("RGB")
    if isinstance(img, dict) and "path" in img:
        return PILImage.open(img["path"]).convert("RGB")
    return img


def get_ours_collate_fn():
    """Return a collate function that produces batches with standard keys."""

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
        return {
            "persona_id": [str(item.get("persona_id", "")) for item in batch],
            "query": [str(item.get("query", "")) for item in batch],
            "image": [_to_pil(item.get("image")) for item in batch],
            "explicit_text_query": [
                str(item.get("explicit_text_query", "")) for item in batch
            ],
            "implicit_text_query": [
                str(item.get("implicit_text_query", "")) for item in batch
            ],
            "malicious_intent": [
                str(item.get("malicious_intent", "")) for item in batch
            ],
        }

    return collate_fn


def get_ours_dataloader(cfg: DictConfig) -> DataLoader:
    """Build a DataLoader over the 'ours' safety evaluation dataset."""
    dataset = load_ours_dataset(cfg)
    batch_size = int(getattr(cfg, "batch_size", 1))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=get_ours_collate_fn(),
    )
