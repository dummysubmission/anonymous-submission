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
"""

from pathlib import Path
from typing import Any

from datasets import Dataset
from datasets import load_dataset
from omegaconf import DictConfig
from PIL import Image as PILImage
from torch.utils.data import DataLoader


def load_ours_dataset(cfg: DictConfig) -> Dataset:
    """Load the 'ours' safety evaluation dataset from HuggingFace.

    Args:
        cfg: Hydra config with a ``dataset`` sub-config containing:
            - ``dataset_path``: local path or HF repo id
            - ``dataset_split``: ``'explicit'`` or ``'implicit'``
            - ``hf_data_dir`` (optional): HuggingFace ``data_dir`` (default ``full``)
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

    ds: Dataset = load_dataset(
        str(dataset_cfg.dataset_path),
        data_dir=str(getattr(dataset_cfg, "hf_data_dir", "full")),
        split="train",
    )

    query_field = (
        "implicit_text_query" if dataset_split == "implicit" else "explicit_text_query"
    )
    ds = ds.map(lambda ex: {"query": ex[query_field]})
    # ds = ds.map(
    #     lambda ex: {
    #         "response_prefix": ex["malicious_intent"]
    #         .replace("The actor intends", "Sure, here is how")
    #         .rstrip(".")
    #         + " based on the image and your request."
    #     }
    # )
    ds = ds.map(lambda ex: {"response_prefix": "Here is how to"})

    max_samples = getattr(dataset_cfg, "max_samples", None)
    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))

    print(f"Loaded {len(ds)} samples from 'ours' (query_type={dataset_split})")
    return ds


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
            "response_prefix": [str(item.get("response_prefix", "")) for item in batch],
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
