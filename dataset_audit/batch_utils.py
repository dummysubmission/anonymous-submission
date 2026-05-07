"""Path helpers for offline dataset audit batch runs."""

from __future__ import annotations

import re
from pathlib import Path

from omegaconf import DictConfig

from dataset_audit.client_utils import is_null_like


def _sanitize_key(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return s.strip("_") or "model"


def get_batch_audit_save_path(cfg: DictConfig, repo_root: Path) -> Path:
    """Canonical JSON path for batch audit output (Slurm-style when ``save_path`` is null)."""
    raw = getattr(cfg, "save_path", None)
    if not is_null_like(raw):
        path = Path(str(raw).strip())
        return path if path.is_absolute() else repo_root / path

    benchmark = str(getattr(cfg, "benchmark", "") or "").strip()
    if not benchmark:
        raise ValueError("benchmark must be set for default batch save_path.")
    model_key = _sanitize_key(str(cfg.model.short_name))
    return repo_root / "output" / "dataset_audit" / f"{benchmark}_{model_key}.json"
