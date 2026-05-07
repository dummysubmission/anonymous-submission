"""Utilities for benchmark-style safety datasets (viewer JSON layout)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_repo_root() -> Path:
    return REPO_ROOT


def samples_json_path_for_benchmark(
    benchmark: str, repo_root: Path | None = None
) -> Path:
    """Return ``data/samples/<benchmark>.json`` under ``repo_root``."""
    root = repo_root if repo_root is not None else _default_repo_root()
    stem = _safe_benchmark_stem(benchmark)
    return root / "data" / "samples" / f"{stem}.json"


def _safe_benchmark_stem(name: str) -> str:
    s = name.strip()
    if not s or ".." in s or "/" in s or "\\" in s:
        raise ValueError(f"Invalid benchmark name: {name!r}")
    if s.lower().endswith(".json"):
        s = Path(s).stem
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", s):
        raise ValueError(
            f"Benchmark name must be a safe filename stem, got: {name!r}"
        )
    return s


def load_safety_dataset(
    *,
    samples_path: str | Path | None = None,
    benchmark: str | None = None,
    repo_root: Path | None = None,
    check_image_files: bool = False,
) -> list[dict[str, Any]]:
    """Load samples from a benchmark JSON list (same layout as online-sample-viewer).

    Each row is normalized to string ``id``, string ``query``, and a repo-relative
    string ``image_path`` (as stored in ``data/samples/*.json``), so
    ``dataset_audit.client_utils.resolve_image_path`` works unchanged.

    Args:
        samples_path: Path to a JSON file containing a list of objects. If relative,
            it is resolved under ``repo_root``.
        benchmark: If set, loads ``repo_root / "data" / "samples" / f"{benchmark}.json"``.
            If both ``samples_path`` and ``benchmark`` are set, ``samples_path`` wins.
        repo_root: Repository root for relative paths (default: repository root).
        check_image_files: If True, warn when ``repo_root / image_path`` is missing.

    Returns:
        List of dicts with keys ``id``, ``query``, ``image_path``.
    """
    root = repo_root if repo_root is not None else _default_repo_root()

    if samples_path is not None:
        path = Path(samples_path)
        if not path.is_absolute():
            path = root / path
    elif benchmark is not None:
        stem = _safe_benchmark_stem(benchmark)
        path = root / "data" / "samples" / f"{stem}.json"
    else:
        raise ValueError("Provide samples_path or benchmark.")

    if not path.is_file():
        raise FileNotFoundError(f"Samples file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, found {type(data).__name__}.")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected every entry in {path} to be a JSON object.")

    out: list[dict[str, Any]] = []
    for raw in data:
        sid = raw.get("id", "")
        if sid is None:
            sid = ""
        elif not isinstance(sid, str):
            sid = str(sid)

        q = raw.get("query", "")
        query = str(q) if q is not None else ""

        img = raw.get("image_path", "")
        image_path = str(img).strip() if img is not None else ""

        row = {**raw, "id": sid, "query": query, "image_path": image_path}
        out.append(row)

        if check_image_files and image_path:
            resolved = root / image_path
            if not resolved.is_file():
                import warnings

                warnings.warn(f"Missing image file: {resolved}", stacklevel=2)

    return out
