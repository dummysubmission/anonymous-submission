"""Load m3risk samples only (malicious intention fidelity requires ``malicious_intent``)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

ALLOWED_BENCHMARK = "m3risk"


def _default_repo_root() -> Path:
    return REPO_ROOT


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


def samples_json_path_m3risk(repo_root: Path | None = None) -> Path:
    """``data/samples/m3risk.json`` under ``repo_root``."""
    root = repo_root if repo_root is not None else _default_repo_root()
    return root / "data" / "samples" / f"{ALLOWED_BENCHMARK}.json"


def load_m3risk_dataset(
    *,
    samples_path: str | Path | None = None,
    benchmark: str | None = None,
    repo_root: Path | None = None,
    check_image_files: bool = False,
) -> list[dict[str, Any]]:
    """Load m3risk JSON only; each row must include non-empty ``malicious_intent``.

    If ``samples_path`` is set, it must resolve to ``.../m3risk.json``. If ``benchmark``
    is set, it must be ``m3risk``. If neither is set, loads the canonical
    ``data/samples/m3risk.json``.

    Each row is normalized to string ``id``, ``query``, ``image_path``, and
    ``malicious_intent`` (string), merged with the original keys.
    """
    root = repo_root if repo_root is not None else _default_repo_root()

    if samples_path is not None:
        path = Path(samples_path)
        if not path.is_absolute():
            path = root / path
        stem = _safe_benchmark_stem(path.stem)
        if stem != ALLOWED_BENCHMARK:
            raise ValueError(
                f"dataset_filter only supports benchmark {ALLOWED_BENCHMARK!r}; "
                f"got samples file stem {stem!r} ({path})."
            )
    elif benchmark is not None:
        stem = _safe_benchmark_stem(benchmark)
        if stem != ALLOWED_BENCHMARK:
            raise ValueError(
                f"dataset_filter only supports benchmark {ALLOWED_BENCHMARK!r}; "
                f"got {stem!r}."
            )
        path = root / "data" / "samples" / f"{stem}.json"
    else:
        path = samples_json_path_m3risk(root)

    if not path.is_file():
        raise FileNotFoundError(f"Samples file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, found {type(data).__name__}.")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected every entry in {path} to be a JSON object.")

    out: list[dict[str, Any]] = []
    for i, raw in enumerate(data):
        sid = raw.get("id", "")
        if sid is None:
            sid = ""
        elif not isinstance(sid, str):
            sid = str(sid)

        q = raw.get("query", "")
        query = str(q) if q is not None else ""

        img = raw.get("image_path", "")
        image_path = str(img).strip() if img is not None else ""

        mi = raw.get("malicious_intent")
        if mi is None or (isinstance(mi, str) and not mi.strip()):
            raise ValueError(
                f"Missing or empty malicious_intent at index {i} in {path} (id={sid!r})."
            )
        malicious_intent = str(mi).strip()

        row = {
            **raw,
            "id": sid,
            "query": query,
            "image_path": image_path,
            "malicious_intent": malicious_intent,
        }
        out.append(row)

        if check_image_files and image_path:
            resolved = root / image_path
            if not resolved.is_file():
                import warnings

                warnings.warn(f"Missing image file: {resolved}", stacklevel=2)

    return out
