#!/usr/bin/env python3
"""Select one diverse harmful intention per persona from an intention bank.

Algorithm
---------
1. Load all intention records from ``source_path`` (a flat JSON list).
2. Embed each record's ``embed_field`` (default: ``harmful_intent``) with a
   SentenceTransformer model.
3. Map string ``persona_id`` values to contiguous integers.
4. Run persona-constrained Adaptive Coverage Sampling (ACS): pick exactly
   one intention per persona while maximising semantic coverage.
5. Save selected records (``raw_generation_response`` stripped by default)
   to ``save_path``.

Usage (Hydra overrides)
-----------------------
    python3 data_curation/select_intention.py \\
        source_path=output/generate_intention_bank/Gemma4-31B-Instruct/intentions_bank_0-1000.json \\
        save_path=output/generate_intention/Gemma4-31B-Instruct/intentions_0-1000.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data_curation.accelerate_constrained_acs import ACSSampler, ScalableACSSampler


# ---------------------------------------------------------------------------
# Stdio configuration (match generate_intention_bank.py convention)
# ---------------------------------------------------------------------------


def _configure_stdio() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


_configure_stdio()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_bank(source_path: Path) -> list[dict[str, Any]]:
    with source_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in {source_path}, got {type(data).__name__}."
        )
    return data


def _embed_texts(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Return (N, D) float32 embeddings via SentenceTransformer."""
    from sentence_transformers import SentenceTransformer  # lazy import

    st_device = None if device == "auto" else device
    model = SentenceTransformer(model_name, device=st_device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def format_embed_text(sample):
    return f"Harmful intent: {sample.get('harmful_intent')}\nExplanation: {sample.get('explanation')}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="../configs", config_name="select_intention", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.source_path is None:
        raise ValueError("source_path must be provided.")
    if cfg.save_path is None:
        raise ValueError("save_path must be provided.")

    source_path = Path(cfg.source_path)
    save_path = Path(cfg.save_path)

    if not source_path.exists():
        raise FileNotFoundError(f"source_path not found: {source_path}")

    if save_path.exists() and not cfg.overwrite:
        print(
            f"Output already exists at {save_path}. " "Pass overwrite=true to re-run."
        )
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load records
    # ------------------------------------------------------------------ #
    print(f"Loading intention bank from {source_path} ...")
    t0 = time.perf_counter()
    records = _load_bank(source_path)
    print(f"  Loaded {len(records)} records  ({time.perf_counter()-t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # 2. Extract texts + build persona_id integer array
    # ------------------------------------------------------------------ #
    embed_field: str = cfg.embed_field
    if embed_field == "custom":
        texts = [format_embed_text(rec) for rec in records]
        str_persona_ids = [rec.get("persona_id") or "" for rec in records]
    else:
        for rec in records:
            if rec.get(embed_field) is None:
                raise ValueError(f"Record {rec.get('persona_id')} missing field '{embed_field}'.")
        texts = [rec.get(embed_field) or "" for rec in records]
        str_persona_ids = [rec.get("persona_id") or "" for rec in records]

    # Map string persona_id → contiguous integer index
    unique_pids = sorted(set(str_persona_ids))
    pid_to_int = {pid: i for i, pid in enumerate(unique_pids)}
    persona_ids = np.array([pid_to_int[p] for p in str_persona_ids], dtype=np.int32)
    n_personas = len(unique_pids)
    n_total = len(records)

    print(
        f"  Unique personas: {n_personas}  |  "
        f"Total intentions: {n_total}  |  "
        f"Avg per persona: {n_total / max(n_personas, 1):.1f}"
    )

    # ------------------------------------------------------------------ #
    # 3. Embed
    # ------------------------------------------------------------------ #
    print(f"\nEmbedding '{embed_field}' with '{cfg.model_name}' ...")
    t0 = time.perf_counter()
    embeddings = _embed_texts(
        texts=texts,
        model_name=cfg.model_name,
        batch_size=cfg.embed_batch_size,
        device=cfg.device,
    )
    print(f"  Embeddings: {embeddings.shape}  ({time.perf_counter()-t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # 4. Persona-constrained ACS
    # ------------------------------------------------------------------ #
    np.random.seed(cfg.seed)

    sampler_kwargs = dict(
        target_coverage=cfg.target_coverage,
        threshold_tol=cfg.threshold_tol,
        max_iter=cfg.max_iter,
        device=cfg.device,
        chunk_size=cfg.chunk_size,
    )

    if cfg.sampler_type == "scalable":
        sampler: ACSSampler = ScalableACSSampler(
            subsample_ratio=cfg.subsample_ratio, **sampler_kwargs
        )
    else:
        sampler = ACSSampler(**sampler_kwargs)

    print(
        f"\nRunning {cfg.sampler_type} ACS  "
        f"(N={n_total}, k={n_personas}, "
        f"target_coverage={cfg.target_coverage}) ..."
    )
    t0 = time.perf_counter()
    selected_indices = sampler.sample(
        embeddings=embeddings,
        k=n_personas,
        persona_ids=persona_ids,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Selected {len(selected_indices)} intentions  ({elapsed:.1f}s)")

    # ------------------------------------------------------------------ #
    # 5. Collect, sort, and save
    # ------------------------------------------------------------------ #
    drop_fields: set[str] = set(list(cfg.drop_fields))

    selected_records = [
        {k: v for k, v in records[int(idx)].items() if k not in drop_fields}
        for idx in selected_indices
    ]

    # Stable ordering by persona_id for reproducibility
    selected_records.sort(key=lambda r: r.get("persona_id", ""))

    save_path.write_text(
        json.dumps(selected_records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved {len(selected_records)} records → {save_path}")


if __name__ == "__main__":
    main()
