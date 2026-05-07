#!/usr/bin/env python3
"""Cluster persona embeddings and select one sample per cluster (closest to centroid)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from datasets import Dataset
from flash_kmeans import FlashKMeans
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))

from data_curation.generate_intention import (  # noqa: E402
    build_persona_input,
    load_persona_dataset,
)


def get_save_path(cfg: DictConfig) -> Path:
    if cfg.save_path is not None:
        return Path(cfg.save_path)
    return (
        REPO_ROOT
        / "output"
        / "select_persona"
        / f"k{int(cfg.sample_size)}"
        / "selected_uuids.json"
    )


def get_hf_export_dir(cfg: DictConfig) -> Path:
    raw = cfg.hf_export_dir
    if raw is None or str(raw) in ("null", "None", ""):
        return REPO_ROOT / "data" / "Persona-10K"
    p = Path(str(raw))
    return p if p.is_absolute() else (REPO_ROOT / p)


def _cfg_full_persona_corpus(cfg: DictConfig) -> DictConfig:
    """Same persona load as clustering, but full train split (no start_index / max_samples)."""
    c = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    c.start_index = 0
    c.max_samples = None
    return c


def _collect_rows_by_uuid(
    chosen_uuids: list[str], cfg: DictConfig
) -> list[dict[str, Any]]:
    """Scan the raw filtered persona split once; return full rows in selection order."""
    want_unique: set[str] = set(chosen_uuids)
    full_cfg = _cfg_full_persona_corpus(cfg)
    dataset = load_persona_dataset(full_cfg)
    by_uuid: dict[str, dict[str, Any]] = {}
    for row in tqdm(dataset, total=len(dataset), desc="fetch full rows"):
        uid = str(row.get("uuid") or "").strip()
        if uid not in want_unique or uid in by_uuid:
            continue
        by_uuid[uid] = dict(row)
        if len(by_uuid) == len(want_unique):
            break

    missing = [u for u in want_unique if u not in by_uuid]
    if missing:
        raise ValueError(
            f"Could not find {len(missing)} uuid(s) in full persona split, "
            f"e.g. {missing[:5]!r}"
        )
    return [by_uuid[u] for u in chosen_uuids]


def _collect_persona_rows(cfg: DictConfig) -> tuple[list[str], list[str]]:
    """Return (uuids, persona_context) for rows with non-empty build_persona_input.

    Parallelises build_persona_input via ``dataset.map`` and filtering via
    ``dataset.filter``, then deduplicates by UUID sequentially.
    """
    dataset = load_persona_dataset(cfg)
    num_proc = min(int(getattr(cfg, "num_workers", 8)), os.cpu_count() or 1)

    def _add_context(batch: dict) -> dict:
        ctxs: list[str] = []
        for i in range(len(batch["uuid"])):
            sample = {k: batch[k][i] for k in batch}
            ctxs.append(build_persona_input(sample))
        return {"_persona_ctx": ctxs}

    dataset = dataset.map(
        _add_context,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        desc="build persona_context",
    )

    dataset = dataset.filter(
        lambda x: bool(str(x.get("uuid") or "").strip()) and bool(x["_persona_ctx"]),
        num_proc=num_proc,
        desc="filter empty",
    )

    all_uuids: list[str] = dataset["uuid"]
    all_ctxs: list[str] = dataset["_persona_ctx"]

    seen: set[str] = set()
    uuids: list[str] = []
    contexts: list[str] = []
    for uid_raw, ctx in zip(all_uuids, all_ctxs):
        uid = str(uid_raw).strip()
        if uid in seen:
            continue
        seen.add(uid)
        uuids.append(uid)
        contexts.append(ctx)

    return uuids, contexts


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= *n* (Triton requires power-of-2 tile dims)."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _encode_contexts(
    model: Any,
    contexts: list[str],
    batch_size: int,
) -> np.ndarray:
    arr = model.encode(
        contexts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    return np.asarray(arr, dtype=np.float32)


def _closest_to_centroid_per_cluster(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    n_clusters: int,
    uuids: list[str],
) -> list[int]:
    """One unique row index per cluster: nearest unused member to that centroid.

    Enforces both row-index and UUID uniqueness so no UUID appears twice.
    """
    used_idx: set[int] = set()
    used_uuids: set[str] = set()
    selected: list[int] = []
    for c in range(n_clusters):
        mask = labels == c
        member_idx = np.flatnonzero(mask)
        pick: int | None = None
        if member_idx.size:
            sub = embeddings[member_idx]
            d = np.sum((sub - centroids[c]) ** 2, axis=1)
            for rank in np.argsort(d):
                candidate = int(member_idx[int(rank)])
                if candidate not in used_idx and uuids[candidate] not in used_uuids:
                    pick = candidate
                    break

        if pick is None:
            d_all = np.sum((embeddings - centroids[c]) ** 2, axis=1)
            for rank in np.argsort(d_all):
                candidate = int(rank)
                if candidate not in used_idx and uuids[candidate] not in used_uuids:
                    pick = candidate
                    break

        if pick is None:
            raise RuntimeError(
                f"Cluster {c}: could not find an unused row with a unique UUID. "
                f"Dataset may have fewer unique UUIDs than sample_size."
            )

        used_idx.add(pick)
        used_uuids.add(uuids[pick])
        selected.append(pick)
    return selected


@hydra.main(config_path="../configs", config_name="select_persona", version_base=None)
def main(cfg: DictConfig) -> None:
    sample_size = int(cfg.sample_size)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    uuids, contexts = _collect_persona_rows(cfg)
    n = len(contexts)
    print(f"Collected {n} non-empty persona_context rows (with uuid).")
    if n < sample_size:
        raise ValueError(
            f"Need at least sample_size={sample_size} rows, but only have {n}."
        )

    cache_path = cfg.embeddings_cache_path
    if cache_path is not None and str(cache_path) not in ("null", "None", ""):
        cache_path = Path(cache_path)
        if cache_path.is_file():
            print(f"Loading embeddings from {cache_path}")
            embeddings = np.load(cache_path)
            if embeddings.shape[0] != n:
                raise ValueError(
                    f"Cache has {embeddings.shape[0]} rows but dataset has {n}."
                )
        else:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                str(cfg.embedding_model_path), device=str(cfg.device)
            )
            embeddings = _encode_contexts(
                model, contexts, batch_size=int(cfg.batch_size)
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
            print(f"Saved embeddings to {cache_path}")
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            str(cfg.embedding_model_path), device=str(cfg.device)
        )
        embeddings = _encode_contexts(model, contexts, batch_size=int(cfg.batch_size))

    seed = int(cfg.seed)
    device = torch.device(str(cfg.device) if torch.cuda.is_available() else "cpu")
    dim = embeddings.shape[1]
    padded_dim = _next_pow2(dim)
    nredo = int(cfg.kmeans_n_init)

    # Triton kernels tile the full D dimension into shared memory and require
    # power-of-2 D.  For large D the tiles exceed GPU shared-memory limits, so
    # we fall back to the chunked PyTorch backend (still GPU-accelerated).
    use_triton = padded_dim <= 256
    if use_triton:
        km_dim = padded_dim
        if padded_dim != dim:
            print(f"Padding embedding dim {dim} -> {padded_dim} (Triton requires power-of-2)")
            data = torch.from_numpy(
                np.pad(embeddings, ((0, 0), (0, padded_dim - dim)))
            )
        else:
            data = torch.from_numpy(embeddings)
    else:
        km_dim = dim
        data = torch.from_numpy(embeddings)
        print(
            f"Embedding dim {dim} (padded {padded_dim}) too large for Triton "
            f"shared memory; using PyTorch GPU backend"
        )

    best_inertia = float("inf")
    best_labels: np.ndarray | None = None
    best_centroids: np.ndarray | None = None

    gpu_mem_bytes = dim * n * 4
    use_fp16 = gpu_mem_bytes > 4 * (1024 ** 3)
    compute_dtype = torch.float16 if use_fp16 else None
    if use_fp16:
        print(
            f"Data footprint ~{gpu_mem_bytes / (1024**3):.1f} GB; "
            f"using float16 compute to reduce GPU memory"
        )

    for attempt in range(nredo):
        km = FlashKMeans(
            d=km_dim,
            k=sample_size,
            niter=int(cfg.kmeans_max_iter),
            seed=seed + attempt,
            verbose=True,
            use_triton=use_triton,
            dtype=compute_dtype,
            chunk_size_data_cpu=max(n + 1, 1048576),
            device=device,
        )
        labels_t = km.fit_predict(data)
        centroids_t = km.centroids_b.squeeze(0)[:, :dim]

        c_np = centroids_t.cpu().numpy().astype(np.float32, copy=False)
        l_np = labels_t.cpu().numpy().ravel()

        assigned = c_np[l_np]
        inertia = float(np.sum((embeddings - assigned) ** 2))

        if nredo > 1:
            print(f"  k-means run {attempt + 1}/{nredo}: inertia = {inertia:.4f}")

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = l_np
            best_centroids = c_np

    centroids = best_centroids
    labels = best_labels

    row_indices = _closest_to_centroid_per_cluster(
        embeddings, labels, centroids, sample_size, uuids
    )
    chosen_uuids = [uuids[i] for i in row_indices]

    assert len(set(chosen_uuids)) == len(chosen_uuids), (
        f"UUID uniqueness violated: {len(chosen_uuids)} selected but only "
        f"{len(set(chosen_uuids))} unique"
    )

    out_path = get_save_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "sample_size": sample_size,
        "num_persona_context_rows": n,
        "uuids": chosen_uuids,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(chosen_uuids)} UUIDs to {out_path}")

    if bool(cfg.export_hf_dataset):
        hf_dir = get_hf_export_dir(cfg)
        rows = _collect_rows_by_uuid(chosen_uuids, cfg)
        hf_ds = Dataset.from_list(rows)
        hf_dir.parent.mkdir(parents=True, exist_ok=True)
        hf_ds.save_to_disk(str(hf_dir))
        print(f"Saved HuggingFace dataset ({len(hf_ds)} rows) to {hf_dir}")


if __name__ == "__main__":
    main()
