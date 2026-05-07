"""
GPU-accelerated Adaptive Coverage Sampling (ACS) with Persona Constraints.

Key optimisations vs. the naive baseline
-----------------------------------------
1. **Chunked top-k similarity** (GPU or CPU)
   Never materialises the full N×N matrix.  Instead, processes rows in
   chunks, runs torch.topk (or numpy argpartition) on each chunk, and
   stores only the top-d_max neighbours per node in an (N, d_max) array.
   Memory: O(N·d_max) instead of O(N²).

2. **Padded adjacency array with sentinel node**
   Neighbour lists are stored as an (N, d_max+1) int32 array.
     • col 0        — self-loop (index i)
     • cols 1..     — top neighbours with sim ≥ threshold; invalid slots
                       hold sentinel index n
   The sentinel node is permanently flagged as "covered", so padding slots
   contribute zero marginal gain without any branching.

3. **Vectorised greedy max-cover**
   Each round computes ALL N node gains in one call:
       gains = (~covered_ext[adj_padded]).sum(axis=1)   # (N,) boolean gather
   and updates coverage with a single boolean scatter.  Replaces an O(N)
   Python inner loop with an O(N·d_max) numpy/torch kernel.

4. **GPU greedy** (when CUDA/MPS is available)
   adj_padded and covered_ext live on-device; only the scalar argmax is
   transferred to the host per round.

5. **Precomputed top-k reused across binary-search iterations**
   The (N, d_max) top-k arrays are computed ONCE; each binary-search
   iteration rebuilds the adjacency via a single vectorised np.where call
   (O(N·d_max)) instead of re-sorting the full similarity matrix.

6. **Persona eligibility**
   Maintained as a flat boolean array.  A precomputed persona→member-
   indices dict allows O(|persona|) updates (≈10 ops) per greedy round
   instead of iterating over all N nodes.

Persona-Constrained Variant
----------------------------
When ``persona_ids`` is supplied each persona contributes at most one
selected sample.  Both ACSSampler and ScalableACSSampler support this.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseSampler(ABC):
    """Abstract base class for data sampling methods."""

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def sample(self, **kwargs) -> np.ndarray:
        """Select a subset of samples; returns array of selected indices."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.config})"


# ---------------------------------------------------------------------------
# ACSSampler
# ---------------------------------------------------------------------------


class ACSSampler(BaseSampler):
    """
    GPU-accelerated Adaptive Coverage Sampling (ACS).

    Parameters
    ----------
    target_coverage : float
        Desired fraction of nodes to cover (default 0.9).
    threshold_tol : float
        Binary-search convergence tolerance (default 1e-2).
    max_iter : int
        Maximum binary-search iterations (default 20).
    use_degree_constraint : bool
        Apply d_max = ⌈2·tc·N/k⌉ degree cap (default True).
    device : str
        ``'auto'`` | ``'cuda'`` | ``'mps'`` | ``'cpu'``.
        ``'auto'`` selects the best available device.
    chunk_size : int
        Number of rows per GPU matmul chunk (default 4096).
        Decrease if you run out of GPU memory.
    """

    def __init__(
        self,
        target_coverage: float = 0.9,
        threshold_tol: float = 1e-2,
        max_iter: int = 20,
        use_degree_constraint: bool = True,
        device: str = "auto",
        chunk_size: int = 4096,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_coverage = target_coverage
        self.threshold_tol = threshold_tol
        self.max_iter = max_iter
        self.use_degree_constraint = use_degree_constraint
        self._device_arg = device
        self.chunk_size = chunk_size
        self._device: Optional[str] = None  # lazily resolved

    # ------------------------------------------------------------------ #
    # Device management
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._resolve_device(self._device_arg)
            print(f"[ACSSampler] device: {self._device}")
        return self._device

    @staticmethod
    def _resolve_device(arg: str) -> str:
        if arg != "auto":
            return arg
        if _HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        if (
            _HAS_TORCH
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------ #
    # Top-k neighbour computation  (replaces full N×N similarity matrix)
    # ------------------------------------------------------------------ #

    def _compute_top_neighbors(
        self,
        embeddings: np.ndarray,
        d_max: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the d_max highest-cosine-similarity neighbours for each node.

        Dispatches to the GPU (torch) or CPU (numpy) implementation.

        Returns
        -------
        top_indices : (N, d_max) int32  — neighbor indices, desc. sim order
        top_sims    : (N, d_max) float32 — corresponding cosine similarities
        """
        n = len(embeddings)
        d_max = min(d_max, n - 1)
        if _HAS_TORCH and self.device != "cpu":
            return self._top_neighbors_torch(embeddings, d_max)
        return self._top_neighbors_numpy(embeddings, d_max)

    def _top_neighbors_torch(
        self,
        embeddings: np.ndarray,
        d_max: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Chunked GPU top-k via torch.mm + torch.topk."""
        dev = self.device
        n = len(embeddings)
        cs = self.chunk_size

        emb = torch.from_numpy(embeddings.astype(np.float32)).to(dev)
        emb = F.normalize(emb, dim=1)

        top_idx = np.empty((n, d_max), dtype=np.int32)
        top_vals = np.empty((n, d_max), dtype=np.float32)

        for start in range(0, n, cs):
            end = min(start + cs, n)
            chunk = emb[start:end]  # (B, D)
            sims = torch.mm(chunk, emb.T)  # (B, N)

            # Mask self-similarities (vectorised diagonal)
            local = torch.arange(end - start, device=dev)
            sims[local, local + start] = -2.0

            vals, idx = torch.topk(sims, k=d_max, dim=1)  # (B, d_max)
            top_idx[start:end] = idx.cpu().numpy().astype(np.int32)
            top_vals[start:end] = vals.cpu().numpy().astype(np.float32)

        return top_idx, top_vals

    def _top_neighbors_numpy(
        self,
        embeddings: np.ndarray,
        d_max: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Chunked CPU top-k via numpy argpartition (O(N) per row)."""
        n = len(embeddings)
        cs = self.chunk_size

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb = (embeddings / np.maximum(norms, 1e-8)).astype(np.float32)

        top_idx = np.empty((n, d_max), dtype=np.int32)
        top_vals = np.empty((n, d_max), dtype=np.float32)

        for start in range(0, n, cs):
            end = min(start + cs, n)
            b = end - start
            rows = np.arange(b)
            sims = (emb[start:end] @ emb.T).astype(np.float32)  # (B, N)
            sims[rows, rows + start] = -2.0  # exclude self

            if d_max < n - 1:
                # argpartition: O(N) — find top-d_max without full sort
                part = np.argpartition(-sims, d_max, axis=1)[:, :d_max]
                psims = sims[rows[:, None], part]
                order = np.argsort(-psims, axis=1)
                idx = part[rows[:, None], order]
                vals = psims[rows[:, None], order]
            else:
                idx = np.argsort(-sims, axis=1)[:, :d_max]
                vals = sims[rows[:, None], idx]

            top_idx[start:end] = idx.astype(np.int32)
            top_vals[start:end] = vals.astype(np.float32)

        return top_idx, top_vals

    def _topk_from_full_sim(
        self,
        similarities: np.ndarray,
        d_max: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a precomputed (N, N) similarity matrix to (N, d_max) top-k."""
        n = len(similarities)
        sims = similarities.astype(np.float32, copy=True)
        np.fill_diagonal(sims, -2.0)

        rows = np.arange(n)[:, None]
        if d_max < n - 1:
            part = np.argpartition(-sims, d_max, axis=1)[:, :d_max]
            psims = sims[rows, part]
            order = np.argsort(-psims, axis=1)
            idx = part[rows, order]
            vals = psims[rows, order]
        else:
            idx = np.argsort(-sims, axis=1)[:, :d_max]
            vals = sims[rows, idx]

        return idx.astype(np.int32), vals.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Padded adjacency matrix
    # ------------------------------------------------------------------ #

    def _build_adj_padded(
        self,
        top_indices: np.ndarray,
        top_sims: np.ndarray,
        threshold: float,
        n: int,
    ) -> np.ndarray:
        """
        Build an (N, d_max+1) int32 adjacency matrix with sentinel node n.

        Layout
        ------
        col 0     — self-loop (index i, always included)
        cols 1..  — top neighbours with sim ≥ threshold; sentinel n otherwise

        The sentinel (index n) is permanently marked 'covered', so padding
        slots contribute zero marginal gain without any branching.

        Complexity: O(N · d_max)  — single vectorised np.where call.
        """
        d_max = top_indices.shape[1]
        masked = np.where(top_sims >= threshold, top_indices, n).astype(np.int32)

        adj = np.empty((n, d_max + 1), dtype=np.int32)
        adj[:, 0] = np.arange(n, dtype=np.int32)  # self-loop
        adj[:, 1:] = masked
        return adj

    # ------------------------------------------------------------------ #
    # Coverage
    # ------------------------------------------------------------------ #

    def _compute_coverage_padded(
        self,
        adj_padded: np.ndarray,
        selected: List[int],
        n: int,
    ) -> float:
        """Coverage fraction from a padded adjacency array."""
        covered_ext = np.zeros(n + 1, dtype=bool)
        covered_ext[n] = True  # sentinel
        for idx in selected:
            covered_ext[adj_padded[idx]] = True
        return float(covered_ext[:n].sum()) / n

    # ------------------------------------------------------------------ #
    # Persona helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_persona_members(
        persona_ids: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """Precompute persona_id → array of member indices (called once)."""
        return {
            int(pid): np.where(persona_ids == pid)[0] for pid in np.unique(persona_ids)
        }

    # ------------------------------------------------------------------ #
    # Greedy max-cover (CPU)
    # ------------------------------------------------------------------ #

    def _greedy_max_cover_fast(
        self,
        adj_padded: np.ndarray,
        n: int,
        k: int,
        persona_ids: Optional[np.ndarray] = None,
    ) -> List[int]:
        """
        Vectorised greedy max-cover on CPU (numpy).

        Each round:
          gains = (~covered_ext[adj_padded]).sum(axis=1)   # (N,) boolean gather
          gains[ineligible] = -1
          best  = argmax(gains)

        Complexity per round: O(N · d_max)  vs.  O(N · avg_degree) Python ops
        in the set-based baseline.
        """
        covered_ext = np.zeros(n + 1, dtype=np.int32)
        covered_ext[n] = 1  # sentinel always covered
        eligible = np.ones(n, dtype=bool)

        persona_members: Optional[Dict[int, np.ndarray]] = None
        if persona_ids is not None:
            persona_members = self._build_persona_members(persona_ids)

        selected: List[int] = []

        for _ in range(k):
            # Vectorised gain: count uncovered (non-sentinel) neighbours
            gains = (covered_ext[adj_padded] == 0).sum(axis=1)  # (N,) int
            gains[~eligible] = -1

            if int(gains.max()) < 0:
                break

            best_idx = int(gains.argmax())
            selected.append(best_idx)

            # Mark neighbours covered + restore sentinel
            covered_ext[adj_padded[best_idx]] = 1
            covered_ext[n] = 1
            eligible[best_idx] = False
            if persona_members is not None:
                eligible[persona_members[int(persona_ids[best_idx])]] = False

            # Early exit when all real nodes are covered
            if covered_ext[:n].all():
                break

        return selected

    # ------------------------------------------------------------------ #
    # Greedy max-cover (GPU)
    # ------------------------------------------------------------------ #

    def _greedy_max_cover_gpu(
        self,
        adj_padded: np.ndarray,
        n: int,
        k: int,
        persona_ids: Optional[np.ndarray] = None,
    ) -> List[int]:
        """
        GPU-accelerated greedy max-cover (torch).

        adj_padded and covered_ext live on-device; only the scalar argmax
        is transferred to host per round (~one .item() call).

        Falls back to _greedy_max_cover_fast if torch is unavailable.
        """
        if not _HAS_TORCH:
            return self._greedy_max_cover_fast(adj_padded, n, k, persona_ids)

        dev = self.device
        adj_t = torch.from_numpy(adj_padded).to(dev, dtype=torch.long)

        covered_ext = torch.zeros(n + 1, dtype=torch.bool, device=dev)
        covered_ext[n] = True
        eligible = torch.ones(n, dtype=torch.bool, device=dev)

        # Precompute persona member tensors on device
        persona_members_gpu: Optional[Dict[int, torch.Tensor]] = None
        if persona_ids is not None:
            persona_members_gpu = {
                int(pid): torch.from_numpy(np.where(persona_ids == pid)[0]).to(dev)
                for pid in np.unique(persona_ids)
            }

        selected: List[int] = []

        for _ in range(k):
            gains = (~covered_ext[adj_t]).sum(dim=1)  # (N,) int
            gains[~eligible] = -1

            if gains.max().item() < 0:
                break

            best_idx = int(gains.argmax().item())
            selected.append(best_idx)

            covered_ext[adj_t[best_idx]] = True
            covered_ext[n] = True
            eligible[best_idx] = False
            if persona_members_gpu is not None:
                pid = int(persona_ids[best_idx])
                eligible[persona_members_gpu[pid]] = False

            if covered_ext[:n].all().item():
                break

        return selected

    def _run_greedy(
        self,
        adj_padded: np.ndarray,
        n: int,
        k: int,
        persona_ids: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Dispatch to GPU or CPU greedy."""
        if _HAS_TORCH and self.device != "cpu":
            return self._greedy_max_cover_gpu(adj_padded, n, k, persona_ids)
        return self._greedy_max_cover_fast(adj_padded, n, k, persona_ids)

    # ------------------------------------------------------------------ #
    # Binary search
    # ------------------------------------------------------------------ #

    def _binary_search_threshold(
        self,
        top_indices: np.ndarray,
        top_sims: np.ndarray,
        n: int,
        k: int,
        target_coverage: float,
        d_max: int,
        persona_ids: Optional[np.ndarray] = None,
    ) -> float:
        """
        Binary search for the highest threshold achieving target_coverage.

        Operates on the precomputed top-k arrays, so adjacency is rebuilt
        in O(N · d_max) per iteration (vectorised np.where) — no re-sorting.
        The greedy step is also GPU-accelerated when available.
        """
        lo = float(top_sims.min())
        hi = float(top_sims.max())
        best = lo  # fallback: densest possible graph

        for it in range(self.max_iter):
            mid = (lo + hi) / 2.0
            adj = self._build_adj_padded(top_indices, top_sims, mid, n)
            sel = self._run_greedy(adj, n, k, persona_ids=persona_ids)
            cov = self._compute_coverage_padded(adj, sel, n)

            print(f"  Iter {it:2d}: threshold={mid:.4f}, coverage={cov:.4f}")

            if cov >= target_coverage:
                best = mid
                lo = mid
            else:
                hi = mid

            if hi - lo < self.threshold_tol:
                break

        return best

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def sample(
        self,
        embeddings: np.ndarray,
        k: int,
        target_coverage: Optional[float] = None,
        precomputed_similarities: Optional[np.ndarray] = None,
        persona_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Select k samples using GPU-accelerated Adaptive Coverage Sampling.

        Parameters
        ----------
        embeddings : (N, D) float array
        k : int
            Number of samples to select.
        target_coverage : float, optional
            Override instance default.
        precomputed_similarities : (N, N) float array, optional
            Skip similarity computation if already available.
        persona_ids : (N,) int array, optional
            When provided, at most one sample per persona is selected
            (one-per-persona constraint).

        Returns
        -------
        np.ndarray of shape (k,) with selected indices.
        """
        if target_coverage is None:
            target_coverage = self.target_coverage

        n = len(embeddings)
        if k >= n:
            return np.arange(n)

        d_max = (
            int(np.ceil(2 * target_coverage * n / k))
            if self.use_degree_constraint
            else n - 1
        )
        print(f"d_max={d_max}, device={self.device}")

        # Step 1: top-k neighbours (once — reused across all binary-search iters)
        if precomputed_similarities is not None:
            print("Converting precomputed similarities to top-k format...")
            top_indices, top_sims = self._topk_from_full_sim(
                precomputed_similarities, d_max
            )
        else:
            print(f"Computing top-{d_max} neighbours per node...")
            top_indices, top_sims = self._compute_top_neighbors(embeddings, d_max)

        # Step 2: binary search for optimal threshold
        print(f"Binary search (target_coverage={target_coverage})...")
        opt_threshold = self._binary_search_threshold(
            top_indices, top_sims, n, k, target_coverage, d_max, persona_ids
        )
        print(f"Optimal threshold: {opt_threshold:.4f}")

        # Step 3: final greedy selection
        adj_padded = self._build_adj_padded(top_indices, top_sims, opt_threshold, n)
        selected = self._run_greedy(adj_padded, n, k, persona_ids)
        coverage = self._compute_coverage_padded(adj_padded, selected, n)
        print(f"Selected {len(selected)} samples with coverage={coverage:.4f}")

        return np.array(selected)

    def compute_coverage_curve(
        self,
        embeddings: np.ndarray,
        k_values: List[int],
        thresholds: List[float],
    ) -> dict:
        """
        Compute coverage for various (k, threshold) combinations.
        Useful for empirical monotonicity validation (Section 4.1).
        """
        n = len(embeddings)
        max_d = max(int(np.ceil(2 * self.target_coverage * n / k)) for k in k_values)
        top_idx, top_sim = self._compute_top_neighbors(embeddings, max_d)

        results = {"k": [], "threshold": [], "coverage": []}
        for threshold in thresholds:
            for k in k_values:
                d_k = int(np.ceil(2 * self.target_coverage * n / k))
                adj = self._build_adj_padded(
                    top_idx[:, :d_k], top_sim[:, :d_k], threshold, n
                )
                sel = self._run_greedy(adj, n, k)
                cov = self._compute_coverage_padded(adj, sel, n)
                results["k"].append(k)
                results["threshold"].append(threshold)
                results["coverage"].append(cov)
        return results


# ---------------------------------------------------------------------------
# ScalableACSSampler
# ---------------------------------------------------------------------------


class ScalableACSSampler(ACSSampler):
    """
    Scalable ACS: tunes the threshold on a pilot subsample, then applies
    it to the full dataset (Hoeffding-bound justification, Appendix B).

    All GPU / vectorisation optimisations from ACSSampler are inherited.

    Persona-constrained pilot
    -------------------------
    When ``persona_ids`` is provided the pilot is drawn by selecting one
    random intention per persona and keeping a ``subsample_ratio`` fraction
    of those personas.  This ensures the threshold is tuned under the same
    one-per-persona constraint as the full run.
    """

    def __init__(self, subsample_ratio: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.subsample_ratio = subsample_ratio

    def sample(
        self,
        embeddings: np.ndarray,
        k: int,
        target_coverage: Optional[float] = None,
        precomputed_similarities: Optional[np.ndarray] = None,
        persona_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Select k samples using the scalable ACS algorithm.

        Parameters
        ----------
        embeddings : (N, D) float array
        k : int
        target_coverage : float, optional
        precomputed_similarities : (N, N) float array, optional
            Applied to the full dataset step; the pilot always recomputes
            its own (smaller) similarity block.
        persona_ids : (N,) int array, optional
            One-per-persona constraint; pilot is also drawn with this
            constraint so threshold tuning is representative.
        """
        if target_coverage is None:
            target_coverage = self.target_coverage

        n = len(embeddings)
        if k >= n:
            return np.arange(n)

        # ---------------------------------------------------------------- #
        # Step 1: build pilot subsample
        # ---------------------------------------------------------------- #
        if persona_ids is not None:
            unique_pids = np.unique(persona_ids)
            n_personas = len(unique_pids)

            # One random intention per persona
            one_per = np.array(
                [np.random.choice(np.where(persona_ids == p)[0]) for p in unique_pids]
            )

            # Keep subsample_ratio fraction of personas for the pilot
            n_pilot = max(1, int(n_personas * self.subsample_ratio))
            pilot_mask = np.random.choice(n_personas, n_pilot, replace=False)
            sub_idx = one_per[pilot_mask]
            sub_pids = persona_ids[sub_idx]
            k_sub = n_pilot  # one selection per pilot persona
        else:
            sub_n = min(int(n * self.subsample_ratio), n)
            sub_idx = np.random.choice(n, sub_n, replace=False)
            sub_pids = None
            k_sub = max(1, int(k * sub_n / n))

        sub_n = len(sub_idx)
        sub_emb = embeddings[sub_idx]
        d_max_sub = int(np.ceil(2 * target_coverage * sub_n / k_sub))

        print(
            f"Tuning threshold on pilot: n={sub_n}, k_sub={k_sub}, "
            f"d_max_sub={d_max_sub}"
        )

        sub_top_idx, sub_top_sim = self._compute_top_neighbors(sub_emb, d_max_sub)
        opt_threshold = self._binary_search_threshold(
            sub_top_idx,
            sub_top_sim,
            sub_n,
            k_sub,
            target_coverage,
            d_max_sub,
            persona_ids=sub_pids,
        )
        print(f"Pilot threshold: {opt_threshold:.4f}")

        # ---------------------------------------------------------------- #
        # Step 2: apply threshold to full dataset
        # ---------------------------------------------------------------- #
        print("Applying threshold to full dataset...")
        d_max = (
            int(np.ceil(2 * target_coverage * n / k))
            if self.use_degree_constraint
            else n - 1
        )

        if precomputed_similarities is not None:
            top_indices, top_sims = self._topk_from_full_sim(
                precomputed_similarities, d_max
            )
        else:
            top_indices, top_sims = self._compute_top_neighbors(embeddings, d_max)

        adj_padded = self._build_adj_padded(top_indices, top_sims, opt_threshold, n)
        selected = self._run_greedy(adj_padded, n, k, persona_ids)
        coverage = self._compute_coverage_padded(adj_padded, selected, n)
        print(f"Selected {len(selected)} samples with coverage={coverage:.4f}")

        return np.array(selected)
