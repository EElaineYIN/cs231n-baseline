"""Visual-text scoring."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)


def _seeded_random_projection(in_dim: int, out_dim: int, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    # scale roughly preserves variance
    proj /= np.sqrt(max(1, in_dim))
    return proj


def score(
    window_features: np.ndarray,
    text_features: np.ndarray,
    *,
    align_dims: bool = True,
    projection_seed: int = 1234,
) -> np.ndarray:
    """Cosine similarity matrix between window features and text queries.

    Args:
      window_features : (N_windows, D_v)
      text_features   : (N_queries, D_t)
      align_dims      : if dims differ, project the higher-d side down with a
                        seeded random projection. This is a known weak
                        no-training cross-modal alignment.

    Returns:
      (N_queries, N_windows) cosine similarities.
    """
    w = np.asarray(window_features, dtype=np.float32)
    t = np.asarray(text_features, dtype=np.float32)

    if w.shape[1] != t.shape[1]:
        if not align_dims:
            raise ValueError(
                f"feature dim mismatch: visual={w.shape[1]} text={t.shape[1]}"
            )
        target = min(w.shape[1], t.shape[1])
        if w.shape[1] != target:
            proj = _seeded_random_projection(w.shape[1], target, projection_seed)
            w = w @ proj
        if t.shape[1] != target:
            proj = _seeded_random_projection(t.shape[1], target, projection_seed + 1)
            t = t @ proj

    w = _l2_normalize(w)
    t = _l2_normalize(t)
    # numpy 2.0 occasionally emits spurious divide/overflow warnings from
    # the BLAS matmul path on small inputs; suppress only here.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = t @ w.T  # (N_queries, N_windows)
    return np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)


__all__ = ["score"]
