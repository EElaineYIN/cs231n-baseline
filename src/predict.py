"""Argmax prediction for the independent-retrieval baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .windows import Window


@dataclass
class Prediction:
    query_text: str
    t_start: float
    t_end: float
    score: float


def predict_intervals(
    queries: Sequence[str],
    windows: Sequence[Window],
    similarities: np.ndarray,
) -> List[Prediction]:
    """For each query, return the highest-scoring window as the predicted span.

    No ordering constraints, no DP — pure independent retrieval.
    """
    if similarities.shape != (len(queries), len(windows)):
        raise ValueError(
            "similarities shape "
            f"{similarities.shape} does not match "
            f"(num_queries={len(queries)}, num_windows={len(windows)})"
        )
    preds: List[Prediction] = []
    if len(windows) == 0:
        return preds
    argmax = similarities.argmax(axis=1)
    for i, q in enumerate(queries):
        idx = int(argmax[i])
        w = windows[idx]
        preds.append(
            Prediction(
                query_text=q,
                t_start=float(w.t_start),
                t_end=float(w.t_end),
                score=float(similarities[i, idx]),
            )
        )
    return preds


__all__ = ["Prediction", "predict_intervals"]
