"""Narration-based temporal prior over clip positions.

Zhukov et al. (CVPR 2019) use the assumption that when a step is being
performed, the narrator tends to mention it. We mirror that with a
simple, training-free signal:

  1. Encode the step text and every sliding *window* of narration
     subtitles into the same embedding space (text encoder of choice).
  2. For each timestep ``t`` (in clip units), aggregate the cosine
     similarity of step text to narration windows that *cover* time
     ``t * clip_stride`` seconds.
  3. Return a ``(T, K)`` matrix that can be added (with a weight λ) to
     the visual ``(T, K)`` similarity matrix before DP decoding.

The "windows of narration" idea matters because individual subtitle
cues are very short (~2-3 s on YouTube auto-subs) and don't usually
contain the full step text. Averaging text in a small sliding window
(default 15 s) gives a denser, more matchable signal.

This is intentionally a *prior*, not a hard constraint — passing
λ = 0 reduces to vision-only. λ ≈ 0.5–1.0 is the typical operating
point for a frozen-CLIP setup; the right value should be picked by the
caller and reported as an ablation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class NarrationSegment:
    start: float
    end: float
    text: str


def load_subtitles(json_path: str) -> List[NarrationSegment]:
    """Load the JSON produced by ``tools/build_subtitles.py``."""
    with open(json_path, "r") as f:
        d = json.load(f)
    segs = []
    for s in d.get("segments", []):
        segs.append(NarrationSegment(float(s["start"]), float(s["end"]), str(s["text"])))
    return segs


def build_narration_windows(
    segments: Sequence[NarrationSegment],
    duration: float,
    *,
    window_seconds: float = 15.0,
    stride_seconds: float = 5.0,
) -> List[Tuple[float, float, str]]:
    """Slide a ``window_seconds``-wide window over the timeline.

    Each emitted window is ``(t_start, t_end, concat_text)`` where the
    concatenated text is the deduplicated union of subtitle cues that
    overlap the window. Empty windows (no overlapping cues) are kept
    with an empty string so the output array has predictable indexing
    — callers can ignore zero-energy rows later if they want.
    """
    if duration <= 0:
        return []
    windows: List[Tuple[float, float, str]] = []
    t = 0.0
    while t < duration:
        w_start = t
        w_end = min(t + window_seconds, duration)
        # collect cues that overlap [w_start, w_end)
        bag: List[str] = []
        seen = set()
        for seg in segments:
            if seg.end <= w_start or seg.start >= w_end:
                continue
            if seg.text not in seen:
                bag.append(seg.text)
                seen.add(seg.text)
        windows.append((w_start, w_end, " ".join(bag)))
        if w_end >= duration:
            break
        t += stride_seconds
    return windows


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)


def narration_score_matrix(
    step_texts: Sequence[str],
    segments: Sequence[NarrationSegment],
    duration: float,
    clip_stride: float,
    num_clips: int,
    text_encoder,
    *,
    window_seconds: float = 15.0,
    stride_seconds: float = 5.0,
) -> np.ndarray:
    """Build the (T, K) narration prior.

    Args:
      step_texts     : K step descriptions in canonical order.
      segments       : narration cues for this video (sorted by start).
      duration       : video duration in seconds.
      clip_stride    : seconds per clip in the visual feature grid.
      num_clips      : T — must match the visual features array length.
      text_encoder   : object with ``encode(list[str]) -> (N, D) float32``.

    The resulting matrix is in cosine-similarity units (roughly [-1, 1])
    so it lives on the same scale as the visual similarity matrix and
    can be added with a small λ.

    Returns:
      ``(T, K)`` float32 array. Rows that fall outside any narration
      window (e.g. silent segments) are zero.
    """
    T = int(num_clips)
    K = len(step_texts)
    if K == 0 or T == 0:
        return np.zeros((T, K), dtype=np.float32)
    if not segments:
        return np.zeros((T, K), dtype=np.float32)

    n_windows = build_narration_windows(
        segments, duration,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
    )
    if not n_windows:
        return np.zeros((T, K), dtype=np.float32)

    # encode steps once
    step_emb = text_encoder.encode(list(step_texts)).astype(np.float32)
    step_emb = _l2_normalize(step_emb)

    # encode windows; map empty text to zero vector (won't contribute)
    win_texts = [w[2] for w in n_windows]
    win_emb = text_encoder.encode(win_texts).astype(np.float32)
    # zero out empty rows so they contribute 0 cosine
    for i, txt in enumerate(win_texts):
        if not txt.strip():
            win_emb[i] = 0.0
    win_emb = _l2_normalize(win_emb + 1e-12)  # avoid NaN from all-zero rows

    # cosine sim per (window, step): (W, D) @ (D, K) -> (W, K)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        win_step_sim = win_emb @ step_emb.T  # (W, K)
    win_step_sim = np.nan_to_num(win_step_sim, nan=0.0, posinf=0.0, neginf=0.0)

    # distribute each window's sim to every clip whose center falls in it
    out = np.zeros((T, K), dtype=np.float32)
    counts = np.zeros((T,), dtype=np.float32)
    for i, (w_start, w_end, _txt) in enumerate(n_windows):
        c_lo = max(0, int(np.floor(w_start / clip_stride)))
        c_hi = min(T, int(np.ceil(w_end / clip_stride)))
        if c_hi <= c_lo:
            continue
        out[c_lo:c_hi] += win_step_sim[i]
        counts[c_lo:c_hi] += 1.0
    nz = counts > 0
    out[nz] /= counts[nz, None]
    return out


def fuse_visual_and_narration(
    visual_scores: np.ndarray,
    narration_scores: np.ndarray,
    lam: float = 0.5,
) -> np.ndarray:
    """Convex combination ``(1 - lam) * V + lam * N``.

    Both inputs are expected on the cosine-similarity scale. The output
    is the (T, K) score matrix to hand to :func:`dp_decoder.decode_ordered`.
    """
    v = np.asarray(visual_scores, dtype=np.float32)
    n = np.asarray(narration_scores, dtype=np.float32)
    if v.shape != n.shape:
        raise ValueError(f"shape mismatch: visual={v.shape} narration={n.shape}")
    lam = float(lam)
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0, 1]; got {lam}")
    return (1.0 - lam) * v + lam * n


__all__ = [
    "NarrationSegment",
    "load_subtitles",
    "build_narration_windows",
    "narration_score_matrix",
    "fuse_visual_and_narration",
]
