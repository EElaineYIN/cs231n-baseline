"""Candidate window generation and pooling.

A *window* is a contiguous span of clip-feature timesteps. We represent
each window by mean-pooling its clip features. Windows of multiple sizes
are produced so steps of different lengths can be matched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class Window:
    clip_start: int        # inclusive
    clip_end: int          # exclusive
    t_start: float         # seconds
    t_end: float           # seconds


def make_windows(
    num_clips: int,
    clip_stride: float,
    window_clip_sizes: Sequence[int],
    stride_clips: int = 1,
) -> List[Window]:
    """Generate multi-scale sliding windows.

    Args:
      num_clips        : total feature timesteps in the video.
      clip_stride      : seconds per feature timestep.
      window_clip_sizes: list of window lengths to enumerate (in clips).
      stride_clips     : shift between consecutive windows (in clips).

    Returns:
      A list of Window objects covering the video at each requested scale.
    """
    if num_clips <= 0:
        return []
    windows: List[Window] = []
    seen = set()  # (start, end) dedupe across scales
    for w in window_clip_sizes:
        w = max(1, min(int(w), num_clips))
        for s in range(0, num_clips - w + 1, max(1, int(stride_clips))):
            e = s + w
            key = (s, e)
            if key in seen:
                continue
            seen.add(key)
            windows.append(
                Window(
                    clip_start=s,
                    clip_end=e,
                    t_start=s * clip_stride,
                    t_end=e * clip_stride,
                )
            )
        # always include the trailing window if the loop's stride skipped it
        last_start = num_clips - w
        if last_start >= 0 and (last_start, last_start + w) not in seen:
            seen.add((last_start, last_start + w))
            windows.append(
                Window(
                    clip_start=last_start,
                    clip_end=last_start + w,
                    t_start=last_start * clip_stride,
                    t_end=(last_start + w) * clip_stride,
                )
            )
    # stable order: by start time, then end time
    windows.sort(key=lambda w: (w.clip_start, w.clip_end))
    return windows


def pool_windows(clip_features: np.ndarray, windows: Sequence[Window]) -> np.ndarray:
    """Mean-pool clip features inside each window.

    Returns a (num_windows, D) float32 array.
    """
    if len(windows) == 0:
        return np.zeros((0, clip_features.shape[1]), dtype=np.float32)
    out = np.empty((len(windows), clip_features.shape[1]), dtype=np.float32)
    for i, w in enumerate(windows):
        out[i] = clip_features[w.clip_start : w.clip_end].mean(axis=0)
    return out


__all__ = ["Window", "make_windows", "pool_windows"]
