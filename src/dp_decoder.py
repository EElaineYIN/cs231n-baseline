"""Dynamic programming for ordered step decoding.

Given a score matrix ``S`` of shape ``(T, K)`` — score of assigning the
t-th frame/clip to the k-th step (steps in their *canonical order*) —
return the highest-scoring assignment of each step to a contiguous
interval ``[a_k, b_k)`` subject to

    0 <= a_0 < b_0 <= a_1 < b_1 <= ... <= a_{K-1} < b_{K-1} <= T

i.e. each step occupies a non-empty contiguous interval and the
intervals appear in order without overlap.

This is the same ordering constraint as Zhukov et al. (CVPR 2019,
§4.3): every step from the procedure's canonical list must be grounded
exactly once, and matches must respect the script order. The score of
an interval is the *sum* of per-frame scores inside it; the optimum is
found in ``O(K · T^2)`` time and ``O(K · T)`` memory by classic
1D interval DP.

Two utility wrappers are provided:

  * :func:`decode_ordered` — full DP, lets steps span multiple frames.
  * :func:`decode_pointwise` — restricts each step to a single frame
    (``b_k = a_k + 1``); this is the discrete monotone-assignment
    variant and is a useful comparison point because it can never beat
    the multi-frame decoder.

Neither does anything stochastic: same inputs always produce the same
intervals (with ties broken by the earliest position).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class StepInterval:
    """One decoded step occupying clips ``[clip_start, clip_end)``.

    ``score`` is the *sum* of per-frame scores inside the interval
    (unnormalized — longer intervals therefore accumulate more, which
    matches the DP objective).
    """
    step_idx: int
    clip_start: int    # inclusive
    clip_end: int      # exclusive
    score: float

    def to_seconds(self, clip_stride: float) -> Tuple[float, float]:
        """Convert clip-index bounds to time bounds in seconds.

        Assumes clip ``i`` covers ``[i * stride, (i + 1) * stride)``.
        """
        return (self.clip_start * clip_stride, self.clip_end * clip_stride)


def _validate_scores(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"scores must be 2D (T, K); got shape {s.shape}")
    return s


def decode_ordered(
    scores: np.ndarray,
    *,
    min_clip_len: int = 1,
    max_clip_len: Optional[int] = None,
) -> List[StepInterval]:
    """Decode K ordered, non-overlapping intervals.

    Args:
      scores         : (T, K). ``scores[t, k]`` = score of frame t under step k.
      min_clip_len   : every interval must span at least this many clips.
      max_clip_len   : optional cap on per-interval length. ``None`` = no cap.

    Returns:
      A list of K :class:`StepInterval`, one per step in order.

    Raises:
      ValueError if no valid assignment exists (e.g. ``T < K * min_clip_len``).
    """
    s = _validate_scores(scores)
    T, K = s.shape
    if K == 0:
        return []
    if T < K * min_clip_len:
        raise ValueError(
            f"cannot fit {K} steps of min length {min_clip_len} into T={T} clips"
        )

    # Precompute per-step prefix sums so interval-sum is O(1).
    # cum[k, t] = sum_{i < t} s[i, k];  interval sum [a, b) = cum[k, b] - cum[k, a].
    cum = np.zeros((K, T + 1), dtype=np.float64)
    cum[:, 1:] = np.cumsum(s.T, axis=1)

    NEG_INF = -np.inf

    # dp[k, b] = best total score using steps 0..k where step k ends at b (exclusive).
    # back[k, b] = best (a, prev_b) recovered as a_k and b_{k-1}.
    dp = np.full((K, T + 1), NEG_INF, dtype=np.float64)
    back_a = np.full((K, T + 1), -1, dtype=np.int64)

    # base case: step 0 spans [a, b) with a in [0, b - min_clip_len]
    # we compute dp[0, b] = max_a cum[0, b] - cum[0, a]
    # constraint: b - a >= min_clip_len, and optionally b - a <= max_clip_len, and a >= 0
    for b in range(min_clip_len, T + 1):
        a_hi = b - min_clip_len  # inclusive
        a_lo = 0 if max_clip_len is None else max(0, b - max_clip_len)
        if a_lo > a_hi:
            continue
        # interval score = cum[0, b] - cum[0, a]; minimize cum[0, a] over a in [a_lo, a_hi]
        a_star = a_lo + int(np.argmin(cum[0, a_lo : a_hi + 1]))
        dp[0, b] = cum[0, b] - cum[0, a_star]
        back_a[0, b] = a_star

    # recursion: dp[k, b] = max over a in [k*min_clip_len, b-min_clip_len]
    #     dp[k-1, a] + (cum[k, b] - cum[k, a])
    # i.e. max_a (dp[k-1, a] - cum[k, a])  +  cum[k, b]
    for k in range(1, K):
        # earliest possible end-of-step-k = (k+1) * min_clip_len
        # latest possible start-of-step-k = b - min_clip_len
        # a must be at least k * min_clip_len (room for prior steps)
        for b in range((k + 1) * min_clip_len, T + 1):
            a_hi = b - min_clip_len
            a_lo = k * min_clip_len
            if max_clip_len is not None:
                a_lo = max(a_lo, b - max_clip_len)
            if a_lo > a_hi:
                continue
            slice_vals = dp[k - 1, a_lo : a_hi + 1] - cum[k, a_lo : a_hi + 1]
            if not np.isfinite(np.max(slice_vals)):
                continue
            a_star = a_lo + int(np.argmax(slice_vals))
            dp[k, b] = dp[k - 1, a_star] + (cum[k, b] - cum[k, a_star])
            back_a[k, b] = a_star

    # find best ending position for the last step
    last_row = dp[K - 1]
    if not np.any(np.isfinite(last_row)):
        raise ValueError("DP failed to find a feasible assignment")
    b_star = int(np.argmax(last_row))
    if not np.isfinite(dp[K - 1, b_star]):
        raise ValueError("DP failed to find a feasible assignment")

    # backtrack to recover (a_k, b_k) for k = K-1 .. 0
    intervals: List[StepInterval] = []
    b_cur = b_star
    for k in range(K - 1, -1, -1):
        a_cur = int(back_a[k, b_cur])
        if a_cur < 0:
            raise ValueError(f"backtrack failed at step k={k}, b={b_cur}")
        seg_score = float(cum[k, b_cur] - cum[k, a_cur])
        intervals.append(StepInterval(step_idx=k, clip_start=a_cur, clip_end=b_cur, score=seg_score))
        b_cur = a_cur
    intervals.reverse()
    return intervals


def decode_pointwise(scores: np.ndarray) -> List[StepInterval]:
    """Restrict each step to a single frame, then DP.

    Faster (``O(K · T)``) special case of :func:`decode_ordered` with
    ``min_clip_len = max_clip_len = 1``. Useful as a sanity check on
    the full decoder and as a reference for ablation tables.
    """
    s = _validate_scores(scores)
    T, K = s.shape
    if K == 0:
        return []
    if T < K:
        raise ValueError(f"need T >= K, got T={T} K={K}")

    NEG_INF = -np.inf
    dp = np.full((K, T), NEG_INF, dtype=np.float64)
    back = np.full((K, T), -1, dtype=np.int64)

    dp[0] = s[:, 0]  # step 0 can be placed at any t
    for k in range(1, K):
        # running max of dp[k-1, :t] as we sweep t
        best_so_far = NEG_INF
        best_idx = -1
        for t in range(k, T):
            val = dp[k - 1, t - 1]
            if val > best_so_far:
                best_so_far = val
                best_idx = t - 1
            if np.isfinite(best_so_far):
                dp[k, t] = best_so_far + s[t, k]
                back[k, t] = best_idx

    last_row = dp[K - 1]
    if not np.any(np.isfinite(last_row)):
        raise ValueError("pointwise DP failed to find a feasible assignment")
    t_star = int(np.argmax(last_row))

    intervals: List[StepInterval] = []
    t_cur = t_star
    for k in range(K - 1, -1, -1):
        intervals.append(StepInterval(step_idx=k, clip_start=t_cur, clip_end=t_cur + 1, score=float(s[t_cur, k])))
        t_cur = int(back[k, t_cur])
        if t_cur < 0 and k > 0:
            raise ValueError(f"backtrack failed at k={k}")
    intervals.reverse()
    return intervals


__all__ = [
    "StepInterval",
    "decode_ordered",
    "decode_pointwise",
]
