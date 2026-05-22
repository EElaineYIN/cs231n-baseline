"""Zhukov-lite baseline: frozen CLIP + ordered DP + narration prior.

End-to-end pipeline that mirrors the recipe in Zhukov et al. (CVPR 2019)
on top of the existing CLIP-feature retrieval baseline:

    visual sim (T, K)               <- frozen CLIP, cosine
    + narration sim (T, K)          <- frozen text-enc over auto-subs
    -> DP-decode K ordered intervals
    -> evaluate against ground truth (IoU + Zhukov recall)

For comparison the same per-clip visual matrix is also decoded with a
naive ``argmax-per-step`` (the existing run_baseline.py's regime), so a
single run produces three numbers side-by-side:

    {argmax_visual,  dp_visual,  dp_visual_plus_narration}

This is a *separate* entry point — it imports loaders and the text
encoder from ``src/`` but does not modify ``run_baseline.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.data import VideoSample, load_crosstask
from src.text_encoder import get_text_encoder
from src.evaluate import evaluate, format_metrics, GroundedExample
from src.dp_decoder import decode_ordered, decode_pointwise, StepInterval
from src.recall_metric import StepPrediction, zhukov_recall, format_recall
from src.narration_constraint import (
    NarrationSegment,
    load_subtitles,
    narration_score_matrix,
    fuse_visual_and_narration,
)


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)


def visual_score_matrix(
    clip_features: np.ndarray, step_emb: np.ndarray
) -> np.ndarray:
    """Cosine similarity per (clip, step). Returns (T, K)."""
    v = _l2_normalize(np.asarray(clip_features, dtype=np.float32))
    t = _l2_normalize(np.asarray(step_emb, dtype=np.float32))
    # numpy 2.0 sometimes warns spuriously on small BLAS matmuls; mirror
    # scoring.py and silence here so the run output stays readable.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = v @ t.T
    return np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def argmax_intervals(
    scores: np.ndarray, clip_stride: float
) -> List[Tuple[float, float]]:
    """Independent per-step argmax (no ordering); each step gets the
    single best clip mapped to a 1-clip-wide interval. Mirrors the
    no-window flavor of the existing baseline's regime when window_size=1.
    """
    T, K = scores.shape
    out: List[Tuple[float, float]] = []
    for k in range(K):
        t_star = int(np.argmax(scores[:, k]))
        out.append((t_star * clip_stride, (t_star + 1) * clip_stride))
    return out


def intervals_to_seconds(
    intervals: List[StepInterval], clip_stride: float
) -> List[Tuple[float, float]]:
    return [iv.to_seconds(clip_stride) for iv in intervals]


def evaluate_run(
    sample_intervals: List[Tuple[VideoSample, List[Tuple[float, float]]]],
) -> Tuple[Dict[str, float], Dict[str, float], List[GroundedExample]]:
    """Compute both IoU-based and Zhukov-recall metrics for one regime.

    ``sample_intervals`` is a list of (sample, intervals_per_step_in_order).
    Returns (iou_metrics, recall_metrics, qualitative_examples).
    """
    iou_pairs: List[Tuple[VideoSample, List]] = []
    recall_preds: List[StepPrediction] = []

    # We reuse evaluate() by constructing a (pred, query) sequence per video.
    # evaluate() expects per_video_pairs = [(sample, [Prediction-like, ...])]
    # but Prediction-like only needs .query_text, .t_start, .t_end attributes
    # for the IoU path. We use a tiny shim.

    @dataclass
    class _P:
        query_text: str
        t_start: float
        t_end: float
        score: float = 0.0

    for sample, ivals in sample_intervals:
        if len(ivals) != len(sample.steps):
            # We always emit one interval per GT step (in order), so this
            # should never trip; assert for debugging.
            raise AssertionError(
                f"video {sample.video_id}: got {len(ivals)} intervals "
                f"for {len(sample.steps)} steps"
            )
        preds = []
        for k, (step, (ts, te)) in enumerate(zip(sample.steps, ivals)):
            preds.append(_P(query_text=step.text, t_start=ts, t_end=te))
            recall_preds.append(
                StepPrediction(
                    video_id=sample.video_id,
                    step_idx=k,
                    step_text=step.text,
                    pred_start=ts,
                    pred_end=te,
                    gt_start=step.start,
                    gt_end=step.end,
                )
            )
        iou_pairs.append((sample.video_id, sample.steps, preds))

    iou_metrics, examples = evaluate(iou_pairs)
    recall_metrics = zhukov_recall(recall_preds)
    return iou_metrics, recall_metrics, examples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--crosstask-root", required=True,
                    help="path to data/crosstask/crosstask_release")
    ap.add_argument("--features-dir", required=True,
                    help="path to clip_features dir with <vid>.npy")
    ap.add_argument("--subs-dir", default=None,
                    help="path to dir with <vid>.json subtitles "
                         "(skip narration if not provided)")
    ap.add_argument("--clip-stride", type=float, default=1.0,
                    help="seconds per clip in the visual feature grid")
    ap.add_argument("--text-encoder", default="clip",
                    choices=["hash", "sentence-transformers", "clip"])
    ap.add_argument("--lam-narration", type=float, default=0.5,
                    help="convex weight on narration prior in fused score")
    ap.add_argument("--min-clip-len", type=int, default=1)
    ap.add_argument("--max-clip-len", type=int, default=None)
    ap.add_argument("--narr-window", type=float, default=15.0,
                    help="narration sliding window length (seconds)")
    ap.add_argument("--narr-stride", type=float, default=5.0,
                    help="narration sliding window stride (seconds)")
    ap.add_argument("--out-dir", default="outputs",
                    help="where to write metrics + predictions")
    args = ap.parse_args()

    print(f"[load] crosstask root={args.crosstask_root}")
    print(f"       features    ={args.features_dir}")
    samples = load_crosstask(
        crosstask_root=args.crosstask_root,
        features_dir=args.features_dir,
        clip_stride=args.clip_stride,
    )
    samples = [s for s in samples if len(s.steps) > 0 and s.num_clips > 0]
    print(f"[load] {len(samples)} usable videos")

    print(f"[setup] text encoder = {args.text_encoder}")
    encoder = get_text_encoder(args.text_encoder)

    use_narration = args.subs_dir is not None
    if use_narration:
        print(f"[setup] narration prior on (lam={args.lam_narration}, "
              f"win={args.narr_window}s, stride={args.narr_stride}s)")
    else:
        print("[setup] narration prior disabled (no --subs-dir)")

    # accumulate per-regime intervals
    by_regime: Dict[str, List[Tuple[VideoSample, List[Tuple[float, float]]]]] = {
        "argmax_visual": [],
        "dp_visual": [],
    }
    if use_narration:
        by_regime["dp_visual_plus_narration"] = []

    n_with_subs = 0
    n_skipped_short = 0
    for s in samples:
        step_texts = [st.text for st in s.steps]
        K = len(step_texts)
        T = s.num_clips
        if T < max(K * args.min_clip_len, K):
            # DP infeasible for this video — skip from DP regimes, but keep
            # argmax (which has no ordering constraint) so the comparison
            # stays apples-to-apples on the same denominator we *can*
            # actually score.
            n_skipped_short += 1

        step_emb = encoder.encode(step_texts).astype(np.float32)
        S_v = visual_score_matrix(s.clip_features, step_emb)

        # regime 1: argmax (independent per-step)
        by_regime["argmax_visual"].append(
            (s, argmax_intervals(S_v, args.clip_stride))
        )

        # regime 2: DP with visual only
        if T >= max(K * args.min_clip_len, K):
            ivals_dp = decode_ordered(
                S_v, min_clip_len=args.min_clip_len, max_clip_len=args.max_clip_len
            )
            by_regime["dp_visual"].append(
                (s, intervals_to_seconds(ivals_dp, args.clip_stride))
            )

        # regime 3: DP with visual + narration
        if use_narration and T >= max(K * args.min_clip_len, K):
            sub_path = os.path.join(args.subs_dir, f"{s.video_id}.json")
            if os.path.exists(sub_path):
                segs = load_subtitles(sub_path)
                n_with_subs += 1
                S_n = narration_score_matrix(
                    step_texts=step_texts,
                    segments=segs,
                    duration=s.duration,
                    clip_stride=args.clip_stride,
                    num_clips=T,
                    text_encoder=encoder,
                    window_seconds=args.narr_window,
                    stride_seconds=args.narr_stride,
                )
                S_fused = fuse_visual_and_narration(S_v, S_n, lam=args.lam_narration)
            else:
                # no subs for this video → fall back to visual-only
                S_fused = S_v
            ivals_fused = decode_ordered(
                S_fused, min_clip_len=args.min_clip_len, max_clip_len=args.max_clip_len
            )
            by_regime["dp_visual_plus_narration"].append(
                (s, intervals_to_seconds(ivals_fused, args.clip_stride))
            )

    if n_skipped_short:
        print(f"[warn] {n_skipped_short} videos had T < K*min_clip_len; "
              "DP regimes drop them, argmax keeps them")
    if use_narration:
        print(f"[info] narration prior applied to {n_with_subs}/{len(samples)} videos")

    # evaluate each regime
    os.makedirs(args.out_dir, exist_ok=True)
    summary: Dict[str, Dict[str, float]] = {}
    for regime, pairs in by_regime.items():
        if not pairs:
            print(f"[skip] regime {regime} has 0 videos")
            continue
        iou_m, rec_m, examples = evaluate_run(pairs)
        summary[regime] = {
            **{f"iou.{k}": v for k, v in iou_m.items()},
            **{f"recall.{k}": v for k, v in rec_m.items()},
        }
        print()
        print(f"=== regime: {regime}  (n_videos={len(pairs)}) ===")
        print(format_metrics(iou_m))
        print(format_recall(rec_m))

    # dump structured outputs
    summary_path = os.path.join(args.out_dir, "metrics_zhukov_lite.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[write] {summary_path}")


if __name__ == "__main__":
    main()
