"""Zhukov 2019-style recall for ordered step localization.

CrossTask evaluation metric (Zhukov et al., CVPR 2019, eq. (10)).

For each (step_text, predicted_interval, gt_interval) triple, the
prediction counts as a hit iff the *center point* of the predicted
interval falls inside the ground-truth interval. Per-video recall is

    recall_v = (# hits in v) / (# annotated steps in v)

and the dataset recall is the unweighted mean over videos that have at
least one annotated step. Reporting average-per-video (rather than
average-per-step) keeps long videos from dominating the score, which is
what Zhukov 2019 Table 2 / CrossTask leaderboards use.

This module is *only* the metric — it does not perform decoding. Pair it
with ``dp_decoder`` (constrained ordered decoding) to reproduce the
paper's experimental setup, or feed it independent argmax predictions
to measure the gap that ordering closes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class StepPrediction:
    """One predicted interval for one (video, step) pair.

    ``step_idx`` is the position of the step inside the *video's* step
    list (0-based). It is what links a prediction back to its GT row.
    """
    video_id: str
    step_idx: int
    step_text: str
    pred_start: float
    pred_end: float
    gt_start: float
    gt_end: float

    @property
    def pred_center(self) -> float:
        return 0.5 * (self.pred_start + self.pred_end)

    @property
    def is_hit(self) -> bool:
        # closed interval — matches the convention in Zhukov 2019
        return self.gt_start <= self.pred_center <= self.gt_end


def zhukov_recall(predictions: Sequence[StepPrediction]) -> Dict[str, float]:
    """Compute Zhukov-style recall, broken down per task and overall.

    Returns a dict with keys:
      * ``recall``           : mean over videos (the headline number)
      * ``recall_micro``     : mean over individual step predictions
      * ``num_videos``       : count of videos contributing
      * ``num_steps``        : count of step predictions
    """
    if not predictions:
        return {
            "recall": 0.0,
            "recall_micro": 0.0,
            "num_videos": 0.0,
            "num_steps": 0.0,
        }

    by_video: Dict[str, List[StepPrediction]] = {}
    for p in predictions:
        by_video.setdefault(p.video_id, []).append(p)

    per_video_recalls: List[float] = []
    total_hits = 0
    total_steps = 0
    for vid, preds in by_video.items():
        hits = sum(1 for p in preds if p.is_hit)
        per_video_recalls.append(hits / max(1, len(preds)))
        total_hits += hits
        total_steps += len(preds)

    return {
        "recall": float(np.mean(per_video_recalls)),
        "recall_micro": total_hits / max(1, total_steps),
        "num_videos": float(len(per_video_recalls)),
        "num_steps": float(total_steps),
    }


def zhukov_recall_by_task(
    predictions: Sequence[StepPrediction],
    video_to_task: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """Same metric, but grouped by task_id then macro-averaged.

    CrossTask reports per-task numbers in their Table 2 / 3; the
    headline number is the *task-averaged* recall (each task contributes
    equally regardless of how many videos it has). This mirrors that.
    """
    by_task: Dict[str, List[StepPrediction]] = {}
    for p in predictions:
        task = video_to_task.get(p.video_id, "unknown")
        by_task.setdefault(task, []).append(p)

    per_task: Dict[str, Dict[str, float]] = {}
    task_recalls: List[float] = []
    for task, preds in by_task.items():
        m = zhukov_recall(preds)
        per_task[task] = m
        task_recalls.append(m["recall"])
    per_task["__macro__"] = {
        "recall": float(np.mean(task_recalls)) if task_recalls else 0.0,
        "num_tasks": float(len(task_recalls)),
    }
    return per_task


def format_recall(metrics: Dict[str, float]) -> str:
    """Pretty-printer for the dataset-level result."""
    return (
        f"Zhukov recall (avg over videos): {metrics['recall']:.4f}  "
        f"[micro={metrics['recall_micro']:.4f}, "
        f"videos={int(metrics['num_videos'])}, "
        f"steps={int(metrics['num_steps'])}]"
    )


__all__ = [
    "StepPrediction",
    "zhukov_recall",
    "zhukov_recall_by_task",
    "format_recall",
]
