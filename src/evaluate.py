"""Evaluation metrics for temporal grounding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .data import Step
from .predict import Prediction


def temporal_iou(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    p_s, p_e = pred
    g_s, g_e = gt
    if p_e <= p_s or g_e <= g_s:
        return 0.0
    inter = max(0.0, min(p_e, g_e) - max(p_s, g_s))
    union = max(p_e, g_e) - min(p_s, g_s)
    if union <= 0:
        return 0.0
    return inter / union


@dataclass
class GroundedExample:
    video_id: str
    step_text: str
    pred_start: float
    pred_end: float
    gt_start: float
    gt_end: float
    iou: float


def evaluate(
    per_video_pairs: Sequence[Tuple[str, Sequence[Step], Sequence[Prediction]]],
    iou_thresholds: Sequence[float] = (0.3, 0.5, 0.7),
) -> Tuple[Dict[str, float], List[GroundedExample]]:
    """Compute Recall@1 at each IoU threshold and mean IoU over all step queries.

    Args:
      per_video_pairs: iterable of (video_id, ground_truth_steps, predictions),
                       where predictions[i] corresponds to ground_truth_steps[i].
      iou_thresholds : IoU cutoffs for Recall@1.

    Returns:
      metrics dict, flat list of GroundedExample for qualitative inspection.
    """
    ious: List[float] = []
    examples: List[GroundedExample] = []

    for video_id, gt_steps, preds in per_video_pairs:
        if len(gt_steps) != len(preds):
            raise ValueError(
                f"video {video_id}: got {len(preds)} predictions for "
                f"{len(gt_steps)} ground-truth steps"
            )
        for step, pred in zip(gt_steps, preds):
            iou = temporal_iou(
                (pred.t_start, pred.t_end),
                (step.start, step.end),
            )
            ious.append(iou)
            examples.append(
                GroundedExample(
                    video_id=video_id,
                    step_text=step.text,
                    pred_start=pred.t_start,
                    pred_end=pred.t_end,
                    gt_start=step.start,
                    gt_end=step.end,
                    iou=iou,
                )
            )

    if not ious:
        return ({"num_queries": 0, "mIoU": 0.0}, examples)

    ious_arr = np.asarray(ious, dtype=np.float64)
    metrics: Dict[str, float] = {"num_queries": float(len(ious))}
    for thr in iou_thresholds:
        metrics[f"R@1@{thr:.1f}"] = float((ious_arr >= thr).mean())
    metrics["mIoU"] = float(ious_arr.mean())
    return metrics, examples


def format_metrics(metrics: Dict[str, float]) -> str:
    lines = [f"  num_queries : {int(metrics.get('num_queries', 0))}"]
    for k in sorted(k for k in metrics if k.startswith("R@1@")):
        lines.append(f"  {k:<12}: {metrics[k] * 100:6.2f} %")
    if "mIoU" in metrics:
        lines.append(f"  {'mIoU':<12}: {metrics['mIoU'] * 100:6.2f} %")
    return "\n".join(lines)


def format_qualitative(examples: Sequence[GroundedExample], n: int = 5) -> str:
    if not examples:
        return "(no examples)"
    sample = list(examples[:n])
    lines = []
    for ex in sample:
        lines.append(
            f"  [{ex.video_id}] \"{ex.step_text}\"\n"
            f"     pred : [{ex.pred_start:7.2f}s, {ex.pred_end:7.2f}s]\n"
            f"     gt   : [{ex.gt_start:7.2f}s, {ex.gt_end:7.2f}s]\n"
            f"     IoU  : {ex.iou:.3f}"
        )
    return "\n".join(lines)


__all__ = [
    "temporal_iou",
    "GroundedExample",
    "evaluate",
    "format_metrics",
    "format_qualitative",
]
