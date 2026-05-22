"""Dataset loading.

Each loader yields VideoSample objects with a uniform schema so the
downstream pipeline does not need to know the dataset format.

Supported loaders:
  * load_dummy_dataset   — synthetic data, no downloads, smoke test
  * load_generic_npz_dir — drop-in for any dataset stored as .npz files
                           (one per video) with the documented schema
  * load_crosstask       — loader for the CrossTask release
                           (https://github.com/DmZhukov/CrossTask)
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class Step:
    """One ground-truth annotated step inside a video."""

    text: str
    start: float  # seconds
    end: float    # seconds


@dataclass
class VideoSample:
    """Uniform per-video record consumed by the baseline pipeline."""

    video_id: str
    duration: float                  # video duration in seconds
    clip_features: np.ndarray        # shape (T, D)
    clip_stride: float               # seconds per clip-feature timestep
    steps: List[Step] = field(default_factory=list)
    task_id: Optional[str] = None    # optional CrossTask task id
    task_name: Optional[str] = None  # optional human-readable task name

    @property
    def num_clips(self) -> int:
        return int(self.clip_features.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.clip_features.shape[1])


# ---------------------------------------------------------------------------
# Dummy / synthetic dataset
# ---------------------------------------------------------------------------

def _stable_token_seed(tok: str, seed: int) -> int:
    """Process-independent 32-bit seed for a token (PYTHONHASHSEED-safe)."""
    h = hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest()
    return (int.from_bytes(h, "little") ^ seed) & 0xFFFFFFFF


def _hash_project(text: str, dim: int, seed: int = 0) -> np.ndarray:
    """Deterministic bag-of-words → fixed-dim vector.

    Used both as a stand-in text encoder (see text_encoder.HashTextEncoder)
    and to seed synthetic visual features that align with their step text.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = [t for t in text.lower().replace(",", " ").split() if t]
    if not tokens:
        return vec
    for tok in tokens:
        sub_rng = np.random.default_rng(_stable_token_seed(tok, seed))
        vec += sub_rng.standard_normal(dim).astype(np.float32)
    n = np.linalg.norm(vec)
    if n > 0:
        vec /= n
    return vec


_DUMMY_STEP_BANK = [
    ("preheat the oven to 350 degrees", 4.0),
    ("dice the onion into small pieces", 7.0),
    ("mince the garlic cloves", 5.0),
    ("heat olive oil in a large skillet", 5.0),
    ("add chopped onions to the pan and saute", 8.0),
    ("stir in the minced garlic", 4.0),
    ("pour in the tomato sauce and simmer", 10.0),
    ("season with salt and black pepper", 4.0),
    ("add cooked pasta to the sauce", 6.0),
    ("toss everything together until well coated", 6.0),
    ("plate the pasta and top with parmesan", 5.0),
    ("garnish with fresh basil leaves", 4.0),
]


def load_dummy_dataset(
    num_videos: int = 8,
    feature_dim: int = 64,
    clip_stride: float = 1.0,
    noise: float = 0.4,
    bg_steps_per_video: int = 2,
    seed: int = 0,
) -> List[VideoSample]:
    """Synthesize a small instructional-video-style dataset.

    Each video has 3-6 ordered steps drawn from a bank of cooking actions.
    For every step we generate ``duration_steps`` worth of clip features by
    hashing the step text into a vector and adding Gaussian noise. Between
    annotated steps we insert short "background" segments seeded from
    distractor text so the retrieval problem is non-trivial.

    The hash-projected text is deterministic, so the HashTextEncoder will
    actually align with the synthesized features and the pipeline produces
    meaningful (non-random) metrics during a smoke test.
    """
    rng = np.random.default_rng(seed)
    videos: List[VideoSample] = []

    for v in range(num_videos):
        num_steps = int(rng.integers(3, 7))
        # sample steps without replacement so each video is distinct
        step_indices = rng.choice(len(_DUMMY_STEP_BANK), size=num_steps, replace=False)
        clip_chunks: List[np.ndarray] = []
        annotated: List[Step] = []
        cursor = 0.0

        for i, idx in enumerate(step_indices):
            text, base_dur = _DUMMY_STEP_BANK[int(idx)]
            duration = float(base_dur + rng.uniform(-1.0, 2.0))
            duration = max(2.0, duration)
            num_clips_for_step = max(2, int(round(duration / clip_stride)))

            base_vec = _hash_project(text, feature_dim)
            step_feats = base_vec[None, :] + noise * rng.standard_normal(
                (num_clips_for_step, feature_dim)
            ).astype(np.float32)

            clip_chunks.append(step_feats)
            start_t = cursor
            end_t = cursor + num_clips_for_step * clip_stride
            annotated.append(Step(text=text, start=start_t, end=end_t))
            cursor = end_t

            # add a short background gap (distractor) between steps
            if i < num_steps - 1 and bg_steps_per_video > 0:
                gap_clips = int(rng.integers(2, 5))
                distractor_text, _ = _DUMMY_STEP_BANK[
                    int(rng.integers(0, len(_DUMMY_STEP_BANK)))
                ]
                gap_vec = _hash_project("background " + distractor_text, feature_dim)
                gap = gap_vec[None, :] + noise * rng.standard_normal(
                    (gap_clips, feature_dim)
                ).astype(np.float32)
                clip_chunks.append(gap)
                cursor += gap_clips * clip_stride

        clip_features = np.concatenate(clip_chunks, axis=0).astype(np.float32)
        videos.append(
            VideoSample(
                video_id=f"dummy_{v:03d}",
                duration=float(clip_features.shape[0] * clip_stride),
                clip_features=clip_features,
                clip_stride=clip_stride,
                steps=annotated,
                task_id="dummy_task",
                task_name="synthetic_cooking",
            )
        )

    return videos


# ---------------------------------------------------------------------------
# Generic NPZ loader (any dataset can target this schema)
# ---------------------------------------------------------------------------

def load_generic_npz_dir(
    npz_dir: str,
    require_steps: bool = True,
) -> List[VideoSample]:
    """Load any dataset that has been preprocessed into per-video .npz files.

    Each .npz must contain:
      * video_id      : str (0-d) or basename of the file
      * duration      : float
      * clip_features : (T, D) float32 array
      * clip_stride   : float seconds-per-feature
      * step_texts    : (N,) array of str
      * step_starts   : (N,) array of float (seconds)
      * step_ends     : (N,) array of float (seconds)
      * task_id       : optional str
      * task_name     : optional str
    """
    if not os.path.isdir(npz_dir):
        raise FileNotFoundError(npz_dir)

    samples: List[VideoSample] = []
    for fn in sorted(os.listdir(npz_dir)):
        if not fn.endswith(".npz"):
            continue
        path = os.path.join(npz_dir, fn)
        z = np.load(path, allow_pickle=True)

        video_id = str(z["video_id"]) if "video_id" in z.files else os.path.splitext(fn)[0]
        duration = float(z["duration"]) if "duration" in z.files else float(
            z["clip_features"].shape[0] * float(z.get("clip_stride", 1.0))
        )
        clip_features = np.asarray(z["clip_features"], dtype=np.float32)
        clip_stride = float(z["clip_stride"]) if "clip_stride" in z.files else 1.0

        steps: List[Step] = []
        if "step_texts" in z.files:
            texts = [str(t) for t in z["step_texts"].tolist()]
            starts = z["step_starts"].astype(float).tolist()
            ends = z["step_ends"].astype(float).tolist()
            steps = [Step(t, s, e) for t, s, e in zip(texts, starts, ends)]
        elif require_steps:
            raise ValueError(f"{path}: missing step_texts/starts/ends")

        samples.append(
            VideoSample(
                video_id=video_id,
                duration=duration,
                clip_features=clip_features,
                clip_stride=clip_stride,
                steps=steps,
                task_id=str(z["task_id"]) if "task_id" in z.files else None,
                task_name=str(z["task_name"]) if "task_name" in z.files else None,
            )
        )
    return samples


# ---------------------------------------------------------------------------
# CrossTask loader
# ---------------------------------------------------------------------------

def _read_crosstask_tasks(tasks_path: str) -> dict:
    """Parse CrossTask ``tasks_primary.txt`` / ``tasks_related.txt``.

    File format (blank-line-separated records):
        task_id
        task_name
        url
        num_steps
        comma-separated step descriptions
    """
    tasks: dict = {}
    if not os.path.exists(tasks_path):
        return tasks
    with open(tasks_path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            task_id = lines[i].strip()
            name = lines[i + 1].strip()
            _url = lines[i + 2].strip()
            _n = int(lines[i + 3].strip())
            step_str = lines[i + 4].strip()
            steps = [s.strip() for s in step_str.split(",") if s.strip()]
            tasks[task_id] = {"name": name, "steps": steps}
            i += 5
        except (IndexError, ValueError):
            i += 1
    return tasks


def _read_crosstask_annotation(csv_path: str) -> List[tuple]:
    """Parse a CrossTask annotation CSV. Each row: step_idx,start,end (seconds)."""
    rows: List[tuple] = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                step_idx = int(row[0])
                start = float(row[1])
                end = float(row[2])
                rows.append((step_idx, start, end))
            except ValueError:
                continue
    return rows


def load_crosstask(
    crosstask_root: str,
    features_dir: Optional[str] = None,
    clip_stride: float = 1.0,
    split: str = "val",
    max_videos: Optional[int] = None,
) -> List[VideoSample]:
    """Load CrossTask validation videos for temporal grounding.

    Expected layout (after downloading the dataset from
    https://github.com/DmZhukov/CrossTask):

        crosstask_root/
          tasks_primary.txt
          tasks_related.txt
          annotations/<task_id>_<video_id>.csv
          (features_dir, e.g. ./crosstask_features/ )
            <video_id>.npy        # (T, D) S3D or similar clip features

    Notes:
      * CrossTask's annotations are at second precision and reference
        ordered step indices defined in tasks_primary.txt.
      * If your features have stride != 1s, pass clip_stride accordingly.
    """
    tasks = {}
    tasks.update(_read_crosstask_tasks(os.path.join(crosstask_root, "tasks_primary.txt")))
    tasks.update(_read_crosstask_tasks(os.path.join(crosstask_root, "tasks_related.txt")))
    ann_dir = os.path.join(crosstask_root, "annotations")
    if not os.path.isdir(ann_dir):
        raise FileNotFoundError(f"annotations/ not found in {crosstask_root}")

    feats_dir = features_dir or os.path.join(crosstask_root, "crosstask_features")

    samples: List[VideoSample] = []
    for fn in sorted(os.listdir(ann_dir)):
        if not fn.endswith(".csv"):
            continue
        stem = fn[:-4]
        if "_" not in stem:
            continue
        task_id, video_id = stem.split("_", 1)
        if task_id not in tasks:
            continue
        feat_path = os.path.join(feats_dir, video_id + ".npy")
        if not os.path.exists(feat_path):
            continue  # skip videos without precomputed features

        clip_features = np.load(feat_path).astype(np.float32)
        duration = float(clip_features.shape[0] * clip_stride)
        ann = _read_crosstask_annotation(os.path.join(ann_dir, fn))
        step_texts = tasks[task_id]["steps"]
        steps = [
            Step(
                text=step_texts[idx - 1] if 0 < idx <= len(step_texts) else f"step_{idx}",
                start=start,
                end=end,
            )
            for idx, start, end in ann
        ]
        samples.append(
            VideoSample(
                video_id=video_id,
                duration=duration,
                clip_features=clip_features,
                clip_stride=clip_stride,
                steps=steps,
                task_id=task_id,
                task_name=tasks[task_id]["name"],
            )
        )
        if max_videos is not None and len(samples) >= max_videos:
            break

    return samples


# Public re-exports
__all__ = [
    "Step",
    "VideoSample",
    "load_dummy_dataset",
    "load_generic_npz_dir",
    "load_crosstask",
]
