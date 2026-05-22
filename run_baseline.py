"""Independent visual-text retrieval baseline for procedure-level
temporal grounding in instructional videos.

For each step description in a video we:
  1. enumerate sliding (multi-scale) candidate windows over clip features
  2. mean-pool the visual features inside each window
  3. embed the step text with a pluggable encoder
     (hash bag-of-words / SentenceTransformer / CLIP)
  4. compute cosine similarity between the step embedding and every window
  5. predict the highest-scoring window as the answer (no ordering, no DP)

Run examples:

  # Smoke test on synthetic data (no downloads, no torch needed):
  python run_baseline.py --dataset dummy --text-encoder hash

  # Real data after preprocessing into the generic .npz schema:
  python run_baseline.py --dataset generic_npz --npz-dir data/youcook2_npz \\
      --text-encoder sentence-transformers

  # CrossTask (https://github.com/DmZhukov/CrossTask) after downloading:
  python run_baseline.py --dataset crosstask \\
      --crosstask-root data/crosstask --features-dir data/crosstask_features \\
      --clip-stride 1.0 --text-encoder clip
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np

# Make ``src`` importable whether we run as `python run_baseline.py` or
# `python -m run_baseline` from the project root.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.data import (  # noqa: E402
    Step,
    VideoSample,
    load_crosstask,
    load_dummy_dataset,
    load_generic_npz_dir,
)
from src.evaluate import (  # noqa: E402
    GroundedExample,
    evaluate,
    format_metrics,
    format_qualitative,
)
from src.predict import Prediction, predict_intervals  # noqa: E402
from src.scoring import score  # noqa: E402
from src.text_encoder import get_text_encoder  # noqa: E402
from src.windows import make_windows, pool_windows  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset",
        choices=["dummy", "generic_npz", "crosstask"],
        default="dummy",
        help="which dataset loader to use",
    )
    # generic / crosstask paths
    p.add_argument("--npz-dir", default=None, help="directory of per-video .npz files")
    p.add_argument("--crosstask-root", default=None, help="root of CrossTask release")
    p.add_argument("--features-dir", default=None, help="dir with <video_id>.npy features")
    p.add_argument("--clip-stride", type=float, default=1.0,
                   help="seconds per clip-feature timestep")
    p.add_argument("--max-videos", type=int, default=None,
                   help="cap the number of videos (useful for quick runs)")

    # dummy-only knobs
    p.add_argument("--dummy-num-videos", type=int, default=8)
    p.add_argument("--dummy-feature-dim", type=int, default=64)
    p.add_argument("--dummy-noise", type=float, default=0.4)
    p.add_argument("--dummy-seed", type=int, default=0)

    # windowing
    p.add_argument(
        "--window-clip-sizes", type=int, nargs="+",
        default=[3, 5, 8, 12, 16],
        help="window lengths to enumerate, in feature timesteps",
    )
    p.add_argument("--window-stride-clips", type=int, default=1,
                   help="stride between consecutive windows, in feature timesteps")

    # text encoder
    p.add_argument(
        "--text-encoder",
        choices=["hash", "sentence-transformers", "clip"],
        default="hash",
    )
    p.add_argument("--hash-dim", type=int, default=None,
                   help="hash-encoder output dim (defaults to visual feature dim)")
    p.add_argument("--st-model", default="all-MiniLM-L6-v2")
    p.add_argument("--clip-model", default="ViT-B-32")
    p.add_argument("--clip-pretrained", default="openai")

    # eval / output
    p.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    p.add_argument("--num-qualitative", type=int, default=5)
    p.add_argument("--output-dir", default=os.path.join(_HERE, "outputs"))
    return p.parse_args()


def load_dataset(args: argparse.Namespace) -> List[VideoSample]:
    if args.dataset == "dummy":
        return load_dummy_dataset(
            num_videos=args.dummy_num_videos,
            feature_dim=args.dummy_feature_dim,
            clip_stride=args.clip_stride,
            noise=args.dummy_noise,
            seed=args.dummy_seed,
        )
    if args.dataset == "generic_npz":
        if not args.npz_dir:
            raise SystemExit("--npz-dir is required when --dataset generic_npz")
        samples = load_generic_npz_dir(args.npz_dir)
        return samples if args.max_videos is None else samples[: args.max_videos]
    if args.dataset == "crosstask":
        if not args.crosstask_root:
            raise SystemExit("--crosstask-root is required when --dataset crosstask")
        return load_crosstask(
            crosstask_root=args.crosstask_root,
            features_dir=args.features_dir,
            clip_stride=args.clip_stride,
            max_videos=args.max_videos,
        )
    raise SystemExit(f"unknown dataset: {args.dataset}")


def run(args: argparse.Namespace) -> Tuple[dict, List[GroundedExample]]:
    samples = load_dataset(args)
    samples = [s for s in samples if s.num_clips > 0 and len(s.steps) > 0]
    if not samples:
        raise SystemExit("loaded 0 usable videos — check dataset paths")

    feature_dim = samples[0].feature_dim
    hash_dim = args.hash_dim if args.hash_dim is not None else feature_dim
    encoder = get_text_encoder(
        args.text_encoder,
        hash_dim=hash_dim,
        st_model=args.st_model,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )
    print(
        f"[setup] dataset={args.dataset}  videos={len(samples)}  "
        f"visual_dim={feature_dim}  text_dim={encoder.dim}  "
        f"text_encoder={args.text_encoder}"
    )

    per_video_pairs = []
    for s in samples:
        windows = make_windows(
            num_clips=s.num_clips,
            clip_stride=s.clip_stride,
            window_clip_sizes=args.window_clip_sizes,
            stride_clips=args.window_stride_clips,
        )
        if not windows:
            continue
        win_feats = pool_windows(s.clip_features, windows)
        queries = [step.text for step in s.steps]
        text_feats = encoder.encode(queries)
        sims = score(win_feats, text_feats)
        preds: List[Prediction] = predict_intervals(queries, windows, sims)
        per_video_pairs.append((s.video_id, s.steps, preds))

    metrics, examples = evaluate(per_video_pairs, iou_thresholds=args.iou_thresholds)
    return metrics, examples


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    metrics, examples = run(args)

    print("\n=== Baseline results ===")
    print(format_metrics(metrics))

    print(f"\n=== Qualitative ({min(args.num_qualitative, len(examples))} examples) ===")
    print(format_qualitative(examples, n=args.num_qualitative))

    # persist
    out_metrics = os.path.join(args.output_dir, f"metrics_{args.dataset}.json")
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    out_preds = os.path.join(args.output_dir, f"predictions_{args.dataset}.jsonl")
    with open(out_preds, "w") as f:
        for ex in examples:
            f.write(json.dumps({
                "video_id": ex.video_id,
                "step_text": ex.step_text,
                "pred_start": ex.pred_start,
                "pred_end": ex.pred_end,
                "gt_start": ex.gt_start,
                "gt_end": ex.gt_end,
                "iou": ex.iou,
            }) + "\n")
    print(f"\n[saved] {out_metrics}")
    print(f"[saved] {out_preds}")


if __name__ == "__main__":
    main()
