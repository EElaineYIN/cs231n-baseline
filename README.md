# CS231N — Independent Visual-Text Retrieval Baseline

Procedure-level temporal grounding in instructional videos. Given an
untrimmed video and an ordered list of step descriptions, predict a
start/end timestamp for each step.

This repository implements **only the retrieval baseline** for the
project (no ASR, no ordering constraints, no DP, no training).

## What the baseline does

For each step query in each video:

1. Split the video into multi-scale sliding **candidate windows** over
   pre-extracted clip features.
2. **Mean-pool** the clip features inside each window.
3. **Embed** the step text with a pluggable encoder
   (`hash` / `sentence-transformers` / `clip`).
4. Compute **cosine similarity** between every window and the query.
5. Predict the **argmax** window as the (start, end) interval.

Steps are scored **independently** — no ordering constraint, no
NMS-style suppression across queries, no DP decoding.

## Repo layout

```
cs231n_temporal_grounding/
├── run_baseline.py         # entry point + CLI
├── src/
│   ├── data.py             # VideoSample + dummy / generic_npz / CrossTask loaders
│   ├── windows.py          # multi-scale sliding windows + mean pooling
│   ├── text_encoder.py     # hash | SentenceTransformer | CLIP (pluggable)
│   ├── scoring.py          # L2-normalized cosine, dim-mismatch handling
│   ├── predict.py          # argmax → (t_start, t_end)
│   └── evaluate.py         # IoU, Recall@1@{0.3,0.5,0.7}, mIoU, qualitative print
├── data/                   # put per-video .npz or CrossTask here
└── outputs/                # metrics.json, predictions.jsonl land here
```

## Quick start (no downloads, no torch)

The dummy dataset is synthesized in NumPy from a bank of cooking
actions. The hash text encoder uses the *same* deterministic projection
that seeded the visual features, so the smoke test produces non-random
metrics and proves every stage of the pipeline is wired correctly.

```bash
# uses ~/door-takeoff-mvp/.venv (numpy 2.0 already installed)
cd ~/code/cs231n_temporal_grounding
~/door-takeoff-mvp/.venv/bin/python run_baseline.py \
    --dataset dummy --text-encoder hash --dummy-seed 0
```

Expected output (deterministic):

```
[setup] dataset=dummy  videos=8  visual_dim=64  text_dim=64  text_encoder=hash

=== Baseline results ===
  num_queries : 39
  R@1@0.3     : 100.00 %
  R@1@0.5     :  97.44 %
  R@1@0.7     :  66.67 %
  mIoU        :  76.02 %

=== Qualitative (5 examples) ===
  [dummy_000] "preheat the oven to 350 degrees"
     pred : [   0.00s,    5.00s]
     gt   : [   0.00s,    5.00s]
     IoU  : 1.000
  ...
```

The numbers above are an upper bound (the dummy features are
text-aligned by construction). They are not a meaningful "model
quality" number — they only confirm the pipeline is correct. A noisier
synthesis (`--dummy-noise 1.5`) drops R@1@0.5 to ~33%, confirming
metrics react to signal quality as expected.

## Running on real datasets

### Option A — generic `.npz` per video (recommended)

Preprocess your dataset (CrossTask, YouCook2, COIN, …) into one `.npz`
file per video with these arrays:

| key             | dtype  | shape   | meaning                                |
|-----------------|--------|---------|----------------------------------------|
| `video_id`      | str    | scalar  | unique id                              |
| `duration`      | float  | scalar  | seconds                                |
| `clip_features` | float32| (T, D)  | pre-extracted clip features            |
| `clip_stride`   | float  | scalar  | seconds per feature timestep           |
| `step_texts`    | str    | (N,)    | ordered step descriptions              |
| `step_starts`   | float  | (N,)    | ground-truth start seconds             |
| `step_ends`     | float  | (N,)    | ground-truth end seconds               |
| `task_id`       | str    | scalar  | optional                               |
| `task_name`     | str    | scalar  | optional                               |

Then:

```bash
python run_baseline.py --dataset generic_npz --npz-dir data/your_dataset_npz \
    --text-encoder sentence-transformers     # or --text-encoder clip
```

### Option B — CrossTask (https://github.com/DmZhukov/CrossTask)

Download CrossTask following the upstream README. You need:

* `tasks_primary.txt` and `tasks_related.txt` (task → step list)
* `annotations/<task_id>_<video_id>.csv` (`step_idx,start,end` in s)
* a directory of per-video `<video_id>.npy` clip features
  (HowTo100M S3D, or anything you re-extract — CLIP-image features
  pair best with the `--text-encoder clip` setting).

Then:

```bash
python run_baseline.py --dataset crosstask \
    --crosstask-root data/crosstask \
    --features-dir   data/crosstask_features \
    --clip-stride 1.0 \
    --text-encoder clip
```

> Note: the official CrossTask release ships S3D / 3D-ResNet features
> that are *not* in CLIP space. With the `clip` text encoder, the
> scorer will random-project the higher-d side down to match
> dimensions — this is a deliberately weak no-training baseline.
> For better numbers you'll want to re-extract CLIP-image features.

## CLI flags worth knowing

| flag                       | default       | what it controls                     |
|----------------------------|---------------|--------------------------------------|
| `--window-clip-sizes`      | `3 5 8 12 16` | window lengths to enumerate (clips)  |
| `--window-stride-clips`    | `1`           | sliding stride (clips)               |
| `--clip-stride`            | `1.0`         | seconds per feature timestep         |
| `--text-encoder`           | `hash`        | `hash` / `sentence-transformers` / `clip` |
| `--iou-thresholds`         | `0.3 0.5 0.7` | thresholds for Recall@1              |
| `--num-qualitative`        | `5`           | qualitative examples to print        |
| `--max-videos`             | `None`        | cap videos (quick smoke runs)        |

## Optional: install real text encoders

The pipeline runs out of the box with `hash`. For real semantic
encoders, install one of:

```bash
# sentence-transformers (~80MB)
pip install sentence-transformers
# or open_clip (~150MB + torch)
pip install open_clip_torch torch
```

If the import fails at runtime the encoder silently falls back to
`hash` with a stderr warning, so the pipeline never crashes on
missing deps.

## Metrics reported

| metric        | definition                                                  |
|---------------|-------------------------------------------------------------|
| `R@1@0.3`     | fraction of queries whose top-1 prediction has IoU ≥ 0.3    |
| `R@1@0.5`     | same, threshold 0.5                                         |
| `R@1@0.7`     | same, threshold 0.7                                         |
| `mIoU`        | mean temporal IoU over all queries                          |

`outputs/metrics_<dataset>.json` is the structured aggregate.
`outputs/predictions_<dataset>.jsonl` is one line per query with
`pred_start`, `pred_end`, `gt_start`, `gt_end`, `iou`.

## Baseline result summary (current run)

Real data was not downloaded in this run. The smoke-test on the
synthetic dataset (8 videos, 39 step queries, `--dummy-seed 0`,
`--text-encoder hash`) yields:

| metric    | value   |
|-----------|---------|
| R@1@0.3   | 100.00% |
| R@1@0.5   |  97.44% |
| R@1@0.7   |  66.67% |
| mIoU      |  76.02% |

These numbers verify pipeline correctness end-to-end (window
generation → pooling → encoding → scoring → argmax → IoU →
aggregation). They are **not** a model-quality number — the dummy
visual features are constructed from the same hash projection used by
the encoder. Once CrossTask / YouCook2 features are available, swap
the dataset flag and rerun.

## What this baseline intentionally does NOT do

* No training of any kind.
* No ASR / Whisper integration.
* No ordering constraint (step *i* may be predicted after step *i+1*).
* No DP decoding to resolve overlaps.
* No NMS across queries.

Those are the planned extensions in the project proposal.
