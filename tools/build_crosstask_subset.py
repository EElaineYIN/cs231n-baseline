"""Build a small real-data CrossTask subset for the baseline.

Pipeline:
  1. Pick N videos from ``videos_val.csv`` whose annotation file exists
     and whose task is in ``tasks_primary.txt``.
  2. Download each video at the lowest available mp4 resolution with
     yt-dlp (skips if mp4 already on disk).
  3. Decode at 1 fps with imageio-ffmpeg, run CLIP ViT-B/32 image
     encoder per frame, save (T, 512) float32 to ``<video_id>.npy``.

Output layout mirrors what ``run_baseline.py --dataset crosstask`` expects.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from typing import List, Tuple

import numpy as np

# repo paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def read_val_csv(path: str) -> List[Tuple[str, str, str]]:
    """Return list of (task_id, video_id, url)."""
    rows = []
    with open(path, "r") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            rows.append((row[0].strip(), row[1].strip(), row[2].strip()))
    return rows


def filter_with_annotations(
    videos: List[Tuple[str, str, str]], ann_dir: str
) -> List[Tuple[str, str, str]]:
    kept = []
    for tid, vid, url in videos:
        if os.path.exists(os.path.join(ann_dir, f"{tid}_{vid}.csv")):
            kept.append((tid, vid, url))
    return kept


def yt_dlp_download(url: str, out_path: str, max_height: int = 240) -> bool:
    """Download a single video as mp4. Returns True on success."""
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        return True
    tmpl = out_path[:-4]  # yt-dlp appends ext
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "-f",
        # smallest mp4 that still encodes the frames we need
        f"bv*[height<={max_height}][ext=mp4]/bv*[height<={max_height}]/wv*",
        "--merge-output-format",
        "mp4",
        "-o",
        f"{tmpl}.%(ext)s",
        url,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300)
    except subprocess.CalledProcessError as exc:
        print(f"  [skip] download failed for {url}: {exc}", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print(f"  [skip] download timed out for {url}", file=sys.stderr)
        return False
    # find the resulting file (yt-dlp may have produced .mp4 / .mkv / .webm)
    for ext in (".mp4", ".mkv", ".webm"):
        cand = tmpl + ext
        if os.path.exists(cand):
            if cand != out_path:
                os.replace(cand, out_path)
            return True
    return False


def extract_clip_features(
    video_path: str,
    model,
    preprocess,
    tokenizer,  # unused but kept for symmetry; image-only here
    device,
    fps_sample: float = 1.0,
    batch_size: int = 16,
) -> np.ndarray:
    """Decode video at fps_sample, run CLIP image encoder per frame.

    Uses ``ffmpeg`` to dump JPEG frames to a tempdir at the desired
    sample rate, then encodes them in mini-batches on CPU.

    Returns (T, D) float32. T is the number of sampled frames; one per
    1/fps_sample seconds of video, matching the standard CrossTask
    1-feature-per-second convention.
    """
    import tempfile  # noqa: WPS433
    import glob  # noqa: WPS433
    import torch  # noqa: WPS433
    from PIL import Image  # noqa: WPS433

    with tempfile.TemporaryDirectory(prefix="ct_frames_") as td:
        # ffmpeg -i video.mp4 -vf fps=1 -q:v 5 frame_%06d.jpg
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-i", video_path,
            "-vf", f"fps={fps_sample}",
            "-q:v", "5",
            os.path.join(td, "frame_%06d.jpg"),
        ]
        subprocess.run(cmd, check=True, timeout=600)
        frame_paths = sorted(glob.glob(os.path.join(td, "frame_*.jpg")))
        if not frame_paths:
            return np.zeros((0, model.visual.output_dim), dtype=np.float32)

        feats: List[np.ndarray] = []
        with torch.no_grad():
            for b in range(0, len(frame_paths), batch_size):
                batch_imgs = [Image.open(p).convert("RGB") for p in frame_paths[b : b + batch_size]]
                batch = torch.stack([preprocess(im) for im in batch_imgs]).to(device)
                emb = model.encode_image(batch)
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                feats.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--release-dir",
        default=os.path.join(_REPO, "data/crosstask/crosstask_release"),
    )
    parser.add_argument(
        "--videos-dir",
        default=os.path.join(_REPO, "data/crosstask/videos"),
    )
    parser.add_argument(
        "--features-dir",
        default=os.path.join(_REPO, "data/crosstask/clip_features"),
    )
    parser.add_argument("--num-videos", type=int, default=5,
                        help="number of validation videos to fetch")
    parser.add_argument("--max-height", type=int, default=240,
                        help="cap video resolution to keep downloads small")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--keep-videos", action="store_true",
                        help="don't delete .mp4 after extraction")
    args = parser.parse_args()

    val_path = os.path.join(args.release_dir, "videos_val.csv")
    ann_dir = os.path.join(args.release_dir, "annotations")
    videos = filter_with_annotations(read_val_csv(val_path), ann_dir)
    print(f"[plan] {len(videos)} val videos have annotations")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(videos)
    targets = videos[: max(args.num_videos * 2, args.num_videos)]  # extras for fallback

    os.makedirs(args.videos_dir, exist_ok=True)
    os.makedirs(args.features_dir, exist_ok=True)

    # lazy import torch + open_clip
    import torch  # noqa: WPS433
    import open_clip  # noqa: WPS433

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device} model={args.clip_model}/{args.clip_pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.clip_model)

    succeeded = 0
    for tid, vid, url in targets:
        if succeeded >= args.num_videos:
            break
        feat_path = os.path.join(args.features_dir, f"{vid}.npy")
        if os.path.exists(feat_path):
            print(f"[ok]    {vid} (features cached)")
            succeeded += 1
            continue

        print(f"[fetch] {tid}/{vid}  {url}")
        mp4 = os.path.join(args.videos_dir, f"{vid}.mp4")
        t0 = time.time()
        if not yt_dlp_download(url, mp4, max_height=args.max_height):
            continue
        t_dl = time.time() - t0

        try:
            t1 = time.time()
            feats = extract_clip_features(
                mp4, model, preprocess, tokenizer, device, fps_sample=1.0
            )
            t_ex = time.time() - t1
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] feature extraction failed: {exc}", file=sys.stderr)
            continue
        if feats.shape[0] == 0:
            print(f"  [skip] {vid} produced 0 frames")
            continue

        np.save(feat_path, feats)
        succeeded += 1
        print(
            f"  -> saved {feats.shape} float32 to {os.path.basename(feat_path)}  "
            f"(dl {t_dl:.1f}s, encode {t_ex:.1f}s)"
        )

        if not args.keep_videos and os.path.exists(mp4):
            try:
                os.remove(mp4)
            except OSError:
                pass

    print(f"\n[done] {succeeded}/{args.num_videos} videos featurized")
    print(f"       features in {args.features_dir}")


if __name__ == "__main__":
    main()
