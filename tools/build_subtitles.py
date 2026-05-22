"""Fetch YouTube auto-generated subtitles for each CrossTask video.

For every ``<vid>.npy`` file in ``--features-dir`` we look up the URL in
``videos_val.csv`` and pull the English auto-subtitle track with yt-dlp.
Output is one JSON per video at ``<subs-dir>/<vid>.json`` with shape:

    {
      "video_id": "<vid>",
      "segments": [
        {"start": 12.3, "end": 14.7, "text": "first you cut the onion"},
        ...
      ]
    }

Auto-subs are what Zhukov 2019 actually uses as the narration channel
(see their §3, "Narrations"). They're noisier than Whisper but
zero-cost and the right thing for a faithful Zhukov-lite baseline.
Whisper can be plugged in later for the full model.

This script does NOT modify any other file; videos already deleted by
``build_crosstask_subset.py`` are re-fetched as subs-only with
``--skip-download`` (yt-dlp downloads the .vtt without touching the
video stream).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def load_url_index(val_csv: str) -> Dict[str, str]:
    """video_id -> url (skips malformed rows)."""
    out: Dict[str, str] = {}
    with open(val_csv, "r") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                out[row[1].strip()] = row[2].strip()
    return out


def list_cached_video_ids(features_dir: str) -> List[str]:
    return sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(features_dir, "*.npy"))
    )


_VTT_TIMING = re.compile(
    r"(\d+):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})\.(\d{3})"
)


def _vtt_time_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(vtt_path: str) -> List[Dict]:
    """Parse a WebVTT file into ``[{start, end, text}, ...]``.

    YouTube auto-subs contain *karaoke-style* per-word timing tags
    (``<00:00:01.500><c>word</c>``) and progressive repetition where the
    next cue restates the previous line. We strip the inline tags and
    deduplicate cues whose text is a prefix of the next cue's text.
    """
    raw_segments: List[Tuple[float, float, str]] = []
    with open(vtt_path, "r", encoding="utf-8", errors="replace") as f:
        block: List[str] = []
        current_times: Optional[Tuple[float, float]] = None
        for line in f:
            line = line.rstrip("\n")
            mt = _VTT_TIMING.search(line)
            if mt:
                current_times = (
                    _vtt_time_to_seconds(*mt.group(1, 2, 3, 4)),
                    _vtt_time_to_seconds(*mt.group(5, 6, 7, 8)),
                )
                block = []
                continue
            if line == "":
                if current_times is not None and block:
                    text = " ".join(block).strip()
                    # strip karaoke tags <00:00:00.000> and <c>...</c>
                    text = re.sub(r"<[^>]+>", "", text)
                    text = re.sub(r"\s+", " ", text).strip()
                    if text:
                        raw_segments.append((current_times[0], current_times[1], text))
                current_times = None
                block = []
            else:
                # skip metadata lines (WEBVTT, NOTE, STYLE, ...)
                if line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE")):
                    continue
                # skip pure cue identifiers (lines that are just digits)
                if line.isdigit():
                    continue
                block.append(line)
        # trailing block w/o blank line terminator
        if current_times is not None and block:
            text = " ".join(block).strip()
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                raw_segments.append((current_times[0], current_times[1], text))

    # YouTube auto-sub deduplication: each successive cue often restates
    # the previous line plus a few new words. Keep only the *final*
    # version of each rolling text segment.
    deduped: List[Dict] = []
    for start, end, text in raw_segments:
        if deduped and text.startswith(deduped[-1]["text"]):
            deduped[-1] = {"start": deduped[-1]["start"], "end": end, "text": text}
        elif deduped and deduped[-1]["text"].startswith(text):
            # current is a prefix of previous => keep previous, drop current
            continue
        else:
            deduped.append({"start": start, "end": end, "text": text})
    return deduped


def fetch_subs_for_video(video_id: str, url: str, out_dir: str, lang: str = "en") -> Optional[str]:
    """Download auto-sub via yt-dlp into a tempdir, return path to .vtt."""
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "--skip-download",
        "--write-auto-sub",
        "--sub-lang", lang,
        "--sub-format", "vtt",
        "-o", os.path.join(out_dir, f"{video_id}.%(ext)s"),
        url,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=120)
    except subprocess.CalledProcessError as exc:
        print(f"  [skip] sub download failed for {video_id}: {exc}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  [skip] sub download timed out for {video_id}", file=sys.stderr)
        return None
    # yt-dlp writes <vid>.<lang>.vtt
    cands = glob.glob(os.path.join(out_dir, f"{video_id}.{lang}*.vtt"))
    if not cands:
        return None
    return sorted(cands)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--release-dir",
        default=os.path.join(_REPO, "data/crosstask/crosstask_release"),
    )
    parser.add_argument(
        "--features-dir",
        default=os.path.join(_REPO, "data/crosstask/clip_features"),
    )
    parser.add_argument(
        "--subs-dir",
        default=os.path.join(_REPO, "data/crosstask/subtitles"),
    )
    parser.add_argument("--lang", default="en")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap number of videos processed (debug)")
    args = parser.parse_args()

    val_csv = os.path.join(args.release_dir, "videos_val.csv")
    url_idx = load_url_index(val_csv)
    vids = list_cached_video_ids(args.features_dir)
    if args.limit:
        vids = vids[: args.limit]
    print(f"[plan] {len(vids)} cached video(s) to fetch subs for")

    os.makedirs(args.subs_dir, exist_ok=True)

    n_ok = 0
    n_missing_url = 0
    n_no_subs = 0
    n_cached = 0
    for vid in vids:
        out_json = os.path.join(args.subs_dir, f"{vid}.json")
        if os.path.exists(out_json):
            print(f"[ok]    {vid} (subs cached)")
            n_ok += 1
            n_cached += 1
            continue
        url = url_idx.get(vid)
        if url is None:
            print(f"[miss]  {vid}: no URL in videos_val.csv")
            n_missing_url += 1
            continue

        print(f"[fetch] {vid}  {url}")
        with tempfile.TemporaryDirectory(prefix="ct_subs_") as td:
            vtt = fetch_subs_for_video(vid, url, td, lang=args.lang)
            if vtt is None:
                print(f"  -> no auto-subs available for {vid}")
                n_no_subs += 1
                continue
            segs = parse_vtt(vtt)
        with open(out_json, "w") as f:
            json.dump({"video_id": vid, "segments": segs}, f, ensure_ascii=False)
        print(f"  -> wrote {len(segs)} segments")
        n_ok += 1

    print()
    print(f"[done] ok={n_ok} (cached={n_cached})  no_subs={n_no_subs}  missing_url={n_missing_url}")
    print(f"       subs in {args.subs_dir}")


if __name__ == "__main__":
    main()
