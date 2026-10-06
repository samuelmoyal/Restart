#!/usr/bin/env python3
"""
Extract the manually annotated approach segments (scripts/hpa_annotate.py) as crops.

For each segment of ``data/hpa/annotations.json`` the best ≤2160p stream is read
directly at [t0, t1] (ffmpeg seeks in the remote stream, the film is not downloaded),
sampled at ``--fps``, cropped to the annotated box and written to
``data/hpa/real/<video>/<video>_s<k>_<n>.jpg`` (downscaled to ``--max-width``).
``data/hpa/real/index.csv`` lists every frame with its time and source size.

    PYTHONPATH=. python scripts/extract_hpa_segments.py

Private use only (channel owner's permission); ``data/hpa`` is gitignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

ROOT = Path("data/hpa")
OUT = ROOT / "real"


def stream_url(vid: str, height: int) -> tuple[str, int, int] | None:
    for _ in range(3):
        r = subprocess.run(
            ["yt-dlp", "--js-runtimes", "node", "-f", f"bv*[height<={height}]", "--no-playlist",
             "--print", "%(width)s\t%(height)s\t%(url)s", f"https://www.youtube.com/watch?v={vid}"],
            capture_output=True, text=True)
        line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        if line.count("\t") == 2:
            w, h, url = line.split("\t")
            return url, int(w), int(h)
    return None


def local_source(vid: str) -> tuple[str, int, int] | None:
    path = ROOT / "preview" / f"{vid}.mp4"
    if not path.exists():
        return None
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    w, h = out.split(",")[:2]
    return str(path), int(w), int(h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--height", type=int, default=2160, help="maximum source height")
    ap.add_argument("--local", action="store_true",
                    help="read the 720p preview copies instead of the remote stream (lower resolution)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--min-seconds", type=float, default=1.5, help="skip shorter segments")
    args = ap.parse_args()

    ann = json.loads((ROOT / "annotations.json").read_text())
    titles = {v["id"]: v["title"] for v in json.loads((ROOT / "videos.json").read_text())}
    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    todo = [(vid, k, s) for vid, a in ann.items() for k, s in enumerate(a.get("segments", []))
            if s["t1"] - s["t0"] >= args.min_seconds]
    for n, (vid, k, s) in enumerate(todo, 1):
        d = out_root / vid
        d.mkdir(exist_ok=True)
        prefix = f"{vid}_s{k}_"
        done = sorted(d.glob(prefix + "*.jpg"))
        x0, y0, x1, y1 = s["crop"]
        src = None
        if not done:
            for attempt in range(3):
                src = local_source(vid) if args.local else stream_url(vid, args.height)
                if src is None:
                    continue
                url, W, H = src
                cw, ch = int((x1 - x0) * W) // 2 * 2, int((y1 - y0) * H) // 2 * 2
                vf = f"fps={args.fps},crop={cw}:{ch}:{int(x0 * W)}:{int(y0 * H)},scale='min({args.max_width},iw)':-2"
                r = subprocess.run(
                    ["ffmpeg", "-v", "error", "-ss", f"{s['t0']:.2f}", "-t", f"{s['t1'] - s['t0']:.2f}", "-i", url,
                     "-vf", vf, "-q:v", "2", str(d / (prefix + "%05d.jpg"))], capture_output=True, text=True)
                done = sorted(d.glob(prefix + "*.jpg"))
                if done:
                    break
                print("  retry", vid, k, r.stderr.strip()[-120:], flush=True)
        crop_px = (f"{int((x1 - x0) * src[1])}x{int((y1 - y0) * src[2])} from {src[1]}x{src[2]}" if src
                   else ("cached" if done else "SOURCE UNAVAILABLE"))
        print(f"[{n}/{len(todo)}] {vid} seg {k} {titles.get(vid, '')[:34]} — {len(done)} frames ({crop_px})", flush=True)
        for i, f in enumerate(done):
            rows.append([f.relative_to(out_root), vid, k, round(s["t0"] + i / args.fps, 2), titles.get(vid, "")])
    with open(out_root / "index.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([["file", "video", "segment", "time_s", "title"]] + rows)
    print(f"{len(rows)} frames from {len({r[1] for r in rows})} videos → {out_root}")


if __name__ == "__main__":
    main()
