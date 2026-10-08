#!/usr/bin/env python3
"""
Build a real-domain image set from edited cockpit landing films (private use only).

Per video (YouTube id):
  1. download the ≤1080p video stream (yt-dlp)
  2. coarse pass: one frame every ``--coarse`` s, scored by YOLO-seg (runway class,
     trained on X-Plane) and CLIP zero-shot ("runway on final approach")
  3. anchors = frames where BOTH agree with high scores AND the runway mask looks
     airborne (does not reach the bottom of the frame, not too wide)
  4. approach segment = [first anchor − ``--lookback`` s, last anchor]; re-extract it
     at ``--fps`` — the look-back recovers the early approach where the runway is too
     small for YOLO, so the set is not biased toward frames that look like the sim
  5. per dense frame: locate the runway (YOLO at low confidence, interpolated in time
     where it misses) and crop a 16:9 window centred on it that (a) does not cross
     edges that stay fixed on screen (picture-in-picture borders, window pillars),
     (b) stays above the caption / logo band, (c) does not look back across a shot
     cut, and (d) still scores as "runway on approach" for CLIP
  6. write crops + ``index.csv`` + contact sheets

    PYTHONPATH=. python scripts/extract_hpa_frames.py --channel            # every video
    PYTHONPATH=. python scripts/extract_hpa_frames.py gi6SeqlgiiU qX7MvzaGVAM

Footage used with the channel owner's permission; ``data/hpa`` is gitignored and must
not be redistributed.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("data/hpa")
CHANNEL = "https://www.youtube.com/@HighPressureAviationFilms/videos"
SCENE_POS = [
    "a photo of an airport runway seen from an airplane cockpit on final approach",
    "a runway ahead with approach lights at night, view from the flight deck before landing",
]
SCENE_NEG = [
    "a photo of clouds and sky from an airplane",
    "a photo of an aircraft cockpit instrument panel",
    "an airplane taxiing on a taxiway at an airport terminal",
    "a city or landscape seen from an airplane with no runway",
    "a title card with large text",
    "an aeronautical chart or map",
]
CLEAN_POS = ["a plain photo with no text and no logo"]
CLEAN_NEG = [
    "a photo with subtitles or caption text on it",
    "a photo with a logo watermark on it",
    "a photo with a small inset picture of a cockpit in the corner",
    "a split-screen video with several camera views",
]


class Scorer:
    def __init__(self, weights: Path):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from ultralytics import YOLO

        self.torch = torch
        self.yolo = YOLO(str(weights))
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        with torch.no_grad():
            self.text = {}
            for name, pos, neg in (("scene", SCENE_POS, SCENE_NEG), ("clean", CLEAN_POS, CLEAN_NEG)):
                t = self.proc(text=pos + neg, return_tensors="pt", padding=True)
                e = self.clip.get_text_features(**t)
                e = e.pooler_output if hasattr(e, "pooler_output") else e
                self.text[name] = (e / e.norm(dim=-1, keepdim=True), len(pos))

    def clip_scores(self, images: list[np.ndarray], which: str, batch: int = 32) -> np.ndarray:
        from PIL import Image

        txt, n_pos = self.text[which]
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), batch):
                pil = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in images[i:i + batch]]
                e = self.clip.get_image_features(**self.proc(images=pil, return_tensors="pt"))
                e = e.pooler_output if hasattr(e, "pooler_output") else e
                e = e / e.norm(dim=-1, keepdim=True)
                p = (100.0 * e @ txt.T).softmax(dim=1).numpy()
                out.append(p[:, :n_pos].sum(1))
        return np.concatenate(out) if out else np.zeros(0)

    def runway(self, bgr: np.ndarray, conf: float) -> tuple[float, tuple[float, float, float, float] | None]:
        """Best runway detection: (confidence, bbox xyxy normalised) or (0, None)."""
        r = self.yolo.predict(source=bgr, conf=conf, imgsz=640, verbose=False, device="cpu")[0]
        if r.boxes is None or len(r.boxes) == 0:
            return 0.0, None
        cls = r.boxes.cls.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()
        idx = np.where(cls == 0)[0]
        if idx.size == 0:
            return 0.0, None
        k = idx[np.argmax(cf[idx])]
        return float(cf[k]), tuple(float(v) for v in r.boxes.xyxyn[k].cpu().numpy())


def list_channel() -> list[str]:
    out = subprocess.run(["yt-dlp", "--js-runtimes", "node", "--flat-playlist", "--print", "%(id)s", CHANNEL],
                         capture_output=True, text=True, check=True).stdout
    return [v for v in out.split() if v]


def download(vid: str) -> Path | None:
    out = ROOT / "raw" / f"{vid}.mp4"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            r = subprocess.run(
                ["yt-dlp", "-q", "--js-runtimes", "node", "-f", "bv*[height<=1080][ext=mp4]/bv*[height<=1080]",
                 "--no-playlist", "-o", str(out.with_suffix(".%(ext)s")), f"https://www.youtube.com/watch?v={vid}"],
                capture_output=True, text=True)
            if out.exists():
                break
        if not out.exists():
            print(vid, "download failed:", r.stderr.strip()[-200:], flush=True)
            return None
    return out


def ffmpeg_frames(src: Path, out: Path, fps: float, start: float | None = None, end: float | None = None) -> list[Path]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{max(start, 0):.2f}"]
    if end is not None:
        cmd += ["-to", f"{end:.2f}"]
    cmd += ["-i", str(src), "-vf", f"fps={fps}", "-q:v", "2", str(out / "%05d.jpg")]
    subprocess.run(cmd, check=True)
    return sorted(out.glob("*.jpg"))


def airborne(box: tuple[float, float, float, float]) -> bool:
    """Runway seen from the air: the mask stays clear of the bottom edge and is not frame-wide."""
    x0, y0, x1, y1 = box
    return y1 < 0.90 and (x1 - x0) < 0.60 and (y1 - y0) > 0.02


def persistent_lines(frames: list[np.ndarray], min_run: float = 0.22) -> tuple[list[float], list[float]]:
    """
    Normalised x of vertical / y of horizontal edges that stay put over the segment:
    picture-in-picture borders, window pillars, glareshield. The outside scene moves,
    so its edges do not persist.
    """
    sw, sh = 480, 270
    g = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (sw, sh), interpolation=cv2.INTER_AREA) for f in frames]
    px = np.mean([np.abs(cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)) > 60 for x in g], axis=0) > 0.6
    py = np.mean([np.abs(cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)) > 60 for x in g], axis=0) > 0.6

    def longest_run(a: np.ndarray) -> np.ndarray:  # along axis 0
        best = np.zeros(a.shape[1], int)
        cur = np.zeros(a.shape[1], int)
        for row in a:
            cur = np.where(row, cur + 1, 0)
            best = np.maximum(best, cur)
        return best

    vx = np.nonzero(longest_run(px) >= min_run * sh)[0] / sw
    hy = np.nonzero(longest_run(py.T) >= min_run * sw)[0] / sh
    return vx.tolist(), hy.tolist()


def crop_window(box, W: int, H: int, min_w: int, zoom: float, vx: list[float], hy: list[float],
                max_bottom: float) -> tuple[int, int, int, int] | None:
    """16:9 window centred on the runway that stays inside the persistent lines and above the caption band."""
    x0, y0, x1, y1 = box
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    lo_x = max([v for v in vx if v < x0 - 0.01], default=0.0)
    hi_x = min([v for v in vx if v > x1 + 0.01], default=1.0)
    lo_y = max([v for v in hy if v < y0 - 0.01], default=0.0)
    hi_y = min([v for v in hy if v > y1 + 0.01] + [max(max_bottom, y1 + 0.02)], default=1.0)
    lo_x, hi_x, lo_y, hi_y = (lo_x + 0.006) * W, (hi_x - 0.006) * W, (lo_y + 0.006) * H, (hi_y - 0.006) * H
    w = max(min_w, zoom * (x1 - x0) * W, zoom * (y1 - y0) * H * 16 / 9)
    w = min(w, hi_x - lo_x, (hi_y - lo_y) * 16 / 9)
    h = w * 9 / 16
    if w < 1.3 * (x1 - x0) * W or h < 1.3 * (y1 - y0) * H or w < 400:
        return None
    x = float(np.clip(cx * W - w / 2, lo_x, hi_x - w))
    y = float(np.clip(cy * H - h / 2, lo_y, hi_y - h))
    return int(x), int(y), int(w), int(h)


def sheet(images, labels, cols=6, tile=(320, 180)) -> np.ndarray:
    tiles = []
    for im, lab in zip(images, labels):
        t = cv2.resize(im, tile, interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (tile[0], 20), (0, 0, 0), -1)
        cv2.putText(t, lab, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def process(vid: str, sc: Scorer, a) -> dict:
    src = download(vid)
    if src is None:
        return {"status": "download_failed"}
    tmp = ROOT / "tmp" / vid

    # ---- coarse pass
    coarse = ffmpeg_frames(src, tmp / "coarse", 1.0 / a.coarse)
    small = [cv2.resize(cv2.imread(str(f)), (640, 360), interpolation=cv2.INTER_AREA) for f in coarse]
    scene = sc.clip_scores(small, "scene")
    det = [sc.runway(im, 0.25) for im in small]
    anchors = [i for i, ((cf, box), s) in enumerate(zip(det, scene))
               if cf >= a.yolo_anchor and s >= a.clip_anchor and box is not None and airborne(box)]
    info = {"status": "ok", "coarse_frames": len(coarse), "anchors": len(anchors)}
    if not anchors:
        info["status"] = "no_approach"
        shutil.rmtree(tmp, ignore_errors=True)
        return info
    # keep the main cluster of anchors (the landing), drop isolated early hits
    t = np.array(anchors) * a.coarse
    t_end = t.max()
    t_first = t[t >= t_end - a.max_segment].min()
    start, end = t_first - a.lookback, t_end + a.coarse / 2
    info.update(segment_s=[round(float(start), 1), round(float(end), 1)])

    # ---- dense pass
    dense = ffmpeg_frames(src, tmp / "dense", a.fps, start, end)
    frames = [cv2.imread(str(f)) for f in dense]
    H, W = frames[0].shape[:2]
    dets = [sc.runway(cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA), a.yolo_low) for im in frames]
    known = [i for i, (cf, box) in enumerate(dets) if box is not None and airborne(box)]
    if not known:
        info["status"] = "no_dense_detection"
        shutil.rmtree(tmp, ignore_errors=True)
        return info
    # interpolate the runway box over time where YOLO misses (constant before the first hit)
    boxes = np.array([dets[i][1] for i in known])
    all_boxes = np.stack([np.interp(np.arange(len(frames)), known, boxes[:, k]) for k in range(4)], axis=1)
    on_ground_from = next((i for i in range(known[-1], len(frames))
                           if dets[i][1] is not None and not airborne(dets[i][1])), len(frames))

    # do not look back across a shot cut
    gray = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
            for f in frames]
    cuts = [i for i in range(1, len(gray)) if np.abs(gray[i] - gray[i - 1]).mean() > a.cut_thr]
    first = max([c for c in cuts if c <= known[0]], default=0)
    vx, hy = persistent_lines(frames[first:on_ground_from])
    info.update(first_after_cut=first, persistent_v=[round(v, 2) for v in vx][:8], persistent_h=[round(v, 2) for v in hy][:8])

    crops, meta, n_overlay = [], [], 0
    for i in range(first, on_ground_from):
        if all_boxes[i][3] > a.max_bottom:  # runway reaches the caption / logo band
            n_overlay += 1
            continue
        win = crop_window(all_boxes[i], W, H, a.min_crop, a.zoom, vx, hy, a.max_bottom + 0.06)
        if win is None:
            n_overlay += 1
            continue
        x, y, w, h = win
        crops.append(cv2.resize(frames[i][y:y + h, x:x + w], (a.out_w, a.out_w * 9 // 16), interpolation=cv2.INTER_AREA))
        meta.append((i, x, y, w, h, dets[i][0], i in known))
    if not crops:
        info["status"] = "no_clean_crop"
        shutil.rmtree(tmp, ignore_errors=True)
        return info
    scene_d = sc.clip_scores(crops, "scene")
    clean = np.ones(len(crops))
    keep = scene_d >= a.scene_thr

    out = ROOT / "crops" / vid
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows = []
    for (i, x, y, w, h, cf, detd), c, k, cl, s in zip(meta, crops, keep, clean, scene_d):
        t_s = start + i / a.fps
        if k:
            cv2.imwrite(str(out / f"{vid}_{i:05d}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append([vid, i, round(float(t_s), 2), x, y, w, h, round(cf, 3), int(detd), round(float(cl), 3),
                     round(float(s), 3), int(k)])
    with open(out / "index.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([["video", "frame", "time_s", "x", "y", "w", "h", "yolo_conf", "yolo_detected",
                                   "clip_clean", "clip_scene", "kept"]] + rows)
    sel = np.linspace(0, len(crops) - 1, min(36, len(crops))).astype(int)
    marked = []
    for j in sel:
        m = crops[j].copy()
        cv2.rectangle(m, (0, 0), (m.shape[1] - 1, m.shape[0] - 1), (0, 200, 0) if keep[j] else (0, 0, 220), 18)
        marked.append(m)
    (ROOT / "sheets").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ROOT / "sheets" / f"{vid}.jpg"),
                sheet(marked, [f"{meta[j][0]:04d} {'K' if keep[j] else 'x'} cl{clean[j]:.2f} sc{scene_d[j]:.2f} y{meta[j][5]:.2f}" for j in sel]))
    info.update(dense_frames=len(frames), airborne_frames=len(crops), kept=int(keep.sum()),
                rejected_overlay=n_overlay, rejected_scene=int((~keep).sum()), yolo_detected=int(sum(m[6] for m in meta)))
    shutil.rmtree(tmp, ignore_errors=True)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--channel", action="store_true", help="process every video of the channel")
    ap.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    ap.add_argument("--coarse", type=float, default=2.0, help="coarse sampling period (s)")
    ap.add_argument("--fps", type=float, default=2.0, help="dense sampling rate in the approach segment")
    ap.add_argument("--lookback", type=float, default=45.0)
    ap.add_argument("--max-segment", type=float, default=150.0)
    ap.add_argument("--yolo-anchor", type=float, default=0.5)
    ap.add_argument("--clip-anchor", type=float, default=0.8)
    ap.add_argument("--yolo-low", type=float, default=0.05)
    ap.add_argument("--max-bottom", type=float, default=0.74,
                    help="drop frames whose runway box goes below this fraction of the height (caption / logo band)")
    ap.add_argument("--cut-thr", type=float, default=35.0, help="mean abs gray difference that marks a shot cut")
    ap.add_argument("--scene-thr", type=float, default=0.3)
    ap.add_argument("--min-crop", type=int, default=800, help="minimum crop width in source pixels")
    ap.add_argument("--zoom", type=float, default=3.0, help="crop width as a multiple of the runway box")
    ap.add_argument("--out-w", type=int, default=960)
    ap.add_argument("--delete-raw", action="store_true", help="delete each downloaded video after processing")
    a = ap.parse_args()

    vids = list_channel() if a.channel else a.videos
    sc = Scorer(a.weights)
    summary_path = ROOT / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for k, vid in enumerate(vids, 1):
        if summary.get(vid, {}).get("status") in ("ok", "no_approach", "no_dense_detection", "no_clean_crop") and (a.channel):
            continue
        try:
            info = process(vid, sc, a)
        except Exception as exc:  # keep the batch going
            info = {"status": f"error: {type(exc).__name__}: {exc}"}
        summary[vid] = info
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=1))
        if a.delete_raw:
            (ROOT / "raw" / f"{vid}.mp4").unlink(missing_ok=True)
        print(f"[{k}/{len(vids)}] {vid} {info}", flush=True)
    kept = sum(v.get("kept", 0) for v in summary.values())
    print(f"total kept: {kept} crops from {sum(1 for v in summary.values() if v.get('kept'))} videos")


if __name__ == "__main__":
    main()
