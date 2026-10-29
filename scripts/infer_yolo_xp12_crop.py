#!/usr/bin/env python3
"""
YOLO-seg on XP12 frames, zoomed on the runway found at t−1 (closed loop), cached like ``infer_yolo_xp12.py``.

At each frame, if the previous frame gave a runway, YOLO runs on a 16:9 crop centred on its bounding box,
``--context`` times its size (at least ``--min-crop`` px wide), resized to ``--imgsz``; the mask is pasted back
into the full frame. With no previous runway, or nothing found in the crop, YOLO runs on the full frame
(re-acquisition). The previous runway always comes from the model's own output, never from the ground truth.

One ``<seq>.npz`` per sequence in ``--output-dir`` with ``indices``, ``masks`` (N, H, W) uint8 (soft mask × 255),
``crops`` (N, 4) x0 y0 x1 y1 (full frame when not cropped) and ``modes`` (0 full frame, 1 crop, 2 crop then full).

    PYTHONPATH=. python scripts/infer_yolo_xp12_crop.py --sequences $(ls data/xp12/masks/yolo | sed 's/.npz//')
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from runway_detection.xp12 import list_sequences, load_sequence
from runway_detection.yolo.inference import load_yolo_model, union_soft_mask_for_class

FULL, CROP, CROP_THEN_FULL = 0, 1, 2


def runway_box(mask: np.ndarray, thr: float = 0.5) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask > thr)
    if xs.size < 4:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def crop_window(box, w: int, h: int, context: float, min_crop: int) -> tuple[int, int, int, int]:
    """16:9 window around ``box``, ``context`` times its size, clamped inside the frame."""
    x0, y0, x1, y1 = box
    cw = max(context * (x1 - x0), context * (y1 - y0) * w / h, min_crop)
    cw = min(cw, w)
    ch = cw * h / w
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(round(np.clip(cx - cw / 2, 0, w - cw)))
    top = int(round(np.clip(cy - ch / 2, 0, h - ch)))
    return left, top, left + int(round(cw)), top + int(round(ch))


def predict(model, img: np.ndarray, args) -> np.ndarray:
    res = model.predict(source=img, conf=args.conf, imgsz=args.imgsz, verbose=False, device=args.device)[0]
    if not args.raw_masks:
        return union_soft_mask_for_class(res, 0, img.shape[:2])
    h, w = img.shape[:2]                                  # max of the raw instance masks, not weighted by conf
    out = np.zeros((h, w), np.float32)
    if res.masks is not None and res.boxes is not None:
        for j in np.where(res.boxes.cls.cpu().numpy().astype(int) == 0)[0]:
            out = np.maximum(out, cv2.resize(res.masks.data[int(j)].cpu().numpy().astype(np.float32), (w, h)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/xp12/xp12_dataset"))
    ap.add_argument("--zip", type=Path, default=Path("xp12_dataset.zip"))
    ap.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/xp12/masks/yolo_crop"))
    ap.add_argument("--sequences", nargs="*", help="explicit sequence ids (default: every --every-th)")
    ap.add_argument("--every", type=int, default=4, help="take every k-th sequence of the sorted list")
    ap.add_argument("--stride", type=int, default=2, help="frame stride within a sequence")
    ap.add_argument("--context", type=float, default=4.0, help="crop size / runway box size")
    ap.add_argument("--min-crop", type=int, default=256, help="minimum crop width (px of the 1280 frame)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--raw-masks", action="store_true",
                    help="keep the raw instance masks; by default they are weighted by conf, which hides detections under 0.5")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seq_ids = args.sequences or list_sequences(args.root)[:: args.every]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_yolo_model(args.weights)

    for k, seq_id in enumerate(seq_ids):
        out = args.output_dir / f"{seq_id}.npz"
        if out.exists():
            continue
        seq = load_sequence(args.root, seq_id, zip_path=args.zip)
        t0, prev = time.time(), None
        indices, masks, crops, modes = [], [], [], []
        for f in seq.frames[:: args.stride]:
            img = seq.read_image(f)
            h, w = img.shape[:2]
            mode, win, soft = FULL, (0, 0, w, h), None
            if prev is not None:
                win = crop_window(prev, w, h, args.context, args.min_crop)
                x0, y0, x1, y1 = win
                part = predict(model, img[y0:y1, x0:x1], args)
                if runway_box(part) is not None:
                    soft = np.zeros((h, w), np.float32)
                    soft[y0:y1, x0:x1] = part
                    mode = CROP
                else:
                    mode, win = CROP_THEN_FULL, (0, 0, w, h)
            if soft is None:
                soft = predict(model, img, args)
            prev = runway_box(soft)
            indices.append(f.index)
            masks.append(np.round(soft * 255).astype(np.uint8))
            crops.append(win)
            modes.append(mode)
        np.savez_compressed(out, indices=np.array(indices), masks=np.stack(masks), crops=np.array(crops), modes=np.array(modes))
        m = np.array(modes)
        print(f"[{k + 1}/{len(seq_ids)}] {seq.label}: {len(m)} frames in {time.time() - t0:.0f}s, "
              f"crop {np.mean(m == CROP):.0%}, re-acquired {np.mean(m == CROP_THEN_FULL):.0%} → {out}", flush=True)


if __name__ == "__main__":
    main()
