#!/usr/bin/env python3
"""
Physically-based fog on XP12 frames (Koschmieder, exact per-pixel ground distance).

    PYTHONPATH=. python scripts/fog_xp12.py --seq 000018 --frame 300            # comparison strip
    PYTHONPATH=. python scripts/fog_xp12.py --yolo-check --weights weights/best.pt  # detection vs visibility
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from evaluation.observations import gt_mask, mask_iou
from evaluation.weather import apply_fog, remove_hud
from runway_detection.xp12 import list_sequences, load_calibrations, load_sequence, pose_from_frame, xp12_intrinsics

ROOT = Path("data/xp12/xp12_dataset")


def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (430, 46), (20, 20, 20), -1)
    cv2.putText(img, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default="000018")
    ap.add_argument("--frame", type=int, default=300)
    ap.add_argument("--visibility", type=float, nargs="+", default=[8000, 3000, 1500, 800])
    ap.add_argument("--zip", type=Path, default=Path("xp12_dataset.zip"))
    ap.add_argument("--out", type=Path, default=Path("data/xp12/fog"))
    ap.add_argument("--yolo-check", action="store_true", help="mask IoU / miss rate vs visibility on sampled frames")
    ap.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    ap.add_argument("--n-seqs", type=int, default=12)
    args = ap.parse_args()

    calibs = load_calibrations("data/xp12/calibration.json")
    K = xp12_intrinsics()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.yolo_check:
        seq = load_sequence(ROOT, args.seq, zip_path=args.zip)
        f = next(x for x in seq.frames if x.index == args.frame)
        pose = pose_from_frame(f, calibs[args.seq])
        clean = remove_hud(seq.read_image(f))
        tiles = [_label(clean, f"clear  (along {pose.along_track_m:.0f} m)")]
        for vis in args.visibility:
            foggy = apply_fog(clean, pose, K, vis)
            cv2.imwrite(str(args.out / f"{args.seq}_{args.frame:06d}_vis{int(vis)}.jpg"), foggy)
            tiles.append(_label(foggy, f"visibility {vis:.0f} m"))
        while len(tiles) % 2:
            tiles.append(np.zeros_like(clean))
        rows = [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
        out = args.out / f"{args.seq}_{args.frame:06d}_fog_compare.jpg"
        cv2.imwrite(str(out), cv2.resize(np.vstack(rows), None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA))
        print(f"→ {out}")
        return

    from runway_detection.yolo.inference import load_yolo_model, union_soft_mask_for_class

    model = load_yolo_model(args.weights)
    levels = [None] + list(args.visibility)
    stats = {v: {"near": [], "far": []} for v in levels}
    seq_ids = list_sequences(ROOT)
    for sid in seq_ids[:: max(1, len(seq_ids) // args.n_seqs)][: args.n_seqs]:
        seq = load_sequence(ROOT, sid, zip_path=args.zip)
        for i in np.linspace(0, len(seq.frames) - 1, 5).astype(int):
            f = seq.frames[i]
            pose = pose_from_frame(f, calibs[sid])
            clean = remove_hud(seq.read_image(f))
            g = gt_mask(f)
            for vis in levels:
                img = clean if vis is None else apply_fog(clean, pose, K, vis)
                res = model.predict(source=img, conf=0.25, imgsz=640, verbose=False, device="cpu")[0]
                m = union_soft_mask_for_class(res, 0, img.shape[:2])
                iou = mask_iou(m, g) if (m > 0.5).any() else 0.0
                stats[vis]["near" if pose.along_track_m < 1500 else "far"].append(iou)
    print(f"{'visibility':>12} | {'IoU <1.5km':>10} {'missed':>8} | {'IoU >1.5km':>10} {'missed':>8}")
    for vis in levels:
        cells = []
        for k in ("near", "far"):
            v = np.array(stats[vis][k])
            cells.append(f"{v.mean():10.3f} {int((v == 0).sum()):4d}/{len(v):<3d}")
        print(f"{'clear' if vis is None else f'{vis:.0f} m':>12} | {cells[0]} | {cells[1]}")


if __name__ == "__main__":
    main()
