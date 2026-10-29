#!/usr/bin/env python3
"""
Compare cached YOLO mask sources on XP12 against the GT runway, by distance to the threshold.

For every frame present in all ``--dirs`` (same ``<seq>.npz`` format as ``infer_yolo_xp12.py``):
detected = some pixel > 0.5; IoU with the filled GT quad; hit = IoU > 0.5. Printed as a Markdown table
per range bin, plus the GT runway width (px) in each bin.

    PYTHONPATH=. python scripts/compare_yolo_masks.py --dirs data/xp12/masks/yolo data/xp12/masks/yolo_crop
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluation.observations import YoloMaskCache, gt_corners_array, gt_mask, mask_iou
from runway_detection.xp12 import load_sequence

BINS = [(-3000, -2000), (-2000, -1000), (-1000, -500), (-500, 0)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/xp12/xp12_dataset"))
    ap.add_argument("--zip", type=Path, default=Path("xp12_dataset.zip"))
    ap.add_argument("--dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--sequences", nargs="*")
    args = ap.parse_args()

    caches = [YoloMaskCache(d) for d in args.dirs]
    seqs = args.sequences or sorted(set.intersection(*[{p.stem for p in d.glob("*.npz")} for d in args.dirs]))
    rows = []  # along, width, then (detected, iou) per source
    for s in seqs:
        seq = load_sequence(args.root, s, zip_path=args.zip)
        common = set.intersection(*[set(c.available_indices(s)) for c in caches])
        for f in seq.frames:
            if f.index not in common:
                continue
            g = gt_mask(f)
            r = [f.x_m, float(np.ptp(gt_corners_array(f)[:, 0]))]
            for c in caches:
                m = c.get(s, f.index)
                r += [float((m > 0.5).any()), mask_iou(m, g)]
            rows.append(r)
    a = np.array(rows)
    names = [d.name for d in args.dirs]
    print(f"{len(seqs)} sequences, {len(a)} frames\n")
    print("| range (m) | frames | GT width (px) | " + " | ".join(f"{n} detected | {n} IoU>0.5 | {n} mean IoU" for n in names) + " |")
    print("|---" * (3 + 3 * len(names)) + "|")
    for lo, hi in BINS + [(-1e9, 1e9)]:
        sel = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if not sel.any():
            continue
        b = a[sel]
        cells = []
        for k in range(len(names)):
            det, iou = b[:, 2 + 2 * k], b[:, 3 + 2 * k]
            cells += [f"{det.mean():.1%}", f"{(iou > 0.5).mean():.1%}", f"{iou.mean():.2f}"]
        label = "all" if lo < -1e8 else f"{lo} → {hi}"
        print(f"| {label} | {sel.sum()} | {np.median(b[:, 1]):.0f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
