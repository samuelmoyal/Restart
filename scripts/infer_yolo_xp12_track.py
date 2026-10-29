#!/usr/bin/env python3
"""
YOLO-seg on XP12 frames through Ultralytics tracking (BoT-SORT / ByteTrack), cached like ``infer_yolo_xp12.py``.

``model.track(..., persist=True)`` is called frame by frame. Ultralytics keeps the raw instance masks of the
detections matched to a confirmed track (only the boxes are replaced by the tracker state), so the same
mask → quad → pose post-processing applies unchanged. Per frame we keep ONE runway: the instance of the
track followed so far, else the most confident track (a new track id is then followed). Unlike
``infer_yolo_xp12.py`` the mask is not weighted by the confidence: tracking feeds detections down to
``--conf`` (0.1 by default in Ultralytics), whose conf-weighted mask would fall under the 0.5 threshold.

``--tracker none`` runs plain ``predict`` with the same selection and threshold (ablation: lower conf, no tracker).

    PYTHONPATH=. python scripts/infer_yolo_xp12_track.py --tracker botsort.yaml --output-dir data/xp12/masks/yolo_botsort
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from runway_detection.xp12 import list_sequences, load_sequence
from runway_detection.yolo.inference import load_yolo_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/xp12/xp12_dataset"))
    ap.add_argument("--zip", type=Path, default=Path("xp12_dataset.zip"))
    ap.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tracker", default="botsort.yaml", help="Ultralytics tracker config, or 'none'")
    ap.add_argument("--sequences", nargs="*", help="explicit sequence ids (default: every --every-th)")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seq_ids = args.sequences or list_sequences(args.root)[:: args.every]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for k, seq_id in enumerate(seq_ids):
        out = args.output_dir / f"{seq_id}.npz"
        if out.exists():
            continue
        model = load_yolo_model(args.weights)  # fresh tracker state per sequence
        seq = load_sequence(args.root, seq_id, zip_path=args.zip)
        t0, followed = time.time(), None
        indices, masks, confs, ids = [], [], [], []
        for f in seq.frames[:: args.stride]:
            img = seq.read_image(f)
            h, w = img.shape[:2]
            kw = dict(source=img, conf=args.conf, imgsz=args.imgsz, verbose=False, device=args.device)
            res = (model.predict(**kw) if args.tracker == "none" else model.track(**kw, persist=True, tracker=args.tracker))[0]
            soft, c, tid = np.zeros((h, w), np.float32), 0.0, -1
            if res.masks is not None and res.boxes is not None and len(res.boxes):
                cls = res.boxes.cls.cpu().numpy().astype(int)
                conf = res.boxes.conf.cpu().numpy()
                tids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.full(len(cls), -1)
                cand = np.where(cls == 0)[0]
                if cand.size:
                    same = cand[tids[cand] == followed] if followed is not None else cand[:0]
                    j = int(same[0]) if same.size else int(cand[np.argmax(conf[cand])])
                    m = res.masks.data[j].cpu().numpy().astype(np.float32)
                    soft = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) if m.shape != (h, w) else m
                    c, tid = float(conf[j]), int(tids[j])
                    if tid >= 0:
                        followed = tid
            indices.append(f.index)
            masks.append(np.round(np.clip(soft, 0, 1) * 255).astype(np.uint8))
            confs.append(c)
            ids.append(tid)
        np.savez_compressed(out, indices=np.array(indices), masks=np.stack(masks), confs=np.array(confs), track_ids=np.array(ids))
        d = np.array(confs) > 0
        print(f"[{k + 1}/{len(seq_ids)}] {seq.label}: {len(d)} frames in {time.time() - t0:.0f}s, runway in {d.mean():.0%}, "
              f"{len(set(ids) - {-1})} track ids → {out}", flush=True)


if __name__ == "__main__":
    main()
