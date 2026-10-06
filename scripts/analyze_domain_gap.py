#!/usr/bin/env python3
"""
Sim (XP12) vs real (HPA crops) distribution study: which gross cues could a
sim-vs-real discriminator overfit to instead of "rendering realism"?

Domains
- ``sim_full`` : XP12 frames, full 1280×720 view
- ``sim_crop`` : XP12 frames cropped around the runway (16:9 window, random zoom/offset)
- ``real``     : manually cropped approach segments from the HPA films

For each image (resized to 512×288): colour / brightness / sharpness statistics,
YOLO runway box (same detector on every domain), and a CLIP embedding. Reports
per-feature medians, the AUC of each single feature as a sim-vs-real classifier,
and logistic-regression probes per feature group with a split by video / sequence.

    PYTHONPATH=. python scripts/analyze_domain_gap.py --real data/hpa/real_720p
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

SIZE = (512, 288)
FEATS = ["brightness", "contrast", "saturation", "mean_b", "mean_g", "mean_r", "dark_frac", "sky_minus_ground",
         "blue_top", "hf_ratio", "lap_var", "edge_density", "blockiness", "src_width"]
GROUPS = {
    "colour only": ["brightness", "contrast", "saturation", "mean_b", "mean_g", "mean_r", "dark_frac",
                    "sky_minus_ground", "blue_top"],
    "sharpness only": ["hf_ratio", "lap_var", "edge_density", "blockiness"],
    "source resolution only": ["src_width"],
    "runway box only": ["det", "box_w", "box_h", "box_cx", "box_cy"],
}


def image_features(bgr: np.ndarray, src_width: int) -> dict:
    im = cv2.resize(bgr, SIZE, interpolation=cv2.INTER_AREA if bgr.shape[1] >= SIZE[0] else cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    v = hsv[..., 2]
    third = SIZE[1] // 3
    f = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean()))) ** 2
    yy, xx = np.indices(f.shape)
    r = np.hypot((yy - f.shape[0] / 2) / f.shape[0], (xx - f.shape[1] / 2) / f.shape[1])
    gx = np.abs(np.diff(gray, axis=1))
    block = gx[:, 7::8].mean() / (gx.mean() + 1e-6)
    b, g, rr = [float(im[..., k].mean()) for k in range(3)]
    return {
        "brightness": float(v.mean()), "contrast": float(gray.std()), "saturation": float(hsv[..., 1].mean()),
        "mean_b": b, "mean_g": g, "mean_r": rr, "dark_frac": float((v < 40).mean()),
        "sky_minus_ground": float(v[:third].mean() - v[-third:].mean()),
        "blue_top": float(im[:third, :, 0].astype(np.float32).mean() - im[:third, :, 2].astype(np.float32).mean()),
        "hf_ratio": float(f[r > 0.25].sum() / (f.sum() + 1e-6)),
        "lap_var": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        "edge_density": float((cv2.Canny(im, 60, 160) > 0).mean()),
        "blockiness": float(block), "src_width": float(src_width),
    }


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg); 0.5 = indistinguishable."""
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for val in np.unique(x[np.isfinite(x)]):  # average ties
        m = x == val
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def logreg_auc(Xtr, ytr, Xte, yte, l2: float = 1e-2, iters: int = 400) -> float:
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w, b = np.zeros(Xtr.shape[1]), 0.0
    lr = 0.5
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Xtr @ w + b)))
        w -= lr * (Xtr.T @ (p - ytr) / len(ytr) + l2 * w)
        b -= lr * float((p - ytr).mean())
    s = Xte @ w + b
    return auc(s[yte == 1], s[yte == 0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", type=Path, default=Path("data/hpa/real_720p"))
    ap.add_argument("--sim-zip", type=Path, default=Path("data/xp12/xp12_sim2real_subset.zip"))
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("data/hpa/domain_gap"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from ultralytics import YOLO

    yolo = YOLO(str(args.weights))
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def describe(bgr, src_width):
        d = image_features(bgr, src_width)
        im = cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA if bgr.shape[1] >= 640 else cv2.INTER_CUBIC)
        r = yolo.predict(source=im, conf=0.25, imgsz=640, verbose=False, device="cpu")[0]
        box = None
        if r.boxes is not None and len(r.boxes):
            cls, cf = r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()
            idx = np.where(cls == 0)[0]
            if idx.size:
                box = r.boxes.xyxyn[idx[np.argmax(cf[idx])]].cpu().numpy()
        d.update(det=float(box is not None),
                 box_w=float(box[2] - box[0]) if box is not None else 0.0,
                 box_h=float(box[3] - box[1]) if box is not None else 0.0,
                 box_cx=float((box[0] + box[2]) / 2) if box is not None else 0.5,
                 box_cy=float((box[1] + box[3]) / 2) if box is not None else 0.5)
        with torch.no_grad():
            e = clip.get_image_features(**proc(images=Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), return_tensors="pt"))
            e = e.pooler_output if hasattr(e, "pooler_output") else e
        return d, (e / e.norm(dim=-1, keepdim=True)).numpy()[0]

    rows, embs, thumbs = [], [], {"sim_full": [], "sim_crop": [], "real": []}

    # ---- real: balanced over videos
    by_video: dict[str, list[Path]] = {}
    for f in sorted(args.real.glob("*/*.jpg")):
        by_video.setdefault(f.parent.name, []).append(f)
    per = max(1, args.n // len(by_video))
    for vid, files in by_video.items():
        for f in rng.choice(files, size=min(per, len(files)), replace=False):
            im = cv2.imread(str(f))
            d, e = describe(im, im.shape[1])
            rows.append({"domain": "real", "group": vid, **d})
            embs.append(e)
            if len(thumbs["real"]) < 48 and rng.random() < 0.06:
                thumbs["real"].append(cv2.resize(im, (320, 180)))
    print("real done", len(rows), flush=True)

    # ---- sim: full frame and runway-centred crop
    with zipfile.ZipFile(args.sim_zip) as z:
        labels = json.loads(z.read("labels.json"))
        names = rng.choice(sorted(labels), size=min(args.n, len(labels)), replace=False)
        for name in names:
            im = cv2.imdecode(np.frombuffer(z.read(f"images/{name}.jpg"), np.uint8), cv2.IMREAD_COLOR)
            H, W = im.shape[:2]
            q = np.array(labels[name]["corners_TR_TL_BL_BR"])
            bw, bh = np.ptp(q[:, 0]), np.ptp(q[:, 1])
            cx, cy = q[:, 0].mean(), q[:, 1].mean()
            zoom = rng.uniform(2.5, 7.0)
            w = float(np.clip(zoom * max(bw, bh * 16 / 9), 240, W))
            h = w * 9 / 16
            x = float(np.clip(cx - w / 2 + rng.uniform(-0.15, 0.15) * w, 0, W - w))
            y = float(np.clip(cy - h / 2 + rng.uniform(-0.15, 0.15) * h, 0, H - h))
            crop = im[int(y):int(y + h), int(x):int(x + w)]
            for dom, img, sw in (("sim_full", im, W), ("sim_crop", crop, crop.shape[1])):
                d, e = describe(img, sw)
                rows.append({"domain": dom, "group": name.split("_")[0], "along_m": labels[name]["along_m"], **d})
                embs.append(e)
                if len(thumbs[dom]) < 48 and rng.random() < 0.05:
                    thumbs[dom].append(cv2.resize(img, (320, 180)))
    print("sim done", len(rows), flush=True)

    E = np.array(embs)
    dom = np.array([r["domain"] for r in rows])
    grp = np.array([r["group"] for r in rows])
    cols = FEATS + ["det", "box_w", "box_h", "box_cx", "box_cy"]
    X = np.array([[r[c] for c in cols] for r in rows])
    with open(args.out / "features.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["domain", "group", "along_m"] + cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    for k, t in thumbs.items():
        if t:
            while len(t) % 8:
                t.append(np.zeros_like(t[0]))
            cv2.imwrite(str(args.out / f"samples_{k}.jpg"), np.vstack([np.hstack(t[i:i + 8]) for i in range(0, len(t), 8)]))

    out = io.StringIO()

    def say(*a):
        print(*a)
        print(*a, file=out)

    say(f"images: real={np.sum(dom == 'real')} ({len(by_video)} videos)  sim={np.sum(dom == 'sim_full')}")
    night = X[:, cols.index("dark_frac")] > 0.5
    say(f"night frames (>50 % dark pixels): real {night[dom == 'real'].mean():.1%}  sim {night[dom == 'sim_full'].mean():.1%}")
    for sim in ("sim_full", "sim_crop"):
        say(f"\n=== real vs {sim} ===")
        say(f"{'feature':18s} {'real med [IQR]':>24s} {'sim med [IQR]':>24s} {'AUC':>6s} {'AUC day':>8s}")
        a, b = dom == "real", dom == sim
        res = []
        for j, c in enumerate(cols):
            ra, sb = X[a, j], X[b, j]
            u = auc(ra, sb)
            ud = auc(X[a & ~night, j], X[b & ~night, j])
            res.append((abs(u - 0.5), c, ra, sb, u, ud))
        for _, c, ra, sb, u, ud in sorted(res, reverse=True):
            q = lambda v: f"{np.median(v):8.3g} [{np.percentile(v, 25):.3g}, {np.percentile(v, 75):.3g}]"
            say(f"{c:18s} {q(ra):>24s} {q(sb):>24s} {u:6.2f} {ud:8.2f}")
        # probes with a split by video / sequence
        groups = np.unique(grp[a | b])
        test_groups = set(groups[rng.permutation(len(groups))[: len(groups) // 3]])
        te = np.array([g in test_groups for g in grp])
        y = a.astype(float)
        say("logistic-regression probes (held-out videos / sequences), AUC all / day-only:")
        for name, feats in list(GROUPS.items()) + [("all hand-crafted", cols), ("CLIP embedding", None)]:
            M = E if feats is None else X[:, [cols.index(f) for f in feats]]
            vals = []
            for keep in (np.ones(len(y), bool), ~night):
                m = (a | b) & keep
                vals.append(logreg_auc(M[m & ~te], y[m & ~te], M[m & te], y[m & te]))
            say(f"  {name:24s} {vals[0]:.3f} / {vals[1]:.3f}")
    (args.out / "report.txt").write_text(out.getvalue())
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
