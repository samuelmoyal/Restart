"""E0: per-image audit of LARD, BARS, RLD and XP12 with one uniform set of descriptors.

Geometry (from labels): number of runway instances, area fraction and bbox size of the
largest runway, its width in pixels once the image is resized to 640 wide.
Photometry (from pixels, on a 320-wide copy): mean luminance, RMS contrast, dark-channel
mean (haze proxy), saturated-pixel fraction (glare proxy), Laplacian variance (blur proxy).
Pose (LARD, XP12 only): slant distance, lateral / vertical path angles, relative yaw.

    python scripts/audit_datasets.py --out data/audit/audit.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import re
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DRIVE = Path.home() / "Google Drive" / "Mon Drive"
LARD_DIR = ROOT / "data" / "LARD"
BARS_DIR = DRIVE / "BARS_yolo"
RLD_ZIP = DRIVE / "Runway Landing Dataset.zip"
XP12_DIR = ROOT / "data" / "xp12" / "xp12_dataset"
XP12_ZIP = ROOT / "xp12_dataset.zip"


def photometry(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    small = cv2.resize(img, (320, max(1, round(320 * h / w))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dark = cv2.erode(small.min(axis=2), np.ones((7, 7), np.uint8))
    return {
        "lum": float(gray.mean()),
        "contrast": float(gray.std()),
        "dark": float(dark.mean()),
        "sat": float((gray >= 250).mean()),
        "sharp": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
    }


def geometry(polys: list[np.ndarray], w: int, h: int) -> dict:
    """polys in pixel coordinates; the largest one is the reference runway."""
    if not polys:
        return {"n_rwy": 0}
    areas = [abs(cv2.contourArea(p.astype(np.float32))) for p in polys]
    p = polys[int(np.argmax(areas))]
    x0, y0 = p.min(axis=0)
    x1, y1 = p.max(axis=0)
    return {
        "n_rwy": len(polys),
        "area_frac": max(areas) / (w * h),
        "bw_frac": (x1 - x0) / w,
        "bh_frac": (y1 - y0) / h,
        "bw_px640": (x1 - x0) / w * 640,
        "cx": (x0 + x1) / 2 / w,
        "cy": (y0 + y1) / 2 / h,
    }


def yolo_polys(text: str, w: int, h: int, cls: int = 0) -> list[np.ndarray]:
    out = []
    for line in text.splitlines():
        v = line.split()
        if len(v) >= 7 and int(float(v[0])) == cls:
            out.append(np.array(v[1:], float).reshape(-1, 2) * (w, h))
    return out


# ---------------------------------------------------------------- per-source jobs

def job_file(args):
    src, path, label, extra = args
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    polys = [label] if isinstance(label, np.ndarray) else yolo_polys(label, w, h)
    return {"dataset": src, "image": str(path), "W": w, "H": h, **extra, **geometry(polys, w, h), **photometry(img)}


def lard_jobs():
    rows = list(csv.DictReader(open(LARD_DIR / "metadata_flsim_train.csv"), delimiter=";"))
    for r in rows:
        quad = np.array([[float(r[f"x_{c}"]), float(r[f"y_{c}"])] for c in ("TL", "TR", "BR", "BL")])
        extra = {
            "split": "train", "airport": r["airport"], "runway": r["runway"],
            "dist_km": float(r["slant_distance"]), "lat_angle": float(r["lateral_path_angle"]),
            "vert_angle": float(r["vertical_path_angle"]), "yaw_rel": np.nan,
            "height_m": float(r["height_above_runway"]),
        }
        yield ("LARD", LARD_DIR / r["image"], quad, extra)


def bars_airport(name: str) -> str:
    stem = re.sub(r"_\d+$", "", Path(name).stem)
    return stem


def bars_jobs():
    for split in ("train", "val", "test"):
        for p in sorted(q for q in (BARS_DIR / "images" / split).iterdir() if q.suffix.lower() in (".jpg", ".png", ".jpeg")):
            lab = BARS_DIR / "labels" / split / (p.stem + ".txt")
            text = lab.read_text() if lab.exists() else ""
            yield ("BARS", p, text, {"split": split, "airport": bars_airport(p.name)})


def rld_rows(_):
    z = zipfile.ZipFile(RLD_ZIP)
    names = sorted(n for n in z.namelist() if n.endswith(".jpg"))
    out = []
    for n in names:
        img = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        ln = n.replace("images_rename", "labels_rename").replace(".jpg", ".txt")
        text = z.read(ln).decode() if ln in z.NameToInfo else ""
        out.append({"dataset": "RLD", "image": n, "W": w, "H": h, "split": "all", "airport": "",
                    **geometry(yolo_polys(text, w, h), w, h), **photometry(img)})
    return out


def xp12_seq(seq: str, step: int = 10):
    z = zipfile.ZipFile(XP12_ZIP)
    d = XP12_DIR / seq
    airport, runway = (d / "meta.txt").read_text().split()[:2]
    out = []
    for lab in sorted((d / "labels").glob("*.txt"))[::step]:
        i = lab.stem
        try:
            img = cv2.imdecode(np.frombuffer(z.read(f"xp12_dataset/{seq}/images/{i}.jpg"), np.uint8), cv2.IMREAD_COLOR)
        except KeyError:
            continue
        h, w = img.shape[:2]
        v = np.array(lab.read_text().split()[1:9], float).reshape(4, 2) * (w, h)
        x, y, zc = map(float, (d / "position" / f"{i}.txt").read_text().split()[:3])
        yaw = float((d / "pose" / f"{i}.txt").read_text().split()[0])
        dist = math.sqrt(x * x + y * y + zc * zc) / 1000
        out.append({
            "dataset": "XP12", "image": f"{seq}/{i}", "W": w, "H": h, "split": "test", "airport": airport,
            "runway": runway, "seq": seq, "dist_km": dist,
            "lat_angle": math.degrees(math.atan2(zc, -x)) if x < 0 else np.nan,
            "vert_angle": math.degrees(math.atan2(y, -x)) if x < 0 else np.nan,
            "yaw_rel": yaw, "height_m": y,
            **geometry([v], w, h), **photometry(img),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "audit" / "audit.csv"))
    ap.add_argument("--only", nargs="*", default=["LARD", "BARS", "RLD", "XP12"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    with Pool(a.workers) as pool:
        for src in a.only:
            if src == "LARD":
                rows = [r for r in pool.imap(job_file, lard_jobs(), chunksize=32) if r]
            elif src == "BARS":
                rows = [r for r in pool.imap(job_file, bars_jobs(), chunksize=32) if r]
            elif src == "RLD":
                rows = rld_rows(None)
            else:
                seqs = sorted(p.name for p in XP12_DIR.iterdir() if (p / "meta.txt").exists())
                rows = [r for rs in pool.imap(xp12_seq, seqs) for r in rs]
            df = pd.DataFrame(rows)
            df.to_csv(out.with_name(f"audit_{src}.csv"), index=False)
            print(src, len(df), flush=True)
            frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)


if __name__ == "__main__":
    main()
