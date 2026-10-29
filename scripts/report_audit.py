"""E0 report: summary table (LaTeX), distribution figures and contact sheets from audit.csv.

    python scripts/report_audit.py --audit data/audit/audit.csv --out paper
"""

from __future__ import annotations

import argparse
import functools
import zipfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["LARD", "BARS", "RLD", "XP12"]
COLORS = {"LARD": "#2a6fdb", "BARS": "#e07a1f", "RLD": "#3a9a5b", "XP12": "#8a4fbf"}

# Attribute flags; thresholds checked on the contact sheets written below.
NIGHT_LUM = 60
HAZE_DARK, HAZE_CONTRAST = 110, 35
GLARE_SAT = 0.01
FAR_PX = 16  # runway narrower than this at 640 px input


def flags(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["night"] = d.lum < NIGHT_LUM
    d["haze"] = (d.dark > HAZE_DARK) & (d.contrast < HAZE_CONTRAST)
    d["glare"] = d.sat > GLARE_SAT
    d["multi"] = d.n_rwy > 1
    d["empty"] = d.n_rwy.fillna(0) == 0
    d["far"] = d.bw_px640 < FAR_PX
    return d


def table(d: pd.DataFrame) -> str:
    rows = []
    for s in ORDER:
        x = d[d.dataset == s]
        if x.empty:
            continue
        pose = "yes" if x.dist_km.notna().any() else "no"
        ap = x.airport.replace("", np.nan).nunique()
        pct = lambda c: f"{100 * x[c].mean():.1f}"  # noqa: E731
        rows.append(
            f"{s} & {len(x):,} & {ap if ap else '--'} & {pose} & {x.bw_px640.median():.0f} & {pct('far')} & "
            f"{pct('night')} & {pct('haze')} & {pct('glare')} & {pct('multi')} & {pct('empty')} \\\\"
        )
    head = (
        "\\begin{table*}[t]\n\\centering\\small\n"
        "\\caption{Dataset audit with uniform descriptors. Width: median runway bounding-box width in pixels at a 640-px-wide input. "
        f"Far: width $<{FAR_PX}$ px. Night: mean luminance $<{NIGHT_LUM}$. Haze: dark-channel mean $>{HAZE_DARK}$ and RMS contrast $<{HAZE_CONTRAST}$. "
        f"Glare: $>{100 * GLARE_SAT:.0f}$\\,\\% saturated pixels. Multi: more than one labelled runway. Empty: no runway label. Percentages of images.}}\n"
        "\\label{tab:audit}\n\\begin{tabular}{lrrlrrrrrrr}\n\\toprule\n"
        "Dataset & Images & Airports & Pose & Width & Far & Night & Haze & Glare & Multi & Empty \\\\\n\\midrule\n"
    )
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"


def hists(d: pd.DataFrame, out: Path):
    specs = [
        ("bw_px640", "runway width at 640 px input (px, log)", np.logspace(0, np.log10(640), 40), True),
        ("dist_km", "slant distance (km)", np.linspace(0, 4.5, 40), False),
        ("lat_angle", "lateral path angle (deg)", np.linspace(-10, 10, 41), False),
        ("lum", "mean luminance", np.linspace(0, 255, 40), False),
        ("dark", "dark-channel mean (haze proxy)", np.linspace(0, 255, 40), False),
        ("cx", "runway centre x (fraction of width)", np.linspace(0, 1, 41), False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, (col, label, bins, logx) in zip(axes.flat, specs):
        for s in ORDER:
            v = d.loc[d.dataset == s, col].dropna()
            if len(v):
                ax.hist(v, bins=bins, histtype="step", density=True, lw=1.6, color=COLORS[s], label=s)
        ax.set_xlabel(label)
        if logx:
            ax.set_xscale("log")
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


@functools.lru_cache(maxsize=None)
def _zip(path: Path) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)


def read_image(row) -> np.ndarray | None:
    s, p = row.dataset, row.image
    if s in ("LARD", "BARS"):
        return cv2.imread(p)
    if s == "RLD":
        z = _zip(Path.home() / "Google Drive" / "Mon Drive" / "Runway Landing Dataset.zip")
        return cv2.imdecode(np.frombuffer(z.read(p), np.uint8), cv2.IMREAD_COLOR)
    seq, i = p.split("/")
    z = _zip(ROOT / "xp12_dataset.zip")
    return cv2.imdecode(np.frombuffer(z.read(f"xp12_dataset/{seq}/images/{i}.jpg"), np.uint8), cv2.IMREAD_COLOR)


def sheet(rows: pd.DataFrame, out: Path, cols: int = 6, w: int = 240):
    tiles = []
    for r in rows.itertuples():
        im = read_image(r)
        if im is None:
            continue
        im = cv2.resize(im, (w, round(w * 9 / 16)))
        cv2.putText(im, r.dataset, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(im)
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(out), grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=str(ROOT / "data" / "audit" / "audit.csv"))
    ap.add_argument("--out", default=str(ROOT / "paper"))
    ap.add_argument("--sheets", default=str(ROOT / "data" / "audit" / "sheets"))
    a = ap.parse_args()
    d = flags(pd.read_csv(a.audit, low_memory=False))
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "tables" / "audit.tex").write_text(table(d))
    hists(d, out / "figures" / "audit_hists.pdf")

    # Airport overlap between BARS splits (leakage check).
    b = d[d.dataset == "BARS"]
    tr, te = set(b[b.split == "train"].airport), set(b[b.split == "test"].airport)
    print(f"BARS airports train={len(tr)} test={len(te)} shared={len(tr & te)}; "
          f"test images at shared airports: {100 * b[(b.split == 'test') & b.airport.isin(tr)].shape[0] / max(1, (b.split == 'test').sum()):.0f}%")
    print(d.groupby("dataset")[["night", "haze", "glare", "multi", "empty", "far"]].mean().round(3))

    sh = Path(a.sheets)
    sh.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for f in ("night", "haze", "glare", "multi"):
        x = d[d[f]]
        sheet(x.sample(min(24, len(x)), random_state=0) if len(x) else x, sh / f"{f}.jpg")
    sheet(d.groupby("dataset").sample(6, random_state=int(rng.integers(1e6))), sh / "random.jpg")


if __name__ == "__main__":
    main()
