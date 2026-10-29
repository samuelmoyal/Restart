"""E0: zero-shot weather / lighting tags with CLIP for every image in audit.csv.

Each attribute is a binary prompt pair; the score is the softmax probability of the positive
prompt (CLIP logit scale), so 0.5 is chance. Thresholds are set by inspecting contact sheets.

    python scripts/clip_tags.py --audit data/audit/audit.csv --out data/audit/clip.csv
"""

from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_audit import read_image  # noqa: E402

DRIVE_AUDIT = Path.home() / "Google Drive" / "Mon Drive" / "capstone_sim2real" / "paper" / "audit"

PAIRS = {
    "night": ("a photo taken at night", "a photo taken during the day"),
    "dusk": ("a photo at sunset or dusk with low sun", "a photo in the middle of the day"),
    "fog": ("a landscape in thick fog with low visibility", "a landscape in clear weather with good visibility"),
    "rain": ("a photo through a windshield covered with raindrops", "a photo through a clean dry windshield"),
    "snow": ("a snowy landscape covered in snow", "a landscape without snow"),
    "glare": ("a photo with strong sun glare and lens flare", "a photo without sun glare"),
    "overcast": ("an overcast grey cloudy sky", "a clear blue sky"),
}


def _tensor(x):
    """transformers >= 5 returns a ModelOutput from get_*_features."""
    return x if torch.is_tensor(x) else x.pooler_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="data/audit/audit.csv")
    ap.add_argument("--out", default="data/audit/clip.csv")
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--no-drive-copy", dest="drive_copy", action="store_false",
                    help="do not copy the finished CSV to the Drive audit folder")
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(a.model).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(a.model)
    prompts = [p for pair in PAIRS.values() for p in pair]
    with torch.no_grad():
        t = _tensor(model.get_text_features(**proc(text=prompts, return_tensors="pt", padding=True).to(dev)))
        t = t / t.norm(dim=-1, keepdim=True)
    d = pd.read_csv(a.audit, low_memory=False)
    out = []
    if Path(a.out).exists():  # resume: keep rows already tagged, skip their images
        prev = pd.read_csv(a.out)
        out = prev.to_dict("records")
        d = d[~d.image.isin(set(prev.image))]
        print(f"resume: {len(prev)} tagged, {len(d)} left", flush=True)

    def load(r):
        im = read_image(r)
        return None if im is None else cv2.cvtColor(cv2.resize(im, (336, 224)), cv2.COLOR_BGR2RGB)

    with ThreadPoolExecutor(8) as ex:
        for i in range(0, len(d), a.batch):
            chunk = d.iloc[i:i + a.batch]
            ims = list(ex.map(load, chunk.itertuples()))
            keep = [j for j, im in enumerate(ims) if im is not None]
            if not keep:
                continue
            with torch.no_grad():
                f = _tensor(model.get_image_features(**proc(images=[ims[j] for j in keep], return_tensors="pt").to(dev)))
                f = f / f.norm(dim=-1, keepdim=True)
                logits = (model.logit_scale.exp() * f @ t.T).view(len(keep), len(PAIRS), 2)
                p = logits.softmax(-1)[..., 0].cpu().numpy()
            for j, row in zip(keep, p):
                out.append({"dataset": chunk.iloc[j].dataset, "image": chunk.iloc[j].image,
                            **{f"clip_{k}": float(v) for k, v in zip(PAIRS, row)}})
            if (i // a.batch) % 20 == 0:
                print(i, len(d), flush=True)
                pd.DataFrame(out).to_csv(a.out, index=False)
    pd.DataFrame(out).to_csv(a.out, index=False)
    if a.drive_copy and DRIVE_AUDIT.parent.exists():
        DRIVE_AUDIT.mkdir(parents=True, exist_ok=True)
        shutil.copy(a.out, DRIVE_AUDIT / Path(a.out).name)


if __name__ == "__main__":
    main()
