#!/usr/bin/env python3
"""
Stream-sample the ``linok-bc/xp12_flights`` archives without storing them.

The two ``.tar.zst`` archives (21.6 GB "easy", 41.3 GB "weather") are read as a
stream (curl | zstd | tar); one frame out of ``--every`` is kept per episode with
its pose / position / labels / corners, plus every per-episode metadata file
(meta.txt, weather.json, weather_trace.txt).

    PYTHONPATH=. python scripts/sample_xp12_flights.py --archive weather --every 30

Output: ``data/xp12_flights/<archive>/<episode>/{images,pose,position,labels,corners}/…``
"""

from __future__ import annotations

import argparse
import subprocess
import tarfile
import time
from pathlib import Path

URL = "https://huggingface.co/datasets/linok-bc/xp12_flights/resolve/main/xp12_dataset_{name}.tar.zst"
PER_FRAME = {"images", "pose", "position", "labels", "corners"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", choices=["easy", "weather"], required=True)
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("data/xp12_flights"))
    args = ap.parse_args()

    out = args.out / args.archive
    out.mkdir(parents=True, exist_ok=True)
    curl = subprocess.Popen(["curl", "-s", "-L", "--retry", "5", URL.format(name=args.archive)], stdout=subprocess.PIPE)
    zstd = subprocess.Popen(["zstd", "-dc", "--long=31"], stdin=curl.stdout, stdout=subprocess.PIPE)
    n = kept = kept_img = 0
    episodes: set[str] = set()
    t0 = time.time()
    with tarfile.open(fileobj=zstd.stdout, mode="r|") as tar:
        for m in tar:
            if not m.isfile():
                continue
            n += 1
            parts = m.name.split("/")
            if len(parts) >= 3 and parts[-2] in PER_FRAME:
                ep, kind, stem = parts[-3], parts[-2], parts[-1].split(".")[0]
                if not stem.isdigit() or (int(stem) + int(ep)) % args.every:
                    continue
                dst = out / ep / kind / parts[-1]
                kept_img += kind == "images"
            else:  # per-episode metadata
                ep = parts[-2]
                dst = out / ep / parts[-1]
            episodes.add(ep)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(tar.extractfile(m).read())
            kept += 1
            if kept_img and kept_img % 500 == 0 and dst.suffix == ".jpg":
                print(f"{n} members read, {kept_img} images kept, {len(episodes)} episodes, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {n} members, {kept_img} images kept from {len(episodes)} episodes in {time.time() - t0:.0f}s → {out}")


if __name__ == "__main__":
    main()
