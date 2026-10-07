"""Stream xp12_flights and write runway-matched sim crops (512x288) + corner labels."""
import sys, tarfile, subprocess, json, os, time, io
import numpy as np, cv2

name, stride, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
allowed = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None   # weather regimes, via flights.csv
os.makedirs(out + "/images", exist_ok=True)
rng = np.random.default_rng(0)
URL = f"https://huggingface.co/datasets/linok-bc/xp12_flights/resolve/main/xp12_dataset_{name}.tar.zst"
OW, OH = 512, 288

def fill_rows(img, x0, x1, y0, y1):
    l = img[y0:y1, x0-6:x0-1].astype(np.float32).mean(1, keepdims=True); r = img[y0:y1, x1+1:x1+6].astype(np.float32).mean(1, keepdims=True)
    w = np.linspace(0, 1, x1-x0)[None, :, None]; img[y0:y1, x0:x1] = np.clip(l*(1-w)+r*w, 0, 255).astype(np.uint8)

def emit(ep, idx, jpg, c, x):
    if c[:, 2].sum() != 4 or not (-600 <= x < 0): return None
    im = cv2.imdecode(np.frombuffer(jpg, np.uint8), 1); H, W = im.shape[:2]
    fill_rows(im, 575, 705, 30, 56); fill_rows(im, 518, 762, 58, 302)            # X-Plane HUD overlay
    q = c[:, :2] * [W, H]; bw = np.ptp(q[:, 0]); cx, cy = q[:, 0].mean(), q[:, 1].mean()
    f = rng.uniform(0.08, 0.35)
    if bw / f < 480: f = bw / 480                                               # no upscaling
    if f > 0.42: return None
    w = min(bw / f, W); h = w * 9 / 16
    x0 = float(np.clip(cx - w/2 + rng.uniform(-.12, .12)*w, 0, W-w)); y0 = float(np.clip(cy - .57*h + rng.uniform(-.08, .08)*h, 0, H-h))
    crop = cv2.resize(im[int(y0):int(y0+h), int(x0):int(x0+w)], (OW, OH), interpolation=cv2.INTER_AREA)
    fn = f"{name}_{ep}_{idx:06d}.jpg"; cv2.imwrite(f"{out}/images/{fn}", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {"file": fn, "episode": ep, "frame": idx, "along_m": -x, "crop_xywh": [x0, y0, w, h],
            "corners": ((q - [x0, y0]) * [OW / w, OH / h]).round(2).tolist()}

LOCAL = f"/content/hf/xp12_dataset_{name}.tar.zst"
zs = subprocess.Popen(["zstd", "-dc", "--long=31", LOCAL], stdout=subprocess.PIPE)
labels, cur, buf, n, t0 = [], None, {}, 0, time.time()
def ready(ep, idx):
    b = buf.get(idx)
    if b and all(k in b for k in ("img", "corners", "position")):
        r = emit(ep, idx, b["img"], b["corners"], b["position"])
        if r: labels.append(r)
        del buf[idx]
try:
    with tarfile.open(fileobj=zs.stdout, mode="r|") as tar:
        for m in tar:
            if not m.isfile(): continue
            n += 1; p = m.name.split("/")
            if len(p) < 3 or p[-2] not in ("images", "corners", "position"): continue
            ep, kind, stem = p[-3], p[-2], p[-1].split(".")[0]
            if not stem.isdigit() or int(stem) % stride: continue
            if ep != cur: cur, buf = ep, {}
            idx = int(stem); data = tar.extractfile(m).read(); b = buf.setdefault(idx, {})
            if kind == "images": b["img"] = data
            elif kind == "corners": b["corners"] = np.array(data.split(), float).reshape(4, 3)
            else: b["position"] = float(data.split()[0])
            ready(ep, idx)
            if n % 100000 < 3: print(n, "members,", len(labels), "crops,", len({l['episode'] for l in labels}), "episodes,", int(time.time()-t0), "s", flush=True)
except Exception as e:
    print("stream ended:", type(e).__name__, e, flush=True)
json.dump(labels, open(f"{out}/labels_{name}.json", "w"))
print("DONE", name, n, "members,", len(labels), "crops from", len({l['episode'] for l in labels}), "episodes in", int(time.time()-t0), "s", flush=True)
