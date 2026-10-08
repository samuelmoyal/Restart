"""Read an xp12_flights archive and write sim crops (512x288) whose runway size follows the real distribution.

    python prep_sim_v2.py <easy|weather> <stride> <out_dir>

Unlike prep_sim.py (last 600 m, runway 12-55 % of the crop width), frames are kept up to 3 km from the threshold and the
runway width (share of the 448x256 centre crop used downstream) is drawn from 70 % the real LARD distribution (training
airports, median 8 %) + 30 % uniform 15-40 %. A crop is 480 to 1280 px wide in the 1280x720 frame (never upscaled).
"""
import sys, tarfile, subprocess, json, os, time
import numpy as np, cv2

name, stride, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
os.makedirs(out + "/images", exist_ok=True)
rng = np.random.default_rng(0)
OW, OH, CENTRE = 512, 288, 512 / 448
REAL_Q = [0.0181, 0.0269, 0.0329, 0.0371, 0.0424, 0.0468, 0.0508, 0.0547, 0.0598, 0.0645, 0.0703, 0.076, 0.082, 0.0898, 0.0965, 0.1035, 0.1137,
          0.1248, 0.1367, 0.1504, 0.1641, 0.1816, 0.2031, 0.2341, 0.2866]          # LARD real (non held-out) runway width / image width, percentiles 1..99


def draw_f(lo, hi):
    """Runway width share of the centre crop, from the target mixture truncated to the feasible range [lo, hi]."""
    for _ in range(30):
        f = float(np.interp(rng.uniform(0, 24), np.arange(25), REAL_Q)) if rng.uniform() < 0.7 else float(rng.uniform(0.15, 0.40))
        if lo <= f <= hi: return f
    return None


def fill_rows(img, x0, x1, y0, y1):
    l = img[y0:y1, x0-6:x0-1].astype(np.float32).mean(1, keepdims=True); r = img[y0:y1, x1+1:x1+6].astype(np.float32).mean(1, keepdims=True)
    w = np.linspace(0, 1, x1-x0)[None, :, None]; img[y0:y1, x0:x1] = np.clip(l*(1-w)+r*w, 0, 255).astype(np.uint8)


def emit(ep, idx, jpg, c, x):
    if c[:, 2].sum() != 4 or not (-3000 <= x < 0): return None
    W, H = 1280, 720; q = c[:, :2] * [W, H]; bw = np.ptp(q[:, 0]); cx, cy = q[:, 0].mean(), q[:, 1].mean()
    if bw < 8: return None
    f = draw_f(bw / W * CENTRE, bw / 480 * CENTRE)                               # crop width between 480 px and the full frame
    if f is None: return None
    im = cv2.imdecode(np.frombuffer(jpg, np.uint8), 1)
    fill_rows(im, 575, 705, 30, 56); fill_rows(im, 518, 762, 58, 302)            # X-Plane HUD overlay
    w = min(bw / (f / CENTRE), W); h = w * 9 / 16
    x0 = float(np.clip(cx - w/2 + rng.uniform(-.2, .2)*w, 0, W-w)); y0 = float(np.clip(cy - rng.uniform(.40, .64)*h, 0, H-h))
    crop = cv2.resize(im[int(y0):int(y0+h), int(x0):int(x0+w)], (OW, OH), interpolation=cv2.INTER_AREA)
    fn = f"{name}_{ep}_{idx:06d}.jpg"; cv2.imwrite(f"{out}/images/{fn}", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {"file": fn, "episode": ep, "frame": idx, "along_m": -x, "width_share": bw / w * CENTRE, "crop_xywh": [x0, y0, w, h],
            "corners": ((q - [x0, y0]) * [OW / w, OH / h]).round(2).tolist()}


zs = subprocess.Popen(["zstd", "-dc", "--long=31", f"/content/hf/xp12_dataset_{name}.tar.zst"], stdout=subprocess.PIPE)
labels, airports, cur, buf, n, t0 = [], {}, None, {}, 0, time.time()


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
            if p[-1] == "meta.txt": airports[p[-2]] = tar.extractfile(m).read().split()[0].decode(); continue
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
for l in labels: l["airport"] = airports.get(l["episode"], "?")
json.dump(labels, open(f"{out}/labels_{name}.json", "w"))
print("DONE", name, n, "members,", len(labels), "crops from", len({l['episode'] for l in labels}), "episodes in", int(time.time()-t0), "s", flush=True)
