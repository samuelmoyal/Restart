"""Appearance (vis / blur) control videos per condition for Cosmos-Transfer2.5 multi-control (edge = structure, vis = colours).

    python cosmos_vis.py <mode> <clip_dir> [...]        mode in: snow, fog, rain

Series 2 lessons: the edge control fixes the layout but not the appearance, and the prompt alone can be misread
(snow prompt "the cleared runway stays darker than the snow" turned a grass clearing into a giant dark runway on CYZF).
The vis control carries colour and luminance, so each condition is rendered here from the sim frame by simple image ops,
with the runway (GT quad, dilated) kept as asphalt:
- snow: vegetation and bare ground (everything but the runway, the sky and very dark pixels) pushed to snow white with
        its texture kept as low-contrast shading; dark pixels (forest) to dark grey; sky to overcast white.
- fog:  blend towards a grey fog colour with a transmission exp(-K / (y - y_h)) (flat ground: distance ~ 1 / rows below
        the horizon, estimated from the far runway corners).
- rain: darker, desaturated, slightly blurred frame, grey sky.
"""
import sys, os, json, numpy as np, cv2

MODE, CLIPS = sys.argv[1], sys.argv[2:]


def runway_mask(g, W, H, grow=0.15):
    m = np.zeros((H, W), np.uint8)
    if g.get("labelled", False):
        q = np.array(g["corners"]); c = q.mean(0); q2 = c + (q - c) * (1 + grow)
        cv2.fillPoly(m, [np.round(q2).astype(np.int32)], 255)
    return m > 0


def horizon(g, H):
    if g.get("labelled", False):
        q = np.array(g["corners"]); return float(min(q[1, 1], q[2, 1])) - 4      # far corners FL, FR
    return H * 0.42


for clip in CLIPS:
    gt = json.load(open(f"{clip}/gt.json")); cap = cv2.VideoCapture(f"{clip}/input.mp4")
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = f"{clip}/{MODE}_vis"; vw = cv2.VideoWriter(f"{out}_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 16, (W, H)); i = 0
    yy = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1)
    while True:
        ok, f = cap.read()
        if not ok: break
        g = gt[min(i, len(gt) - 1)]; rw = runway_mask(g, W, H); yh = horizon(g, H)
        x = f.astype(np.float32); gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
        sky = yy < yh
        if MODE == "snow":
            shade = (gray - gray.mean()) * 0.35
            snow = np.clip(225 + shade, 150, 250).repeat(3, 2)
            dark = (gray[..., 0] < 55)[..., None]
            o = np.where(dark, np.clip(gray * 0.9 + 25, 30, 110).repeat(3, 2), snow)
            o = np.where(sky[..., None], np.full_like(x, 215), o)
            o = np.where(rw[..., None], x * 0.9, o)
        elif MODE == "fog":
            # v2 (first fog vis too light: haze only on the horizon): transmission exp(-K / rows below the horizon),
            # distance ~ 1 / (y - y_h) on flat ground; K=46: t=0.10 at 20 px below the horizon, 0.79 at 200 px
            t = np.exp(-float(os.environ.get("FOGK", 46)) / np.maximum(yy - yh, 1e-3)) * (yy > yh)
            fogc = np.array([200, 198, 195], np.float32)
            o = x * t[..., None] + fogc * (1 - t[..., None])
        elif MODE == "rain":
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32); hsv[..., 1] *= 0.45; hsv[..., 2] *= 0.6
            o = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
            o = cv2.GaussianBlur(o, (0, 0), 1.2); o = np.where(sky[..., None], np.full_like(x, 120), o)
        else:
            raise SystemExit(f"unknown mode {MODE}")
        vw.write(np.clip(o, 0, 255).astype(np.uint8)); i += 1
    vw.release()
    os.system(f"ffmpeg -loglevel error -y -i {out}_raw.mp4 -c:v libx264 -pix_fmt yuv420p -crf 14 {out}.mp4 && rm {out}_raw.mp4")
    print(clip, MODE, i, "frames", flush=True)
