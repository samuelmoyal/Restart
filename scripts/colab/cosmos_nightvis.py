"""Night appearance control for Cosmos-Transfer2.5 multi-control (edge + vis): <clip>/night_vis.mp4.

    python cosmos_nightvis.py <clip_dir> [...]

Series 2 lessons: the prompt alone does not create runway light rows (day Canny edges carry no lights), and light points
drawn into the *edge* control are rendered as painted ground markings, not as emitters: edge control is structure only.
The vis (blur) control carries colour and luminance, so the lights go there: the sim frame darkened to night levels
(x0.12, slight blue cast) plus glowing light points at their true positions on the runway plane (homography of the GT
corners, as in cosmos_lights.py): white edge rows, green threshold row, red runway-end row, white approach bars.
Lights are drawn as Gaussian glows (sigma grows with the local scale) so the blur control keeps them as bright spots.
"""
import sys, os, json, numpy as np, cv2

N_EDGE, N_THR, APP, N_APP = 40, 14, 0.3, 12
DARK = float(os.environ.get("DARK", 0.12))


def rows():
    p = [(0.0, v, "w") for v in np.linspace(0, 1, N_EDGE)] + [(1.0, v, "w") for v in np.linspace(0, 1, N_EDGE)]
    p += [(u, 0.0, "g") for u in np.linspace(0, 1, N_THR)] + [(u, 1.0, "r") for u in np.linspace(0, 1, N_THR)]
    for v in np.linspace(-APP, -APP / N_APP, N_APP):
        p += [(u, v, "w") for u in np.linspace(0.42, 0.58, 4)]
    p += [(u, -APP / 3, "w") for u in np.linspace(0.15, 0.85, 10)]
    return p


COL = {"w": (235, 245, 255), "g": (120, 255, 120), "r": (60, 60, 255)}   # BGR
R = rows(); UV = np.array([[a, b] for a, b, _ in R], np.float64)

for clip in sys.argv[1:]:
    gt = json.load(open(f"{clip}/gt.json")); cap = cv2.VideoCapture(f"{clip}/input.mp4")
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(f"{clip}/night_vis_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 16, (W, H)); i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        base = (f.astype(np.float32) * DARK * np.array([1.15, 1.0, 0.9], np.float32))     # darker, slightly blue
        glow = np.zeros_like(base)
        g = gt[min(i, len(gt) - 1)]
        if g.get("labelled", False):
            q = np.array(g["corners"], np.float64)
            Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [0, 1], [1, 1], [1, 0]]), q.astype(np.float32)).astype(np.float64)
            sgn = np.sign((np.array([[0, 0, 1.0]]) @ Hm.T)[0, 2])
            h = np.c_[UV, np.ones(len(UV))] @ Hm.T; P = h[:, :2] / h[:, 2:3]; front = h[:, 2] * sgn > 1e-6
            hA = np.c_[np.zeros(len(UV)), UV[:, 1], np.ones(len(UV))] @ Hm.T; hB = np.c_[np.ones(len(UV)), UV[:, 1], np.ones(len(UV))] @ Hm.T
            wpx = np.linalg.norm(hA[:, :2] / hA[:, 2:3] - hB[:, :2] / hB[:, 2:3], axis=1)
            for (x, y), w, ok_, (_, _, c) in zip(P, wpx, front, R):
                if ok_ and -10 <= x < W + 10 and -10 <= y < H + 10 and np.isfinite(w):
                    # v2 (ENRY vis30: approach bars rendered well, edge-light rows too faint): bigger, brighter points
                    cv2.circle(glow, (int(round(x)), int(round(y))), int(np.clip(w / 15, 1, 6)), COL[c], -1, cv2.LINE_AA)
            glow = cv2.GaussianBlur(glow, (0, 0), 2.5) * 3.5 + glow
        vw.write(np.clip(base + glow, 0, 255).astype(np.uint8)); i += 1
    vw.release()
    os.system(f"ffmpeg -loglevel error -y -i {clip}/night_vis_raw.mp4 -c:v libx264 -pix_fmt yuv420p -crf 14 {clip}/night_vis.mp4 && rm {clip}/night_vis_raw.mp4")
    print(clip, i, "frames", flush=True)
