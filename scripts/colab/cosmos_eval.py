"""Runway alignment of Cosmos-Transfer outputs against the X-Plane ground truth, frame by frame, plus a contact sheet.

    python cosmos_eval.py <in_dir> <out_dir> <sheet.jpg> <clip>_<prompt> [...]

<in_dir>/<clip>/{input.mp4, gt.json} come from cosmos_prep.py, <out_dir>/<clip>_<prompt>.mp4 is the generated video.
The lateral shift of each long runway edge (sim frame vs generated frame) is measured by cross-correlation of the gradient
profiles normal to the edge (same metric as gen_dataset_t2i.py), in pixels of the 1280x720 frame.
"""
import sys, os, json, numpy as np, cv2

IN, OUT, SHEET = sys.argv[1:4]; NAMES = sys.argv[4:]


def frames(path):
    cap, fr = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok: return fr
        fr.append(f)


def edge_shift(a, b, p0, p1, half=8):
    """Lateral shift (px) of the image edge along p0-p1 between gray images a and b, and its correlation."""
    d = (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-9); nrm = np.array([-d[1], d[0]]); offs = np.arange(-half * 2, half * 2 + 1); prof = []
    for im in (a, b):
        rows = []
        for t in np.linspace(0.15, 0.85, 25):
            pts = (p0 + t * (p1 - p0))[None] + offs[:, None] * nrm[None]
            rows.append(cv2.remap(im, pts[:, 0].astype(np.float32)[None], pts[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0])
        prof.append(np.abs(np.gradient(np.mean(rows, axis=0))))
    pa, pb = prof; best, bs = 0, -1e9
    for s in range(-half, half + 1):
        u = pa[half: -half]; v = pb[half + s: len(pb) - half + s]
        c = float(np.dot(u - u.mean(), v - v.mean()) / (np.linalg.norm(u - u.mean()) * np.linalg.norm(v - v.mean()) + 1e-9))
        if c > bs: bs, best = c, s
    return best, bs


rows, report = [], {}
for name in NAMES:
    clip = name.rsplit("_", 1)[0]; gt = json.load(open(f"{IN}/{clip}/gt.json"))
    sim, gen = frames(f"{IN}/{clip}/input.mp4"), frames(f"{OUT}/{name}.mp4")
    n = min(len(sim), len(gen), len(gt)); H, W = sim[0].shape[:2]
    gen = [cv2.resize(g, (W, H)) if g.shape[:2] != (H, W) else g for g in gen]
    s, c, wpx = [], [], []
    for i in range(n):
        if not gt[i]["visible"]: continue
        q = np.array(gt[i]["corners"]); ga, gb = (cv2.cvtColor(x[i], cv2.COLOR_BGR2GRAY).astype(np.float32) for x in (sim, gen))
        for p0, p1 in ((q[0], q[1]), (q[3], q[2])):                                    # NL-FL and NR-FR long edges
            if np.linalg.norm(p1 - p0) < 20: continue
            a, b = edge_shift(ga, gb, p0, p1); s.append(abs(a)); c.append(b)
        wpx.append(np.ptp(q[:, 0]))
    s = np.array(s); report[name] = dict(frames=n, sim_frames=len(sim), gen_frames=len(gen), gen_size=list(gen[0].shape[:2]), runway_width_px=float(np.median(wpx)),
                                         shift_median=float(np.median(s)), shift_p90=float(np.percentile(s, 90)), off_gt2px=float(np.mean(s > 2)), corr_median=float(np.median(c)))
    print(name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in report[name].items()}, flush=True)
    row = []
    for i in (0, n // 2, n - 1):
        g = gen[i].copy(); cv2.polylines(g, [np.round(np.array(gt[i]["corners"])).astype(np.int32)], True, (0, 255, 0), 1)
        row += [cv2.resize(sim[i], (480, 270)), cv2.resize(g, (480, 270))]
    r = np.hstack(row); cv2.putText(r, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2); rows.append(r)
cv2.imwrite(SHEET, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); json.dump(report, open(SHEET[:-4] + ".json", "w"), indent=1); print("saved", SHEET)
