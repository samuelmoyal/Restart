"""Package every finished Cosmos-Transfer video with its annotations and copy it to Drive (idempotent, run it in a loop).

    python cosmos_post.py <drive_dir> <batch>=<out_dir> [...]

For each <out_dir>/<name>.mp4 (and the <name>.json arguments Cosmos writes next to it), writes to <drive_dir>/<batch>/<name>/:
  generated.mp4, input.mp4, edge.mp4, edge_light.mp4 and mask.mp4 (if any), gt.json, args.json, style.jpg (real style image, if any),
  frames.zip  - images/<name>_<i>.jpg (generated frames) + labels/<name>_<i>.txt (YOLO-seg runway polygon, clipped to the frame),
                only for the frames whose runway can be labelled (gt.json "labelled"),
  align.csv   - per frame: distance to threshold, runway width (px), lateral shift of each long edge vs the sim frame (px), correlation,
  sheet.jpg   - sim / generated (GT in green) at 6 instants,
  summary.json.
"""
import sys, os, json, csv, shutil, zipfile, time, numpy as np, cv2

DRIVE = sys.argv[1]; BATCHES = dict(a.split("=", 1) for a in sys.argv[2:])


def frames(path):
    cap, fr = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok: return fr
        fr.append(f)


def clip_poly(poly, w, h):
    """Sutherland-Hodgman clipping of a polygon to the image rectangle."""
    for ax, lim, keep in ((0, 0, 1), (0, w, -1), (1, 0, 1), (1, h, -1)):
        out = []
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]; ia, ib = keep * (a[ax] - lim) >= 0, keep * (b[ax] - lim) >= 0
            if ia: out.append(a)
            if ia != ib: out.append(a + (b - a) * (lim - a[ax]) / (b[ax] - a[ax]))
        poly = out
        if len(poly) < 3: return []
    return poly


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


for batch, out in BATCHES.items():
    if not os.path.isdir(out): continue
    for f in sorted(os.listdir(out)):
        if not f.endswith(".mp4") or "_control_" in f or "_mask" in f: continue
        name = f[:-4]; dst = f"{DRIVE}/{batch}/{name}"
        if os.path.exists(f"{dst}/summary.json") or not os.path.exists(f"{out}/{name}.json"): continue
        if time.time() - os.path.getmtime(f"{out}/{f}") < 60: continue                       # still being written
        args = json.load(open(f"{out}/{name}.json")); args = args.get("args", args); clip = os.path.dirname(args["video_path"])
        gt = json.load(open(f"{clip}/gt.json")); sim, gen = frames(f"{clip}/input.mp4"), frames(f"{out}/{f}")
        if not gen: continue
        H, W = sim[0].shape[:2]; gen = [cv2.resize(g, (W, H)) if g.shape[:2] != (H, W) else g for g in gen]; n = min(len(sim), len(gen), len(gt))
        os.makedirs(dst, exist_ok=True); tmp = f"/content/post_tmp/{batch}_{name}"; os.makedirs(tmp, exist_ok=True)
        rows, shifts, nlab = [], [], 0
        with zipfile.ZipFile(f"{tmp}/frames.zip", "w", zipfile.ZIP_STORED) as z:
            for i in range(n):
                g = gt[i]; q = np.array(g["corners"]); lab = g.get("labelled", g["visible"]); r = [i, g["frame"], g["along_m"], round(float(np.ptp(q[:, 0])), 1), lab]
                if lab:
                    ga, gb = (cv2.cvtColor(x[i], cv2.COLOR_BGR2GRAY).astype(np.float32) for x in (sim, gen))
                    for p0, p1 in ((q[0], q[1]), (q[3], q[2])):                              # NL-FL and NR-FR long edges
                        ok = np.linalg.norm(p1 - p0) >= 20 and all(0 <= p[0] < W and 0 <= p[1] < H for p in (p0 + 0.15 * (p1 - p0), p0 + 0.85 * (p1 - p0)))
                        a, c = edge_shift(ga, gb, p0, p1) if ok else (None, None); r += [a, None if c is None else round(c, 3)]
                        if a is not None: shifts.append(abs(a))
                    poly = clip_poly(list(q), W, H)
                    if poly:
                        nlab += 1; ok, b = cv2.imencode(".jpg", gen[i], [cv2.IMWRITE_JPEG_QUALITY, 92])
                        z.writestr(f"images/{name}_{i:04d}.jpg", b.tobytes())
                        z.writestr(f"labels/{name}_{i:04d}.txt", "0 " + " ".join(f"{x / W:.5f} {y / H:.5f}" for x, y in poly))
                else: r += [None, None, None, None]
                rows.append(r)
        with open(f"{tmp}/align.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["i", "sim_frame", "along_m", "runway_width_px", "labelled", "shift_left_px", "corr_left", "shift_right_px", "corr_right"]); w.writerows(rows)
        sheet = []
        for i in np.linspace(0, n - 1, 6).astype(int):
            g = gen[i].copy(); q = np.array(gt[i]["corners"])
            if gt[i].get("labelled", gt[i]["visible"]): cv2.polylines(g, [np.round(q).astype(np.int32)], True, (0, 255, 0), 1)
            t = np.hstack([cv2.resize(sim[i], (640, 360)), cv2.resize(g, (640, 360))])
            cv2.putText(t, f"{name}  #{i}  {gt[i]['along_m']:.0f} m" if gt[i]["along_m"] is not None else name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2); sheet.append(t)
        cv2.imwrite(f"{tmp}/sheet.jpg", np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 85])
        s = np.array(shifts) if shifts else np.array([np.nan])
        summary = dict(name=name, batch=batch, clip=os.path.basename(clip), frames_sim=len(sim), frames_gen=len(gen), frames_used=n, frames_labelled=nlab,
                       shift_median_px=float(np.nanmedian(s)), shift_p90_px=float(np.nanpercentile(s, 90)), edges_off_gt2px=float(np.nanmean(s > 2)),
                       prompt=args.get("prompt"), style_image=args.get("image_context_path"), seed=args.get("seed"))
        for k in ("frames.zip", "align.csv", "sheet.jpg"): shutil.copy(f"{tmp}/{k}", f"{dst}/{k}")
        shutil.copy(f"{out}/{f}", f"{dst}/generated.mp4"); json.dump(args, open(f"{dst}/args.json", "w"), indent=1)
        for k in ("input.mp4", "edge.mp4", "edge_light.mp4", "mask.mp4", "gt.json"):
            if os.path.exists(f"{clip}/{k}"): shutil.copy(f"{clip}/{k}", f"{dst}/{k}")
        if args.get("image_context_path") and os.path.exists(args["image_context_path"]): shutil.copy(args["image_context_path"], f"{dst}/style.jpg")
        json.dump(summary, open(f"{dst}/summary.json", "w"), indent=1); shutil.rmtree(tmp)
        print(batch, name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in summary.items() if k not in ("prompt", "style_image")}, flush=True)
