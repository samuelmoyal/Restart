"""Compare Cosmos outputs (series2 folders on Drive) on a common distance range: edge shift, ring fidelity, temporal flicker.

    python cosmos_compare.py <series_dir> <max_dist_m> <name> [...]

Flicker = median over frames of the mean absolute difference between consecutive generated frames, outside the runway
(dilated GT quad), divided by the same quantity on the sim input: 1 means the generated video changes as much as the
sim from frame to frame (camera motion), > 1 means extra flicker / texture popping. Frames are restricted to
along_m >= -max_dist_m so clips with different strides are compared on the same part of the approach.
"""
import sys, os, csv, json, numpy as np, cv2

SD, DMAX, NAMES = sys.argv[1], float(sys.argv[2]), sys.argv[3:]


def frames(p):
    cap, out = cv2.VideoCapture(p), []
    while True:
        ok, f = cap.read()
        if not ok: return out
        out.append(cv2.cvtColor(cv2.resize(f, (640, 360)), cv2.COLOR_BGR2GRAY).astype(np.float32))


for n in NAMES:
    d = f"{SD}/{n}"; gt = json.load(open(f"{d}/gt.json")); rows = list(csv.DictReader(open(f"{d}/align.csv")))
    sim, gen = frames(f"{d}/input.mp4"), frames(f"{d}/generated.mp4"); k = min(len(sim), len(gen), len(gt))
    keep = [i for i in range(k) if gt[i]["along_m"] is not None and gt[i]["along_m"] >= -DMAX]
    fl = []
    for i in keep[1:]:
        if i - 1 not in keep: continue
        m = np.ones((360, 640), np.uint8); q = np.round(np.array(gt[i]["corners"]) / 2).astype(np.int32)
        cv2.fillPoly(m, [q], 0); m = cv2.erode(m, np.ones((15, 15), np.uint8)) > 0
        ds, dg = np.abs(sim[i] - sim[i - 1])[m].mean(), np.abs(gen[i] - gen[i - 1])[m].mean()
        if ds > 0.5: fl.append(dg / ds)
    sh = [abs(float(r[c])) for r in rows if int(r["i"]) in keep for c in ("shift_left_px", "shift_right_px") if r[c] not in ("", "None")]
    rp = [float(r["ring_precision"]) for r in rows if int(r["i"]) in keep and r.get("ring_precision") not in (None, "", "None")]
    rr = [float(r["ring_recall"]) for r in rows if int(r["i"]) in keep and r.get("ring_recall") not in (None, "", "None")]
    print(f"{n:40s} frames {len(keep):3d}  shift med {np.median(sh) if sh else np.nan:.1f} p90 {np.percentile(sh, 90) if sh else np.nan:.1f}  "
          f"ring prec {np.median(rp) if rp else np.nan:.2f} rec {np.median(rr) if rr else np.nan:.2f}  flicker {np.median(fl) if fl else np.nan:.2f}")
