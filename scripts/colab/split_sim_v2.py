"""Pick the crops of prep_sim_v2.py so that the runway width distribution matches the target, split by airport.

    python split_sim_v2.py <prep_dir> <out_dir> <old_sim_labels.json> [n_train] [n_test]

Test airports = those of the previous test split. Writes <out>/{train_A,test_A}/ and <out>/sim_labels.json.
"""
import sys, os, json, shutil, numpy as np, cv2
src, out, old = sys.argv[1:4]; NTR = int(sys.argv[4]) if len(sys.argv) > 4 else 3090; NTE = int(sys.argv[5]) if len(sys.argv) > 5 else 252
rng = np.random.default_rng(0); L = [l for f in os.listdir(src) if f.startswith("labels_") for l in json.load(open(f"{src}/{f}"))]
def visible(l):
    """Share of the runway quad that falls inside the 448x256 centre crop used for generation."""
    a = np.zeros((288 * 3, 512 * 3), np.uint8); cv2.fillPoly(a, [np.round(np.array(l["corners"]) + [512, 288]).astype(np.int32)], 1)
    return a[288 + 16: 288 + 272, 512 + 32: 512 + 480].sum() / max(1, a.sum())


n0 = len(L); L = [l for l in L if visible(l) > 0.7]; print(n0, "crops,", len(L), "with more than 70 % of the runway inside the centre crop", flush=True)
test_ap = {l["airport"] for l in json.load(open(old))["test"]}
REAL_Q = [0.0181, 0.0269, 0.0329, 0.0371, 0.0424, 0.0468, 0.0508, 0.0547, 0.0598, 0.0645, 0.0703, 0.076, 0.082, 0.0898, 0.0965, 0.1035, 0.1137,
          0.1248, 0.1367, 0.1504, 0.1641, 0.1816, 0.2031, 0.2341, 0.2866]
u = rng.uniform(size=200000); target = np.where(u < 0.7, np.interp(rng.uniform(0, 24, 200000), np.arange(25), REAL_Q), rng.uniform(0.15, 0.40, 200000))
edges = np.array([0, 0.03, 0.045, 0.06, 0.08, 0.10, 0.14, 0.20, 0.28, 1.0]); share = np.histogram(target, edges)[0] / len(target)
res = {}
for split, n, sel in (("train", NTR, [l for l in L if l["airport"] not in test_ap]), ("test", NTE, [l for l in L if l["airport"] in test_ap])):
    w = np.array([l["width_share"] for l in sel]); b = np.digitize(w, edges) - 1; keep = []
    for k in range(len(edges) - 1):
        idx = np.flatnonzero(b == k); q = int(round(share[k] * n)); keep += rng.choice(idx, min(q, len(idx)), replace=False).tolist()
        print(split, f"width {edges[k]:.3f}-{edges[k+1]:.3f}: target {q}, available {len(idx)}", flush=True)
    res[split] = [sel[i] for i in sorted(keep)]; os.makedirs(f"{out}/{split}_A", exist_ok=True)
    for l in res[split]: shutil.copy(f"{src}/images/{l['file']}", f"{out}/{split}_A/{l['file']}")
    ws = np.array([l["width_share"] for l in res[split]]); al = np.array([l["along_m"] for l in res[split]])
    print(split, len(res[split]), "crops,", len({l["airport"] for l in res[split]}), "airports | width share pct 5/25/50/75/95:", np.percentile(ws, [5, 25, 50, 75, 95]).round(3),
          "| along_m pct 5/50/95:", np.percentile(al, [5, 50, 95]).round())
json.dump(res, open(f"{out}/sim_labels.json", "w"))
