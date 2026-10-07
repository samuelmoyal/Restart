"""YOLOv8n-seg trained on sim only / generated only / sim + generated, tested on sim, generated and held-out real (LARD).

    python yolo_exp.py <yolo_ds> <out_dir> [epochs]

<yolo_ds> is the output of gen_dataset_t2i.py. The three runs get the same number of gradient steps
(the mixed set has twice the images, so half the epochs). Real test = LARD real crops of the 30 % airports
never seen by the LoRA. Reports mask mAP, detection rate (conf >= 0.25) and mask IoU of the best detection.
"""
import sys, os, json, numpy as np, cv2
from ultralytics import YOLO

DS, OUT = sys.argv[1], sys.argv[2]; EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
D = "/content/drive/MyDrive/capstone_sim2real"; R = "/content/real_all/lard"; os.makedirs(OUT, exist_ok=True)

# held-out real test set in YOLO-seg format
real = f"{DS}/real_lard"; held = set(json.load(open(f"{D}/lard_real_split.json"))["test_files"])
if not os.path.exists(real):
    os.makedirs(f"{real}/images/test"); os.makedirs(f"{real}/labels/test")
    for m in json.load(open(f"{D}/lard_real_crops_meta.json")):
        if m["file"] not in held or not os.path.exists(f"{R}/{m['file']}"): continue
        im = cv2.imread(f"{R}/{m['file']}"); h, w = im.shape[:2]
        hull = cv2.convexHull(np.array(m["corners"], np.float32)).reshape(-1, 2); hull[:, 0] = hull[:, 0].clip(0, w); hull[:, 1] = hull[:, 1].clip(0, h)
        cv2.imwrite(f"{real}/images/test/{m['file']}", im)
        open(f"{real}/labels/test/{m['file'][:-4]}.txt", "w").write("0 " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in hull))
print("real test images:", len(os.listdir(f"{real}/images/test")), flush=True)


def yaml(name, train, val):
    p = f"{OUT}/{name}.yaml"; tr = "\n".join(f"  - {t}" for t in train)
    open(p, "w").write(f"train:\n{tr}\nval: {val}\nnames:\n  0: runway\n"); return p


SIM, GEN = f"{DS}/sim/images", f"{DS}/gen/images"
TRAIN = {"sim": ([f"{SIM}/train"], EPOCHS), "gen": ([f"{GEN}/train"], EPOCHS), "sim+gen": ([f"{SIM}/train", f"{GEN}/train"], EPOCHS // 2)}
TEST = {"sim": f"{SIM}/test", "gen": f"{GEN}/test", "real": f"{real}/images/test"}


def best_iou(model, img_dir):
    """Detection rate and mask IoU of the most confident detection against the GT polygon."""
    ious, det = [], 0; files = sorted(os.listdir(img_dir))
    for f in files:
        im = cv2.imread(f"{img_dir}/{f}"); h, w = im.shape[:2]
        lab = open(f"{img_dir.replace('/images/', '/labels/')}/{f[:-4]}.txt").read().split()[1:]
        gt = np.zeros((h, w), np.uint8); cv2.fillPoly(gt, [np.round(np.array(lab, float).reshape(-1, 2) * [w, h]).astype(np.int32)], 1)
        r = model.predict(im, conf=0.25, imgsz=640, verbose=False)[0]
        if r.masks is None or len(r.masks.xy) == 0 or len(r.masks.xy[0]) < 3: ious.append(0.0); continue
        det += 1; pm = np.zeros((h, w), np.uint8); cv2.fillPoly(pm, [np.round(r.masks.xy[int(r.boxes.conf.argmax())]).astype(np.int32)], 1)
        ious.append(float((pm & gt).sum() / max(1, (pm | gt).sum())))
    return det / len(files), float(np.mean(ious)), float(np.mean(np.array(ious) > 0.5))


res = json.load(open(f"{OUT}/results.json")) if os.path.exists(f"{OUT}/results.json") else {}
for name, (train, ep) in TRAIN.items():
    w = f"{OUT}/{name}/weights/last.pt"
    if not os.path.exists(w):
        YOLO("yolov8n-seg.pt").train(data=yaml(name, train, TEST["sim"]), epochs=ep, imgsz=640, batch=32, project=OUT, name=name, exist_ok=True, seed=0, verbose=False, plots=False, workers=4)
    model = YOLO(w)
    for tname, tdir in TEST.items():
        if f"{name}|{tname}" in res: continue
        v = model.val(data=yaml(f"val_{tname}", [tdir], tdir), imgsz=640, batch=32, verbose=False, plots=False, project=OUT, name=f"val_{name}_{tname}", exist_ok=True)
        dr, iou, ok = best_iou(model, tdir)
        res[f"{name}|{tname}"] = dict(mask_map50=float(v.seg.map50), mask_map=float(v.seg.map), det_rate=dr, mean_iou=iou, iou_gt_50=ok)
        print(name, "->", tname, {k: round(x, 3) for k, x in res[f"{name}|{tname}"].items()}, flush=True); json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
print("DONE"); print(json.dumps(res, indent=1))
