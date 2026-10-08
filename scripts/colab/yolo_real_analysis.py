"""Per-image analysis of the three YOLO models (yolo_exp.py) on the held-out real LARD crops:
detection rate / IoU at several confidence thresholds, bootstrap CI, split by runway size."""
import os, cv2, numpy as np
from ultralytics import YOLO
img_dir = "/content/yolo_ds/real_lard/images/test"; files = sorted(os.listdir(img_dir)); per = {}; rng = np.random.default_rng(0)
for name in ("sim", "gen", "sim+gen"):
    m = YOLO(f"/content/yolo_exp/{name}/weights/last.pt"); rec = []
    for f in files:
        im = cv2.imread(f"{img_dir}/{f}"); h, w = im.shape[:2]; lab = open(f"{img_dir.replace('/images/', '/labels/')}/{f[:-4]}.txt").read().split()[1:]
        gt = np.zeros((h, w), np.uint8); cv2.fillPoly(gt, [np.round(np.array(lab, float).reshape(-1, 2) * [w, h]).astype(np.int32)], 1)
        r = m.predict(im, conf=0.02, imgsz=640, verbose=False)[0]
        if r.masks is None or len(r.masks.xy) == 0: rec.append((0.0, 0.0, 0, gt.sum() / (h * w))); continue
        i = int(r.boxes.conf.argmax()); pm = np.zeros((h, w), np.uint8)
        if len(r.masks.xy[i]) >= 3: cv2.fillPoly(pm, [np.round(r.masks.xy[i]).astype(np.int32)], 1)
        rec.append((float(r.boxes.conf[i]), float((pm & gt).sum() / max(1, (pm | gt).sum())), int((r.boxes.conf > 0.25).sum()), gt.sum() / (h * w)))
    per[name] = np.array(rec); c, iou, nd, area = per[name].T
    bs = [np.mean((iou * (c >= 0.25))[rng.integers(0, len(iou), len(iou))]) for _ in range(1000)]
    print(f"{name:8s}", " | ".join(f"conf>={t}: det {np.mean(c >= t):.3f} IoU {np.mean(iou * (c >= t)):.3f} ok {np.mean((iou > 0.5) & (c >= t)):.3f}" for t in (0.05, 0.1, 0.25, 0.5)),
          f"| IoU of best mask regardless of conf {iou.mean():.3f} | 95% CI IoU@0.25 [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}] | mean nb detections@0.25 {nd.mean():.2f}")
area = per["sim"][:, 3]; small = area < np.median(area)
for name in per: print(name, "small runways: ok@0.25", round(float(np.mean((per[name][small, 1] > 0.5) & (per[name][small, 0] >= 0.25))), 3), "| large:", round(float(np.mean((per[name][~small, 1] > 0.5) & (per[name][~small, 0] >= 0.25))), 3))
print("median runway area share:", round(float(np.median(area)), 4)); np.save("/content/drive/MyDrive/capstone_sim2real/yolo_exp/per_image_real.npy", per)
