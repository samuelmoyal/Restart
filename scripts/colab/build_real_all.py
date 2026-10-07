"""Combined real set (playlist + HPA 1440p + LARD real): CLIP filter for cockpit structure + automatic captions."""
import os, glob, json, zipfile, random, collections, numpy as np, cv2, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
random.seed(0); dev = "cuda"; D = "/content/drive/MyDrive/capstone_sim2real"; R = "/content/real_all"
if not os.path.exists(f"{R}/hpa"):
    with zipfile.ZipFile(f"{D}/real_hpa1440.zip") as z: z.extractall(f"{R}/hpa")
src = {"playlist": sorted(glob.glob("/content/img2img-turbo/data/xp2real/train_B/*.jpg")), "lard": sorted(glob.glob(f"{R}/lard/*.jpg"))}
hpa = []
for v in sorted(glob.glob(f"{R}/hpa/*/")):
    fs = sorted(glob.glob(v + "*.jpg")); hpa += random.sample(fs, min(140, len(fs)))        # balance the films
src["hpa"] = hpa
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval().half(); proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
G = {
 "bad": ["an unobstructed aerial view of an airport runway and the landscape", "a view through an aircraft windshield with the window frame or cockpit pillar in the picture",
         "a photo with an aircraft instrument panel or glareshield in the picture", "a video frame with a small inset picture or text overlay", "a very blurry or overexposed washed-out image"],
 "time": ["in daylight", "at dusk with warm sunset light", "at night with runway lights"],
 "sky": ["under a clear blue sky", "with scattered clouds", "under an overcast grey sky", "in haze or fog with low visibility"],
 "env": ["surrounded by green fields", "snow covered ground", "by the sea or a lake", "surrounded by a city", "in a dry desert landscape", "surrounded by mountains"],
}
def feats(txt):
    with torch.no_grad():
        e = model.get_text_features(**proc(text=txt, return_tensors="pt", padding=True).to(dev)); e = e.pooler_output if hasattr(e, "pooler_output") else e
    return e / e.norm(dim=-1, keepdim=True)
T = {k: feats([("" if k == "bad" else "an airport runway seen from an aircraft ") + p for p in v]) for k, v in G.items()}
rows = []
for name, files in src.items():
    for i in range(0, len(files), 64):
        batch = files[i:i + 64]
        with torch.no_grad():
            e = model.get_image_features(**proc(images=[Image.open(f).convert("RGB") for f in batch], return_tensors="pt").to(dev, torch.float16)); e = e.pooler_output if hasattr(e, "pooler_output") else e
            e = e / e.norm(dim=-1, keepdim=True); P = {k: (100 * e @ t.T).float().softmax(-1).cpu().numpy() for k, t in T.items()}
        for j, f in enumerate(batch):
            ok = P["bad"][j, 0]; t = G["time"][P["time"][j].argmax()]; s = G["sky"][P["sky"][j].argmax()]
            env = G["env"][P["env"][j].argmax()] if P["env"][j].max() > 0.45 else ""
            if "night" in t: s = ""
            cap = "rwyreal photo of an airport runway seen from an aircraft cockpit on final approach, " + ", ".join(x for x in (t, s, env) if x)
            rows.append({"file": f, "source": name, "clean": float(ok), "caption": cap, "time": t, "sky": s, "env": env})
thr = float(os.environ.get("CLEAN_THR", "0.5")); keep = [r for r in rows if r["clean"] >= thr]
json.dump(rows, open(f"{R}/all_scored.json", "w")); json.dump(keep, open(f"{R}/train_captions.json", "w"))
for name in src:
    a = [r for r in rows if r["source"] == name]; k = [r for r in a if r["clean"] >= thr]; print(f"{name}: {len(k)}/{len(a)} kept ({len(k)/len(a):.0%})")
for key in ("time", "sky", "env"): print(key, dict(collections.Counter(r[key] for r in keep).most_common()))
def sheet(rs, path, n=48):
    rs = random.sample(rs, min(n, len(rs))); t = []
    for r in rs:
        im = cv2.resize(cv2.imread(r["file"]), (320, 180)); cv2.rectangle(im, (0, 0), (320, 16), (0, 0, 0), -1)
        cv2.putText(im, f"{r['clean']:.2f} {r['time'][3:12]}|{r['sky'][-14:]}|{r['env'][:10]}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1); t.append(im)
    while len(t) % 8: t.append(np.zeros_like(t[0]))
    cv2.imwrite(path, np.vstack([np.hstack(t[i:i + 8]) for i in range(0, len(t), 8)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
os.makedirs(f"{D}/gen/real_all", exist_ok=True)
sheet([r for r in keep if r["source"] != "playlist"], f"{D}/gen/real_all/kept.jpg"); sheet([r for r in rows if r["clean"] < thr], f"{D}/gen/real_all/rejected.jpg"); print("sheets saved; total kept", len(keep))
