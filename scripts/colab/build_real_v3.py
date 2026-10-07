"""Real set v3: playlist + HPA unfiltered (manually cropped), LARD filtered with 30 % of airports held out; BLIP captions + CLIP time/sky tags."""
import os, glob, json, zipfile, random, collections, numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, BlipProcessor, BlipForConditionalGeneration
random.seed(0); dev = "cuda"; D = "/content/drive/MyDrive/capstone_sim2real"; R = "/content/real_all"
os.makedirs(R, exist_ok=True)
if not os.path.exists(f"{R}/hpa"):
    with zipfile.ZipFile(f"{D}/real_hpa1440.zip") as z: z.extractall(f"{R}/hpa")
if not os.path.exists(f"{R}/lard"):
    with zipfile.ZipFile(f"{D}/lard_real_crops.zip") as z: z.extractall(R)
if not os.path.exists(f"{R}/pl/train_B"):
    with zipfile.ZipFile(f"{D}/real_pl_cyclegan.zip") as z: z.extractall(f"{R}/pl")
lard_meta = json.load(open(f"{D}/lard_real_crops_meta.json"))
airports = sorted({m["airport"] for m in lard_meta}); random.Random(1).shuffle(airports); held = set(airports[: int(0.3 * len(airports))])
json.dump({"held_out_airports": sorted(held), "test_files": [m["file"] for m in lard_meta if m["airport"] in held]}, open(f"{D}/lard_real_split.json", "w"))
prev = {os.path.basename(r["file"]): r["clean"] for r in json.load(open(f"{R}/all_scored.json"))} if os.path.exists(f"{R}/all_scored.json") else {}
items = [("playlist", f) for f in sorted(glob.glob(f"{R}/pl/train_B/*.jpg"))]
for v in sorted(glob.glob(f"{R}/hpa/*/")):
    fs = sorted(glob.glob(v + "*.jpg")); items += [("hpa", f) for f in random.sample(fs, min(250, len(fs)))]
lard_train = [m for m in lard_meta if m["airport"] not in held]
items += [("lard", f"{R}/lard/{m['file']}") for m in lard_train if prev.get(m["file"], 1.0) >= 0.5]       # LARD keeps the cockpit-structure filter
print(collections.Counter(s for s, _ in items), "| LARD held-out airports:", len(held), "of", len(airports), "→", sum(m["airport"] in held for m in lard_meta), "test images", flush=True)
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval().half(); cproc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
G = {"time": ["in daylight", "at dusk with warm sunset light", "at night with runway lights"],
     "sky": ["under a clear blue sky", "with scattered clouds", "under an overcast grey sky", "in haze or fog with low visibility"],
     "wx": ["dry clear view", "rain drops on the windshield", "snow covered ground"]}
def tf(txt):
    with torch.no_grad():
        e = clip.get_text_features(**cproc(text=txt, return_tensors="pt", padding=True).to(dev)); e = e.pooler_output if hasattr(e, "pooler_output") else e
    return e / e.norm(dim=-1, keepdim=True)
T = {k: tf(["an airport runway seen from an aircraft " + p for p in v]) for k, v in G.items()}
bproc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large"); blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to(dev).eval()
PREFIX = "an aerial view of an airport runway"
out = []
for i in range(0, len(items), 48):
    batch = items[i:i + 48]; ims = [Image.open(f).convert("RGB") for _, f in batch]
    with torch.no_grad():
        e = clip.get_image_features(**cproc(images=ims, return_tensors="pt").to(dev, torch.float16)); e = e.pooler_output if hasattr(e, "pooler_output") else e
        e = e / e.norm(dim=-1, keepdim=True); P = {k: (100 * e @ t.T).float().softmax(-1).cpu().numpy() for k, t in T.items()}
        ids = blip.generate(**bproc(images=ims, text=[PREFIX] * len(ims), return_tensors="pt").to(dev, torch.float16), max_new_tokens=40, num_beams=3, repetition_penalty=1.3)
    caps = bproc.batch_decode(ids, skip_special_tokens=True)
    for j, (srcn, f) in enumerate(batch):
        t = G["time"][P["time"][j].argmax()]; s = "" if "night" in t else G["sky"][P["sky"][j].argmax()]
        wx = G["wx"][P["wx"][j].argmax()] if P["wx"][j].max() > 0.6 and P["wx"][j].argmax() > 0 else ""
        desc = caps[j].strip()
        cap = "rwyreal photo, " + desc + ", " + ", ".join(x for x in (t, s, wx) if x)
        out.append({"file": f, "source": srcn, "caption": cap, "blip": desc, "time": t, "sky": s, "wx": wx})
    if i % 960 == 0: print(i, "/", len(items), "|", out[-1]["caption"], flush=True)
json.dump(out, open(f"{R}/train_captions_v3.json", "w")); json.dump(out, open(f"{D}/real_all_captions_v3.json", "w"))
print("total", len(out)); 
for k in ("time", "sky", "wx"): print(k, dict(collections.Counter(r[k] for r in out).most_common()))
random.seed(3)
for r in random.sample(out, 24): print(f"[{r['source']}] {r['caption']}")
words = collections.Counter(w for r in out for w in r["blip"].replace(",", " ").split()); print("frequent words:", [w for w, _ in words.most_common(60)])
