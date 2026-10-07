"""Generated training set for YOLO-seg: one realistic image per sim crop (text-to-image, control = sim edges + GT runway quad).

    python gen_dataset_t2i.py <out_dir> <lora_dir> <captions.json> <train|test> [max_n]

Each prompt is a caption drawn from the LoRA training captions, so the generated scenes follow the real distribution.
Writes <out>/{sim,gen}/{images,labels}/<split>/ (YOLO-seg polygons, same label for the sim crop and its generated twin)
and <out>/gen_<split>.csv with the prompt and the runway edge shift between the sim crop and the generated image.
"""
import sys, os, json, csv, random, numpy as np, cv2, torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

OUT, LORA, CAPS, SPLIT = sys.argv[1:5]; MAXN = int(sys.argv[5]) if len(sys.argv) > 5 else 10 ** 9
DS = "/content/img2img-turbo/data/xp2real"; SRC = f"{DS}/{'train_A' if SPLIT == 'train' else 'test_A'}"
NEG = "video game, flight simulator, cgi, 3d render, cartoon, blurry, text, watermark, window frame, cockpit pillar, windshield wiper, instrument panel"
W, H, BS = 896, 512, 4
random.seed(0); caps = sorted({c["caption"] for c in json.load(open(CAPS))}); pool = [c["caption"] for c in json.load(open(CAPS))]
labels = {l["file"]: l for l in json.load(open(f"{DS}/sim_labels.json"))[SPLIT]}; files = sorted(labels)[:MAXN]
for d in ("sim", "gen"):
    for k in ("images", "labels"): os.makedirs(f"{OUT}/{d}/{k}/{SPLIT}", exist_ok=True)
print(len(files), "sim crops,", len(caps), "distinct captions", flush=True)


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


cn = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn, torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config); pipe.set_progress_bar_config(disable=True)
if LORA != "none": pipe.load_lora_weights(LORA)

log = f"{OUT}/gen_{SPLIT}.csv"; new = not os.path.exists(log); fh = open(log, "a", newline=""); wr = csv.writer(fh)
if new: wr.writerow(["file", "prompt", "shift_left_px", "corr_left", "shift_right_px", "corr_right"])
todo = [f for f in files if not os.path.exists(f"{OUT}/gen/images/{SPLIT}/{f}")]
for i in range(0, len(todo), BS):
    batch, ctrls, bigs, quads, prompts = todo[i: i + BS], [], [], [], []
    for f in batch:
        sim = cv2.imread(f"{SRC}/{f}")[16:272, 32:480]; q = (np.array(labels[f]["corners"], float) - [32, 16]) * 2        # NL, FL, FR, NR at 896x512
        big = cv2.resize(sim, (W, H), interpolation=cv2.INTER_CUBIC); ctrl = cv2.Canny(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY), 60, 160)
        cv2.polylines(ctrl, [np.round(q).astype(np.int32)], True, 255, 1)
        poly = clip_poly(list(q), W, H); txt = "0 " + " ".join(f"{x / W:.5f} {y / H:.5f}" for x, y in poly) if poly else ""
        for d in ("sim", "gen"): open(f"{OUT}/{d}/labels/{SPLIT}/{f[:-4]}.txt", "w").write(txt)
        cv2.imwrite(f"{OUT}/sim/images/{SPLIT}/{f}", sim, [cv2.IMWRITE_JPEG_QUALITY, 95])
        ctrls.append(Image.fromarray(np.stack([ctrl] * 3, -1))); bigs.append(big); quads.append(q); prompts.append(random.choice(pool))
    res = pipe(prompt=prompts, negative_prompt=[NEG] * len(batch), image=ctrls, width=W, height=H, controlnet_conditioning_scale=1.0, guidance_scale=6.5,
               num_inference_steps=24, generator=torch.Generator("cpu").manual_seed(i)).images
    for f, im, big, q, p in zip(batch, res, bigs, quads, prompts):
        gen = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR); cv2.imwrite(f"{OUT}/gen/images/{SPLIT}/{f}", gen, [cv2.IMWRITE_JPEG_QUALITY, 92])
        ga, gb = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY).astype(np.float32)
        (sl, cl), (sr, cr) = edge_shift(ga, gb, q[0], q[1]), edge_shift(ga, gb, q[3], q[2])
        wr.writerow([f, p, sl, round(cl, 3), sr, round(cr, 3)])
    fh.flush()
    if i % 200 == 0: print(f"{i + len(batch)}/{len(todo)}", flush=True)
print("DONE", SPLIT, len(todo), "generated", flush=True)
