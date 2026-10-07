"""Sim -> real with SD 1.5 + ControlNet (canny + GT runway quad) + real-footage LoRA. Measures runway edge shift."""
import sys, os, json, glob, numpy as np, cv2, torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler

lora, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
strength, scale_up = float(sys.argv[4]), int(sys.argv[5])          # img2img strength, inference upscaling factor
cscale = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
DS = "/content/img2img-turbo/data/xp2real"
PROMPT = "rwyreal photo of an airport runway seen from an aircraft cockpit on final approach"
NEG = "video game, flight simulator, cgi, 3d render, cartoon, blurry, text, watermark"
os.makedirs(out, exist_ok=True)
labels = {l["file"]: l for l in json.load(open(f"{DS}/sim_labels.json"))["test"]}
files = sorted(labels)[:: max(1, len(labels) // n)][:n]
cn = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn, torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config); pipe.set_progress_bar_config(disable=True)
if lora != "none": pipe.load_lora_weights(lora)

def edge_shift(a, b, p0, p1, half=6):
    """Lateral shift (px) of the image edge along segment p0-p1 between gray images a and b (cross-correlation of normal profiles)."""
    d = (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-9); nrm = np.array([-d[1], d[0]]); offs = np.arange(-half * 2, half * 2 + 1)
    prof = []
    for im in (a, b):
        rows = []
        for t in np.linspace(0.15, 0.85, 25):
            pts = (p0 + t * (p1 - p0))[None] + offs[:, None] * nrm[None]
            rows.append(cv2.remap(im, pts[:, 0].astype(np.float32)[None], pts[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0])
        prof.append(np.gradient(np.mean(rows, axis=0)))
    pa, pb = prof; best, bs = 0, -1e9
    for s in range(-half, half + 1):
        seg_a = pa[half: -half]; seg_b = pb[half + s: len(pb) - half + s]
        c = float(np.dot(seg_a - seg_a.mean(), seg_b - seg_b.mean()) / (np.linalg.norm(seg_a - seg_a.mean()) * np.linalg.norm(seg_b - seg_b.mean()) + 1e-9))
        if c > bs: bs, best = c, s
    return best, bs

W, H = 448 * scale_up, 256 * scale_up; shifts, tiles = [], []
g = torch.Generator("cpu").manual_seed(0)
for f in files:
    sim = cv2.imread(f"{DS}/test_A/{f}")[16:272, 32:480]                       # centre crop 448x256
    q = (np.array(labels[f]["corners"]) - [32, 16]) * scale_up                  # NL, FL, FR, NR
    big = cv2.resize(sim, (W, H), interpolation=cv2.INTER_CUBIC)
    ctrl = cv2.Canny(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY), 60, 160)
    cv2.polylines(ctrl, [np.round(q).astype(np.int32)], True, 255, 1)
    res = pipe(prompt=PROMPT, negative_prompt=NEG, image=Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB)), control_image=Image.fromarray(np.stack([ctrl] * 3, -1)),
               strength=strength, controlnet_conditioning_scale=cscale, guidance_scale=5.0, num_inference_steps=24, generator=g).images[0]
    gen = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2BGR)
    ga, gb = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for i, j in ((0, 1), (3, 2)):                                               # left and right runway edges
        s, c = edge_shift(ga, gb, q[i], q[j]); shifts.append((s / scale_up, c))
    cv2.imwrite(f"{out}/{f[:-4]}_gen.jpg", gen, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if len(tiles) < 8:
        ov = gen.copy(); cv2.polylines(ov, [np.round(q).astype(np.int32)], True, (0, 255, 0), 1)
        tiles.append(np.hstack([cv2.resize(x, (448, 256), interpolation=cv2.INTER_AREA) for x in (big, gen, ov)]))
cv2.imwrite(f"{out}/sheet.jpg", np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
s = np.array(shifts); ok = s[:, 1] > 0.3
print(f"{out}: {len(files)} images | runway edge shift (px at 448x256): median |shift| {np.median(np.abs(s[ok, 0])):.2f}, p90 {np.percentile(np.abs(s[ok, 0]), 90):.2f}, "
      f"edges matched {ok.mean():.0%} (corr>0.3) | edges lost {1 - ok.mean():.0%}")
