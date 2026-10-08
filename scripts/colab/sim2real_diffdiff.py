"""One-pass generation with a per-pixel strength map (Differential Diffusion, Levin & Fried 2023) + ControlNet + LoRA.

    python sim2real_diffdiff.py <out_dir> <lora_dir>

The whole image starts from noise. Pixels of the runway (strength s < 1) are tied to the noised sim latents during the first
(1 - s) share of the steps, then denoised freely with the rest of the image; the background (strength 1) is free from the start.
Compared with the plain one-pass text-to-image mode; the runway edge shift against the sim frame is measured for each mode.
"""
import sys, os, json, numpy as np, cv2, torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DDIMScheduler
DS = "/content/img2img-turbo/data/xp2real"; OUT, LORA = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
BASE = "rwyreal photo, an aerial view of an airport runway"
NEG = "video game, flight simulator, cgi, 3d render, cartoon, blurry, text, watermark, window frame, cockpit pillar, windshield wiper, instrument panel"
VARIANTS = [("water", " and a body of water, in daylight, with scattered clouds"),
            ("city", " with a city in the background, in daylight, under a clear blue sky"),
            ("desert", " in the middle of the desert, in daylight, under a clear blue sky"),
            ("snow", " in winter with snow on the ground, in daylight, under an overcast grey sky, snow covered ground"),
            ("night", " at night with the lights on, at night with runway lights")]
W, H, STEPS = 672, 384, 30
MODES = [("1 passe", None), ("force piste 0.5", 0.5), ("force piste 0.7", 0.7), ("force piste 0.85", 0.85)]      # runway strength (background = 1)
labels = {l["file"]: l for l in json.load(open(f"{DS}/sim_labels.json"))["test"]}; files = [sorted(labels)[i] for i in (0, 60, 120, 200)]
cn = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn, torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config); pipe.set_progress_bar_config(disable=True)      # single-step scheduler: latents can be overwritten between steps
if LORA != "none": pipe.load_lora_weights(LORA)
tag = lambda im, s: cv2.putText(im, s, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


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


shifts = {m: [] for m, _ in MODES}; grid = []
for k, f in enumerate(files):
    sim = cv2.imread(f"{DS}/test_A/{f}")[16:272, 32:480]; q = (np.array(labels[f]["corners"], float) - [32, 16]) * (W / 448); qi = np.round(q).astype(np.int32)
    big = cv2.resize(sim, (W, H), interpolation=cv2.INTER_CUBIC); ctrl = cv2.Canny(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY), 60, 160); cv2.polylines(ctrl, [qi], True, 255, 1)
    C = Image.fromarray(np.stack([ctrl] * 3, -1)); ga = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x = torch.from_numpy(cv2.cvtColor(big, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].to("cuda", torch.float16) / 127.5 - 1
    with torch.no_grad(): z0 = pipe.vae.encode(x).latent_dist.mode() * pipe.vae.config.scaling_factor                # sim latents (1, 4, H/8, W/8)
    inside = np.zeros((H, W), np.float32); cv2.fillPoly(inside, [qi], 1.0); inside = cv2.dilate(inside, np.ones((7, 7), np.uint8))
    soft = cv2.resize(cv2.GaussianBlur(inside, (0, 0), 5), (W // 8, H // 8), interpolation=cv2.INTER_AREA)            # 1 on the runway, 0 far from it, soft border
    rows = {m: [tag(big.copy(), "simule") if i == 0 else tag(cv2.cvtColor(ctrl, cv2.COLOR_GRAY2BGR), "contours") if i == 1 else np.zeros((H, W, 3), np.uint8)] for i, (m, _) in enumerate(MODES)}
    for j, (name, suffix) in enumerate(VARIANTS):
        for mode, s_r in MODES:
            g = torch.Generator("cpu").manual_seed(100 * k + j); noise = torch.randn(z0.shape, generator=g).to("cuda", torch.float16); kw = {}
            if s_r is not None:
                smap = torch.from_numpy(1.0 - (1.0 - s_r) * soft)[None, None].to("cuda")                             # per-pixel strength

                def cb(p, i, t, d, smap=smap, noise=noise):
                    done = (i + 1) / STEPS                                                                             # pixels with strength s are released once done >= 1 - s
                    if i + 1 < len(p.scheduler.timesteps):
                        frozen = (smap < 1.0 - done).to(d["latents"].dtype); ref = p.scheduler.add_noise(z0, noise, p.scheduler.timesteps[i + 1: i + 2])
                        d["latents"] = frozen * ref + (1 - frozen) * d["latents"]
                    return d
                kw = dict(callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
            res = pipe(prompt=BASE + suffix, negative_prompt=NEG, image=C, width=W, height=H, latents=noise.clone(), guidance_scale=6.5, num_inference_steps=STEPS,
                       controlnet_conditioning_scale=1.0, **kw).images[0]
            im = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2BGR); gb = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
            shifts[mode] += [edge_shift(ga, gb, q[0], q[1]), edge_shift(ga, gb, q[3], q[2])]
            cv2.imwrite(f"{OUT}/{k}_{name}_{mode.replace(' ', '_')}.jpg", im); cv2.polylines(im, [qi], True, (0, 255, 0), 1); rows[mode].append(tag(im, f"{name} - {mode}"))
    grid += [np.hstack(rows[m]) for m, _ in MODES]
cv2.imwrite(f"{OUT}/grid_diffdiff.jpg", np.vstack(grid), [cv2.IMWRITE_JPEG_QUALITY, 88])
for m, v in shifts.items():
    s = np.abs(np.array([a for a, _ in v])) * 448 / W; c = np.array([b for _, b in v])
    print(f"{m:18s} edge shift px@448x256: median {np.median(s):.2f} p90 {np.percentile(s, 90):.2f} max {s.max():.2f} | edges off by > 2 px: {np.mean(s > 2):.2f} | corr median {np.median(c):.2f}", flush=True)
print("saved", f"{OUT}/grid_diffdiff.jpg")
