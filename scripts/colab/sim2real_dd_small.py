"""Per-pixel strength on small runways: compare strength maps (runway strength, context ring around it, background strength).

    python sim2real_dd_small.py <out_dir> <lora_dir> <sim_dataset_dir>
"""
import sys, os, json, numpy as np, cv2, torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DDIMScheduler
OUT, LORA, DS = sys.argv[1:4]; os.makedirs(OUT, exist_ok=True)
BASE = "rwyreal photo, an aerial view of an airport runway"
NEG = "video game, flight simulator, cgi, 3d render, cartoon, blurry, text, watermark, window frame, cockpit pillar, windshield wiper, instrument panel"
VARIANTS = [("water", " and a body of water, in daylight, with scattered clouds"), ("desert", " in the middle of the desert, in daylight, under a clear blue sky"),
            ("snow", " in winter with snow on the ground, in daylight, under an overcast grey sky, snow covered ground"), ("night", " at night with the lights on, at night with runway lights")]
W, H, STEPS = 672, 384, 20
MODES = [("piste 0.5", 0.5, 0.0, 1.0), ("piste 0.3 + contexte", 0.3, 0.6, 1.0), ("piste 0.3 + contexte, fond 0.9", 0.3, 0.6, 0.9),
         ("piste 0.3 + contexte large, fond 0.9", 0.3, 1.5, 0.9)]                    # name, runway strength, context radius (x runway width), background strength
L = sorted(json.load(open(f"{DS}/sim_labels.json"))["test"], key=lambda l: l["width_share"]); small = [l for l in L if 0.025 <= l["width_share"] <= 0.08]
picks = small[:: max(1, len(small) // 5)][:5]
cn = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn, torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)
if LORA != "none": pipe.load_lora_weights(LORA)
tag = lambda im, s: cv2.putText(im, s, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
grid = []
for k, l in enumerate(picks):
    sim = cv2.imread(f"{DS}/test_A/{l['file']}")[16:272, 32:480]; q = (np.array(l["corners"], float) - [32, 16]) * (W / 448); qi = np.round(q).astype(np.int32)
    big = cv2.resize(sim, (W, H), interpolation=cv2.INTER_CUBIC); ctrl = cv2.Canny(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY), 60, 160); cv2.polylines(ctrl, [qi], True, 255, 1)
    C = Image.fromarray(np.stack([ctrl] * 3, -1)); bw = float(np.ptp(q[:, 0]))
    x = torch.from_numpy(cv2.cvtColor(big, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].to("cuda", torch.float16) / 127.5 - 1
    with torch.no_grad(): z0 = pipe.vae.encode(x).latent_dist.mode() * pipe.vae.config.scaling_factor
    inside = np.zeros((H, W), np.float32); cv2.fillPoly(inside, [qi], 1.0)
    for mode, s_r, ctx, s_bg in MODES:
        r = int(max(4, ctx * bw)); core = cv2.dilate(inside, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
        hard = (cv2.resize(core, (W // 8, H // 8), interpolation=cv2.INTER_AREA) > 0).astype(np.float32)
        soft = np.maximum(hard, cv2.resize(cv2.GaussianBlur(core, (0, 0), max(5, r / 2)), (W // 8, H // 8), interpolation=cv2.INTER_AREA))
        smap = torch.from_numpy(s_bg - (s_bg - s_r) * soft)[None, None].to("cuda"); row = [tag(big.copy(), f"simule {100 * l['width_share']:.0f}% {l['along_m']:.0f}m")]
        for j, (name, suffix) in enumerate(VARIANTS):
            noise = torch.randn(z0.shape, generator=torch.Generator("cpu").manual_seed(100 * k + j)).to("cuda", torch.float16)

            def cb(p, i, t, d, smap=smap, noise=noise):
                if i + 1 < len(p.scheduler.timesteps):
                    frozen = (smap < 1.0 - (i + 1) / STEPS).to(d["latents"].dtype); ref = p.scheduler.add_noise(z0, noise, p.scheduler.timesteps[i + 1: i + 2])
                    d["latents"] = frozen * ref + (1 - frozen) * d["latents"]
                return d
            res = pipe(prompt=BASE + suffix, negative_prompt=NEG, image=C, width=W, height=H, latents=noise.clone(), guidance_scale=6.5, num_inference_steps=STEPS,
                       controlnet_conditioning_scale=1.0, callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"]).images[0]
            im = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2BGR); x0, y0 = int(max(0, qi[:, 0].min() - 30)), int(max(0, qi[:, 1].min() - 30))
            cv2.rectangle(im, (x0, y0), (int(min(W - 1, qi[:, 0].max() + 30)), int(min(H - 1, qi[:, 1].max() + 30))), (0, 255, 0), 1)     # box around the labelled runway, runway itself left visible
            row.append(tag(im, f"{name} - {mode}"))
        grid.append(np.hstack(row))
cv2.imwrite(f"{OUT}/grid_dd_small.jpg", np.vstack(grid), [cv2.IMWRITE_JPEG_QUALITY, 88]); print("saved", f"{OUT}/grid_dd_small.jpg")
