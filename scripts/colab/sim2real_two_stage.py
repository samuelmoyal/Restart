"""Two-stage generation: (1) restyle the sim frame with img2img so the runway keeps its exact geometry, (2) repaint everything
outside the runway from noise (inpainting, same prompt, sim edges as control), (3) optional: edges around the runway removed
from the control and a light img2img pass over the whole image. Compared with the one-pass text-to-image mode.

    python sim2real_two_stage.py <out_dir> <lora_dir>
"""
import sys, os, json, numpy as np, cv2, torch
from PIL import Image
from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline,
                       StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler)
DS = "/content/img2img-turbo/data/xp2real"; OUT, LORA = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
BASE = "rwyreal photo, an aerial view of an airport runway"
NEG = "video game, flight simulator, cgi, 3d render, cartoon, blurry, text, watermark, window frame, cockpit pillar, windshield wiper, instrument panel"
VARIANTS = [("water", " and a body of water, in daylight, with scattered clouds"),
            ("city", " with a city in the background, in daylight, under a clear blue sky"),
            ("desert", " in the middle of the desert, in daylight, under a clear blue sky"),
            ("snow", " in winter with snow on the ground, in daylight, under an overcast grey sky, snow covered ground"),
            ("night", " at night with the lights on, at night with runway lights")]
W, H, STEPS, S1, S3 = 672, 384, 20, 0.55, 0.35                          # img2img strength of stage 1 (runway) and stage 3 (harmonisation)
labels = {l["file"]: l for l in json.load(open(f"{DS}/sim_labels.json"))["test"]}; files = [sorted(labels)[i] for i in (0, 60, 120, 200)]
cn = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
t2i = StableDiffusionControlNetPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn, torch_dtype=torch.float16, safety_checker=None).to("cuda")
t2i.scheduler = UniPCMultistepScheduler.from_config(t2i.scheduler.config)
if LORA != "none": t2i.load_lora_weights(LORA)
i2i = StableDiffusionControlNetImg2ImgPipeline(**t2i.components); inp = StableDiffusionControlNetInpaintPipeline(**t2i.components)
for p in (t2i, i2i, inp): p.set_progress_bar_config(disable=True)
tag = lambda im, s: cv2.putText(im, s, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
rows = []
for k, f in enumerate(files):
    sim = cv2.imread(f"{DS}/test_A/{f}")[16:272, 32:480]; q = (np.array(labels[f]["corners"], float) - [32, 16]) * (W / 448)
    big = cv2.resize(sim, (W, H), interpolation=cv2.INTER_CUBIC); ctrl = cv2.Canny(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY), 60, 160)
    qi = np.round(q).astype(np.int32); cv2.polylines(ctrl, [qi], True, 255, 1); C = Image.fromarray(np.stack([ctrl] * 3, -1))
    keep = np.zeros((H, W), np.uint8); cv2.fillPoly(keep, [qi], 255); keep = cv2.dilate(keep, np.ones((5, 5), np.uint8))   # runway (+2 px) is kept
    mask = Image.fromarray(255 - keep); src = Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
    ring = cv2.dilate(keep, np.ones((45, 45), np.uint8)) > 0; inside = np.zeros((H, W), np.uint8); cv2.fillPoly(inside, [qi], 255)
    ctrl2 = ctrl.copy(); ctrl2[ring & (inside == 0)] = 0; cv2.polylines(ctrl2, [qi], True, 255, 1); C2 = Image.fromarray(np.stack([ctrl2] * 3, -1))   # no competing outline around the runway
    top, bot, har = [tag(big.copy(), "simule")], [tag(cv2.cvtColor(ctrl, cv2.COLOR_GRAY2BGR), "contours")], [tag(cv2.cvtColor(ctrl2, cv2.COLOR_GRAY2BGR), "contours nettoyes")]
    for j, (name, suffix) in enumerate(VARIANTS):
        kw = dict(prompt=BASE + suffix, negative_prompt=NEG, guidance_scale=6.5, num_inference_steps=STEPS, controlnet_conditioning_scale=1.0)
        g = lambda: torch.Generator("cpu").manual_seed(100 * k + j)
        one = t2i(image=C, width=W, height=H, generator=g(), **kw).images[0]
        p1 = i2i(image=src, control_image=C, strength=S1, generator=g(), **kw).images[0]                                # stage 1: runway restyled in place
        two = inp(image=p1, mask_image=mask, control_image=C, strength=1.0, width=W, height=H, generator=g(), **kw).images[0]   # stage 2: background from noise
        two2 = inp(image=p1, mask_image=mask, control_image=C2, strength=1.0, width=W, height=H, generator=g(), **kw).images[0]
        three = i2i(image=two2, control_image=C2, strength=S3, generator=g(), **kw).images[0]                           # stage 3: light pass over the whole image (seam, lighting)
        a, b, c = (cv2.cvtColor(np.asarray(x), cv2.COLOR_RGB2BGR) for x in (one, two, three)); cv2.imwrite(f"{OUT}/{k}_{name}_three.jpg", c)
        for im in (a, b): cv2.polylines(im, [qi], True, (0, 255, 0), 1)
        top.append(tag(a, name + " - 1 passe")); bot.append(tag(b, name + " - 2 etapes")); har.append(tag(c, name + " - nettoye + harmonise"))
    rows += [np.hstack(top), np.hstack(bot), np.hstack(har)]
cv2.imwrite(f"{OUT}/grid_two_stage.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); print("saved", f"{OUT}/grid_two_stage.jpg")
