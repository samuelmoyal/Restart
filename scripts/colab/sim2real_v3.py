"""Sim -> real, v2: control edges only around the runway, free surroundings, weather via the prompt."""
import sys, os, json, numpy as np, cv2, torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler
DS = "/content/img2img-turbo/data/xp2real"; OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
LORA = "/content/drive/MyDrive/capstone_sim2real/lora_real_pl/step3000"
BASE = "rwyreal photo of an airport runway seen from an aircraft cockpit on final approach"
NEG = "video game, flight simulator, cgi, 3d render, cartoon, blurry, text, watermark"
VARIANTS = [  # name, control mode, strength, control end, prompt suffix
    ("edges 50% s0.75", "full", 0.75, 0.5, ""),
    ("edges 50% s0.85", "full", 0.85, 0.5, ""),
    ("overcast", "full", 0.80, 0.5, ", dark overcast sky, flat grey light"),
    ("rain", "full", 0.80, 0.5, ", heavy rain, wet asphalt reflections, dark overcast sky"),
    ("fog", "full", 0.80, 0.5, ", dense morning fog, low visibility, grey sky"),
    ("sunset", "full", 0.80, 0.5, ", golden hour sunset, warm low sun, long shadows"),
]
labels = {l["file"]: l for l in json.load(open(f"{DS}/sim_labels.json"))["test"]}
files = [sorted(labels)[i] for i in (0, 60, 120, 200)]
cn = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn, torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config); pipe.set_progress_bar_config(disable=True); pipe.load_lora_weights(LORA)
W, H = 896, 512; rows = []
for f in files:
    sim = cv2.imread(f"{DS}/test_A/{f}")[16:272, 32:480]; q = (np.array(labels[f]["corners"]) - [32, 16]) * 2
    big = cv2.resize(sim, (W, H), interpolation=cv2.INTER_CUBIC); edges = cv2.Canny(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY), 60, 160)
    region = np.zeros((H, W), np.uint8); cv2.fillPoly(region, [np.round(q).astype(np.int32)], 255); region = cv2.dilate(region, np.ones((41, 41), np.uint8))
    horizon = np.zeros((H, W), np.uint8)   # keep the horizon line too, so the sky/ground split stays put
    gy = np.abs(np.diff(cv2.GaussianBlur(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY), (0, 0), 6).astype(np.float32).mean(1))); hy = int(np.argmax(gy[: int(q[:, 1].min())])) if q[:, 1].min() > 20 else 0
    tile = [cv2.putText(cv2.resize(big, (448, 256)), "simule", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)]
    for name, mode, st, cend, suffix in VARIANTS:
        ctrl = edges.copy() if mode == "full" else cv2.bitwise_and(edges, region)
        if mode == "runway": cv2.line(ctrl, (0, hy), (W, hy), 255, 1)
        cv2.polylines(ctrl, [np.round(q).astype(np.int32)], True, 255, 1)
        res = pipe(prompt=BASE + suffix, negative_prompt=NEG, image=Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB)), control_image=Image.fromarray(np.stack([ctrl] * 3, -1)),
                   strength=st, controlnet_conditioning_scale=1.0, control_guidance_end=cend, guidance_scale=6.0, num_inference_steps=24, generator=torch.Generator("cpu").manual_seed(0)).images[0]
        gen = cv2.resize(cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2BGR), (448, 256), interpolation=cv2.INTER_AREA)
        cv2.polylines(gen, [np.round(q / 2).astype(np.int32)], True, (0, 255, 0), 1) if name == "edges 50% s0.75" else None
        tile.append(cv2.putText(gen, name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2))
    rows.append(np.hstack(tile))
cv2.imwrite(f"{OUT}/grid_v3.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); print("saved", f"{OUT}/grid_v3.jpg")
