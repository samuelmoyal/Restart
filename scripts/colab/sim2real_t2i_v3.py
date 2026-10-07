
import sys, os, json, numpy as np, cv2, torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
DS = "/content/img2img-turbo/data/xp2real"; OUT, LORA = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
BASE = "rwyreal photo, an aerial view of an airport runway"  # caption style of build_real_v3.py
NEG = "video game, flight simulator, cgi, 3d render, cartoon, blurry, text, watermark, window frame, cockpit pillar, windshield wiper, instrument panel"
VARIANTS = [("plain, clear", ", in daylight, under a clear blue sky"),
            ("water", " and a body of water, in daylight, with scattered clouds"),
            ("city", " with a city in the background, in daylight, under a clear blue sky"),
            ("mountains", " with mountains in the background, in daylight, under an overcast grey sky"),
            ("desert", " in the middle of the desert, in daylight, under a clear blue sky"),
            ("snow", " in winter with snow on the ground, in daylight, under an overcast grey sky, snow covered ground"),
            ("fog", ", in daylight, in haze or fog with low visibility"),
            ("night", " at night with the lights on, at night with runway lights")]
labels = {l["file"]: l for l in json.load(open(f"{DS}/sim_labels.json"))["test"]}; files = [sorted(labels)[i] for i in (0, 60, 120, 200)]
cn = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn, torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config); pipe.set_progress_bar_config(disable=True)
if LORA != "none": pipe.load_lora_weights(LORA)
W, H = 896, 512; rows = []
for k, f in enumerate(files):
    sim = cv2.imread(f"{DS}/test_A/{f}")[16:272, 32:480]; q = (np.array(labels[f]["corners"]) - [32, 16]) * 2
    big = cv2.resize(sim, (W, H), interpolation=cv2.INTER_CUBIC); gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY); edges = cv2.Canny(gray, 60, 160)
    ctrl = edges.copy(); cv2.polylines(ctrl, [np.round(q).astype(np.int32)], True, 255, 1)
    tile = [cv2.putText(cv2.resize(big, (448, 256)), "simule (non utilise)", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2),
            cv2.putText(cv2.cvtColor(cv2.resize(ctrl, (448, 256), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR), "contours imposes", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)]
    for j, (name, suffix) in enumerate(VARIANTS):
        res = pipe(prompt=BASE + suffix, negative_prompt=NEG, image=Image.fromarray(np.stack([ctrl] * 3, -1)), width=W, height=H, controlnet_conditioning_scale=1.0,
                   guidance_scale=6.5, num_inference_steps=24, generator=torch.Generator("cpu").manual_seed(100 * k + j)).images[0]
        gen = cv2.resize(cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2BGR), (448, 256), interpolation=cv2.INTER_AREA)
        if j == 0: cv2.polylines(gen, [np.round(q / 2).astype(np.int32)], True, (0, 255, 0), 1)
        tile.append(cv2.putText(gen, name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2))
    rows.append(np.hstack(tile))
cv2.imwrite(f"{OUT}/grid_t2i_v3.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); print("saved", f"{OUT}/grid_t2i_v3.jpg")
