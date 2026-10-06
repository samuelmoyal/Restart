#!/usr/bin/env python3
"""
Sim-to-real restyling of XP12 frames with Stable Diffusion 1.5 + ControlNet (Canny).

Geometry is kept from the simulator (so the pose / corner labels stay valid):
the frame's Canny edges, with the GT runway quad drawn in, drive the ControlNet,
and img2img starts from the simulator frame so the layout cannot drift far.

    PYTHONPATH=. python scripts/restyle_xp12_controlnet.py --seq 000018 --frame 400

Writes ``<out>/<seq>_<frame>_{before,after,compare}.jpg``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from evaluation.weather import remove_hud
from runway_detection.xp12 import load_sequence

PROMPT = (
    "photograph taken from an airliner cockpit on final approach, real airport runway ahead, "
    "asphalt with worn white markings, natural daylight, atmospheric haze, real trees and fields, "
    "telephoto lens, sharp focus, photorealistic, 35mm photo"
)
NEGATIVE = (
    "video game, flight simulator, cgi, 3d render, cartoon, illustration, painting, oversaturated, "
    "low poly, text, watermark, hud, blurry, deformed"
)


def control_image(bgr: np.ndarray, quad: np.ndarray, low: int, high: int) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    cv2.polylines(edges, [np.round(quad).astype(np.int32)], True, 255, 1, cv2.LINE_8)
    return np.stack([edges] * 3, axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default="000018")
    ap.add_argument("--frame", type=int, default=400)
    ap.add_argument("--root", type=Path, default=Path("data/xp12/xp12_dataset"))
    ap.add_argument("--zip", type=Path, default=Path("xp12_dataset.zip"))
    ap.add_argument("--out", type=Path, default=Path("data/xp12/restyle"))
    ap.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--controlnet", default="lllyasviel/control_v11p_sd15_canny")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--strength", type=float, default=0.55, help="img2img denoise (0 = keep sim frame)")
    ap.add_argument("--control-scale", type=float, default=1.0)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--canny", type=int, nargs=2, default=(60, 160))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler

    seq = load_sequence(args.root, args.seq, zip_path=args.zip)
    frame = next(f for f in seq.frames if f.index == args.frame)
    bgr = remove_hud(seq.read_image(frame))
    quad = np.array([frame.corners_px[n] for n in ("TR", "TL", "BL", "BR")])

    size = (args.width, args.height)
    scale = np.array([args.width / bgr.shape[1], args.height / bgr.shape[0]])
    small = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    ctrl = control_image(small, quad * scale, *args.canny)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype, variant="fp16")
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.base, controlnet=controlnet, torch_dtype=dtype, variant="fp16", safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.enable_attention_slicing()

    out = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        image=Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)),
        control_image=Image.fromarray(ctrl),
        strength=args.strength,
        controlnet_conditioning_scale=args.control_scale,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    after = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.seq}_{args.frame:06d}{args.tag}"
    cv2.imwrite(str(args.out / f"{stem}_before.jpg"), small)
    cv2.imwrite(str(args.out / f"{stem}_after.jpg"), after)
    cv2.imwrite(str(args.out / f"{stem}_control.png"), ctrl)
    cv2.imwrite(str(args.out / f"{stem}_compare.jpg"), np.hstack([small, after]))
    print(f"→ {args.out / f'{stem}_compare.jpg'}  ({device}, {dtype})")


if __name__ == "__main__":
    main()
