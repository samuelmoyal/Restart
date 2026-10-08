# Colab scripts — sim-to-real restyling (run on a GPU runtime)

Written and run in a Colab session (L4), copied here for the record. Paths point to
`/content` and to the Drive folder `capstone_sim2real/`.

| script | role |
|---|---|
| `prep_sim.py` | read an `xp12_flights` archive, keep the last 600 m of each flight, write runway-matched 512×288 crops + corner labels |
| `build_real_all.py` | real set = playlist + HPA 1440p + LARD real; CLIP filter for cockpit structure; automatic captions (time / sky / environment) |
| `train_lora_real.py` | LoRA (rank 16, UNet attention) on SD 1.5 with one fixed caption — playlist only |
| `train_lora_real_v2.py` | same with per-image captions (all real sources) |
| `build_real_v3.py` | real set v3: manual HPA / playlist extractions kept unfiltered, 30 % of the LARD real airports held out for testing (`lard_real_split.json`), BLIP caption + CLIP tags per image |
| `clean_captions_v3.py` | caption clean-up: drop objects invented by BLIP (planes, cars), take day / night from the caption, keep the rain tag only when the caption mentions it |
| `val_lora.py` | denoising error per checkpoint on held-out vs training videos (convergence / memorisation check) |
| `sim2real_controlnet.py` | ControlNet (canny + GT runway quad) img2img + LoRA; measures runway edge shift |
| `sim2real_v2.py` / `_v3.py` / `_v4.py` | control restricted to the runway / control on the first 50 % of steps / weather prompts matching the training captions |
| `sim2real_v4_nolora.py` | same prompts as v4 with the LoRA optional (pass `none`) |
| `sim2real_t2i.py` | no source image: text-to-image from noise, control = runway quad + its markings + horizon line; environment / weather from the prompt |
| `sim2real_t2i_full.py` | no source image, control = all Canny edges of the sim frame + runway quad (best mode so far: plausible scene, free colours / textures / weather) |
| `sim2real_t2i_v3.py` | same mode with prompts in the v3 caption style (water, city, mountains, desert, snow, fog, night) |
| `sim2real_two_stage.py` | runway restyled in place by img2img, then background repainted from noise by inpainting (runway geometry kept by construction); optional cleaned control + light img2img pass, which over-sharpens at strength 0.35 |
| `sim2real_diffdiff.py` | one pass with a per-pixel strength map (Differential Diffusion): runway tied to the noised sim latents for the first steps, background free; on 20 images, runway strength 0.5 halves the share of edges off by more than 2 px (5 % vs 10 %) |
| `prep_sim_v2.py` / `split_sim_v2.py` | sim crops up to 3 km from the threshold, runway width drawn from the real LARD distribution (median 12 % of the image after mixing, 3–36 % for the 5–95 % range), same airport split as before |
| `sim2real_dd_small.py` | strength maps compared on small runways: runway 0.3 + context ring + background 0.9 removes most invented runways, images slightly hazier |
| `gen_dataset_dd.py` | generated YOLO-seg set with the per-pixel strength map (runway 0.5, or 0.3 when narrower than 10 % of the image; background 0.9) |
| `gen_dataset_t2i.py` | generated YOLO-seg training set: one realistic twin per sim crop, prompt drawn from the real captions, same polygon label, per-image runway edge shift logged |
| `yolo_exp.py` | YOLOv8n-seg trained on sim only / generated only / sim + generated (equal gradient steps), tested on sim, generated and held-out real LARD crops |
| `cg_init.py` | CycleGAN-Turbo generator at initialisation (no training) |
| `xp2real_repo.patch` | patch for `GaParmar/img2img-turbo`: 16:9 448×256 transforms, split cycle backward (halves peak memory), optional bf16 autocast |

Measured on an L4: CycleGAN-Turbo 4.5–5 s/iteration at 448×256 (peak 17 GB after the patch);
LoRA training 0.63 s/iteration (batch 8); generation 4.3 s/image at 896×512 with 24 steps (batch 4), 1.9 s/image at 672×384 with 20 steps.

## YOLO experiment (LoRA v3, one-pass generation, 2026-10-07)

YOLOv8n-seg, 3,090 sim crops and their generated twins (672×384), equal gradient steps, one seed.
Real test = 474 LARD real crops from 32 airports held out of the LoRA training (`results/lard_real_split.json`).
Full numbers in `results/yolo_exp_v3.json`; per-image analysis with `yolo_real_analysis.py`.

| trained on | real mask mAP50 | real mask mAP | detected (conf ≥ 0.25) | IoU > 0.5 | small runways IoU > 0.5 | large runways IoU > 0.5 |
|---|---|---|---|---|---|---|
| sim | 0.605 | 0.433 | 81.4 % | 64.8 % | 53.2 % | 76.4 % |
| generated | 0.721 | 0.533 | 73.8 % | 64.8 % | 46.0 % | 83.5 % |
| sim + generated | 0.651 | 0.490 | 73.4 % | 62.0 % | 43.0 % | 81.0 % |

No clear gain on the real set: the generated-only model is more selective (higher mAP, better on large runways)
but worse on small runways (below 2 % of the image); bootstrap intervals of the mean IoU overlap.
Generated set: median runway edge shift 0 px, 10.6 % of images with an edge off by more than 2 px (448×256 scale).

## YOLO experiment v4 (size-matched crops, per-pixel strength generation, 2026-10-08)

Same protocol and same real test (474 held-out LARD crops); sim crops from `prep_sim_v2.py` / `split_sim_v2.py`,
generated twins from `gen_dataset_dd.py`. Numbers in `results/yolo_exp_v4.json`.

| trained on | real mask mAP50 | real mask mAP | detected (conf ≥ 0.25) | IoU > 0.5 |
|---|---|---|---|---|
| sim | 0.721 | 0.502 | 88.2 % | 76.2 % |
| generated | 0.756 | 0.540 | 81.9 % | 73.0 % |
| sim + generated | 0.842 | 0.641 | 85.7 % | 77.8 % |

Share of real runways with IoU > 0.5 by runway width (share of the image width), previous experiment → this one:

| width | n | sim | generated | sim + generated |
|---|---|---|---|---|
| below 6 % | 74 | 0.16 → 0.74 | 0.09 → 0.72 | 0.08 → 0.73 |
| 6–10 % | 82 | 0.66 → 0.78 | 0.43 → 0.63 | 0.43 → 0.65 |
| 10–20 % | 175 | 0.75 → 0.76 | 0.81 → 0.70 | 0.74 → 0.78 |
| above 20 % | 143 | 0.77 → 0.76 | 0.87 → 0.83 | 0.86 → 0.87 |

The gain comes from the runway size distribution of the sim crops (sim only: +11.4 points, paired bootstrap 95 % CI [7.8, 15.0]).
Adding generated images raises the mask mAP (0.50 → 0.64) and the large-runway score, but the share of well-segmented
runways moves by +1.7 points only (CI [-2.1, 5.5]); one seed per model.

Note: `src_w` in `lard_real_crops_meta.json` is the width of the source crop before resizing to 512 px, not the image width.
Widths must be divided by 512. The target quantiles in `prep_sim_v2.py` were computed with `src_w` and are therefore too
small (median 8 % instead of 13 %); the 30 % uniform component brings the final mix (median 12 %) close to the true real
distribution (median 13 %, 5–95 % range 3.5–32 %).
