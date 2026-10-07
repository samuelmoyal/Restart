# Colab scripts — sim-to-real restyling (run on a GPU runtime)

Written and run in a Colab session (L4), copied here for the record. Paths point to
`/content` and to the Drive folder `capstone_sim2real/`.

| script | role |
|---|---|
| `prep_sim.py` | read an `xp12_flights` archive, keep the last 600 m of each flight, write runway-matched 512×288 crops + corner labels |
| `build_real_all.py` | real set = playlist + HPA 1440p + LARD real; CLIP filter for cockpit structure; automatic captions (time / sky / environment) |
| `train_lora_real.py` | LoRA (rank 16, UNet attention) on SD 1.5 with one fixed caption — playlist only |
| `train_lora_real_v2.py` | same with per-image captions (all real sources) |
| `val_lora.py` | denoising error per checkpoint on held-out vs training videos (convergence / memorisation check) |
| `sim2real_controlnet.py` | ControlNet (canny + GT runway quad) img2img + LoRA; measures runway edge shift |
| `sim2real_v2.py` / `_v3.py` / `_v4.py` | control restricted to the runway / control on the first 50 % of steps / weather prompts matching the training captions |
| `cg_init.py` | CycleGAN-Turbo generator at initialisation (no training) |
| `xp2real_repo.patch` | patch for `GaParmar/img2img-turbo`: 16:9 448×256 transforms, split cycle backward (halves peak memory), optional bf16 autocast |

Measured on an L4: CycleGAN-Turbo 4.5–5 s/iteration at 448×256 (peak 17 GB after the patch);
LoRA training 0.63 s/iteration (batch 8); generation ~1.5 s/image at 896×512.
