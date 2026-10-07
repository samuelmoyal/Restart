
import torch, glob, random, numpy as np, torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from diffusers import StableDiffusionPipeline, DDPMScheduler
BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"; L = "/content/drive/MyDrive/capstone_sim2real/lora_real_pl"; DS = "/content/img2img-turbo/data/xp2real"
PROMPT = "rwyreal photo of an airport runway seen from an aircraft cockpit on final approach"; dev = "cuda"
tf = T.Compose([T.CenterCrop((256, 448)), T.ToTensor(), T.Normalize([0.5], [0.5])])
random.seed(0)
sets = {"held-out videos": sorted(glob.glob(DS + "/test_B/*.jpg"))[::2][:128], "training videos": random.sample(sorted(glob.glob(DS + "/train_B/*.jpg")), 128)}
sched = DDPMScheduler.from_pretrained(BASE, subfolder="scheduler")
for ck in (0, 500, 1000, 1500, 2000, 2500, 3000):
    pipe = StableDiffusionPipeline.from_pretrained(BASE, torch_dtype=torch.float16, safety_checker=None).to(dev)
    if ck: pipe.load_lora_weights(f"{L}/step{ck}")
    emb = pipe.encode_prompt(PROMPT, dev, 1, False)[0]; out = []
    for name, files in sets.items():
        g = torch.Generator(dev).manual_seed(0); tot = []
        for i in range(0, len(files), 16):
            x = torch.stack([tf(Image.open(f).convert("RGB")) for f in files[i:i + 16]]).to(dev, torch.float16)
            with torch.no_grad():
                lat = pipe.vae.encode(x).latent_dist.mode() * pipe.vae.config.scaling_factor
                for t in (100, 300, 500, 700, 900):
                    noise = torch.randn(lat.shape, generator=g, device=dev, dtype=lat.dtype); ts = torch.full((len(lat),), t, device=dev).long()
                    pred = pipe.unet(sched.add_noise(lat, noise, ts), ts, encoder_hidden_states=emb.expand(len(lat), -1, -1)).sample
                    tot.append(F.mse_loss(pred.float(), noise.float()).item())
        out.append(f"{name} {np.mean(tot):.4f}")
    print(f"step {ck:4d}: " + " | ".join(out), flush=True); del pipe; torch.cuda.empty_cache()
