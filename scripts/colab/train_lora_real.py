"""LoRA on SD 1.5 UNet attention, trained on the real playlist crops (learn the 'real footage' look)."""
import os, glob, sys, time, random, torch, torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers

BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DATA, OUT = sys.argv[1], sys.argv[2]
STEPS, BS, LR, RANK = int(sys.argv[3]), 8, 1e-4, 16
PROMPT = "rwyreal photo of an airport runway seen from an aircraft cockpit on final approach"
torch.manual_seed(0); random.seed(0); dev = "cuda"
tok = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer")
te = CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder").to(dev).eval()
vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae").to(dev).eval()
unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet").to(dev)
sched = DDPMScheduler.from_pretrained(BASE, subfolder="scheduler")
unet.requires_grad_(False)
unet.add_adapter(LoraConfig(r=RANK, lora_alpha=RANK, init_lora_weights="gaussian", target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
params = [p for p in unet.parameters() if p.requires_grad]
print("trainable M:", round(sum(p.numel() for p in params) / 1e6, 2), flush=True)
opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-2)
files = sorted(glob.glob(f"{DATA}/*.jpg")); print(len(files), "images", flush=True)
tf = T.Compose([T.RandomCrop((256, 448)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([0.5], [0.5])])
with torch.no_grad():
    emb = te(tok([PROMPT], padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids.to(dev))[0]
    emb_empty = te(tok([""], padding="max_length", max_length=tok.model_max_length, return_tensors="pt").input_ids.to(dev))[0]
t0 = time.time()
for step in range(1, STEPS + 1):
    x = torch.stack([tf(Image.open(random.choice(files)).convert("RGB")) for _ in range(BS)]).to(dev)
    with torch.no_grad():
        lat = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
    noise = torch.randn_like(lat); ts = torch.randint(0, sched.config.num_train_timesteps, (BS,), device=dev)
    noisy = sched.add_noise(lat, noise, ts)
    cond = torch.where(torch.rand(BS, 1, 1, device=dev) < 0.1, emb_empty, emb.expand(BS, -1, -1))   # 10 % prompt dropout for CFG
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = unet(noisy, ts, encoder_hidden_states=cond).sample
    loss = F.mse_loss(pred.float(), noise)
    loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
    if step % 50 == 0: print(f"step {step}/{STEPS} loss {loss.item():.4f} {(time.time()-t0)/step:.2f} s/it", flush=True)
    if step % 500 == 0 or step == STEPS:
        d = f"{OUT}/step{step}"; os.makedirs(d, exist_ok=True)
        StableDiffusionPipeline.save_lora_weights(d, unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)))
print("DONE", flush=True)
