
import torch, cv2, numpy as np, json, sys
sys.path.insert(0, "/content/img2img-turbo/src")
from cyclegan_turbo import CycleGAN_Turbo, VAE_encode, VAE_decode, initialize_unet, initialize_vae
from diffusers import DDPMScheduler
from transformers import AutoTokenizer, CLIPTextModel
import copy
dev = "cuda"; M = "stabilityai/sd-turbo"
tok = AutoTokenizer.from_pretrained(M, subfolder="tokenizer"); te = CLIPTextModel.from_pretrained(M, subfolder="text_encoder").to(dev)
unet, *_ = initialize_unet(128, return_lora_module_names=True); vae_a2b, _ = initialize_vae(4, return_lora_module_names=True)
unet.to(dev).eval(); vae_a2b.to(dev).eval(); vae_b2a = copy.deepcopy(vae_a2b)
enc, dec = VAE_encode(vae_a2b, vae_b2a=vae_b2a), VAE_decode(vae_a2b, vae_b2a=vae_b2a)
sched = DDPMScheduler.from_pretrained(M, subfolder="scheduler"); sched.set_timesteps(1, device=dev); sched.alphas_cumprod = sched.alphas_cumprod.to(dev)
emb = te(tok("real photograph of an airport runway seen from an aircraft cockpit on final approach", max_length=tok.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(dev))[0]
DS = "/content/img2img-turbo/data/xp2real"; labels = sorted(l["file"] for l in json.load(open(DS + "/sim_labels.json"))["test"]); tiles = []
for i in (0, 60, 120, 200):
    sim = cv2.imread(f"{DS}/test_A/{labels[i]}")[16:272, 32:480]
    x = torch.from_numpy(cv2.cvtColor(sim, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(dev)
    with torch.no_grad():
        y = CycleGAN_Turbo.forward_with_networks(x, "a2b", enc, unet, dec, sched, torch.tensor([999], device=dev).long(), emb)
    out = cv2.cvtColor(((y[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).byte().cpu().numpy(), cv2.COLOR_RGB2BGR)
    tiles.append(np.hstack([sim, out]))
cv2.imwrite(sys.argv[1], np.vstack(tiles)); print("saved", sys.argv[1])
