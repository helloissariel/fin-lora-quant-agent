"""
Stable Diffusion LoRA 微调脚本 (基于 diffusers 官方训练范式简化版)。

用途:
  在 8~32 张小数据集上把 SD 微调到特定风格 (如 "blueprint of trading charts",
  "neon cyberpunk fintech illustration")，比 prompt engineering 更稳定。

数据准备:
  data/sd_finance/
      img_001.png
      img_001.txt   # caption: "a blueprint style chart of S&P 500 candlesticks"
      img_002.png
      img_002.txt
      ...

设计要点 (面试可讲):
  - UNet 的 cross-attention 层挂 LoRA (q/k/v/out)，约 ~3M 可训参数；
  - rank=4 already enough for style transfer (rank 大易过拟合到具体内容);
  - 学习率比 LLM 大: 1e-4 ~ 5e-4，因为 SD UNet 是 zero-init residual；
  - 训练 800-1500 步即可，过长容易丢失原模型能力。

运行:
  python multimodal/train_sd_lora.py \\
      --data-dir data/sd_finance \\
      --output-dir checkpoints/sd-finance-lora \\
      --max-steps 1000
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class FinanceImageDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer, size: int = 512):
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.items = []
        for img_path in sorted(Path(data_dir).glob("*.png")):
            txt_path = img_path.with_suffix(".txt")
            caption = txt_path.read_text().strip() if txt_path.exists() else ""
            self.items.append((img_path, caption))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, caption = self.items[idx]
        image = self.transform(Image.open(img_path).convert("RGB"))
        ids = self.tokenizer(
            caption, padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt",
        ).input_ids.squeeze(0)
        return {"pixel_values": image, "input_ids": ids}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", default="checkpoints/sd-finance-lora")
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)

    from diffusers import (
        AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel,
    )
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("[sd-lora] loading base components...")
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder").to(device, dtype)
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae").to(device, dtype)
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet").to(device, dtype)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 给 UNet cross-attn 挂 LoRA
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_cfg)
    cast_training_params(unet, dtype=torch.float32)

    trainable = [p for p in unet.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[sd-lora] trainable params: {n_trainable/1e6:.2f}M")

    dataset = FinanceImageDataset(args.data_dir, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"[sd-lora] dataset size: {len(dataset)}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    step = 0
    pbar = tqdm(total=args.max_steps)
    while step < args.max_steps:
        for batch in loader:
            with torch.no_grad():
                pv = batch["pixel_values"].to(device, dtype)
                latents = vae.encode(pv).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                          (latents.shape[0],), device=device).long()
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)

                enc_hidden = text_encoder(batch["input_ids"].to(device))[0]

            pred = unet(noisy, timesteps, enc_hidden).sample
            loss = F.mse_loss(pred.float(), noise.float())

            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if step >= args.max_steps:
                break

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # 用 diffusers 自带的 save_lora_weights，输出 pytorch_lora_weights.safetensors
    StableDiffusionPipeline.save_lora_weights(
        save_directory=args.output_dir,
        unet_lora_layers={k: v for k, v in unet.state_dict().items() if "lora" in k},
    )
    print(f"[sd-lora] saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
