"""
Stable Diffusion 文生图 — 为研究报告 / 量化策略生成封面图。

默认模型: stable-diffusion-v1-5 (兼容性最广)
可切换:   sd-turbo (4 步出图，速度优先) / sdxl-base-1.0 (画质优先)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class GenerationConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 768
    seed: int | None = 42


_PIPE = None


def _get_pipe(cfg: GenerationConfig):
    """Lazy 加载 SD pipeline；支持挂载 LoRA adapter。"""
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    from diffusers import StableDiffusionPipeline

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model_id, torch_dtype=dtype, safety_checker=None,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()

    # 可选: 挂载本仓库训练的 LoRA adapter (sd-finance-lora)
    lora_path = os.environ.get("SD_LORA_ADAPTER")
    if lora_path and Path(lora_path).exists():
        print(f"[SD] loading LoRA adapter -> {lora_path}")
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora()

    _PIPE = pipe
    return pipe


def generate_report_cover(
    prompt: str,
    output_path: str = "outputs/report_cover.png",
    negative_prompt: str = "low quality, blurry, watermark, text, logo",
    cfg: GenerationConfig | None = None,
) -> str:
    """根据 prompt 生成研究报告封面图，返回保存路径。"""
    cfg = cfg or GenerationConfig()
    pipe = _get_pipe(cfg)

    generator = None
    if cfg.seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(cfg.seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        height=cfg.height,
        width=cfg.width,
        generator=generator,
    ).images[0]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return str(out)


def finance_themed_prompt(topic: str, mood: str = "professional") -> str:
    """为金融场景构造高质量 prompt。"""
    style = {
        "professional": "professional financial research report cover, elegant, minimalistic",
        "bullish": "rising stock market trend, golden light, optimistic atmosphere",
        "bearish": "declining stock market, stormy clouds, dramatic lighting",
        "tech": "futuristic AI trading floor, holographic candlestick charts, neon blue",
    }.get(mood, mood)
    return (
        f"{topic}, {style}, ultra high quality, 4k, sharp focus, "
        f"cinematic lighting, financial data visualization aesthetic"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="quantitative trading strategy on AI stocks")
    parser.add_argument("--mood", default="tech")
    parser.add_argument("--out", default="outputs/report_cover.png")
    args = parser.parse_args()

    prompt = finance_themed_prompt(args.topic, args.mood)
    print(f"[SD] prompt: {prompt}")
    path = generate_report_cover(prompt, output_path=args.out)
    print(f"[SD] saved -> {path}")
