"""
K 线图理解 — 用 CLIP 做 zero-shot 分类。

思路:
  CLIP 把图像和文本投到同一向量空间，给定一张 K 线图 + 几个候选 caption
  ("a bullish candlestick chart...", "a bearish..."), 算余弦相似度选最高。

为什么这样设计 (面试可讲):
  - 不需要标注 K 线图数据集；
  - 同样的范式可换 prompt 做 "牛市形态识别"、"双底/头肩顶判断" 等更复杂的任务；
  - 落地量化场景: 把每日 K 线图过一遍 CLIP，得到的 image embedding 可以拼到
    数值因子上做 ensemble，是一种轻量的视觉信号融合。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


CLIP_MODEL_ID = "openai/clip-vit-base-patch16"

DEFAULT_LABELS: dict[str, str] = {
    "BULLISH":   "a candlestick stock chart in a strong uptrend with rising prices and green bars",
    "BEARISH":   "a candlestick stock chart in a downtrend with falling prices and red bars",
    "SIDEWAYS":  "a candlestick stock chart moving sideways in a tight range without clear direction",
    "VOLATILE":  "a candlestick stock chart with extreme volatility and large wicks in both directions",
}


@dataclass
class ChartClassification:
    label: str
    scores: dict[str, float]
    top2_gap: float


_PROCESSOR: CLIPProcessor | None = None
_MODEL: CLIPModel | None = None


def _load():
    global _PROCESSOR, _MODEL
    if _MODEL is None:
        _PROCESSOR = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _MODEL = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        _MODEL.eval()
        if torch.cuda.is_available():
            _MODEL.cuda()
    return _PROCESSOR, _MODEL


@torch.no_grad()
def classify_kline(
    image_path: str | Path,
    labels: dict[str, str] | None = None,
) -> ChartClassification:
    """对一张 K 线图做 zero-shot 分类。

    Args:
        image_path: K 线图 PNG/JPG 路径。
        labels: {label_name: caption}，默认 BULLISH/BEARISH/SIDEWAYS/VOLATILE。
    """
    processor, model = _load()
    labels = labels or DEFAULT_LABELS

    image = Image.open(image_path).convert("RGB")
    captions = list(labels.values())
    names = list(labels.keys())

    inputs = processor(text=captions, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image           # [1, n_labels]
    probs = logits_per_image.softmax(dim=-1).squeeze(0).cpu().tolist()

    scored = dict(zip(names, probs))
    sorted_pairs = sorted(scored.items(), key=lambda kv: -kv[1])
    top_label = sorted_pairs[0][0]
    top2_gap = sorted_pairs[0][1] - sorted_pairs[1][1]

    return ChartClassification(label=top_label, scores=scored, top2_gap=top2_gap)


if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser()
    p.add_argument("image", help="K 线图路径")
    args = p.parse_args()

    result = classify_kline(args.image)
    print(f"Label: {result.label}  (top2 gap = {result.top2_gap:.3f})")
    print(json.dumps(result.scores, indent=2))
