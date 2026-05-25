"""
封装微调后的 Qwen2.5-1.5B + FinLoRA adapter，提供金融文本情感分类接口。

设计:
  - 单例加载，避免每次 Agent 调用都重新载入 3GB 模型
  - 推理时合并 LoRA adapter 进 base weights，速度比挂着 adapter 快 ~20%
  - 输出结构化 (label, rationale) 而非自由文本，便于下游 Agent 解析
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, "FinLoRASentiment"] = {}

LABEL_PATTERN = re.compile(r"Sentiment\s*:\s*(POSITIVE|NEGATIVE|NEUTRAL)", re.IGNORECASE)
RATIONALE_PATTERN = re.compile(r"Rationale\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)

SYSTEM_PROMPT = (
    "You are FinLoRA, a financial analyst assistant. "
    "Given a piece of news, social post, or earnings statement, output:\n"
    "1) Sentiment label among {POSITIVE, NEGATIVE, NEUTRAL}.\n"
    "2) A one-sentence rationale grounded in the text.\n"
    "Keep answers concise and avoid speculation beyond the evidence."
)


@dataclass
class SentimentResult:
    label: str
    rationale: str
    raw: str

    def to_dict(self):
        return {"label": self.label, "rationale": self.rationale, "raw": self.raw}


class FinLoRASentiment:
    """金融情感分析器 (LoRA 微调 Qwen)，单例使用。"""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: str | None = None,
        device: str | None = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map=device or ("auto" if torch.cuda.is_available() else None),
            trust_remote_code=True,
        )
        if adapter_path and Path(adapter_path).exists():
            from peft import PeftModel
            print(f"[FinLoRA] loading adapter -> {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
        self.model.eval()

    @classmethod
    def get(cls, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
            adapter_path: str | None = None) -> "FinLoRASentiment":
        key = f"{base_model}::{adapter_path}"
        with _MODEL_LOCK:
            if key not in _MODEL_CACHE:
                _MODEL_CACHE[key] = cls(base_model, adapter_path)
            return _MODEL_CACHE[key]

    @torch.no_grad()
    def __call__(self, text: str, max_new_tokens: int = 96) -> SentimentResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze the financial sentiment of the following text:\n\n{text.strip()}"},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        ).to(self.model.device)
        out = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        raw = self.tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True).strip()

        label_m = LABEL_PATTERN.search(raw)
        rationale_m = RATIONALE_PATTERN.search(raw)
        return SentimentResult(
            label=label_m.group(1).upper() if label_m else "UNKNOWN",
            rationale=rationale_m.group(1).strip() if rationale_m else raw,
            raw=raw,
        )
