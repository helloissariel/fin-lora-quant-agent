"""
对比 base 模型 vs LoRA 微调模型 在金融情感测试集上的表现。

输出:
  eval/results_baseline.json
  eval/results_lora.json
  控制台打印 accuracy / F1 / 混淆矩阵
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer

LABEL_PATTERN = re.compile(r"Sentiment\s*:\s*(POSITIVE|NEGATIVE|NEUTRAL)", re.IGNORECASE)
LABELS = ["POSITIVE", "NEGATIVE", "NEUTRAL"]


def parse_label(text: str) -> str | None:
    m = LABEL_PATTERN.search(text)
    if m:
        return m.group(1).upper()
    # 兜底: 找第一个 label 词
    for lab in LABELS:
        if lab in text.upper():
            return lab
    return None


def load_model(base_model: str, adapter_path: str | None, load_in_4bit: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(device_map="auto", trust_remote_code=True)
    if load_in_4bit:
        # 32B 在 bf16 下 64GB 装不下 48GB A6000，必须 4bit NF4 量化
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)

    if adapter_path:
        print(f"[eval] loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        # 4bit 下不能 merge_and_unload (会 dequant→merge→requant，数值漂移)
        if not load_in_4bit:
            model = model.merge_and_unload()
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate(tokenizer, model, messages, max_new_tokens=64):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    out = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    completion = tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)
    return completion


def evaluate(tokenizer, model, eval_file: str, max_samples: int | None):
    y_true, y_pred, dumps = [], [], []
    with open(eval_file) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            obj = json.loads(line)
            messages = obj["messages"]
            gold_response = messages[-1]["content"]
            gold_label = parse_label(gold_response)
            prompt_messages = messages[:-1]   # 去掉 assistant 回复

            completion = generate(tokenizer, model, prompt_messages)
            pred_label = parse_label(completion)

            if gold_label and pred_label:
                y_true.append(gold_label)
                y_pred.append(pred_label)
            dumps.append({
                "input": messages[1]["content"][:200],
                "gold": gold_label,
                "pred": pred_label,
                "completion": completion[:200],
            })
            if (i + 1) % 20 == 0:
                print(f"  [{i + 1}] running acc = "
                      f"{accuracy_score(y_true, y_pred):.3f}" if y_true else "")

    if not y_true:
        print("[eval] WARNING: no parseable predictions")
        return {"accuracy": 0.0, "n": 0}, dumps

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=LABELS, digits=3, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()

    print(f"\n[eval] N = {len(y_true)} samples (parsed)")
    print(f"[eval] Accuracy = {acc:.4f}")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=3))
    print(f"[eval] Confusion matrix (rows=gold, cols=pred, order={LABELS}):")
    for row in cm:
        print("  ", row)

    return {
        "accuracy": acc,
        "n": len(y_true),
        "report": report,
        "confusion_matrix": cm,
        "labels": LABELS,
    }, dumps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default=None, help="LoRA adapter 路径；不传则评 base 模型")
    p.add_argument("--eval-data", default="data/processed/eval.jsonl")
    p.add_argument("--output", default=None, help="评测结果 json 输出路径")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--load-in-4bit", action="store_true",
                   help="32B 等大模型必须开启，NF4 量化加载")
    args = p.parse_args()

    tokenizer, model = load_model(args.base_model, args.adapter, load_in_4bit=args.load_in_4bit)
    metrics, dumps = evaluate(tokenizer, model, args.eval_data, args.max_samples)

    if args.output is None:
        tag = "lora" if args.adapter else "baseline"
        args.output = f"eval/results_{tag}.json"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        {"metrics": metrics, "predictions": dumps},
        indent=2, ensure_ascii=False,
    ))
    print(f"\n[eval] saved -> {args.output}")


if __name__ == "__main__":
    main()
