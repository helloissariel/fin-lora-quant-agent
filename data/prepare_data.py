"""
准备金融领域指令微调数据集。

支持两种数据源:
  1) HuggingFace 公开数据集 (默认 FinGPT/fingpt-sentiment-train，中英混合可加财经新闻)
  2) 本地 samples/*.jsonl 兜底 (网络不可用时仍能跑通 pipeline)

输出: data/processed/train.jsonl, data/processed/eval.jsonl
格式: ChatML messages 列表，可直接喂 trl.SFTTrainer / LLaMA-Factory
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"
SAMPLES = ROOT / "samples"

SYSTEM_PROMPT = (
    "You are FinLoRA, a financial analyst assistant. "
    "Given a piece of news, social post, or earnings statement, output:\n"
    "1) Sentiment label among {POSITIVE, NEGATIVE, NEUTRAL}.\n"
    "2) A one-sentence rationale grounded in the text.\n"
    "Keep answers concise and avoid speculation beyond the evidence."
)


def format_example(text: str, label: str, rationale: str | None = None) -> dict:
    """把 (text, label) 转成 ChatML 三轮对话。"""
    user = f"Analyze the financial sentiment of the following text:\n\n{text.strip()}"
    if rationale:
        assistant = f"Sentiment: {label}\nRationale: {rationale.strip()}"
    else:
        assistant = f"Sentiment: {label}\nRationale: {label.lower()} signal inferred from the wording."
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def from_huggingface(dataset_name: str, max_train: int, max_eval: int):
    """从 HF 拉取 FinGPT 情感数据 (英文为主，含金融关键词)。"""
    from datasets import load_dataset

    print(f"[data] loading from HF: {dataset_name}")
    ds = load_dataset(dataset_name)
    split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

    rows = []
    for r in split:
        text = r.get("input") or r.get("text") or r.get("sentence")
        label = (r.get("output") or r.get("label") or "").strip().upper()
        if not text or label not in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
            continue
        rows.append(format_example(text, label))
    random.shuffle(rows)
    return rows[:max_train], rows[max_train : max_train + max_eval]


def from_samples():
    """兜底: 用 samples/ 下的手工样例 (中英混合)。"""
    print("[data] falling back to hand-crafted samples")
    train, eval_ = [], []
    for f in sorted(SAMPLES.glob("*.jsonl")):
        with f.open() as fp:
            for line in fp:
                obj = json.loads(line)
                ex = format_example(obj["text"], obj["label"], obj.get("rationale"))
                (eval_ if "eval" in f.stem else train).append(ex)
    return train, eval_


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[data] wrote {len(rows):>5d} rows -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="FinGPT/fingpt-sentiment-train",
                   help="HF 数据集名；填 'samples' 强制走样例")
    p.add_argument("--max-train", type=int, default=2000)
    p.add_argument("--max-eval", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    if args.source == "samples":
        train, eval_ = from_samples()
    else:
        try:
            train, eval_ = from_huggingface(args.source, args.max_train, args.max_eval)
            if not train:
                raise RuntimeError("empty after filtering")
        except Exception as e:
            print(f"[data] HF source failed ({e}); falling back to samples")
            train, eval_ = from_samples()

    write_jsonl(PROCESSED / "train.jsonl", train)
    write_jsonl(PROCESSED / "eval.jsonl", eval_)

    if train:
        print("\n[data] sanity check — first training example:")
        print(json.dumps(train[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
