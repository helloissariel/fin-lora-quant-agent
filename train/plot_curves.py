"""
从 log_history.json 生成训练曲线图 (loss / eval_loss / lr)。
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="checkpoints/finlora-qwen2.5-1.5b/log_history.json")
    p.add_argument("--out", default="assets/training_curves.png")
    args = p.parse_args()

    history = json.loads(Path(args.log).read_text())

    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    lr_steps, lr_vals = [], []
    for r in history:
        if "loss" in r and "eval_loss" not in r:
            train_steps.append(r["step"])
            train_loss.append(r["loss"])
        if "eval_loss" in r:
            eval_steps.append(r["step"])
            eval_loss.append(r["eval_loss"])
        if "learning_rate" in r and "eval_loss" not in r:
            lr_steps.append(r["step"])
            lr_vals.append(r["learning_rate"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_steps, train_loss, label="train loss", linewidth=1.2)
    if eval_loss:
        axes[0].plot(eval_steps, eval_loss, "o-", label="eval loss", color="C3")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training / Eval Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(lr_steps, lr_vals, color="C2")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("learning rate")
    axes[1].set_title("LR Schedule (cosine + warmup)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=140)
    print(f"[plot] saved -> {args.out}")


if __name__ == "__main__":
    main()
