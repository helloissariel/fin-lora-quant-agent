"""
QLoRA 微调 Qwen2.5-32B-Instruct，在单卡 A6000 (48GB) 上跑出 32B 模型的金融情感专家。

技术栈 (面试可讲的几个点):
  - **4-bit NF4 量化**: 权重压缩到 4bit, 32B 模型从 64GB 压到 ~16GB，单卡能装下。
  - **double quantization**: 把量化常数本身再量化一次, 额外省 ~0.4 bit/param。
  - **bfloat16 compute**: 前向/反向计算时 dequant 到 bf16，保留训练稳定性。
  - **paged AdamW 8bit**: 优化器状态用 8bit + paging (类似 swap)，再省一截显存。
  - **gradient checkpointing**: 不存中间激活，反向时重算，时间换显存。
  - 训练参数仍只挂 LoRA r=16/alpha=32 到 q/k/v/o，可训参数 ~30M / 32B ≈ 0.09%。

显存预算 (A6000 48GB):
  量化权重 ~16GB + LoRA 参数 + 梯度 ~0.2GB + 8bit AdamW ~0.5GB
  + 激活 (batch=2 + grad ckpt) ~5GB ≈ 总 22-26GB，留出 ~20GB 余量做更长 seq。

运行:
  python train/train_qlora.py \\
      --base-model Qwen/Qwen2.5-32B-Instruct \\
      --output-dir checkpoints/finlora-qwen2.5-32b-qlora
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--train-data", default="data/processed/train.jsonl")
    p.add_argument("--eval-data", default="data/processed/eval.jsonl")
    p.add_argument("--output-dir", default="checkpoints/finlora-qwen2.5-32b-qlora")

    # LoRA 超参 — 32B 模型上保持和 1.5B 一致以便对比
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # 训练超参 — 模型 20× 大，batch 必须降下来；用更大 grad accum 维持有效 batch
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)        # 有效 batch = 16
    p.add_argument("--lr", type=float, default=1e-4)           # 32B 用更小 lr 比 1.5B 稳
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ---------- 4-bit 量化配置 ----------
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",                 # NF4: QLoRA 原作推荐, 比 fp4 收敛更好
        bnb_4bit_compute_dtype=torch.bfloat16,     # 算的时候 dequant 到 bf16
        bnb_4bit_use_double_quant=True,            # 量化常数二次量化, 再省 ~0.4 bit/param
    )

    print(f"[qlora] loading 4bit base from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"[qlora] loaded, gpu mem alloc: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # kbit 训练前必备: 关闭 cache、提升数值精度、用 grad ckpt 兼容的 forward
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ---------- LoRA ----------
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---------- 数据 ----------
    raw = load_dataset("json", data_files={"train": args.train_data, "eval": args.eval_data})

    # ---------- SFTConfig ----------
    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=0.3,                          # QLoRA 论文推荐 0.3，比常规 1.0 小

        max_length=args.max_seq_len,
        packing=False,

        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        optim="paged_adamw_8bit",                   # 8bit AdamW + paging，配合 QLoRA 省显存

        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,                         # 32B adapter 也只占 ~120MB，但仍只留 1 个最佳
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to=["none"],
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=raw["train"],
        eval_dataset=raw["eval"],
        processing_class=tokenizer,
    )

    print("[qlora] starting training...")
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    Path(args.output_dir, "train_metrics.json").write_text(
        json.dumps(train_result.metrics, indent=2, ensure_ascii=False))
    Path(args.output_dir, "log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, ensure_ascii=False))
    print(f"[qlora] saved adapter -> {args.output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
