# Eval Summary — Scaling Comparison

| Model | Trainable | Acc | Macro-F1 | POS F1 | NEG F1 | NEU F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B (base) | — | 0.5350 | 0.5115 | 0.640 | 0.507 | 0.388 |
| Qwen2.5-1.5B + LoRA | 4.36M (0.28%) | 0.6250 | 0.6336 | 0.656 | 0.791 | 0.454 |
| Qwen2.5-32B (base, 4bit NF4) | — | 0.6300 | 0.6360 | 0.660 | 0.821 | 0.427 |
| Qwen2.5-32B + **QLoRA** | ~30M (~0.09%) | 0.8250 | 0.8291 | 0.823 | 0.857 | 0.807 |

*N=200 samples · deterministic decoding · 4bit eval uses NF4 + double quant + bf16 compute.*
