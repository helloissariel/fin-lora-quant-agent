# FinLoRA-Agent

> A complete *fine-tune → agent → multimodal* pipeline for quantitative finance research.

[中文](./README.md) · [English]

## What's in the box

| Module | Stack | One-line summary |
| --- | --- | --- |
| **Fine-tune** | Qwen2.5-1.5B-Instruct + LoRA (PEFT/TRL) | LoRA SFT on ~2K financial sentiment instructions. 3 epochs / A6000 / 30 min. |
| **Agent** | LangChain ReAct | Wraps the fine-tuned model + yfinance + RSI/MACD + SD into a tool-using agent. |
| **Chart understanding** | CLIP zero-shot | Classifies K-line images as BULLISH / BEARISH / SIDEWAYS / VOLATILE. |
| **Image generation** | Stable Diffusion + LoRA | SD-1.5 cover generation, plus a SD-LoRA training script for style transfer. |
| **Demo** | Gradio | 4-tab web UI covering all capabilities. |

## Why this project

Off-the-shelf LLMs on financial text tend to (a) over-predict NEUTRAL sentiment and
(b) refuse to chain tool calls. This repo addresses both:

- **LoRA fine-tune** to sharpen sentiment judgments on finance-specific phrasing.
- **ReAct agent** that delegates sentiment to the fine-tuned specialist while
  using a stronger general LLM (OpenAI / DeepSeek / local Qwen) for planning.

## Quickstart

```bash
pip install -r requirements.txt
python data/prepare_data.py
python train/train_lora.py --output-dir checkpoints/finlora-qwen2.5-1.5b
python train/eval_model.py --adapter checkpoints/finlora-qwen2.5-1.5b
python demo/app.py --adapter checkpoints/finlora-qwen2.5-1.5b
```

See [README.md](./README.md) for the full Chinese write-up.

## License

MIT
