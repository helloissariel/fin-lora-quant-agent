"""
FinLoRA-Agent Gradio Demo — 一站式展示:
  Tab 1: 金融情感分析 (微调 vs base 模型对比)
  Tab 2: 量化助手 Agent (Q&A，走 LangChain)
  Tab 3: K 线图理解 (CLIP zero-shot)
  Tab 4: 报告封面生成 (Stable Diffusion)

启动:
  python demo/app.py --adapter checkpoints/finlora-qwen2.5-1.5b
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 允许从仓库根目录直接 import agent / multimodal
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from agent.sentiment_llm import FinLoRASentiment
from agent.tools import generate_kline_chart
from multimodal.chart_understanding import classify_kline


# ============================================================
# Tab 1: 情感分析对比
# ============================================================
_BASE_MODEL = None
_LORA_MODEL = None


def get_models(base_id: str, adapter: str | None):
    global _BASE_MODEL, _LORA_MODEL
    if _BASE_MODEL is None:
        _BASE_MODEL = FinLoRASentiment(base_id, adapter_path=None)
    if _LORA_MODEL is None and adapter:
        _LORA_MODEL = FinLoRASentiment(base_id, adapter_path=adapter)
    return _BASE_MODEL, _LORA_MODEL


def sentiment_compare(text, base_id, adapter):
    if not text.strip():
        return "请输入文本", "请输入文本"
    base, lora = get_models(base_id, adapter)
    base_out = base(text)
    if lora is None:
        return base_out.raw, "(未提供 adapter，无对比)"
    lora_out = lora(text)
    return base_out.raw, lora_out.raw


# ============================================================
# Tab 2: Agent
# ============================================================
def run_agent_q(question, backend):
    from agent.finance_agent import run_agent
    try:
        out = run_agent(question, backend=backend)
        steps = []
        for i, (act, obs) in enumerate(out.get("intermediate_steps", [])):
            steps.append(f"[{i+1}] {act.tool}({act.tool_input}) ->\n{str(obs)[:300]}")
        return out["output"], "\n\n".join(steps) or "(无工具调用)"
    except Exception as e:
        return f"Agent 调用失败: {e}\n(本地 backend 需先 pip install langchain-huggingface)", ""


# ============================================================
# Tab 3: K 线图理解
# ============================================================
def kline_pipeline(ticker, days):
    chart_path = generate_kline_chart.invoke({"ticker": ticker, "days": int(days)})
    if chart_path.startswith("ERROR"):
        return None, chart_path
    result = classify_kline(chart_path)
    scores_md = "\n".join(f"- **{k}**: {v:.3f}" for k, v in
                          sorted(result.scores.items(), key=lambda kv: -kv[1]))
    summary = f"### CLIP 判定: **{result.label}**  (置信度差 {result.top2_gap:.3f})\n\n{scores_md}"
    return chart_path, summary


# ============================================================
# Tab 4: SD 封面生成
# ============================================================
def sd_pipeline(topic, mood, steps):
    from multimodal.sd_generate import (
        finance_themed_prompt, generate_report_cover, GenerationConfig,
    )
    cfg = GenerationConfig(num_inference_steps=int(steps))
    prompt = finance_themed_prompt(topic, mood)
    path = generate_report_cover(prompt, output_path="outputs/demo_cover.png", cfg=cfg)
    return path, prompt


# ============================================================
# UI
# ============================================================
def build_ui(base_model_id: str, adapter: str | None):
    with gr.Blocks(title="FinLoRA-Agent Demo") as demo:
        gr.Markdown(
            f"# FinLoRA-Agent · 金融领域 LoRA 微调 + Agent + 多模态\n"
            f"Base model: `{base_model_id}` · "
            f"LoRA adapter: `{adapter or '(none)'}`"
        )

        with gr.Tab("情感分析对比"):
            txt = gr.Textbox(
                label="输入金融文本 (新闻 / 财报 / 帖子)", lines=4,
                value="Apple reported record Q4 revenue of $94.9B, beating estimates by 6%.",
            )
            with gr.Row():
                b1 = gr.Textbox(label="Base 模型输出", lines=4, interactive=False)
                b2 = gr.Textbox(label="FinLoRA 微调输出", lines=4, interactive=False)
            btn = gr.Button("Compare", variant="primary")
            btn.click(
                lambda t: sentiment_compare(t, base_model_id, adapter),
                inputs=[txt], outputs=[b1, b2],
            )

        with gr.Tab("量化助手 Agent"):
            q = gr.Textbox(
                label="提问",
                value="Should I worry about TSLA? Check its RSI and any recent news sentiment.",
            )
            backend = gr.Dropdown(
                choices=["openai", "deepseek", "local"], value="local",
                label="Agent LLM backend",
            )
            with gr.Row():
                final = gr.Textbox(label="Final Answer", lines=6, interactive=False)
                trace = gr.Textbox(label="Tool calls trace", lines=6, interactive=False)
            gr.Button("Run").click(run_agent_q, inputs=[q, backend], outputs=[final, trace])

        with gr.Tab("K 线图理解"):
            ticker = gr.Textbox(label="Ticker", value="AAPL")
            days = gr.Slider(20, 180, value=60, step=5, label="近 N 日")
            img = gr.Image(label="K 线图")
            md = gr.Markdown()
            gr.Button("Run").click(kline_pipeline, inputs=[ticker, days], outputs=[img, md])

        with gr.Tab("研究报告封面 (SD)"):
            topic = gr.Textbox(label="主题", value="quantitative momentum strategy in AI sector")
            mood = gr.Dropdown(
                choices=["professional", "bullish", "bearish", "tech"],
                value="tech", label="风格",
            )
            steps = gr.Slider(10, 50, value=25, step=1, label="采样步数")
            cover = gr.Image(label="生成结果")
            prompt_view = gr.Textbox(label="使用的 prompt", interactive=False)
            gr.Button("Generate").click(sd_pipeline, [topic, mood, steps], [cover, prompt_view])

    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default=os.environ.get(
        "FINLORA_ADAPTER", "checkpoints/finlora-qwen2.5-1.5b"))
    p.add_argument("--server-name", default="0.0.0.0")
    p.add_argument("--server-port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    if args.adapter and not Path(args.adapter).exists():
        print(f"[demo] WARNING: adapter not found at {args.adapter}, "
              f"falling back to base-only.")
        args.adapter = None

    demo = build_ui(args.base_model, args.adapter)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
