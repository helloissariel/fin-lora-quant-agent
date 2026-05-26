# FinLoRA-Agent · 模块思维导图

> GitHub 原生渲染 mermaid 思维导图，点开就能看完整层级。

```mermaid
mindmap
  root((FinLoRA-Agent))
    数据
      prepare_data.py
        HF FinGPT/fingpt-sentiment-train
        samples/ 中英 35 条 fallback
        ChatML messages 格式
      train.jsonl 2000 条
      eval.jsonl 200 条
    训练
      train_lora.py
        Qwen2.5-1.5B-Instruct
        LoRA r=16 α=32 q/k/v/o
        TRL SFTTrainer
        3 epoch / 3.8 min / A6000
      train_qlora.py
        Qwen2.5-32B-Instruct
        4bit NF4 + double quant
        paged AdamW 8bit
        gradient checkpointing
      eval_model.py
        accuracy / F1 / 混淆矩阵
        --load-in-4bit 32B 评测
      plot_curves.py
      configs/qwen_lora.yaml LLaMA-Factory
    Agent
      finance_agent.py
        LangChain ReAct
        backend OpenAI/DeepSeek/Local
      sentiment_llm.py
        微调模型推理单例
        LoRA merge_and_unload
      tools.py 6 个工具
        get_stock_quote yfinance
        calculate_rsi 14日
        calculate_macd 12/26/9
        analyze_sentiment 调FinLoRA
        generate_kline_chart matplotlib
        czsc_analyze 缠论
      czsc_tool.py
        akshare A股 / yfinance 美股
        笔/分型/中枢 解析
        多 endpoint 冗余
      prompts.py system prompt
    多模态
      chart_understanding.py
        CLIP zero-shot
        BULLISH/BEARISH/SIDEWAYS/VOLATILE
      sd_generate.py
        Stable Diffusion 1.5
        finance_themed_prompt
      train_sd_lora.py
        UNet cross-attn LoRA
        rank=4 风格迁移
    Demo
      app.py Gradio 4 Tab
        情感分析 Base vs LoRA 对比
        Agent Q&A 走工具链
        K线图 CLIP 理解
        SD 报告封面生成
    自动化脚本
      01_install.sh
      02_prepare_data.sh
      03_train.sh
      04_eval.sh
      05_demo.sh
      06_post_train.sh 1.5B 后处理链
      07_post_train_32b.sh 32B 后处理链
      _auto_chain_32b.sh 下载-训练-eval 全链
    评测产物
      results_baseline.json 1.5B base
      results_lora.json 1.5B LoRA acc 62.5%
      SUMMARY.md 对比汇总
      results_32b_*.json 训练中
    资源与文档
      assets/architecture.md mermaid
      assets/training_curves.png
      README.md 中文
      README_EN.md 英文
      requirements.txt
      LICENSE MIT
```

## 模块速查表

| 模块 | 解决的问题 | 关键技术 |
| --- | --- | --- |
| **数据** | 把 FinGPT 数据转成可微调的 ChatML | datasets, samples fallback |
| **训练 (1.5B LoRA)** | 让 base 模型敢做方向性判断 | PEFT + TRL.SFTTrainer |
| **训练 (32B QLoRA)** | 单卡 A6000 上微调 32B 大模型 | NF4 + double quant + paged AdamW + grad ckpt |
| **Agent** | 把工具链 + 微调模型串成可问答智能体 | LangChain ReAct |
| **czsc 工具** | 用国内主流的缠论范式做技术分析 | czsc + akshare |
| **多模态-图像理解** | K 线图分类、视觉信号融合 | CLIP zero-shot |
| **多模态-图像生成** | 给研究报告自动生成配图 | Stable Diffusion + LoRA |
| **Demo** | 把全套能力可视化、能现场跑 | Gradio 4-Tab |
| **自动化** | 本地断网/关机也能跑完整 pipeline | nohup chain + post-train hooks |
| **评测** | 量化证明微调有效、scale up 有效 | accuracy/F1/confusion matrix |

也可参考 [`architecture.md`](./architecture.md) 看数据流向架构图。
