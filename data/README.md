# Data

## 数据来源

| 优先级 | 来源 | 规模 | 说明 |
| --- | --- | --- | --- |
| 1 | `FinGPT/fingpt-sentiment-train` (HuggingFace) | ~76K | 英文为主，主流金融情感分类基准 |
| 2 | `financial_phrasebank` (HuggingFace) | ~5K | 经典金融情感数据集 |
| 3 | `samples/*.jsonl` (本仓库) | 35 条 | 手工构造的中英混合样例，离线兜底 |

## 数据格式

最终输出为 ChatML messages 列表，可直接喂给 `trl.SFTTrainer` 或 LLaMA-Factory：

```json
{
  "messages": [
    {"role": "system", "content": "You are FinLoRA, a financial analyst assistant..."},
    {"role": "user", "content": "Analyze the financial sentiment of: ..."},
    {"role": "assistant", "content": "Sentiment: POSITIVE\nRationale: ..."}
  ]
}
```

## 使用

```bash
# 默认: 拉 FinGPT，自动 fallback 到 samples
python data/prepare_data.py

# 强制只用本地 samples (无网络环境)
python data/prepare_data.py --source samples

# 自定义规模
python data/prepare_data.py --max-train 5000 --max-eval 500
```

输出：`data/processed/{train,eval}.jsonl`
