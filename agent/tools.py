"""
LangChain Tools — Agent 可调用的金融工具集。

工具列表:
  - get_stock_quote(ticker)        最新价 + 涨跌幅
  - get_price_history(ticker, days) 历史价 (供后续指标计算)
  - calculate_rsi(ticker, period)  RSI 14
  - calculate_macd(ticker)         MACD (12,26,9)
  - analyze_sentiment(text)        调用微调 FinLoRA 模型做情感分类
  - generate_kline_chart(ticker)   生成 K 线图 PNG，返回路径
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from langchain.tools import tool

ASSET_DIR = Path("outputs/charts")
ASSET_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------- 行情 ----------------------

def _fetch_history(ticker: str, days: int = 90) -> pd.DataFrame:
    """用 yfinance 拉历史价格。失败时抛出，调用方负责转 string 返回给 Agent。"""
    import yfinance as yf
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"yfinance returned empty data for {ticker}")
    # yfinance 新版可能返回 MultiIndex columns，flatten 一下
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@tool
def get_stock_quote(ticker: str) -> str:
    """获取股票最新价格与今日涨跌幅。输入: 标的代码 (如 'AAPL', 'TSLA', '0700.HK')。"""
    try:
        df = _fetch_history(ticker, days=5)
        last = df["Close"].iloc[-1]
        prev = df["Close"].iloc[-2]
        chg = (last - prev) / prev * 100
        return f"{ticker} latest close = {last:.2f}, day change = {chg:+.2f}%"
    except Exception as e:
        return f"ERROR fetching quote for {ticker}: {e}"


@tool
def calculate_rsi(ticker: str, period: int = 14) -> str:
    """计算 RSI 指标 (默认 14 日)。RSI > 70 超买，RSI < 30 超卖。"""
    try:
        df = _fetch_history(ticker, days=period * 4 + 10)
        close = df["Close"]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - 100 / (1 + rs)
        latest = float(rsi.iloc[-1])
        signal = "OVERBOUGHT" if latest > 70 else "OVERSOLD" if latest < 30 else "NEUTRAL"
        return f"{ticker} RSI({period}) = {latest:.1f} [{signal}]"
    except Exception as e:
        return f"ERROR computing RSI for {ticker}: {e}"


@tool
def calculate_macd(ticker: str) -> str:
    """计算 MACD(12,26,9)。返回 DIF / DEA / 柱状值与金叉死叉信号。"""
    try:
        df = _fetch_history(ticker, days=120)
        close = df["Close"]
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        hist = dif - dea
        cross = ""
        if dif.iloc[-2] < dea.iloc[-2] and dif.iloc[-1] > dea.iloc[-1]:
            cross = " — GOLDEN CROSS (bullish)"
        elif dif.iloc[-2] > dea.iloc[-2] and dif.iloc[-1] < dea.iloc[-1]:
            cross = " — DEATH CROSS (bearish)"
        return (f"{ticker} MACD: DIF={float(dif.iloc[-1]):.3f}, "
                f"DEA={float(dea.iloc[-1]):.3f}, "
                f"HIST={float(hist.iloc[-1]):.3f}{cross}")
    except Exception as e:
        return f"ERROR computing MACD for {ticker}: {e}"


# ---------------------- LLM 情感分析 ----------------------

@tool
def analyze_sentiment(text: str) -> str:
    """用微调后的 FinLoRA 模型分析金融文本情感，返回 label + rationale。

    输入: 金融新闻、社交媒体帖子、财报段落等文本。
    """
    from .sentiment_llm import FinLoRASentiment
    adapter = os.environ.get("FINLORA_ADAPTER", "checkpoints/finlora-qwen2.5-1.5b")
    model = FinLoRASentiment.get(adapter_path=adapter)
    result = model(text)
    return f"Sentiment: {result.label}\nRationale: {result.rationale}"


# ---------------------- 绘图 ----------------------

@tool
def generate_kline_chart(ticker: str, days: int = 60) -> str:
    """生成股票最近 N 日的 K 线图 (蜡烛 + 成交量 + MA20)，保存为 PNG。

    返回: 图片本地路径 (供 Agent 进一步用于多模态理解或展示)。
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        df = _fetch_history(ticker, days=days + 30)
        df = df.tail(days).reset_index()

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        ma20 = df["Close"].rolling(20).mean()
        for i, row in df.iterrows():
            o, c, h, l = row["Open"], row["Close"], row["High"], row["Low"]
            color = "#d33" if c < o else "#2a7"
            ax1.plot([i, i], [l, h], color=color, linewidth=0.7)
            ax1.add_patch(Rectangle((i - 0.3, min(o, c)), 0.6,
                                    abs(c - o), color=color))
        ax1.plot(df.index, ma20, label="MA20", color="orange", linewidth=1.0)
        ax1.set_title(f"{ticker} — last {days} days")
        ax1.grid(alpha=0.3)
        ax1.legend()

        colors = ["#d33" if c < o else "#2a7"
                  for o, c in zip(df["Open"], df["Close"])]
        ax2.bar(df.index, df["Volume"], color=colors, width=0.6)
        ax2.set_ylabel("Volume")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        path = ASSET_DIR / f"{ticker.replace('.', '_')}_kline.png"
        plt.savefig(path, dpi=120)
        plt.close()
        return str(path)
    except Exception as e:
        return f"ERROR generating chart for {ticker}: {e}"


# ---------------------- 工具集合 ----------------------

ALL_TOOLS = [
    get_stock_quote,
    calculate_rsi,
    calculate_macd,
    analyze_sentiment,
    generate_kline_chart,
]
