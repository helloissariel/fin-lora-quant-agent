"""
czsc (缠中说禅) 缠论分析 — LangChain Tool。

为什么加这个工具:
  - RSI/MACD 是西方 TA，缠论是国内量化研究的本土主流框架之一；
  - 蝶威、九坤、明汯等国内基金做技术面研究都用过缠论；
  - 让 Agent 同时具备 "西方指标 + 中国结构" 两种视角，决策时能交叉验证。

数据源策略:
  - A 股 (代码 6 位数字, 如 "000001" "600000"): 走 akshare
  - 港股 (XXXX.HK): 走 akshare 港股接口
  - 美股 (字母代码, 如 "AAPL"): 走 yfinance
  - 全部失败时返回 ERROR 字符串，Agent 会自己跳过这个工具

输出:
  最近若干"笔" + 当前"未完成笔"方向 + 最近"分型"，给 Agent 做趋势判断。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from langchain.tools import tool


# -----------------------------------------------------------------------------
# 数据获取
# -----------------------------------------------------------------------------

_A_SHARE_RE = re.compile(r"^\d{6}$")


def _is_a_share(ticker: str) -> bool:
    return bool(_A_SHARE_RE.match(ticker))


def _a_share_market_prefix(ticker: str) -> str:
    """000xxx/300xxx 是深圳，6xxxxx 是上海。"""
    return "sh" if ticker.startswith("6") else "sz"


def _fetch_a_share(ticker: str, days: int) -> pd.DataFrame:
    """A 股数据 — 优先 stock_zh_a_daily (更稳)，失败 fallback 到 stock_zh_a_hist。"""
    import akshare as ak
    symbol = f"{_a_share_market_prefix(ticker)}{ticker}"
    try:
        df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
        if df is not None and not df.empty:
            df = df.tail(days).reset_index(drop=True)   # 截取最近 N 天
            df["date"] = pd.to_datetime(df["date"])
            return df.rename(columns={"volume": "vol"})
    except Exception:
        pass
    # fallback
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    df = ak.stock_zh_a_hist(
        symbol=ticker, period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust="qfq",
    )
    if df is None or df.empty:
        raise ValueError(f"akshare empty for A-share {ticker}")
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low",
        "成交量": "vol", "成交额": "amount",
    })
    df["date"] = pd.to_datetime(df["date"])
    return df


def _fetch_ohlcv(ticker: str, days: int) -> pd.DataFrame:
    """返回 DataFrame, 列: date open close high low vol amount."""
    if _is_a_share(ticker):
        return _fetch_a_share(ticker, days)

    import yfinance as yf
    df = yf.download(ticker, period=f"{days}d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"yfinance empty for {ticker}")
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={
        "Date": "date", "Open": "open", "Close": "close",
        "High": "high", "Low": "low", "Volume": "vol",
    })
    df["amount"] = df["close"] * df["vol"]
    return df


# -----------------------------------------------------------------------------
# CZSC 构建
# -----------------------------------------------------------------------------

def _build_czsc(ticker: str, df: pd.DataFrame):
    from czsc import CZSC, RawBar, Freq
    bars = [
        RawBar(
            symbol=ticker,
            dt=row["date"].to_pydatetime(),
            freq=Freq.D,
            open=float(row["open"]),
            close=float(row["close"]),
            high=float(row["high"]),
            low=float(row["low"]),
            vol=float(row["vol"]),
            amount=float(row["amount"]),
            id=i,
        )
        for i, row in df.iterrows()
    ]
    return CZSC(bars)


# -----------------------------------------------------------------------------
# 结果格式化
# -----------------------------------------------------------------------------

def _format_summary(ticker: str, c, last_n_bi: int = 5, last_n_fx: int = 3) -> str:
    lines = [f"=== 缠论结构 (czsc) {ticker} ==="]
    lines.append(f"K 线 {len(c.bars_raw)} 根 · 分型 {len(c.fx_list)} 个 · 笔 {len(c.bi_list)} 段")

    # 最近 N 笔
    if c.bi_list:
        lines.append(f"\n最近 {last_n_bi} 笔:")
        for bi in c.bi_list[-last_n_bi:]:
            arrow = "↑" if bi.direction.value == "向上" else "↓"
            lines.append(
                f"  {arrow} {bi.direction.value}  "
                f"{bi.fx_a.dt.date()} → {bi.fx_b.dt.date()}  "
                f"{bi.low:.2f} → {bi.high:.2f}  "
                f"({bi.length} bars)"
            )

    # 未完成笔
    if c.ubi:
        ubi = c.ubi
        direction = ubi.get("direction") if isinstance(ubi, dict) else getattr(ubi, "direction", None)
        if direction is not None:
            dir_str = direction.value if hasattr(direction, "value") else str(direction)
            lines.append(f"\n当前未完成笔方向: {dir_str}")
        else:
            lines.append("\n当前存在未完成笔 (方向待确认)")
    else:
        lines.append("\n无未完成笔，最近一笔已成型")

    # 最近分型 — 重点是顶/底分型的转折信号
    if c.fx_list:
        lines.append(f"\n最近 {last_n_fx} 个分型:")
        for fx in c.fx_list[-last_n_fx:]:
            lines.append(f"  {fx.mark.value}  {fx.dt.date()}  价格 {fx.fx:.2f}")

    # 极简交易解读
    if len(c.bi_list) >= 2:
        last, prev = c.bi_list[-1], c.bi_list[-2]
        if last.direction.value == "向上" and prev.direction.value == "向下":
            lines.append("\n[解读] 最近一笔由下转上，可能形成局部反转 (需结合更高级别确认)。")
        elif last.direction.value == "向下" and prev.direction.value == "向上":
            lines.append("\n[解读] 最近一笔由上转下，警惕高位回落。")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# LangChain Tool
# -----------------------------------------------------------------------------

@tool
def czsc_analyze(ticker: str, days: int = 180) -> str:
    """缠论 (czsc) 技术结构分析 — 对中国本土量化研究友好的 TA 框架。

    输入:
      ticker: 标的代码。A 股填 6 位数字 (如 "000001" 平安银行、"600519" 茅台);
              美股填字母代码 (如 "AAPL" "TSLA")。
      days: 取近 N 个日历日的日线 (默认 180, 推荐 90-360)。

    输出: 最近若干笔的方向/价格/长度、未完成笔方向、最近分型, 以及一句趋势解读。
    适合: 在 RSI/MACD 之外提供"结构性"视角, 看清当前是上涨笔/下跌笔的哪一段。
    """
    try:
        df = _fetch_ohlcv(ticker, days)
    except Exception as e:
        return f"ERROR fetching OHLCV for {ticker}: {e}"

    if len(df) < 30:
        return f"ERROR: too few bars ({len(df)}) for czsc analysis on {ticker}"

    try:
        c = _build_czsc(ticker, df)
    except Exception as e:
        return f"ERROR building CZSC for {ticker}: {e}"

    return _format_summary(ticker, c)


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "000001"
    print(czsc_analyze.invoke({"ticker": ticker, "days": 365}))
