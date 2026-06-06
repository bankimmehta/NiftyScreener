
"""
Nifty Screener Pro (Upgraded Skeleton)
Key upgrades:
- Order Block detection
- Fair Value Gap (FVG) detection
- Break of Structure (BOS)
- Institutional Score
- Buy / Neutral / Caution verdict
- Uses final forward return instead of peak return for hit-rate
- Designed as a replacement starting point for the original script
"""

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS_PER_MONTH = 21


def download_history(ticker, period="5y"):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def add_indicators(df, short=50, long=200):
    df = df.copy()
    df["SMA_short"] = df["Close"].rolling(short).mean()
    df["SMA_long"] = df["Close"].rolling(long).mean()
    return df


def detect_crossovers(df):
    diff = df["SMA_short"] - df["SMA_long"]
    sign = np.sign(diff)
    prev = sign.shift(1)
    golden = (prev <= 0) & (sign > 0)
    death = (prev >= 0) & (sign < 0)
    return golden.fillna(False), death.fillna(False)


def detect_order_blocks(df, lookback=120):
    obs = []
    recent = df.tail(lookback)

    for i in range(20, len(recent) - 5):
        candle = recent.iloc[i]

        body = abs(candle["Close"] - candle["Open"])

        avg_body = (
            abs(recent["Close"] - recent["Open"])
            .rolling(20)
            .mean()
            .iloc[i]
        )

        if pd.isna(avg_body) or avg_body == 0:
            continue

        bearish = candle["Close"] < candle["Open"]
        large_body = body > avg_body * 1.5

        if bearish and large_body:
            future_high = recent["Close"].iloc[i + 1:i + 6].max()
            impulse = (future_high / candle["Close"]) - 1

            if impulse > 0.05:
                obs.append({
                    "date": recent.index[i],
                    "low": float(candle["Low"]),
                    "high": float(candle["Open"]),
                    "strength": round(impulse * 100, 2),
                })

    return obs


def detect_fvg(df):
    gaps = []

    for i in range(2, len(df)):
        c1 = df.iloc[i - 2]
        c3 = df.iloc[i]

        if c3["Low"] > c1["High"]:
            gaps.append({
                "type": "bullish",
                "low": float(c1["High"]),
                "high": float(c3["Low"]),
            })

    return gaps


def detect_bos(df):
    recent_high = df["High"].rolling(20).max().shift(1)
    if pd.isna(recent_high.iloc[-1]):
        return False
    return bool(df["Close"].iloc[-1] > recent_high.iloc[-1])


def institutional_score(df):
    score = 0

    if len(detect_order_blocks(df)):
        score += 40

    if detect_bos(df):
        score += 30

    if len(detect_fvg(df)):
        score += 30

    return min(score, 100)


def backtest_forward(df, golden, horizon, target):
    closes = df["Close"].to_numpy()
    idxs = np.where(golden.to_numpy())[0]

    hits = 0
    total = 0

    for i in idxs:
        end = i + horizon

        if end >= len(closes):
            continue

        entry = closes[i]
        final_ret = closes[end] / entry - 1.0

        total += 1

        if final_ret >= target:
            hits += 1

    return {
        "signals": total,
        "hit_rate": 100 * hits / total if total else 0
    }


def verdict(score, inst_score, hit_rate):

    if score >= 80 and inst_score >= 70 and hit_rate >= 60:
        return "BUY"

    if score >= 60:
        return "NEUTRAL"

    return "CAUTION"


def screen_stock(ticker):
    df = download_history(ticker)

    if len(df) < 250:
        return None

    df = add_indicators(df)

    golden, _ = detect_crossovers(df)

    bt = backtest_forward(
        df,
        golden,
        horizon=6 * TRADING_DAYS_PER_MONTH,
        target=0.20,
    )

    inst = institutional_score(df)

    momentum = (
        df["Close"].iloc[-1] /
        df["Close"].iloc[-63] - 1
    ) * 100

    score = (
        0.40 * bt["hit_rate"]
        + 0.20 * min(momentum / 30, 1) * 100
        + 0.40 * inst
    )

    return {
        "ticker": ticker,
        "score": round(score, 1),
        "institutional_score": inst,
        "hit_rate": round(bt["hit_rate"], 1),
        "order_blocks": len(detect_order_blocks(df)),
        "verdict": verdict(score, inst, bt["hit_rate"]),
    }


if __name__ == "__main__":
    universe = ["RELIANCE.NS", "ICICIBANK.NS", "BEL.NS"]

    rows = []

    for t in universe:
        result = screen_stock(t)
        if result:
            rows.append(result)

    print(pd.DataFrame(rows).sort_values("score", ascending=False))
