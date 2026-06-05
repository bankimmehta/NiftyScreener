"""
Nifty 50 SMA-Crossover Screener  +  3-Year Forward Backtest  +  News/Geo Overlay
=================================================================================

What it does (on demand, only when you click "Run screener"):

1. UNIVERSE      : current Nifty 50 constituents (NSE rebalance of Sept 2025).
2. SIGNAL        : per-stock short/long SMA crossover. A "golden cross" (short SMA
                   crossing above long SMA) marks a bullish regime.
3. SHORTLIST     : a stock qualifies for "20%-in-6-months potential" if it is
                   CURRENTLY in a bullish regime AND its OWN 3-year history shows
                   golden crosses frequently reached the target move within the
                   forward horizon. "Potential" = empirical hit-rate, NOT a forecast.
4. BACKTEST      : for every past golden cross over the look-back window, measure
                   peak forward return, whether it hit the target, days-to-target
                   and end-of-window return -> per-stock hit-rate + avg peak return.
5. OVERLAY       : current news headlines (yfinance, always on) + an OPTIONAL Gemini
                   layer (Google-Search-grounded) for news / geopolitics / order-book
                   context on the shortlisted names only.

This is a screening & analysis tool, not investment advice. SMA crossovers are a
trend signal, not a magnitude predictor; treat the hit-rate as a historical
frequency, not a guarantee.

-------------------------------------------------------------------------------
Run:
    pip install streamlit yfinance pandas numpy plotly
    # optional (for the Gemini overlay):
    pip install google-genai
    streamlit run nifty_sma_screener.py
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------- #
#  Universe : current Nifty 50 (NSE constituents as of the Sept-2025 rebalance)  #
#  Keyed by yfinance symbol (NSE listings use the ".NS" suffix).                 #
# ----------------------------------------------------------------------------- #
NIFTY50: dict[str, str] = {
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports & SEZ",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "ASIANPAINT.NS": "Asian Paints",
    "AXISBANK.NS": "Axis Bank",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "BEL.NS": "Bharat Electronics",
    "BHARTIARTL.NS": "Bharti Airtel",
    "CIPLA.NS": "Cipla",
    "COALINDIA.NS": "Coal India",
    "DRREDDY.NS": "Dr. Reddy's Laboratories",
    "EICHERMOT.NS": "Eicher Motors",
    "ETERNAL.NS": "Eternal (ex-Zomato)",
    "GRASIM.NS": "Grasim Industries",
    "HCLTECH.NS": "HCLTech",
    "HDFCBANK.NS": "HDFC Bank",
    "HDFCLIFE.NS": "HDFC Life",
    "HINDALCO.NS": "Hindalco Industries",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ICICIBANK.NS": "ICICI Bank",
    "INDIGO.NS": "InterGlobe Aviation (IndiGo)",
    "INFY.NS": "Infosys",
    "ITC.NS": "ITC",
    "JIOFIN.NS": "Jio Financial Services",
    "JSWSTEEL.NS": "JSW Steel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "M&M.NS": "Mahindra & Mahindra",
    "MARUTI.NS": "Maruti Suzuki",
    "MAXHEALTH.NS": "Max Healthcare",
    "NESTLEIND.NS": "Nestle India",
    "NTPC.NS": "NTPC",
    "ONGC.NS": "Oil & Natural Gas Corp.",
    "POWERGRID.NS": "Power Grid",
    "RELIANCE.NS": "Reliance Industries",
    "SBILIFE.NS": "SBI Life Insurance",
    "SHRIRAMFIN.NS": "Shriram Finance",
    "SBIN.NS": "State Bank of India",
    "SUNPHARMA.NS": "Sun Pharma",
    "TCS.NS": "Tata Consultancy Services",
    "TATACONSUM.NS": "Tata Consumer Products",
    "TMPV.NS": "Tata Motors Passenger Vehicles",
    "TATASTEEL.NS": "Tata Steel",
    "TECHM.NS": "Tech Mahindra",
    "TITAN.NS": "Titan Company",
    "TRENT.NS": "Trent",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "WIPRO.NS": "Wipro",
}

TRADING_DAYS_PER_MONTH = 21  # ~252 / 12


# ============================================================================= #
#  DATA                                                                         #
# ============================================================================= #
@st.cache_data(show_spinner=False, ttl=60 * 60)
def download_history(ticker: str, period: str) -> pd.DataFrame:
    """Download daily OHLCV. Cached so re-runs with identical params are instant."""
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    # yfinance can return a MultiIndex (one level per ticker) on single downloads.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
    return df


@st.cache_data(show_spinner=False, ttl=30 * 60)
def fetch_news(ticker: str, limit: int = 6) -> list[dict]:
    """Best-effort recent headlines. yfinance's news schema has changed over time,
    so we read both the legacy flat shape and the newer nested 'content' shape."""
    out: list[dict] = []
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return out
    for item in raw[:limit]:
        title = item.get("title")
        publisher = item.get("publisher")
        link = item.get("link")
        ts = item.get("providerPublishTime")
        # newer nested shape
        content = item.get("content")
        if content and isinstance(content, dict):
            title = content.get("title") or title
            prov = content.get("provider") or {}
            publisher = (prov.get("displayName") if isinstance(prov, dict) else None) or publisher
            cu = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
            link = (cu.get("url") if isinstance(cu, dict) else None) or link
            ts = content.get("pubDate") or ts
        when = ""
        if isinstance(ts, (int, float)):
            when = datetime.fromtimestamp(ts).strftime("%d %b %Y")
        elif isinstance(ts, str):
            when = ts[:10]
        if title:
            out.append({"title": title, "publisher": publisher or "", "link": link or "", "when": when})
    return out


# ============================================================================= #
#  SIGNAL + BACKTEST                                                            #
# ============================================================================= #
def add_smas(df: pd.DataFrame, short: int, long: int) -> pd.DataFrame:
    df = df.copy()
    df["SMA_short"] = df["Close"].rolling(short).mean()
    df["SMA_long"] = df["Close"].rolling(long).mean()
    return df


def detect_crossovers(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return boolean Series for golden (short crosses above long) and death crosses."""
    diff = df["SMA_short"] - df["SMA_long"]
    sign = np.sign(diff)
    prev = sign.shift(1)
    golden = (prev <= 0) & (sign > 0)
    death = (prev >= 0) & (sign < 0)
    return golden.fillna(False), death.fillna(False)


def backtest_forward(df: pd.DataFrame, golden: pd.Series, horizon: int, target: float) -> pd.DataFrame:
    """For each COMPLETED golden cross, measure the forward window outcome."""
    closes = df["Close"].to_numpy()
    dates = df.index
    idxs = np.where(golden.to_numpy())[0]
    n = len(closes)
    rows = []
    for i in idxs:
        end = i + horizon
        if end >= n:  # forward window not fully observed yet -> exclude from stats
            continue
        entry = closes[i]
        if not np.isfinite(entry) or entry <= 0:
            continue
        window = closes[i + 1: end + 1]
        peak_ret = window.max() / entry - 1.0
        final_ret = window[-1] / entry - 1.0
        hit = bool(peak_ret >= target)
        days_to_hit = None
        if hit:
            days_to_hit = int(np.argmax(window >= entry * (1 + target)) + 1)
        rows.append({
            "cross_date": dates[i].date(),
            "entry": round(float(entry), 2),
            "peak_fwd_return": round(float(peak_ret) * 100, 1),
            "final_fwd_return": round(float(final_ret) * 100, 1),
            "hit_target": hit,
            "days_to_target": days_to_hit,
        })
    return pd.DataFrame(rows)


def current_state(df: pd.DataFrame, golden: pd.Series, recency_days: int) -> dict:
    """Snapshot of the present regime for the most recent bar."""
    last = df.iloc[-1]
    short, long, close = last["SMA_short"], last["SMA_long"], last["Close"]
    bullish_regime = bool(short > long)
    above_short = bool(close > short)
    gap_pct = float((short / long - 1) * 100) if long else np.nan

    g_idx = np.where(golden.to_numpy())[0]
    days_since = int(len(df) - 1 - g_idx[-1]) if len(g_idx) else None
    recent_cross = days_since is not None and days_since <= recency_days

    # simple momentum: trailing ~3-month return
    mom_lb = min(63, len(df) - 1)
    mom_3m = float(close / df["Close"].iloc[-mom_lb - 1] - 1) * 100 if mom_lb > 0 else np.nan

    return {
        "close": round(float(close), 2),
        "bullish_regime": bullish_regime,
        "above_short_sma": above_short,
        "sma_gap_pct": round(gap_pct, 2) if np.isfinite(gap_pct) else np.nan,
        "days_since_golden": days_since,
        "recent_cross": recent_cross,
        "mom_3m_pct": round(mom_3m, 1) if np.isfinite(mom_3m) else np.nan,
    }


def score_stock(state: dict, bt: pd.DataFrame, min_signals: int) -> dict | None:
    """Combine current regime with historical hit-rate into a screening score.
    Returns None if the stock fails the hard gates."""
    n_signals = len(bt)
    if n_signals < min_signals:
        return None
    # must be trending up right now
    if not (state["bullish_regime"] and (state["recent_cross"] or state["above_short_sma"])):
        return None

    hit_rate = float(bt["hit_target"].mean())
    avg_peak = float(bt["peak_fwd_return"].mean())
    avg_final = float(bt["final_fwd_return"].mean())
    med_days = bt.loc[bt["hit_target"], "days_to_target"].median()
    med_days = float(med_days) if pd.notna(med_days) else None

    # normalised momentum component (cap so one runaway name doesn't dominate)
    mom = state["mom_3m_pct"] if np.isfinite(state["mom_3m_pct"]) else 0.0
    mom_norm = max(0.0, min(mom / 30.0, 1.0))           # 0..1, saturates at +30% / 3m
    peak_norm = max(0.0, min(avg_peak / 40.0, 1.0))     # 0..1, saturates at +40% avg peak

    score = 0.50 * hit_rate + 0.30 * peak_norm + 0.20 * mom_norm

    return {
        "n_signals": n_signals,
        "hit_rate": round(hit_rate * 100, 1),
        "avg_peak_return": round(avg_peak, 1),
        "avg_final_return": round(avg_final, 1),
        "median_days_to_target": med_days,
        "score": round(score * 100, 1),
    }


# ============================================================================= #
#  OPTIONAL GEMINI OVERLAY (news / geopolitics / order book)                    #
# ============================================================================= #
def gemini_overlay(company: str, ticker: str, api_key: str, model: str) -> str:
    """Search-grounded qualitative note. Defensive: any failure returns a message
    instead of breaking the run. Requires `pip install google-genai`."""
    try:
        from google import genai
        from google.genai import types
    except Exception:
        return "google-genai not installed. Run: pip install google-genai"

    prompt = (
        f"You are an equity research assistant. For {company} (NSE: {ticker.replace('.NS','')}), "
        "use up-to-date sources and give a concise, structured note covering ONLY:\n"
        "1) Recent company news (last ~4 weeks) and overall tone (Positive / Neutral / Negative).\n"
        "2) Order-book / new-order / capacity / guidance updates if any are public.\n"
        "3) Sector & geopolitical factors that could affect the next ~6 months (tariffs, policy, "
        "commodity/currency, demand).\n"
        "End with one line: 'Net read for 6-month upside: Supportive / Mixed / Cautious'.\n"
        "Be factual, cite nothing you cannot find, and keep it under 180 words. "
        "This is not investment advice."
    )
    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
            ),
        )
        return (resp.text or "").strip() or "No response text returned."
    except Exception as e:
        return f"Gemini call failed ({type(e).__name__}): {e}. Check the model name / API key."


# ============================================================================= #
#  CORE PIPELINE                                                                #
# ============================================================================= #
def run_screener(cfg: dict, progress_cb=None) -> dict:
    period = f"{cfg['lookback_years'] + 1}y"   # +1y buffer so long SMA is valid early on
    horizon = cfg["horizon_months"] * TRADING_DAYS_PER_MONTH
    target = cfg["target_pct"] / 100.0
    min_bars = cfg["long_sma"] + horizon + 5   # need enough history for SMA + one fwd window

    tickers = list(NIFTY50.items())
    if cfg["max_stocks"]:
        tickers = tickers[: cfg["max_stocks"]]

    analysed, shortlisted, skipped, per_stock = [], [], [], {}

    for k, (ticker, company) in enumerate(tickers):
        if progress_cb:
            progress_cb(k / len(tickers), f"Analysing {company} ({ticker})")
        df = download_history(ticker, period)
        if df.empty or len(df) < min_bars:
            skipped.append({"ticker": ticker, "company": company,
                            "reason": f"insufficient history ({0 if df.empty else len(df)} bars; need {min_bars})"})
            continue

        df = add_smas(df, cfg["short_sma"], cfg["long_sma"]).dropna(subset=["SMA_short", "SMA_long"])
        if len(df) < horizon + 5:
            skipped.append({"ticker": ticker, "company": company, "reason": "too few bars after SMA warm-up"})
            continue

        golden, death = detect_crossovers(df)
        bt = backtest_forward(df, golden, horizon, target)
        state = current_state(df, golden, cfg["recency_days"])
        analysed.append(ticker)

        per_stock[ticker] = {
            "company": company, "df": df, "golden": golden, "death": death,
            "backtest": bt, "state": state, "scored": None,
        }

        scored = score_stock(state, bt, cfg["min_signals"])
        if scored and scored["hit_rate"] >= cfg["hit_rate_threshold"]:
            per_stock[ticker]["scored"] = scored
            row = {"ticker": ticker.replace(".NS", ""), "company": company, **scored,
                   "price": state["close"], "mom_3m_%": state["mom_3m_pct"],
                   "days_since_cross": state["days_since_golden"]}
            shortlisted.append(row)

    if progress_cb:
        progress_cb(1.0, "Done")

    shortlist_df = pd.DataFrame(shortlisted)
    if not shortlist_df.empty:
        shortlist_df = shortlist_df.sort_values("score", ascending=False).reset_index(drop=True)

    return {
        "config": cfg,
        "shortlist": shortlist_df,
        "skipped": pd.DataFrame(skipped),
        "per_stock": per_stock,
        "analysed_count": len(analysed),
        "universe_count": len(tickers),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ============================================================================= #
#  CHARTS                                                                       #
# ============================================================================= #
def price_chart(rec: dict, short: int, long: int) -> go.Figure:
    df, golden, death = rec["df"], rec["golden"], rec["death"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(color="#1f77b4", width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_short"], name=f"SMA {short}",
                             line=dict(color="#ff7f0e", width=1.2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_long"], name=f"SMA {long}",
                             line=dict(color="#7f7f7f", width=1.2)))
    g = df.index[golden.to_numpy()]
    d = df.index[death.to_numpy()]
    fig.add_trace(go.Scatter(x=g, y=df.loc[g, "Close"], mode="markers", name="Golden cross",
                             marker=dict(symbol="triangle-up", color="green", size=11)))
    fig.add_trace(go.Scatter(x=d, y=df.loc[d, "Close"], mode="markers", name="Death cross",
                             marker=dict(symbol="triangle-down", color="red", size=11)))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.12), title="Price & SMA crossovers")
    return fig


def distribution_chart(bt: pd.DataFrame, target: float) -> go.Figure:
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Histogram(x=bt["peak_fwd_return"], nbinsx=20, name="Peak fwd return %",
                               marker_color="#2ca02c"))
    fig.add_vline(x=target, line_dash="dash", line_color="black",
                  annotation_text=f"target +{target:.0f}%")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                      title="Distribution of peak forward returns after past golden crosses",
                      xaxis_title="Peak forward return within horizon (%)", yaxis_title="# crosses")
    return fig


# ============================================================================= #
#  STREAMLIT UI                                                                 #
# ============================================================================= #
st.set_page_config(page_title="Nifty 50 SMA Screener", layout="wide")
st.title("Nifty 50 — SMA-Crossover Screener & 3-Year Backtest")
st.caption(
    "Screening & analysis tool, not investment advice. SMA crossover is a trend signal, "
    "not a magnitude predictor; the hit-rate is a historical frequency, not a forecast."
)

with st.sidebar:
    st.header("Parameters")
    short_sma = st.number_input("Short SMA (days)", 5, 100, 50, step=1)
    long_sma = st.number_input("Long SMA (days)", 20, 300, 200, step=5)
    st.divider()
    target_pct = st.slider("Target move (%)", 5, 50, 20, step=1)
    horizon_months = st.slider("Forward horizon (months)", 1, 12, 6, step=1)
    lookback_years = st.slider("Backtest look-back (years)", 1, 7, 3, step=1)
    st.divider()
    st.subheader("Shortlist gates")
    hit_rate_threshold = st.slider("Min historical hit-rate (%)", 0, 100, 50, step=5)
    min_signals = st.number_input("Min # past golden crosses", 1, 20, 2, step=1)
    recency_days = st.number_input("Treat cross as 'recent' if within (days)", 5, 250, 60, step=5)
    st.divider()
    st.subheader("Speed / scope")
    max_stocks = st.number_input("Limit universe (0 = all 50)", 0, 50, 0, step=1)
    st.divider()
    st.subheader("Optional Gemini overlay")
    use_gemini = st.checkbox("Add news / geo / order-book note for shortlist", value=False)
    gemini_key = st.text_input("Gemini API key", type="password", disabled=not use_gemini)
    gemini_model = st.text_input("Gemini model", value="gemini-2.5-flash", disabled=not use_gemini)

    if short_sma >= long_sma:
        st.error("Short SMA must be smaller than Long SMA.")

cfg = dict(
    short_sma=int(short_sma), long_sma=int(long_sma), target_pct=int(target_pct),
    horizon_months=int(horizon_months), lookback_years=int(lookback_years),
    hit_rate_threshold=int(hit_rate_threshold), min_signals=int(min_signals),
    recency_days=int(recency_days), max_stocks=int(max_stocks),
)

# ---- The ONLY trigger. Nothing runs on load or on widget changes. ----------- #
run = st.button("Run screener", type="primary", disabled=(short_sma >= long_sma))

if run:
    bar = st.progress(0.0, text="Starting…")

    def cb(frac, msg):
        bar.progress(min(frac, 1.0), text=msg)

    t0 = time.time()
    st.session_state["results"] = run_screener(cfg, progress_cb=cb)
    st.session_state["results"]["elapsed"] = round(time.time() - t0, 1)
    bar.empty()

# ---- Render persisted results (survives later UI interaction) --------------- #
res = st.session_state.get("results")
if not res:
    st.info("Set your parameters in the sidebar and click **Run screener**. "
            "It runs only on demand — no auto-refresh.")
    st.stop()

c = res["config"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Universe", res["universe_count"])
m2.metric("Analysed", res["analysed_count"])
m3.metric("Shortlisted", 0 if res["shortlist"].empty else len(res["shortlist"]))
m4.metric("Run time (s)", res.get("elapsed", "—"))
st.caption(
    f"Signal: SMA {c['short_sma']}/{c['long_sma']}  •  Target +{c['target_pct']}% within "
    f"{c['horizon_months']}m  •  Look-back {c['lookback_years']}y  •  "
    f"Min hit-rate {c['hit_rate_threshold']}%  •  Generated {res['generated_at']}"
)

st.subheader("Shortlist — currently bullish + historically reached the target")
if res["shortlist"].empty:
    st.warning("No stocks passed the gates. Try a lower target, lower hit-rate threshold, "
               "a faster SMA pair (e.g. 20/50), or a longer look-back.")
else:
    st.dataframe(res["shortlist"], use_container_width=True, hide_index=True)
    st.download_button(
        "Download shortlist (CSV)",
        res["shortlist"].to_csv(index=False).encode(),
        file_name=f"nifty_sma_shortlist_{res['generated_at'].replace(' ', '_').replace(':','')}.csv",
        mime="text/csv",
    )
    st.caption(
        "score = 0.50·hit-rate + 0.30·(avg peak return, capped) + 0.20·(3-month momentum, capped). "
        "hit_rate = % of past golden crosses that reached the target within the horizon."
    )

    # ---- Per-stock detail ---------------------------------------------------- #
    st.subheader("Detail")
    choices = {f"{r['company']} ({r['ticker']})": r["ticker"] + ".NS"
               for _, r in res["shortlist"].iterrows()}
    pick = st.selectbox("Choose a shortlisted stock", list(choices.keys()))
    tkr = choices[pick]
    rec = res["per_stock"][tkr]
    s, bt = rec["state"], rec["backtest"]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Last close", f"₹{s['close']:,}")
    d2.metric("Days since golden cross", s["days_since_golden"])
    d3.metric("3-month momentum", f"{s['mom_3m_pct']}%")
    d4.metric("SMA gap (short vs long)", f"{s['sma_gap_pct']}%")

    st.plotly_chart(price_chart(rec, c["short_sma"], c["long_sma"]), use_container_width=True)

    if not bt.empty:
        st.plotly_chart(distribution_chart(bt, c["target_pct"]), use_container_width=True)
        ch, ci, cj = st.columns(3)
        ch.metric("Hit-rate", f"{rec['scored']['hit_rate']}%",
                  help="Past golden crosses that reached the target within the horizon")
        ci.metric("Avg peak fwd return", f"{rec['scored']['avg_peak_return']}%")
        med = rec["scored"]["median_days_to_target"]
        cj.metric("Median days to target", "—" if med is None else int(med))
        with st.expander("Per-crossover backtest table"):
            st.dataframe(bt, use_container_width=True, hide_index=True)

    # ---- News + optional Gemini overlay ------------------------------------- #
    st.markdown("**Recent headlines**")
    news = fetch_news(tkr)
    if news:
        for n in news:
            meta = " · ".join(x for x in [n["publisher"], n["when"]] if x)
            if n["link"]:
                st.markdown(f"- [{n['title']}]({n['link']})  \n  <small>{meta}</small>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"- {n['title']}  \n  <small>{meta}</small>", unsafe_allow_html=True)
    else:
        st.caption("No headlines returned by the data source right now.")

    if use_gemini and gemini_key:
        with st.spinner("Gemini: news / geopolitics / order-book…"):
            note = gemini_overlay(rec["company"], tkr, gemini_key, gemini_model)
        st.markdown("**Gemini overlay (news / geo / order book)**")
        st.info(note)
    elif use_gemini and not gemini_key:
        st.caption("Enter a Gemini API key in the sidebar to enable the overlay.")

# ---- Skipped names (transparency) ------------------------------------------- #
if not res["skipped"].empty:
    with st.expander(f"Skipped ({len(res['skipped'])}) — insufficient/clean history"):
        st.caption("Common cause: recent symbol changes (e.g. Tata Motors demerger -> TMPV, "
                   "Zomato -> ETERNAL, Jio Financial) lack a clean multi-year series.")
        st.dataframe(res["skipped"], use_container_width=True, hide_index=True)