import time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
)
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="QUANTUM-AI // CRYPTO & SOLANA TERMINAL",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 50% 10%, #0d172b 0%, #05080f 75%);
        color: #d8e4ff;
        font-family: 'Courier New', monospace;
    }
    .agent-card {
        background: #070c17;
        border: 1px solid #16223a;
        border-radius: 6px;
        padding: 9px 11px;
        margin-bottom: 8px;
    }
    .verdict-box {
        text-align: center;
        background: rgba(7, 12, 24, 0.92);
        border: 1px solid #1f3356;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .big-green { color: #00ffa3; font-size: 25px; font-weight: 900; }
    .big-red { color: #ff3b5c; font-size: 25px; font-weight: 900; }
    .big-amber { color: #ffb800; font-size: 25px; font-weight: 900; }
    .muted { color: #6c80a6; font-size: 11px; }
</style>
""", unsafe_allow_html=True)

if "paper_balance" not in st.session_state:
    st.session_state.paper_balance = 10000.0
if "paper_positions" not in st.session_state:
    st.session_state.paper_positions = []

PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "DOGEUSDT", "PEPEUSDT", "AVAXUSDT",
    "ADAUSDT", "LINKUSDT", "SUIUSDT", "TONUSDT",
]
BINANCE_BASES = ["https://data-api.binance.vision", "https://api.binance.com"]

@st.cache_data(ttl=25)
def fetch_market_data(symbol: str, interval: str, limit: int = 500):
    for base in BINANCE_BASES:
        try:
            kr = requests.get(f"{base}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}", timeout=5)
            dr = requests.get(f"{base}/api/v3/depth?symbol={symbol}&limit=25", timeout=5)
            tr = requests.get(f"{base}/api/v3/ticker/24hr", timeout=5)

            if kr.status_code == 200 and dr.status_code == 200:
                df = pd.DataFrame(kr.json(), columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"
                ])
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)
                df["time"] = pd.to_datetime(df["open_time"], unit="ms")

                raw_d = dr.json()
                ob = {
                    "bids": [(float(p), float(q)) for p, q in raw_d["bids"]],
                    "asks": [(float(p), float(q)) for p, q in raw_d["asks"]],
                }
                tickers = {}
                if tr.status_code == 200:
                    for item in tr.json():
                        if item["symbol"] in PAIRS:
                            tickers[item["symbol"]] = {
                                "price": float(item["lastPrice"]),
                                "change": float(item["priceChangePercent"]),
                            }
                return df, ob, tickers, True
        except Exception:
            continue

    np.random.seed(int(time.time() // 30))
    bases = {
        "BTCUSDT": 89500.0, "ETHUSDT": 3120.0, "SOLUSDT": 195.0, "BNBUSDT": 620.0,
        "XRPUSDT": 2.35, "DOGEUSDT": 0.31, "PEPEUSDT": 0.000018, "AVAXUSDT": 34.0,
        "ADAUSDT": 0.78, "LINKUSDT": 16.5, "SUIUSDT": 3.10, "TONUSDT": 5.40,
    }
    p0 = bases.get(symbol, 100.0)
    rets = np.random.normal(0.0001, 0.0045, limit)
    closes = p0 * np.cumprod(1 + rets)
    opens = np.roll(closes, 1)
    opens[0] = p0
    highs = np.maximum(opens, closes) * (1 + np.random.uniform(0, 0.003, limit))
    lows = np.minimum(opens, closes) * (1 - np.random.uniform(0, 0.003, limit))
    vols = np.random.lognormal(mean=5, sigma=0.6, size=limit)
    df = pd.DataFrame({
        "time": pd.date_range(end=pd.Timestamp.utcnow(), periods=limit, freq="5min"),
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols,
    })
    last_p = closes[-1]
    ob = {
        "bids": [(last_p * (1 - i * 0.0005), float(np.random.uniform(2, 20))) for i in range(1, 21)],
        "asks": [(last_p * (1 + i * 0.0005), float(np.random.uniform(2, 20))) for i in range(1, 21)],
    }
    tickers = {s: {"price": bases[s], "change": float(np.random.uniform(-5, 5))} for s in PAIRS}
    tickers[symbol]["price"] = float(last_p)
    return df, ob, tickers, False


@st.cache_data(ttl=300)
def fetch_fear_and_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=7&format=json", timeout=5)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                return int(data[0]["value"]), data[0]["value_classification"], [int(x["value"]) for x in reversed(data)]
    except Exception:
        pass
    return 50, "Neutral (Offline)", [48, 52, 50, 49, 51, 50, 50]


@st.cache_data(ttl=60)
def fetch_solana_memecoins():
    try:
        b_res = requests.get("https://api.dexscreener.com/token-boosts/latest/v1", timeout=6)
        if b_res.status_code != 200:
            return []
        boosts = [b for b in b_res.json() if b.get("chainId") == "solana"]
        addrs = []
        for b in boosts:
            addr = b.get("tokenAddress")
            if addr and addr not in addrs:
                addrs.append(addr)
            if len(addrs) >= 15:
                break
        if not addrs:
            return []

        p_res = requests.get(f"https://api.dexscreener.com/latest/dex/tokens/{','.join(addrs)}", timeout=6)
        if p_res.status_code != 200:
            return []
        pairs = p_res.json().get("pairs") or []

        best_by_token = {}
        for pr in pairs:
            if pr.get("chainId") != "solana":
                continue
            addr = (pr.get("baseToken") or {}).get("address")
            liq = float((pr.get("liquidity") or {}).get("usd") or 0)
            if addr and (addr not in best_by_token or liq > best_by_token[addr]["liq"]):
                tx1h = (pr.get("txns") or {}).get("h1") or {}
                buys = int(tx1h.get("buys") or 0)
                sells = int(tx1h.get("sells") or 0)
                vol24 = float((pr.get("volume") or {}).get("h24") or 0)
                fdv = float(pr.get("fdv") or pr.get("marketCap") or 0)
                chg1h = float((pr.get("priceChange") or {}).get("h1") or 0)

                liq_ratio = (liq / fdv) if fdv > 0 else 0
                buy_ratio = buys / max(buys + sells, 1)
                score = 50.0
                if liq >= 25000: score += 15
                elif liq < 5000: score -= 30
                if liq_ratio >= 0.08: score += 12
                elif liq_ratio < 0.02: score -= 20
                if buy_ratio > 0.56 and (buys + sells) > 30: score += 15
                elif buy_ratio < 0.42: score -= 15
                if vol24 > 100000: score += 10
                score = max(0, min(100, score))

                verdict = "BUY / SPECULATE" if score >= 68 else "WATCH" if score >= 48 else "SKIP / RUG RISK"
                best_by_token[addr] = {
                    "symbol": (pr.get("baseToken") or {}).get("symbol", "UNKNOWN"),
                    "dex": pr.get("dexId", "solana"),
                    "price": float(pr.get("priceUsd") or 0),
                    "liq": liq,
                    "fdv": fdv,
                    "vol24": vol24,
                    "chg1h": chg1h,
                    "buys_sells": f"{buys}/{sells}",
                    "ai_score": round(score, 1),
                    "verdict": verdict,
                    "url": pr.get("url", ""),
                }
        return sorted(best_by_token.values(), key=lambda x: x["ai_score"], reverse=True)
    except Exception:
        return []


@st.cache_data(ttl=180)
def fetch_crypto_news():
    bull_words = {"surge", "rally", "etf", "adoption", "bull", "soar", "gain", "record", "approve", "boost", "institutional", "buy"}
    bear_words = {"crash", "dump", "hack", "ban", "sec", "lawsuit", "bear", "drop", "plunge", "wipe", "fear", "scam", "liquidation"}
    items = []
    net_sent = 0
    try:
        r = requests.get("https://cointelegraph.com/rss", timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            for node in root.findall(".//item")[:10]:
                title = (node.findtext("title") or "").strip()
                link = (node.findtext("link") or "").strip()
                pub = (node.findtext("pubDate") or "").strip()
                words = set(title.lower().replace(",", "").replace(".", "").split())
                s = len(words & bull_words) - len(words & bear_words)
                net_sent += s
                tag = "BULLISH" if s > 0 else "BEARISH" if s < 0 else "NEUTRAL"
                items.append({"title": title, "link": link, "pub": pub, "tag": tag})
    except Exception:
        pass
    return items, net_sent


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["close"]
    d["ret_1"] = c.pct_change(1)
    d["ret_3"] = c.pct_change(3)
    d["ret_6"] = c.pct_change(6)
    d["ret_12"] = c.pct_change(12)
    d["body_pct"] = (d["close"] - d["open"]) / (d["open"] + 1e-12)
    d["wick_ratio"] = (d["high"] - d["low"]) / (np.abs(d["close"] - d["open"]) + 1e-12)

    d["ema9"] = c.ewm(span=9, adjust=False).mean()
    d["ema21"] = c.ewm(span=21, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()
    d["ema_spread_9_21"] = (d["ema9"] - d["ema21"]) / (c + 1e-12)
    d["dist_ema50"] = (c - d["ema50"]) / (c + 1e-12)

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-12)))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["macd"] = (ema12 - ema26) / (c + 1e-12)
    d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]

    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    d["bb_z"] = (c - ma20) / (std20 + 1e-12)
    d["bb_width"] = (4 * std20) / (ma20 + 1e-12)

    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - c.shift(1)).abs(),
        (d["low"] - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["atr_pct"] = d["atr"] / (c + 1e-12)

    vol_ma = d["volume"].rolling(20).mean()
    vol_std = d["volume"].rolling(20).std()
    d["vol_z"] = (d["volume"] - vol_ma) / (vol_std + 1e-12)
    d["target"] = (c.shift(-1) > c).astype(int)
    return d


def run_ai_models(df_feat: pd.DataFrame, ob: dict):
    feature_cols = [
        "ret_1", "ret_3", "ret_6", "ret_12", "body_pct", "wick_ratio",
        "ema_spread_9_21", "dist_ema50", "rsi", "macd", "macd_hist",
        "bb_z", "bb_width", "atr_pct", "vol_z",
    ]
    clean = df_feat.dropna().copy()
    latest_row = clean.iloc[-1:]
    train_df = clean.iloc[:-1]

    X = train_df[feature_cols].values
    y = train_df["target"].values
    X_live = latest_row[feature_cols].values

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_live_s = scaler.transform(X_live)

    if len(np.unique(y_train)) < 2:
        y_train[0] = 1 - y_train[0]

    rf = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42).fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_prob = float(rf.predict_proba(X_live)[0][1])

    gb = HistGradientBoostingClassifier(max_iter=100, max_depth=4, learning_rate=0.05, random_state=42).fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test))
    gb_prob = float(gb.predict_proba(X_live)[0][1])

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.02, max_iter=350, random_state=42).fit(X_train_s, y_train)
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test_s))
    mlp_prob = float(mlp.predict_proba(X_live_s)[0][1])

    cluster_feats = scaler.transform(X)[:, [12, 13, 14]]
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42).fit(cluster_feats)
    live_c = int(kmeans.predict(X_live_s[:, [12, 13, 14]])[0])

    iso = IsolationForest(contamination=0.05, random_state=42).fit(X_train_s)
    is_anomaly = int(iso.predict(X_live_s)[0]) == -1

    atr_pct_now = float(latest_row["atr_pct"].iloc[0]) * 100
    ema_spread_now = float(latest_row["ema_spread_9_21"].iloc[0])
    if is_anomaly:
        regime = f"ANOMALY SHOCK (C{live_c})"
    elif atr_pct_now > 0.9:
        regime = f"HIGH VOLATILITY (C{live_c})"
    elif abs(ema_spread_now) > 0.002:
        regime = ("BULL TREND" if ema_spread_now > 0 else "BEAR TREND") + f" (C{live_c})"
    else:
        regime = f"RANGING / CHOP (C{live_c})"

    reg_prob = float(0.5 + np.tanh(ema_spread_now * 120) * 0.35)

    bid_vol = sum(q for _, q in ob["bids"])
    ask_vol = sum(q for _, q in ob["asks"])
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-12)
    ob_prob = float(0.5 + np.tanh(imbalance * 1.8) * 0.45)
    veto = is_anomaly and atr_pct_now > 1.5

    ens_prob = rf_prob * 0.25 + gb_prob * 0.28 + mlp_prob * 0.25 + reg_prob * 0.10 + ob_prob * 0.12
    test_probs = (
        rf.predict_proba(X_test)[:, 1] * 0.35
        + gb.predict_proba(X_test)[:, 1] * 0.35
        + mlp.predict_proba(X_test_s)[:, 1] * 0.30
    )
    ens_acc = accuracy_score(y_test, (test_probs >= 0.5).astype(int))
    test_rets = train_df["ret_1"].iloc[split_idx:].shift(-1).fillna(0).values
    strat_dir = np.where(test_probs > 0.54, 1, np.where(test_probs < 0.46, -1, 0))
    equity_curve = np.cumprod(1 + strat_dir * test_rets) * 10000

    if veto: verdict = "VETO · ANOMALY RISK"
    elif ens_prob >= 0.62: verdict = "STRONG BUY"
    elif ens_prob >= 0.54: verdict = "BUY (LONG)"
    elif ens_prob <= 0.38: verdict = "STRONG SELL"
    elif ens_prob <= 0.46: verdict = "SELL (SHORT)"
    else: verdict = "WAIT / NEUTRAL"

    last_price = float(latest_row["close"].iloc[0])
    atr_val = float(latest_row["atr"].iloc[0])
    is_short = ens_prob < 0.5
    tp = last_price - atr_val * 2.6 if is_short else last_price + atr_val * 2.6
    sl = last_price + atr_val * 1.4 if is_short else last_price - atr_val * 1.4

    return {
        "last_price": last_price, "rsi": float(latest_row["rsi"].iloc[0]),
        "atr_pct": atr_pct_now, "imbalance": imbalance, "regime": regime,
        "is_anomaly": is_anomaly, "veto": veto, "ensemble_prob_up": ens_prob,
        "score": (ens_prob - 0.5) * 2, "verdict": verdict, "tp": tp, "sl": sl,
        "ens_acc": ens_acc, "equity_curve": equity_curve,
        "importances": dict(zip(feature_cols, rf.feature_importances_)),
        "agents": [
            {"name": "01 // RANDOM FOREST AI", "prob": rf_prob, "info": f"Test Acc: {rf_acc*100:.1f}% | 120 Trees"},
            {"name": "02 // GRADIENT BOOSTING AI", "prob": gb_prob, "info": f"Test Acc: {gb_acc*100:.1f}% | Boosted"},
            {"name": "03 // NEURAL NETWORK (MLP)", "prob": mlp_prob, "info": f"Test Acc: {mlp_acc*100:.1f}% | Layers 64-32"},
            {"name": "04 // KMEANS & ANOMALY AI", "prob": reg_prob, "info": regime},
            {"name": "05 // ORDERBOOK & RISK VETO", "prob": ob_prob, "info": f"OB Imb: {imbalance*100:+.1f}% | {'VETO!' if veto else 'Risk OK'}"},
            {"name": "06 // META-ENSEMBLE CORE", "prob": ens_prob, "info": f"OOS Ensemble Acc: {ens_acc*100:.1f}%"},
        ],
    }


def ask_llm_analyst(api_key: str, base_url: str, model_name: str, symbol: str, res: dict, fng_val: int, fng_label: str, news_sent: int, lang: str):
    prompt = (
        f"You are an expert quantitative crypto analyst. Analyze {symbol}:\n"
        f"- Price: ${res['last_price']}, Ensemble P(UP): {res['ensemble_prob_up']*100:.1f}% -> {res['verdict']}\n"
        f"- Regime: {res['regime']}, RSI: {res['rsi']:.1f}, ATR%: {res['atr_pct']:.2f}%\n"
        f"- Crypto Fear & Greed Index: {fng_val}/100 ({fng_label}), News Sentiment Score: {news_sent}\n"
        f"- Suggested TP: ${res['tp']}, SL: ${res['sl']}\n"
        f"Give a concise 4-bullet trading & risk assessment in {lang}."
    )
    r = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 350},
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def make_orb_figure(score: float, prob_up: float):
    n = 260
    phi = np.arccos(1 - 2 * np.linspace(0.01, 0.99, n))
    theta = np.pi * (1 + 5 ** 0.5) * np.arange(n)
    r = 1.0 + 0.12 * np.sin(theta * 3 + score * 4)
    color = "#00ffa3" if prob_up >= 0.54 else "#ff3b5c" if prob_up <= 0.46 else "#ffb800"
    fig = go.Figure(data=[
        go.Scatter3d(
            x=r * np.sin(phi) * np.cos(theta),
            y=r * np.sin(phi) * np.sin(theta),
            z=r * np.cos(phi),
            mode="markers",
            marker=dict(size=3.5, color=r * np.cos(phi), colorscale=[[0, "#00f0ff"], [0.5, color], [1, "#ffffff"]], opacity=0.85),
            hoverinfo="none",
        )
    ])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), height=280, showlegend=False,
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def make_radar_figure(ob: dict):
    bids, asks = ob["bids"][:12], ob["asks"][:12]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[q for _, q in bids], theta=np.linspace(190, 350, len(bids)),
        mode="markers+lines", name="BIDS", line=dict(color="#00ffa3", width=1),
        marker=dict(size=7, color="#00ffa3"), fill="toself", fillcolor="rgba(0,255,163,0.12)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[q for _, q in asks], theta=np.linspace(10, 170, len(asks)),
        mode="markers+lines", name="ASKS", line=dict(color="#ff3b5c", width=1),
        marker=dict(size=7, color="#ff3b5c"), fill="toself", fillcolor="rgba(255,59,92,0.12)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=230,
        margin=dict(l=20, r=20, t=20, b=20),
        polar=dict(bgcolor="#070c17", radialaxis=dict(visible=True, gridcolor="#182742", tickfont=dict(size=8, color="#6c80a6")), angularaxis=dict(visible=False)),
        legend=dict(orientation="h", y=-0.15, font=dict(size=10, color="#d8e4ff")),
    )
    return fig


def make_candle_figure(df_feat: pd.DataFrame):
    view = df_feat.tail(80)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(
        x=view["time"], open=view["open"], high=view["high"], low=view["low"], close=view["close"],
        name="Price", increasing_line_color="#00ffa3", decreasing_line_color="#ff3b5c",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["time"], y=view["ema9"], line=dict(color="#00f0ff", width=1.2), name="EMA 9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["time"], y=view["ema21"], line=dict(color="#ffb800", width=1.2), name="EMA 21"), row=1, col=1)
    fig.add_trace(go.Bar(x=view["time"], y=view["volume"], marker_color="#25406b", name="Volume"), row=2, col=1)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#070c17", height=320,
        margin=dict(l=5, r=5, t=10, b=5), xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.06, font=dict(size=10, color="#d8e4ff")), font=dict(color="#6c80a6", size=10),
    )
    fig.update_xaxes(gridcolor="#121e33")
    fig.update_yaxes(gridcolor="#121e33")
    return fig


def fmt_p(p: float) -> str:
    if p >= 100: return f"${p:,.2f}"
    if p >= 1: return f"${p:,.4f}"
    return f"${p:.8f}"


with st.sidebar:
    st.subheader("🤖 Optional LLM Reasoning Agent")
    st.caption("Works with OpenAI, Groq, OpenRouter, or any OpenAI-compatible API.")
    llm_base = st.text_input("API Base URL", value="https://api.openai.com/v1")
    llm_model = st.text_input("Model Name", value="gpt-4o-mini")
    llm_key = st.text_input("API Key (Optional)", type="password")
    llm_lang = st.selectbox("Report Language", ["Moroccan Darija / Arabic", "English", "French"])

c1, c2, c3, c4 = st.columns([3, 1.2, 1.1, 1.2])
with c1:
    st.markdown("### 🧠 QUANTUM-AI // CRYPTO & SOLANA AI TERMINAL")
with c2:
    symbol = st.selectbox("Pair", PAIRS, index=0, label_visibility="collapsed")
with c3:
    interval = st.selectbox("TF", ["1m", "5m", "15m", "1h", "4h"], index=1, label_visibility="collapsed")
with c4:
    if st.button("⟳ REFRESH AI", use_container_width=True):
        st.cache_data.clear()

fng_val, fng_label, fng_hist = fetch_fear_and_greed()
news_items, news_sent = fetch_crypto_news()

tab1, tab2, tab3 = st.tabs([
    "🧠 AI Terminal (12 Cryptos)",
    "🚀 Solana Memecoin AI (DexScreener)",
    "📰 Crypto News & Fear/Greed AI",
])

with tab1:
    with st.spinner("Training 6 AI models on live crypto candles..."):
        df_raw, ob, tickers, is_live = fetch_market_data(symbol, interval, limit=500)
        df_feat = build_features(df_raw)
        res = run_ai_models(df_feat, ob)

    status_txt = "🟢 LIVE BINANCE" if is_live else "🟡 SIMULATION MODE"
    st.caption(f"{status_txt} | Fear & Greed: **{fng_val}/100 ({fng_label})** | Regime: **{res['regime']}** | OOS Acc: **{res['ens_acc']*100:.1f}%**")

    col_l, col_c, col_r = st.columns([1.15, 1.5, 1.1])
    with col_l:
        st.markdown("**SIX AI AGENTS · LIVE MODEL VOTES**")
        for ag in res["agents"]:
            p = ag["prob"] * 100
            bc = "#00ffa3" if p >= 54 else "#ff3b5c" if p <= 46 else "#ffb800"
            bias = "BULL" if p >= 54 else "BEAR" if p <= 46 else "NEUT"
            st.markdown(f"""
            <div class="agent-card">
                <div style="display:flex;justify-content:space-between;">
                    <span style="font-size:11px;font-weight:bold;">{ag['name']}</span>
                    <span style="color:{bc};font-size:11px;font-weight:bold;">{bias} {p:.1f}%</span>
                </div>
                <div style="background:#131f36;height:5px;border-radius:3px;margin:5px 0;">
                    <div style="width:{p:.0f}%;background:{bc};height:5px;border-radius:3px;"></div>
                </div>
                <div class="muted">{ag['info']}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_c:
        st.plotly_chart(make_orb_figure(res["score"], res["ensemble_prob_up"]), use_container_width=True)
        v_cls = "big-green" if res["ensemble_prob_up"] >= 0.54 else "big-red" if res["ensemble_prob_up"] <= 0.46 else "big-amber"
        st.markdown(f"""
        <div class="verdict-box">
            <div class="muted">ENSEMBLE AI CONSENSUS · P(UP) = {res['ensemble_prob_up']*100:.1f}%</div>
            <div class="{v_cls}">{res['verdict']}</div>
            <div style="font-size:12px;">Price: <b>{fmt_p(res['last_price'])}</b> | TP: <b style="color:#00ffa3">{fmt_p(res['tp'])}</b> | SL: <b style="color:#ff3b5c">{fmt_p(res['sl'])}</b></div>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(make_candle_figure(df_feat), use_container_width=True)

    with col_r:
        st.markdown("**LIQUIDITY ENTANGLEMENT RADAR**")
        st.plotly_chart(make_radar_figure(ob), use_container_width=True)
        st.markdown("**DECISION STACK · 12 CRYPTO PAIRS**")
        for p in PAIRS:
            tk = tickers.get(p, {"price": 0.0, "change": 0.0})
            chg = tk["change"]
            col_c_hex = "#00ffa3" if chg > 0 else "#ff3b5c"
            tag = "BUY" if chg > 1.5 else "SHORT" if chg < -1.5 else "WAIT"
            st.markdown(f"""
            <div class="agent-card" style="display:flex;justify-content:space-between;align-items:center;padding:5px 9px;margin-bottom:5px;">
                <span><b>{p.replace('USDT','')}</b></span>
                <span class="muted">{fmt_p(tk['price'])}</span>
                <span style="color:{col_c_hex};font-weight:bold;">{chg:+.2f}% ({tag})</span>
            </div>
            """, unsafe_allow_html=True)

    b1, b2 = st.columns([1.2, 1])
    with b1:
        imp_s = pd.Series(res["importances"]).sort_values(ascending=True).tail(8)
        fig_diag = make_subplots(rows=1, cols=2, subplot_titles=("Feature Importance", "OOS Test Equity ($)"))
        fig_diag.add_trace(go.Bar(x=imp_s.values, y=imp_s.index, orientation="h", marker_color="#00f0ff"), row=1, col=1)
        fig_diag.add_trace(go.Scatter(y=res["equity_curve"], mode="lines", line=dict(color="#00ffa3", width=2)), row=1, col=2)
        fig_diag.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#070c17", height=230, margin=dict(l=10, r=10, t=30, b=10), showlegend=False, font=dict(color="#d8e4ff", size=10))
        st.plotly_chart(fig_diag, use_container_width=True)

    with b2:
        st.markdown(f"**PAPER TRADING SIMULATOR · BALANCE: `${st.session_state.paper_balance:,.2f}`**")
        pc1, pc2, pc3 = st.columns(3)
        if pc1.button("▲ LONG $1,000", use_container_width=True) and st.session_state.paper_balance >= 1000:
            st.session_state.paper_balance -= 1000
            st.session_state.paper_positions.append({"sym": symbol, "side": "LONG", "entry": res["last_price"], "size": 1000.0})
            st.rerun()
        if pc2.button("▼ SHORT $1,000", use_container_width=True) and st.session_state.paper_balance >= 1000:
            st.session_state.paper_balance -= 1000
            st.session_state.paper_positions.append({"sym": symbol, "side": "SHORT", "entry": res["last_price"], "size": 1000.0})
            st.rerun()
        if pc3.button("✕ CLOSE ALL", use_container_width=True):
            for pos in st.session_state.paper_positions:
                cur = tickers.get(pos["sym"], {}).get("price", res["last_price"])
                pct = (cur - pos["entry"]) / pos["entry"] if pos["side"] == "LONG" else (pos["entry"] - cur) / pos["entry"]
                st.session_state.paper_balance += pos["size"] * (1 + pct)
            st.session_state.paper_positions = []
            st.rerun()

        if not st.session_state.paper_positions:
            st.info("No open paper trades.")
        else:
            for pos in st.session_state.paper_positions:
                cur = tickers.get(pos["sym"], {}).get("price", res["last_price"])
                pct = (cur - pos["entry"]) / pos["entry"] if pos["side"] == "LONG" else (pos["entry"] - cur) / pos["entry"]
                st.write(f"**{pos['sym']}** `{pos['side']}` | Entry: {fmt_p(pos['entry'])} → Now: {fmt_p(cur)} | PnL: **${pos['size']*pct:+.2f} ({pct*100:+.2f}%)**")

    if llm_key and st.button("🤖 Generate LLM Written Report"):
        with st.spinner("Generating LLM report..."):
            try:
                st.success(ask_llm_analyst(llm_key, llm_base, llm_model, symbol, res, fng_val, fng_label, news_sent, llm_lang))
            except Exception as e:
                st.error(f"LLM Error: {e}")

with tab2:
    st.subheader("🚀 Solana Memecoin & DEX AI Risk Scanner (Live via DexScreener)")
    st.caption("Inspects real-time liquidity, FDV ratio, 1H Buy/Sell pressure, and scores Rug-Pull Risk vs Momentum.")
    sol_tokens = fetch_solana_memecoins()
    if not sol_tokens:
        st.warning("Could not fetch live Solana tokens right now (offline or rate-limited). Click Refresh AI.")
    else:
        sol_df = pd.DataFrame(sol_tokens)
        st.dataframe(
            sol_df[["symbol", "verdict", "ai_score", "price", "liq", "fdv", "vol24", "chg1h", "buys_sells", "dex"]],
            use_container_width=True,
            hide_index=True,
        )

with tab3:
    st.subheader("📰 Crypto Macro Sentiment & Live Headlines")
    m1, m2, m3 = st.columns(3)
    m1.metric("Fear & Greed Index", f"{fng_val} / 100", fng_label)
    m2.metric("News Sentiment Score", f"{news_sent:+d}", "Bullish" if news_sent > 0 else "Bearish" if news_sent < 0 else "Neutral")
    m3.metric("Headlines Scanned", len(news_items))

    fig_fg = go.Figure(go.Scatter(y=fng_hist, mode="lines+markers", line=dict(color="#00f0ff", width=2)))
    fig_fg.update_layout(title="7-Day Crypto Fear & Greed Trend", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#070c17", height=220, font=dict(color="#d8e4ff"))
    st.plotly_chart(fig_fg, use_container_width=True)

    for item in news_items:
        color = "#00ffa3" if item["tag"] == "BULLISH" else "#ff3b5c" if item["tag"] == "BEARISH" else "#ffb800"
        st.markdown(f"""
        <div class="agent-card">
            <span style="color:{color};font-weight:bold;">[{item['tag']}]</span>
            <a href="{item['link']}" target="_blank" style="color:#d8e4ff;text-decoration:none;font-weight:bold;"> {item['title']}</a>
            <div class="muted">{item['pub']}</div>
        </div>
        """, unsafe_allow_html=True)
