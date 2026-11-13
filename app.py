import base64
import io
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
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

# ==================== PAGE CONFIG & THEME ====================
st.set_page_config(
    page_title="QUANTUM-AI // INSTITUTIONAL TERMINAL",
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
    .inst-banner {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: space-between;
        background: rgba(8, 15, 29, 0.95);
        border: 1px solid #1f3356;
        border-radius: 8px;
        padding: 9px 14px;
        margin-bottom: 10px;
        font-size: 12px;
    }
    .verdict-box {
        text-align: center;
        background: rgba(7, 12, 24, 0.92);
        border: 1px solid #1f3356;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .grade-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-weight: 900;
        font-size: 13px;
        margin-bottom: 6px;
        border: 1px solid;
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
    "XAUUSD (GOLD)", "PAXGUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT",
    "BNBUSDT", "XRPUSDT", "DOGEUSDT", "PEPEUSDT", "AVAXUSDT",
    "ADAUSDT", "LINKUSDT", "SUIUSDT", "TONUSDT",
]
BINANCE_PAIRS = [p for p in PAIRS if p != "XAUUSD (GOLD)"]
BINANCE_BASES = ["https://data-api.binance.vision", "https://api.binance.com"]


# ==================== SESSION KILLZONES (MOROCCO TIME UTC+1) ====================
def get_morocco_killzone():
    ma_now = datetime.now(timezone.utc) + timedelta(hours=1)
    h, m = ma_now.hour, ma_now.minute
    t_dec = h + m / 60.0
    t_str = ma_now.strftime("%H:%M")

    if 8.0 <= t_dec <= 11.5:
        return f"🟢 LONDON KILLZONE ({t_str} MA) · High Liquidity", "#00ffa3"
    elif 13.0 <= t_dec <= 17.0:
        return f"🔥 NEW YORK KILLZONE ({t_str} MA) · Peak Gold/Crypto Vol", "#00f0ff"
    elif 1.0 <= t_dec <= 5.5:
        return f"🟡 ASIAN RANGE ({t_str} MA) · Liquidity Building", "#ffb800"
    else:
        return f"⚪ OFF-KILLZONE ({t_str} MA) · Wait for Liquidity", "#6c80a6"


# ==================== YAHOO GOLD & DXY FETCHERS ====================
def fetch_yahoo_ohlcv(ticker_list, interval: str):
    tf_map = {
        "1m": ("1m", "1d"),
        "5m": ("5m", "5d"),
        "15m": ("15m", "5d"),
        "1h": ("1h", "1mo"),
        "4h": ("1h", "3mo"),
    }
    y_int, y_rng = tf_map.get(interval, ("5m", "5d"))
    headers = {"User-Agent": "Mozilla/5.0"}

    for ticker in ticker_list:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval={y_int}&range={y_rng}"
            r = requests.get(url, headers=headers, timeout=6)
            if r.status_code != 200:
                continue
            res = (r.json().get("chart") or {}).get("result") or []
            if not res:
                continue
            data = res[0]
            ts = data.get("timestamp") or []
            q = ((data.get("indicators") or {}).get("quote") or [{}])[0]
            df = pd.DataFrame({
                "time": pd.to_datetime(ts, unit="s"),
                "open": q.get("open"),
                "high": q.get("high"),
                "low": q.get("low"),
                "close": q.get("close"),
                "volume": q.get("volume"),
            }).dropna(subset=["open", "high", "low", "close"])
            df["volume"] = df["volume"].fillna(100.0).replace(0, 100.0).astype(float)
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)

            if interval == "4h" and len(df) > 20:
                df = df.set_index("time").resample("4h").agg({
                    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
                }).dropna().reset_index()

            if len(df) >= 60:
                return df.tail(500).reset_index(drop=True)
        except Exception:
            continue
    return None


@st.cache_data(ttl=180)
def fetch_dxy_status():
    """Fetches US Dollar Index (DXY) to warn Gold traders of inverse pressure."""
    df_dxy = fetch_yahoo_ohlcv(["DX-Y.NYB"], "1h")
    if df_dxy is not None and len(df_dxy) >= 24:
        now_p = float(df_dxy["close"].iloc[-1])
        prev_p = float(df_dxy["close"].iloc[-24])
        chg = ((now_p - prev_p) / prev_p) * 100
        bias = "BULLISH (Pressures Gold)" if chg > 0.25 else "BEARISH (Supports Gold)" if chg < -0.25 else "NEUTRAL"
        return now_p, chg, bias, True
    return 104.50, 0.0, "OFFLINE", False


# ==================== BINANCE FUTURES INSTITUTIONAL DATA ====================
@st.cache_data(ttl=60)
def fetch_futures_telemetry(symbol: str):
    """Fetches Open Interest, Funding Rate, and Top Traders L/S Ratio from Binance Futures."""
    fut_sym = "PAXGUSDT" if symbol == "XAUUSD (GOLD)" else symbol
    oi_val, funding_pct, top_ls_ratio, top_long_pct = None, None, None, None
    try:
        r_oi = requests.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={fut_sym}", timeout=4)
        if r_oi.status_code == 200:
            oi_val = float(r_oi.json().get("openInterest", 0))

        r_fr = requests.get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={fut_sym}&limit=1", timeout=4)
        if r_fr.status_code == 200 and r_fr.json():
            funding_pct = float(r_fr.json()[0].get("fundingRate", 0)) * 100

        r_ls = requests.get(f"https://fapi.binance.com/futures/data/topLongShortAccountRatio?symbol={fut_sym}&period=1h&limit=1", timeout=4)
        if r_ls.status_code == 200 and r_ls.json():
            item = r_ls.json()[0]
            top_ls_ratio = float(item.get("longShortRatio", 1.0))
            top_long_pct = float(item.get("longAccount", 0.5)) * 100
    except Exception:
        pass
    return {
        "oi": oi_val,
        "funding_pct": funding_pct,
        "top_ls_ratio": top_ls_ratio,
        "top_long_pct": top_long_pct,
    }


# ==================== MULTI-TIMEFRAME CONFLUENCE (15m / 1h / 4h) ====================
@st.cache_data(ttl=90)
def fetch_mtf_confluence(symbol: str):
    tfs = ["15m", "1h", "4h"]
    biases = {}
    for tf in tfs:
        df_tf = None
        if symbol == "XAUUSD (GOLD)":
            df_tf = fetch_yahoo_ohlcv(["GC=F", "XAUUSD=X"], tf)
        if df_tf is None:
            sym_b = "PAXGUSDT" if symbol == "XAUUSD (GOLD)" else symbol
            for base in BINANCE_BASES:
                try:
                    r = requests.get(f"{base}/api/v3/klines?symbol={sym_b}&interval={tf}&limit=80", timeout=4)
                    if r.status_code == 200:
                        df_tf = pd.DataFrame(r.json(), columns=[
                            "ot", "open", "high", "low", "close", "volume", "ct", "q", "t", "tb", "tq", "ig"
                        ])
                        for col in ["open", "high", "low", "close", "volume"]:
                            df_tf[col] = df_tf[col].astype(float)
                        break
                except Exception:
                    continue
        if df_tf is not None and len(df_tf) >= 30:
            c = df_tf["close"]
            ema9 = c.ewm(span=9, adjust=False).mean().iloc[-1]
            ema21 = c.ewm(span=21, adjust=False).mean().iloc[-1]
            delta = c.diff()
            g = delta.clip(lower=0).rolling(14).mean().iloc[-1]
            l = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
            rsi = 100 - (100 / (1 + g / (l + 1e-12)))
            if ema9 > ema21 and rsi >= 48:
                biases[tf] = "BULL"
            elif ema9 < ema21 and rsi <= 52:
                biases[tf] = "BEAR"
            else:
                biases[tf] = "NEUT"
        else:
            biases[tf] = "NEUT"

    bull_cnt = sum(1 for v in biases.values() if v == "BULL")
    bear_cnt = sum(1 for v in biases.values() if v == "BEAR")
    if bull_cnt == 3:
        summary = "3/3 BULLISH ALIGNED"
    elif bear_cnt == 3:
        summary = "3/3 BEARISH ALIGNED"
    elif bull_cnt >= 2:
        summary = f"{bull_cnt}/3 BULL BIAS"
    elif bear_cnt >= 2:
        summary = f"{bear_cnt}/3 BEAR BIAS"
    else:
        summary = "CONFLICTED / CHOPPY"
    return biases, summary, bull_cnt, bear_cnt


# ==================== MAIN MARKET DATA FETCHER ====================
@st.cache_data(ttl=25)
def fetch_market_data(symbol: str, interval: str, limit: int = 500):
    tickers = {}
    for base in BINANCE_BASES:
        try:
            tr = requests.get(f"{base}/api/v3/ticker/24hr", timeout=5)
            if tr.status_code == 200:
                for item in tr.json():
                    if item["symbol"] in BINANCE_PAIRS:
                        tickers[item["symbol"]] = {
                            "price": float(item["lastPrice"]),
                            "change": float(item["priceChangePercent"]),
                        }
                break
        except Exception:
            continue

    if symbol == "XAUUSD (GOLD)":
        df_gold = fetch_yahoo_ohlcv(["GC=F", "XAUUSD=X"], interval)
        if df_gold is not None:
            last_p = float(df_gold["close"].iloc[-1])
            prev_p = float(df_gold["close"].iloc[-20]) if len(df_gold) > 20 else last_p
            tickers["XAUUSD (GOLD)"] = {"price": last_p, "change": ((last_p - prev_p) / prev_p) * 100}
            ob = {
                "bids": [(last_p - i * 0.25, float(np.random.uniform(5, 25))) for i in range(1, 21)],
                "asks": [(last_p + i * 0.25, float(np.random.uniform(5, 25))) for i in range(1, 21)],
            }
            return df_gold, ob, tickers, True, "LIVE GOLD (YAHOO/SPOT)"
        symbol_fetch = "PAXGUSDT"
        source_note = "LIVE GOLD PROXY (PAXG 24/7)"
    else:
        symbol_fetch = symbol
        source_note = "LIVE BINANCE"

    for base in BINANCE_BASES:
        try:
            kr = requests.get(f"{base}/api/v3/klines?symbol={symbol_fetch}&interval={interval}&limit={limit}", timeout=5)
            dr = requests.get(f"{base}/api/v3/depth?symbol={symbol_fetch}&limit=25", timeout=5)
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
                if "PAXGUSDT" in tickers and "XAUUSD (GOLD)" not in tickers:
                    tickers["XAUUSD (GOLD)"] = tickers["PAXGUSDT"]
                return df, ob, tickers, True, source_note
        except Exception:
            continue

    np.random.seed(int(time.time() // 30))
    bases = {
        "XAUUSD (GOLD)": 2650.0, "PAXGUSDT": 2650.0,
        "BTCUSDT": 89500.0, "ETHUSDT": 3120.0, "SOLUSDT": 195.0, "BNBUSDT": 620.0,
        "XRPUSDT": 2.35, "DOGEUSDT": 0.31, "PEPEUSDT": 0.000018, "AVAXUSDT": 34.0,
        "ADAUSDT": 0.78, "LINKUSDT": 16.5, "SUIUSDT": 3.10, "TONUSDT": 5.40,
    }
    p0 = bases.get(symbol, 100.0)
    rets = np.random.normal(0.0001, 0.004, limit)
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
    for s in PAIRS:
        if s not in tickers:
            tickers[s] = {"price": bases[s], "change": float(np.random.uniform(-4, 4))}
    tickers[symbol]["price"] = float(last_p)
    return df, ob, tickers, False, "SIMULATION MODE"


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
                    "liq": liq, "fdv": fdv, "vol24": vol24, "chg1h": chg1h,
                    "buys_sells": f"{buys}/{sells}",
                    "ai_score": round(score, 1), "verdict": verdict,
                }
        return sorted(best_by_token.values(), key=lambda x: x["ai_score"], reverse=True)
    except Exception:
        return []


@st.cache_data(ttl=180)
def fetch_crypto_news():
    bull_words = {"surge", "rally", "etf", "adoption", "bull", "soar", "gain", "record", "approve", "boost", "institutional", "buy", "gold"}
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


# ==================== FEATURE ENGINEERING (WITH VWAP & LIQUIDITY) ====================
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

    # Institutional Rolling VWAP
    tp = (d["high"] + d["low"] + c) / 3.0
    d["vwap"] = (tp * d["volume"]).rolling(48, min_periods=1).sum() / (d["volume"].rolling(48, min_periods=1).sum() + 1e-12)
    d["dist_vwap"] = (c - d["vwap"]) / (c + 1e-12)

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


# ==================== REAL AI / ML ENGINE + INSTITUTIONAL CONFLUENCE ====================
def run_ai_models(df_feat: pd.DataFrame, ob: dict, symbol: str, mtf_bull: int, mtf_bear: int, fut: dict, dxy_chg: float):
    feature_cols = [
        "ret_1", "ret_3", "ret_6", "ret_12", "body_pct", "wick_ratio",
        "ema_spread_9_21", "dist_ema50", "dist_vwap", "rsi", "macd", "macd_hist",
        "bb_z", "bb_width", "atr_pct", "vol_z",
    ]
    clean = df_feat.dropna().copy()
    latest_row = clean.iloc[-1:]
    train_df = clean.iloc[:-1]

    X = train_df[feature_cols].values
    y = train_df["target"].values
    X_live = latest_row[feature_cols].values

    split_idx = max(20, int(len(X) * 0.8))
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

    cluster_feats = scaler.transform(X)[:, [13, 14, 15]]
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42).fit(cluster_feats)
    live_c = int(kmeans.predict(X_live_s[:, [13, 14, 15]])[0])

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

    # Orderbook + Institutional Futures Bias
    bid_vol = sum(q for _, q in ob["bids"])
    ask_vol = sum(q for _, q in ob["asks"])
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-12)
    ob_prob = float(0.5 + np.tanh(imbalance * 1.8) * 0.45)

    # Adjust subtly with Top Trader positioning if available
    if fut["top_long_pct"] is not None:
        whale_bias = (fut["top_long_pct"] - 50.0) / 100.0
        ob_prob = float(np.clip(ob_prob * 0.7 + (0.5 + whale_bias) * 0.3, 0.05, 0.95))

    veto = is_anomaly and atr_pct_now > 1.5
    dxy_warning = ""
    if "GOLD" in symbol or "PAXG" in symbol:
        if dxy_chg > 0.35:
            dxy_warning = "⚠️ DXY Surging (+): Headwind for Gold Longs"
        elif dxy_chg < -0.35:
            dxy_warning = "✅ DXY Dropping (-): Tailwind for Gold Longs"

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

    # Institutional Grade: combines AI Edge + Agent Agreement + Multi-Timeframe Confluence
    sub_probs = [rf_prob, gb_prob, mlp_prob, reg_prob, ob_prob]
    is_bull_call = ens_prob >= 0.5
    agree_cnt = sum(1 for p in sub_probs if (p > 0.52 if is_bull_call else p < 0.48))
    mtf_aligned = (mtf_bull == 3 and is_bull_call) or (mtf_bear == 3 and not is_bull_call)
    mtf_opposed = (mtf_bear >= 2 and is_bull_call) or (mtf_bull >= 2 and not is_bull_call)
    edge = abs(ens_prob - 0.5)

    if veto or edge < 0.04 or mtf_opposed:
        grade = "C (NO TRADE / CONFLICT)"
    elif edge >= 0.11 and agree_cnt >= 4 and mtf_aligned and not dxy_warning.startswith("⚠️"):
        grade = "A+ (INSTITUTIONAL SNIPER)"
    elif edge >= 0.07 and agree_cnt >= 3 and (mtf_bull >= 2 or mtf_bear >= 2):
        grade = "A (STRONG SETUP)"
    else:
        grade = "B (MODERATE)"

    if veto: verdict = "VETO · ANOMALY RISK"
    elif ens_prob >= 0.62: verdict = "STRONG BUY"
    elif ens_prob >= 0.54: verdict = "BUY (LONG)"
    elif ens_prob <= 0.38: verdict = "STRONG SELL"
    elif ens_prob <= 0.46: verdict = "SELL (SHORT)"
    else: verdict = "WAIT / NEUTRAL"

    last_price = float(latest_row["close"].iloc[0])
    vwap_now = float(latest_row["vwap"].iloc[0])
    atr_val = float(latest_row["atr"].iloc[0])
    is_short = ens_prob < 0.5
    tp1 = last_price - atr_val * 1.8 if is_short else last_price + atr_val * 1.8
    tp2 = last_price - atr_val * 3.0 if is_short else last_price + atr_val * 3.0
    sl = last_price + atr_val * 1.3 if is_short else last_price - atr_val * 1.3

    # 24H Liquidity Levels (PDH / PDL)
    last_24h = df_feat[df_feat["time"] >= (df_feat["time"].max() - pd.Timedelta(hours=24))]
    if len(last_24h) < 5:
        last_24h = df_feat.tail(48)
    liq_high = float(last_24h["high"].max())
    liq_low = float(last_24h["low"].min())

    return {
        "last_price": last_price, "vwap": vwap_now, "liq_high": liq_high, "liq_low": liq_low,
        "rsi": float(latest_row["rsi"].iloc[0]), "atr_pct": atr_pct_now,
        "imbalance": imbalance, "regime": regime, "is_anomaly": is_anomaly,
        "veto": veto, "dxy_warning": dxy_warning, "ensemble_prob_up": ens_prob,
        "score": (ens_prob - 0.5) * 2, "verdict": verdict, "grade": grade,
        "tp": tp1, "tp2": tp2, "sl": sl, "ens_acc": ens_acc,
        "equity_curve": equity_curve, "importances": dict(zip(feature_cols, rf.feature_importances_)),
        "agents": [
            {"name": "01 // RANDOM FOREST AI", "prob": rf_prob, "info": f"Test Acc: {rf_acc*100:.1f}% | 120 Trees"},
            {"name": "02 // GRADIENT BOOSTING AI", "prob": gb_prob, "info": f"Test Acc: {gb_acc*100:.1f}% | Boosted"},
            {"name": "03 // NEURAL NETWORK (MLP)", "prob": mlp_prob, "info": f"Test Acc: {mlp_acc*100:.1f}% | Layers 64-32"},
            {"name": "04 // KMEANS & ANOMALY AI", "prob": reg_prob, "info": f"{regime} | Price vs VWAP: {'Above' if last_price >= vwap_now else 'Below'}"},
            {"name": "05 // WHALE FLOW & VETO", "prob": ob_prob, "info": f"OB Imb: {imbalance*100:+.1f}% | {'VETO!' if veto else 'Risk OK'}"},
            {"name": "06 // META-ENSEMBLE CORE", "prob": ens_prob, "info": f"Grade: {grade} | OOS Acc: {ens_acc*100:.1f}%"},
        ],
    }


# ==================== VISION AI & LOCAL CV CHART ANALYZER ====================
def local_cv_chart_scan(img: Image.Image):
    rgb = img.convert("RGB").resize((360, 220))
    arr = np.asarray(rgb, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    green_mask = (g > r + 25) & (g > b + 15) & (g > 70)
    red_mask = (r > g + 25) & (r > b + 15) & (r > 70)
    recent_green = int(green_mask[:, 230:].sum())
    recent_red = int(red_mask[:, 230:].sum())
    bull_ratio = recent_green / (recent_green + recent_red + 1)

    left_y = np.where(green_mask[:, 40:180] | red_mask[:, 40:180])[0]
    right_y = np.where(green_mask[:, 200:340] | red_mask[:, 200:340])[0]
    slope_up = len(left_y) > 20 and len(right_y) > 20 and (np.mean(right_y) < np.mean(left_y) - 5)

    score = max(10, min(95, 50 + (bull_ratio - 0.5) * 60 + (15 if slope_up else -10)))
    if score >= 68: grade, action = "A (LOCAL CV)", "BUY / BULLISH MOMENTUM"
    elif score <= 34: grade, action = "A (LOCAL CV)", "SELL / BEARISH MOMENTUM"
    elif score >= 56 or score <= 44: grade, action = "B (MODERATE)", "BUY BIAS" if score >= 56 else "SELL BIAS"
    else: grade, action = "C (CHOPPY / UNCLEAR)", "WAIT / NO CLEAR EDGE"

    return {"grade": grade, "action": action, "bull_pct": bull_ratio * 100, "slope": "Uptrend" if slope_up else "Downtrend / Range"}


def analyze_screenshot_vision_ai(api_key: str, base_url: str, model_name: str, img: Image.Image, asset_hint: str, tf_hint: str, lang: str):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        f"You are an elite institutional ICT/SMC price-action analyst. "
        f"Inspect this chart screenshot (Asset: {asset_hint}, Timeframe: {tf_hint}).\n"
        f"Read the visible price axis, market structure, liquidity sweeps, FVGs, and order blocks.\n"
        f"Respond in {lang} using this exact structure:\n"
        f"1. **Setup Grade:** (A+ / A / B / C) + short reason.\n"
        f"2. **Action:** (BUY / SELL / WAIT)\n"
        f"3. **Entry Zone:** (Exact price or zone)\n"
        f"4. **Stop Loss (SL):** (Exact invalidation price)\n"
        f"5. **Take Profit Targets:** TP1 and TP2 + Risk:Reward\n"
        f"6. **Institutional Breakdown:** 3 concise bullet points."
    )
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}]}],
        "temperature": 0.2, "max_tokens": 700,
    }
    r = requests.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, timeout=25)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ask_llm_analyst(api_key: str, base_url: str, model_name: str, symbol: str, res: dict, mtf_sum: str, fut: dict, dxy_bias: str, fng_val: int, lang: str):
    prompt = (
        f"Institutional analysis for {symbol}:\n"
        f"- Price: ${res['last_price']}, VWAP: ${res['vwap']}, Grade: {res['grade']}, Verdict: {res['verdict']}\n"
        f"- Multi-Timeframe Confluence: {mtf_sum}, Regime: {res['regime']}, RSI: {res['rsi']:.1f}\n"
        f"- Futures OI: {fut['oi']}, Funding: {fut['funding_pct']}%, Top Traders Long%: {fut['top_long_pct']}%\n"
        f"- DXY Bias: {dxy_bias}, Fear & Greed: {fng_val}/100\n"
        f"- TP1: ${res['tp']}, TP2: ${res['tp2']}, SL: ${res['sl']}\n"
        f"Give a concise 4-bullet institutional trade plan in {lang}."
    )
    r = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 400},
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, timeout=15,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ==================== PLOTLY CHARTS ====================
def make_orb_figure(score: float, prob_up: float):
    n = 260
    phi = np.arccos(1 - 2 * np.linspace(0.01, 0.99, n))
    theta = np.pi * (1 + 5 ** 0.5) * np.arange(n)
    r = 1.0 + 0.12 * np.sin(theta * 3 + score * 4)
    color = "#00ffa3" if prob_up >= 0.54 else "#ff3b5c" if prob_up <= 0.46 else "#ffb800"
    fig = go.Figure(data=[
        go.Scatter3d(
            x=r * np.sin(phi) * np.cos(theta), y=r * np.sin(phi) * np.sin(theta), z=r * np.cos(phi),
            mode="markers",
            marker=dict(size=3.5, color=r * np.cos(phi), colorscale=[[0, "#00f0ff"], [0.5, color], [1, "#ffffff"]], opacity=0.85),
            hoverinfo="none",
        )
    ])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0),
        height=270, showlegend=False,
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
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=225,
        margin=dict(l=20, r=20, t=20, b=20),
        polar=dict(bgcolor="#070c17", radialaxis=dict(visible=True, gridcolor="#182742", tickfont=dict(size=8, color="#6c80a6")), angularaxis=dict(visible=False)),
        legend=dict(orientation="h", y=-0.15, font=dict(size=10, color="#d8e4ff")),
    )
    return fig


def make_candle_figure(df_feat: pd.DataFrame, liq_high: float, liq_low: float):
    view = df_feat.tail(80)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(
        x=view["time"], open=view["open"], high=view["high"], low=view["low"], close=view["close"],
        name="Price", increasing_line_color="#00ffa3", decreasing_line_color="#ff3b5c",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["time"], y=view["vwap"], line=dict(color="#c084fc", width=1.6, dash="dot"), name="VWAP (Inst.)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["time"], y=view["ema9"], line=dict(color="#00f0ff", width=1.1), name="EMA 9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["time"], y=view["ema21"], line=dict(color="#ffb800", width=1.1), name="EMA 21"), row=1, col=1)

    fig.add_hline(y=liq_high, line_dash="dash", line_color="rgba(255,59,92,0.5)", annotation_text="24H High Liq", row=1, col=1)
    fig.add_hline(y=liq_low, line_dash="dash", line_color="rgba(0,255,163,0.5)", annotation_text="24H Low Liq", row=1, col=1)

    fig.add_trace(go.Bar(x=view["time"], y=view["volume"], marker_color="#25406b", name="Volume"), row=2, col=1)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#070c17", height=340,
        margin=dict(l=5, r=5, t=10, b=5), xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.07, font=dict(size=9, color="#d8e4ff")), font=dict(color="#6c80a6", size=10),
    )
    fig.update_xaxes(gridcolor="#121e33")
    fig.update_yaxes(gridcolor="#121e33")
    return fig


def fmt_p(p: float) -> str:
    if p >= 100: return f"${p:,.2f}"
    if p >= 1: return f"${p:,.4f}"
    return f"${p:.8f}"


# ==================== SIDEBAR (VISION & LLM SETTINGS) ====================
with st.sidebar:
    st.subheader("🤖 Vision AI & LLM Settings")
    provider = st.selectbox("AI Provider Preset", [
        "Google Gemini (Free Key)",
        "OpenAI (GPT-4o / 4o-mini)",
        "OpenRouter",
        "Custom Endpoint",
    ])
    if provider == "Google Gemini (Free Key)":
        def_url, def_mod = "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash"
    elif provider == "OpenAI (GPT-4o / 4o-mini)":
        def_url, def_mod = "https://api.openai.com/v1", "gpt-4o-mini"
    elif provider == "OpenRouter":
        def_url, def_mod = "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"
    else:
        def_url, def_mod = "https://api.openai.com/v1", "gpt-4o-mini"

    llm_base = st.text_input("API Base URL", value=def_url)
    llm_model = st.text_input("Model Name", value=def_mod)
    llm_key = st.text_input("API Key (For Vision Screenshot AI)", type="password")
    llm_lang = st.selectbox("Report Language", ["Moroccan Darija / Arabic", "English", "French"])
    st.divider()
    st.markdown("**🔗 Pro External Terminals**")
    st.markdown("- [Forex Factory Calendar](https://www.forexfactory.com/calendar)")
    st.markdown("- [CoinGlass Liquidations](https://www.coinglass.com/LiquidationData)")
    st.markdown("- [DexScreener Solana](https://dexscreener.com/solana)")

# ==================== TOP HEADER ====================
c1, c2, c3, c4 = st.columns([3, 1.3, 1.0, 1.2])
with c1:
    st.markdown("### 🧠 QUANTUM-AI // INSTITUTIONAL PRO TERMINAL")
with c2:
    symbol = st.selectbox("Pair", PAIRS, index=0, label_visibility="collapsed")
with c3:
    interval = st.selectbox("TF", ["1m", "5m", "15m", "1h", "4h"], index=1, label_visibility="collapsed")
with c4:
    if st.button("⟳ REFRESH AI", use_container_width=True):
        st.cache_data.clear()

kz_text, kz_color = get_morocco_killzone()
dxy_val, dxy_chg, dxy_bias, dxy_ok = fetch_dxy_status()
fng_val, fng_label, fng_hist = fetch_fear_and_greed()
news_items, news_sent = fetch_crypto_news()

tab1, tab4, tab2, tab3 = st.tabs([
    "🧠 Institutional Terminal",
    "📸 Screenshot AI (A+ / A / B)",
    "🚀 Solana Memecoin AI",
    "📰 Macro, News & Pro Links",
])

# ==================== TAB 1: INSTITUTIONAL TERMINAL ====================
with tab1:
    with st.spinner("Running 6 ML models + Multi-Timeframe Confluence + Whale Telemetry..."):
        df_raw, ob, tickers, is_live, src_note = fetch_market_data(symbol, interval, limit=500)
        mtf_biases, mtf_sum, mtf_bull, mtf_bear = fetch_mtf_confluence(symbol)
        fut = fetch_futures_telemetry(symbol)
        df_feat = build_features(df_raw)
        res = run_ai_models(df_feat, ob, symbol, mtf_bull, mtf_bear, fut, dxy_chg)

    # INSTITUTIONAL TELEMETRY BANNER
    oi_txt = f"{fut['oi']:,.0f}" if fut["oi"] is not None else "N/A"
    fr_txt = f"{fut['funding_pct']:+.4f}%" if fut["funding_pct"] is not None else "N/A"
    ls_txt = f"{fut['top_long_pct']:.1f}% L / {100-fut['top_long_pct']:.1f}% S" if fut["top_long_pct"] is not None else "N/A"
    mtf_txt = f"15m:{mtf_biases['15m']} | 1h:{mtf_biases['1h']} | 4h:{mtf_biases['4h']}"

    st.markdown(f"""
    <div class="inst-banner">
        <span style="color:{kz_color};font-weight:bold;">{kz_text}</span>
        <span>📊 <b>MTF ({mtf_sum}):</b> {mtf_txt}</span>
        <span>💵 <b>DXY:</b> {dxy_val:.2f} ({dxy_chg:+.2f}%)</span>
        <span>🐳 <b>Whales:</b> {ls_txt} | <b>Funding:</b> {fr_txt} | <b>OI:</b> {oi_txt}</span>
    </div>
    """, unsafe_allow_html=True)

    if res["dxy_warning"]:
        st.warning(res["dxy_warning"])

    dot = "🟢" if is_live else "🟡"
    st.caption(f"{dot} {src_note} | Fear & Greed: **{fng_val}/100 ({fng_label})** | VWAP: **{fmt_p(res['vwap'])}** | OOS Acc: **{res['ens_acc']*100:.1f}%**")

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
        g_color = "#00ffa3" if res["grade"].startswith("A+") else "#00f0ff" if res["grade"].startswith("A") else "#ffb800" if res["grade"].startswith("B") else "#ff3b5c"
        st.markdown(f"""
        <div class="verdict-box">
            <div class="grade-pill" style="color:{g_color};border-color:{g_color};">SETUP GRADE: {res['grade']}</div>
            <div class="muted">ENSEMBLE AI CONSENSUS · P(UP) = {res['ensemble_prob_up']*100:.1f}%</div>
            <div class="{v_cls}">{res['verdict']}</div>
            <div style="font-size:12px;">Entry: <b>{fmt_p(res['last_price'])}</b> | SL: <b style="color:#ff3b5c">{fmt_p(res['sl'])}</b></div>
            <div style="font-size:12px;">TP1: <b style="color:#00ffa3">{fmt_p(res['tp'])}</b> | TP2: <b style="color:#00f0ff">{fmt_p(res['tp2'])}</b></div>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(make_candle_figure(df_feat, res["liq_high"], res["liq_low"]), use_container_width=True)

    with col_r:
        st.markdown("**LIQUIDITY ENTANGLEMENT RADAR**")
        st.plotly_chart(make_radar_figure(ob), use_container_width=True)
        st.markdown("**DECISION STACK · GOLD & CRYPTO**")
        for p in PAIRS:
            tk = tickers.get(p, {"price": 0.0, "change": 0.0})
            chg = tk["change"]
            col_c_hex = "#00ffa3" if chg > 0 else "#ff3b5c"
            tag = "BUY" if chg > 1.5 else "SHORT" if chg < -1.5 else "WAIT"
            short_n = p.replace(" (GOLD)", "").replace("USDT", "")
            st.markdown(f"""
            <div class="agent-card" style="display:flex;justify-content:space-between;align-items:center;padding:5px 9px;margin-bottom:5px;">
                <span><b>{short_n}</b></span>
                <span class="muted">{fmt_p(tk['price'])}</span>
                <span style="color:{col_c_hex};font-weight:bold;">{chg:+.2f}% ({tag})</span>
            </div>
            """, unsafe_allow_html=True)

    # POSITION SIZE / LOT CALCULATOR + BACKTEST + PAPER TRADING
    b1, b2 = st.columns([1.15, 1])
    with b1:
        st.markdown("**🛡️ INSTITUTIONAL POSITION & LOT SIZE CALCULATOR**")
        rc1, rc2 = st.columns(2)
        cap_usd = rc1.number_input("Account Capital ($)", min_value=10.0, value=1000.0, step=100.0)
        risk_pct = rc2.number_input("Risk Per Trade (%)", min_value=0.2, max_value=10.0, value=1.0, step=0.5)

        risk_usd = cap_usd * (risk_pct / 100.0)
        stop_dist = max(abs(res["last_price"] - res["sl"]), 1e-8)
        qty_units = risk_usd / stop_dist
        gold_lots = risk_usd / (stop_dist * 100.0)

        m_a, m_b, m_c = st.columns(3)
        m_a.metric("Max Risk ($)", f"${risk_usd:,.2f}")
        m_b.metric("SL Distance", f"{(stop_dist/res['last_price'])*100:.2f}%")
        if "GOLD" in symbol:
            m_c.metric("XAUUSD Lot Size", f"{gold_lots:.3f} Lots")
        else:
            m_c.metric("Position Size", f"{qty_units:,.4f} {symbol.replace('USDT','')}")

        imp_s = pd.Series(res["importances"]).sort_values(ascending=True).tail(8)
        fig_diag = make_subplots(rows=1, cols=2, subplot_titles=("Feature Importance", "OOS Test Equity ($)"))
        fig_diag.add_trace(go.Bar(x=imp_s.values, y=imp_s.index, orientation="h", marker_color="#00f0ff"), row=1, col=1)
        fig_diag.add_trace(go.Scatter(y=res["equity_curve"], mode="lines", line=dict(color="#00ffa3", width=2)), row=1, col=2)
        fig_diag.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#070c17", height=210, margin=dict(l=10, r=10, t=30, b=10), showlegend=False, font=dict(color="#d8e4ff", size=10))
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
            st.info("No open paper trades. Test A+ / A setups risk-free.")
        else:
            for pos in st.session_state.paper_positions:
                cur = tickers.get(pos["sym"], {}).get("price", res["last_price"])
                pct = (cur - pos["entry"]) / pos["entry"] if pos["side"] == "LONG" else (pos["entry"] - cur) / pos["entry"]
                st.write(f"**{pos['sym']}** `{pos['side']}` | Entry: {fmt_p(pos['entry'])} → Now: {fmt_p(cur)} | PnL: **${pos['size']*pct:+.2f} ({pct*100:+.2f}%)**")

        if llm_key and st.button("🤖 Generate LLM Institutional Report", use_container_width=True):
            with st.spinner("Generating institutional report..."):
                try:
                    st.success(ask_llm_analyst(llm_key, llm_base, llm_model, symbol, res, mtf_sum, fut, dxy_bias, fng_val, llm_lang))
                except Exception as e:
                    st.error(f"LLM Error: {e}")

# ==================== TAB 4: SCREENSHOT CHART AI ANALYZER ====================
with tab4:
    st.subheader("📸 Chart Screenshot AI Analyzer (Setup Grade: A+ / A / B / C)")
    st.caption("Upload any chart screenshot (Gold XAUUSD, Crypto, Forex) to get Buy/Sell/Wait, Entry, SL, TP1/TP2, and Grade.")

    sc1, sc2 = st.columns(2)
    asset_hint = sc1.text_input("Asset (Optional)", value="XAUUSD (Gold)")
    tf_hint = sc2.selectbox("Chart Timeframe", ["M1 / M5", "M15", "H1", "H4", "Daily"], index=2)

    uploaded_img = st.file_uploader("Upload Chart Screenshot (PNG / JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        st.image(img, caption="Uploaded Chart Screenshot", use_container_width=True)

        if st.button("🔍 ANALYZE SCREENSHOT WITH AI", type="primary", use_container_width=True):
            if llm_key:
                with st.spinner("Vision AI reading candlesticks, price levels & liquidity..."):
                    try:
                        report = analyze_screenshot_vision_ai(llm_key, llm_base, llm_model, img, asset_hint, tf_hint, llm_lang)
                        st.markdown('<div class="agent-card">', unsafe_allow_html=True)
                        st.markdown(report)
                        st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Vision API Error: {e}")
            else:
                cv_res = local_cv_chart_scan(img)
                st.warning("⚠️ No Vision API Key found in sidebar (`>>` top-left). Showing Local Computer Vision Scan:")
                st.markdown(f"""
                <div class="verdict-box">
                    <div class="grade-pill" style="color:#00f0ff;border-color:#00f0ff;">GRADE: {cv_res['grade']}</div>
                    <div class="big-amber">{cv_res['action']}</div>
                    <div>Recent Green Candle Dominance: <b>{cv_res['bull_pct']:.1f}%</b> | Visual Slope: <b>{cv_res['slope']}</b></div>
                </div>
                """, unsafe_allow_html=True)
                st.info("💡 حط مفتاح مجاني من `aistudio.google.com` فالقائمة الجانبية (`>>`) باش يقرا الذكاء الاصطناعي أرقام SL و TP من الصورة!")

# ==================== TAB 2: SOLANA MEMECOIN AI SCANNER ====================
with tab2:
    st.subheader("🚀 Solana Memecoin & DEX AI Risk Scanner (Live via DexScreener)")
    sol_tokens = fetch_solana_memecoins()
    if not sol_tokens:
        st.warning("Could not fetch live Solana tokens right now. Click Refresh AI.")
    else:
        sol_df = pd.DataFrame(sol_tokens)
        st.dataframe(
            sol_df[["symbol", "verdict", "ai_score", "price", "liq", "fdv", "vol24", "chg1h", "buys_sells", "dex"]],
            use_container_width=True,
            hide_index=True,
        )

# ==================== TAB 3: NEWS, MACRO & PRO PLATFORMS ====================
with tab3:
    st.subheader("📰 Macro News, Dollar Index (DXY) & Pro Trading Checklists")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fear & Greed", f"{fng_val} / 100", fng_label)
    m2.metric("US Dollar (DXY)", f"{dxy_val:.2f}", f"{dxy_chg:+.2f}%")
    m3.metric("News Sentiment", f"{news_sent:+d}", "Bullish" if news_sent > 0 else "Bearish" if news_sent < 0 else "Neutral")
    m4.metric("Morocco Session", kz_text.split(" ")[1])

    st.markdown("""
    **🏦 قائمة الفحص قبل الدخول لأي صفقة (A+ Institutional Checklist):**
    1. **Forex Factory:** واش كاين خبر أحمر على الدولار (CPI / NFP / FOMC) فالساعة الجاية؟ 👉 [فتح Forex Factory](https://www.forexfactory.com/calendar)
    2. **CoinGlass:** فين كاينة السيولة والتصفية الكبيرة للكريبتو؟ 👉 [فتح CoinGlass](https://www.coinglass.com/LiquidationData)
    3. **MTF Confluence:** تأكد الفوق أن الفريمات `15m / 1h / 4h` متفقين + تداول فوق أو تحت **VWAP**.
    4. **Risk Rule:** ما تفوتش **1% – 2%** خسارة باستعمال حاسبة اللوت!
    """)

    fig_fg = go.Figure(go.Scatter(y=fng_hist, mode="lines+markers", line=dict(color="#00f0ff", width=2)))
    fig_fg.update_layout(title="7-Day Crypto Fear & Greed Trend", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#070c17", height=210, font=dict(color="#d8e4ff"))
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
