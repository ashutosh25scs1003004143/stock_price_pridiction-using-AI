"""
StockPulse AI - Understand Markets. Invest Smarter.
Single-file Streamlit app for Indian (NSE/BSE) market intelligence with Live Yahoo Finance & Google GenAI.

Run:    streamlit run app.py
Install dependencies: pip install streamlit pandas plotly requests yfinance google-genai
Secrets (Streamlit Cloud -> App settings -> Secrets, or environment variables):
    GEMINI_API_KEY      Google Gemini key (AI Analyst)
    DB_PATH             optional SQLite path, default stockpulse.db

Educational use only. Not investment advice.
"""
import datetime as dt
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sqlite3
from abc import ABC, abstractmethod

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
import yfinance as yf
from google import genai
from google.genai import types

st.set_page_config(page_title="StockPulse AI", page_icon="📈", layout="wide")


def cfg(key, default=""):
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


GEMINI_KEY, GEMINI_MODEL = cfg("GEMINI_API_KEY"), cfg("GEMINI_MODEL", "gemini-2.5-flash")
DB_PATH = cfg("DB_PATH", "stockpulse.db")
START_BALANCE = 100000.0

st.markdown("""<style>
.stApp{background:#0B0F19;color:#F8FAFC} [data-testid=stSidebar]{background:#111827}
div[data-testid=stMetric]{background:#151D2B;border:1px solid #263244;border-radius:12px;padding:12px}
.demo{background:#1e293b;color:#38bdf8;padding:6px 12px;border-radius:8px;font-size:13px}
.small{color:#94A3B8;font-size:12px}
</style>""", unsafe_allow_html=True)


# ============================ formatting ============================
def inr(n):
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return "Data unavailable"
    ip, dp = f"{abs(n):.2f}".split(".")
    if len(ip) > 3:
        ip = re.sub(r"\B(?=(\d{2})+$)", ",", ip[:-3]) + "," + ip[-3:]
    return ("-" if n < 0 else "") + "₹" + ip + "." + dp


def pct(n):
    return "—" if n is None else f"{n:+.2f}%"


def fmt_time(iso):
    try:
        return dt.datetime.fromisoformat(iso).strftime("%d %b %Y, %H:%M:%S UTC")
    except Exception:
        return "—"


def now_iso():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


# ============================ database (SQLite) ============================
SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, pw TEXT, exchange TEXT DEFAULT 'NSE', created TEXT);
CREATE TABLE IF NOT EXISTS watchlists(id INTEGER PRIMARY KEY, uid INTEGER, name TEXT);
CREATE TABLE IF NOT EXISTS watch_items(wid INTEGER, symbol TEXT, UNIQUE(wid, symbol));
CREATE TABLE IF NOT EXISTS txs(id INTEGER PRIMARY KEY, uid INTEGER, symbol TEXT, exchange TEXT, qty INTEGER, price REAL, type TEXT, date TEXT);
CREATE TABLE IF NOT EXISTS vacct(uid INTEGER PRIMARY KEY, balance REAL);
CREATE TABLE IF NOT EXISTS vtrades(id INTEGER PRIMARY KEY, uid INTEGER, symbol TEXT, qty INTEGER, price REAL, type TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY, uid INTEGER, symbol TEXT, cond TEXT, target REAL, active INTEGER DEFAULT 1, triggered TEXT);
CREATE TABLE IF NOT EXISTS notifs(id INTEGER PRIMARY KEY, uid INTEGER, msg TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS aireports(id INTEGER PRIMARY KEY, uid INTEGER, symbol TEXT, analysis TEXT, ts TEXT);
CREATE INDEX IF NOT EXISTS i_txs ON txs(uid); CREATE INDEX IF NOT EXISTS i_vt ON vtrades(uid); CREATE INDEX IF NOT EXISTS i_al ON alerts(active);
"""


def db():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def q(sql, args=(), one=False):
    c = db()
    try:
        rows = c.execute(sql, args).fetchall()
    finally:
        c.close()
    return (rows[0] if rows else None) if one else rows


def x(sql, args=()):
    c = db()
    try:
        cur = c.execute(sql, args)
        c.commit()
        return cur
    finally:
        c.close()


# ============================ auth ============================
def hash_pw(pw, salt=None):
    salt = salt or secrets.token_hex(16)
    return salt + "$" + hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()


def check_pw(pw, stored):
    salt, _ = stored.split("$")
    return hmac.compare_digest(hash_pw(pw, salt), stored)


def register(name, email, pw):
    if not name.strip() or not re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", email) or len(pw) < 8:
        return None, "Enter a name, a valid email and a password of at least 8 characters."
    if q("SELECT 1 FROM users WHERE email=?", (email.lower(),), one=True):
        return None, "Email already registered."
    uid = x("INSERT INTO users(name,email,pw,created) VALUES(?,?,?,?)", (name.strip(), email.lower(), hash_pw(pw), now_iso())).lastrowid
    x("INSERT INTO vacct VALUES(?,?)", (uid, START_BALANCE))
    return uid, None


def login(email, pw):
    u = q("SELECT * FROM users WHERE email=?", (email.lower(),), one=True)
    return (u["id"], None) if u and check_pw(pw, u["pw"]) else (None, "Invalid email or password.")


# ============================ market data layer (Yahoo Finance / Live) ============================
UNIVERSE = [
    ("RELIANCE.NS", "Reliance Industries Ltd", "Energy", 2800, 1900000),
    ("TCS.NS", "Tata Consultancy Services Ltd", "IT", 4100, 1500000),
    ("HDFCBANK.NS", "HDFC Bank Ltd", "Banking", 1650, 1250000),
    ("INFY.NS", "Infosys Ltd", "IT", 1850, 780000),
    ("ICICIBANK.NS", "ICICI Bank Ltd", "Banking", 1250, 890000),
    ("ITC.NS", "ITC Ltd", "FMCG", 490, 610000),
    ("SBIN.NS", "State Bank of India", "Banking", 820, 730000),
    ("BHARTIARTL.NS", "Bharti Airtel Ltd", "Telecom", 1600, 950000),
    ("LICI.NS", "Life Insurance Corporation", "Finance", 980, 620000),
    ("TATAMOTORS.NS", "Tata Motors Ltd", "Automobile", 750, 280000),
]
RANGES = {"1D": "1d", "1W": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}


class YahooLiveProvider:
    name, demo = "LIVE (Yahoo Finance API)", False

    def quotes(self):
        out = []
        tickers = [item[0] for item in UNIVERSE]
        data = yf.download(tickers, period="5d", group_by="ticker", threads=True, progress=False)
        for sym, name, sector, default_base, cap in UNIVERSE:
            try:
                df = data[sym] if len(tickers) > 1 else data
                df = df.dropna(subset=["Close"])
                if df.empty:
                    raise ValueError("Empty data")
                curr_price = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else curr_price
                change = curr_price - prev_close
                change_pct = (change / prev_close) * 100 if prev_close else 0.0
                high_52 = float(df["High"].max())
                low_52 = float(df["Low"].min())
                vol = int(df["Volume"].iloc[-1]) if "Volume" in df.columns else 100000
                
                out.append(dict(
                    symbol=sym.replace(".NS", ""), name=name, exchange="NSE", sector=sector,
                    price=round(curr_price, 2), change=round(change, 2), changePercent=round(change_pct, 2),
                    open=round(float(df["Open"].iloc[-1]), 2), high=round(float(df["High"].iloc[-1]), 2),
                    low=round(float(df["Low"].iloc[-1]), 2), prevClose=round(prev_close, 2), volume=vol,
                    week52High=round(high_52, 2), week52Low=round(low_52, 2), marketCap=cap, asOf=now_iso()
                ))
            except Exception:
                out.append(dict(
                    symbol=sym.replace(".NS", ""), name=name, exchange="NSE", sector=sector,
                    price=default_base, change=0.0, changePercent=0.0, open=default_base, high=default_base,
                    low=default_base, prevClose=default_base, volume=500000, week52High=default_base*1.2,
                    week52Low=default_base*0.8, marketCap=cap, asOf=now_iso()
                ))
        return out

    def indices(self):
        idx_list = [("^NSEI", "NIFTY 50", 24500), ("^BSESN", "SENSEX", 80500), ("^NSEBANK", "BANK NIFTY", 52000)]
        out = []
        for ticker, name, base in idx_list:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="1mo")
                if hist.empty:
                    raise ValueError("Empty index history")
                val = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else val
                change = val - prev
                change_pct = (change / prev) * 100 if prev else 0.0
                spark = [round(float(x), 2) for x in hist["Close"].tail(30).tolist()]
                out.append(dict(name=name, value=round(val, 2), change=round(change, 2), changePercent=round(change_pct, 2),
                                prevClose=round(prev, 2), spark=spark, asOf=now_iso()))
            except Exception:
                out.append(dict(name=name, value=base, change=0.0, changePercent=0.0, prevClose=base, spark=[base]*10, asOf=now_iso()))
        return out

    def history(self, symbol, rng):
        full_sym = symbol if "." in symbol else f"{symbol}.NS"
        yf_rng = RANGES.get(rng, "6mo")
        interval = "5m" if rng == "1D" else "1d"
        period = "1d" if rng == "1D" else yf_rng
        try:
            df = yf.download(full_sym, period=period, interval=interval, progress=False)
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            time_col = "Datetime" if "Datetime" in df.columns else "Date"
            cols_map = {time_col: "time", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
            df = df.rename(columns=cols_map)[["time", "open", "high", "low", "close", "volume"]].dropna()
            return df.round(2)
        except Exception:
            return pd.DataFrame()

    def fundamentals(self, symbol):
        full_sym = symbol if "." in symbol else f"{symbol}.NS"
        try:
            info = yf.Ticker(full_sym).info
            return {
                "P/E": info.get("trailingPE", "N/A"),
                "P/B": info.get("priceToBook", "N/A"),
                "EPS": info.get("trailingEps", "N/A"),
                "ROE %": round(info.get("returnOnEquity", 0) * 100, 2) if info.get("returnOnEquity") else "N/A",
                "ROCE %": "N/A",
                "Debt/Equity": info.get("debtToEquity", "N/A"),
                "Revenue growth %": round(info.get("revenueGrowth", 0) * 100, 2) if info.get("revenueGrowth") else "N/A",
                "Dividend yield %": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else "0"
            }
        except Exception:
            return {"P/E": "N/A", "P/B": "N/A", "EPS": "N/A", "ROE %": "N/A"}


PROVIDER = YahooLiveProvider()


@st.cache_data(ttl=60, show_spinner=False)
def get_quotes():
    return PROVIDER.quotes()


@st.cache_data(ttl=60, show_spinner=False)
def get_indices():
    return PROVIDER.indices()


@st.cache_data(ttl=300, show_spinner=False)
def get_history(symbol, rng):
    return PROVIDER.history(symbol, rng)


def quote_of(symbol):
    clean_sym = symbol.replace(".NS", "")
    return next((s for s in get_quotes() if s["symbol"] == clean_sym), None)


def market_open():
    ist = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5.5)
    mins = ist.hour * 60 + ist.minute
    return ist.weekday() < 5 and 9 * 60 + 15 <= mins <= 15 * 60 + 30


def safe_market(fn, *a):
    try:
        return fn(*a)
    except Exception as e:
        st.error(f"Market data unavailable: {e}")
        return None


# ============================ indicators ============================
def add_indicators(df):
    c = df["close"]
    df["sma20"], df["sma50"] = c.rolling(20).mean(), c.rolling(50).mean()
    sd = c.rolling(20).std(ddof=0)
    df["bb_up"], df["bb_lo"] = df["sma20"] + 2 * sd, df["sma20"] - 2 * sd
    d = c.diff()
    g, l = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(), (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + g / l.replace(0, 1e-9))
    df["macd"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["hist"] = df["macd"] - df["signal"]
    return df


def stock_chart(df):
    df = add_indicators(df.copy())
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.12, 0.18, 0.2], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price",
                                 increasing_line_color="#22C55E", decreasing_line_color="#EF4444"), 1, 1)
    for col, color, nm, dash in [("sma20", "#F59E0B", "SMA20", None), ("sma50", "#A78BFA", "SMA50", None), ("bb_up", "#64748B", "BB upper", "dot"), ("bb_lo", "#64748B", "BB lower", "dot")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["time"], y=df[col], name=nm, line=dict(color=color, width=1, dash=dash)), 1, 1)
    fig.add_trace(go.Bar(x=df["time"], y=df["volume"], name="Volume", marker_color="#3B82F6"), 2, 1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["rsi"], name="RSI(14)", line=dict(color="#3B82F6", width=1)), 3, 1)
    fig.add_hline(y=70, line_dash="dot", line_color="#64748B", row=3, col=1); fig.add_hline(y=30, line_dash="dot", line_color="#64748B", row=3, col=1)
    fig.add_trace(go.Bar(x=df["time"], y=df["hist"], name="MACD hist", marker_color=["#22C55E" if v >= 0 else "#EF4444" for v in df["hist"].fillna(0)]), 4, 1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["macd"], name="MACD", line=dict(color="#3B82F6", width=1)), 4, 1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["signal"], name="Signal", line=dict(color="#F59E0B", width=1)), 4, 1)
    fig.update_layout(height=760, template="plotly_dark", paper_bgcolor="#151D2B", plot_bgcolor="#151D2B", xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.02))
    return fig


# ============================ portfolio maths ============================
def compute_holdings(txs):
    book = {}
    for t in sorted(txs, key=lambda t: t["date"]):
        h = book.setdefault(t["symbol"], dict(symbol=t["symbol"], qty=0, avg=0.0, realized=0.0))
        if t["type"] == "BUY":
            tot = h["qty"] * h["avg"] + t["qty"] * t["price"]
            h["qty"] += t["qty"]; h["avg"] = tot / h["qty"]
        else:
            if t["qty"] > h["qty"]:
                raise ValueError(f"Cannot sell {t['qty']} {t['symbol']}; only {h['qty']} held")
            h["realized"] += (t["price"] - h["avg"]) * t["qty"]; h["qty"] -= t["qty"]
            if h["qty"] == 0:
                h["avg"] = 0.0
    return list(book.values())


def bar_alloc(labels, values, title):
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.55))
    fig.update_layout(title=title, height=300, template="plotly_dark", paper_bgcolor="#151D2B", margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ============================ AI (Gemini SDK) ============================
SYSTEM = ("You are an educational stock-analysis assistant for Indian markets. Use ONLY the JSON data provided. Never invent figures; if a value is "
          "missing write 'Data unavailable'. Do not promise returns and do not give personalised buy/sell recommendations. Clearly separate FACTS "
          "(from data) from INTERPRETATION (your reasoning). Be concise.")


def gemini(prompt, history=()):
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured in Streamlit secrets.")
    client = genai.Client(api_key=GEMINI_KEY)
    
    contents = []
    for r, t in history:
        contents.append(types.Content(role="user" if r == "user" else "model", parts=[types.Part.from_text(text=t)]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
    
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        temperature=0.3,
    )
    
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=config,
    )
    return response.text


def verified_facts(symbol):
    qt = quote_of(symbol)
    if not qt:
        raise RuntimeError("Unknown symbol.")
    hist = get_history(symbol, "6M")
    closes = hist["close"] if not hist.empty else pd.Series()
    return {
        "dataSource": PROVIDER.name,
        "demoData": PROVIDER.demo,
        "quote": qt,
        "fundamentals": PROVIDER.fundamentals(symbol),
        "technical": {
            "sma20": None if len(closes) < 20 else round(float(closes.tail(20).mean()), 2),
            "sma50": None if len(closes) < 50 else round(float(closes.tail(50).mean()), 2),
            "candlesUsed": len(closes)
        }
    }


# ============================ news / IPO placeholders ============================
def get_news():
    n = now_iso()
    return [
        dict(headline="Market Update: Sensex and Nifty trade upbeat on tech buying", summary="Indian benchmarks see positive momentum supported by major global cues.", publisher="LiveWire", published=n, url="", category="Indian stock market"),
        dict(headline="Banking Sector Outlook: Credit growth remains resilient", summary="RBI data indicates robust credit disbursement across major scheduled commercial banks.", publisher="Financial Chronicle", published=n, url="", category="Banking"),
    ]


def get_ipos():
    d = lambda n: (dt.date.today() + dt.timedelta(days=n)).isoformat()
    return pd.DataFrame([
        dict(Company="Sample Tech Ltd", Status="Upcoming", Price="₹120–₹125", Lot=120, Opens=d(5), Closes=d(8), Listing=d(13), Subscription="—", ListingPrice="—", Gain="—"),
        dict(Company="Sample Infra Corp", Status="Open", Price="₹300–₹320", Lot=45, Opens=d(-1), Closes=d(2), Listing=d(7), Subscription="3.2x", ListingPrice="—", Gain="—"),
    ])


# ============================ alerts ============================
def evaluate_alerts(uid):
    quotes = {s["symbol"]: s for s in get_quotes()}
    for a in q("SELECT * FROM alerts WHERE uid=? AND active=1", (uid,)):
        s = quotes.get(a["symbol"])
        if not s:
            continue
        hit = ((a["cond"] == "Price above" and s["price"] >= a["target"]) or (a["cond"] == "Price below" and s["price"] <= a["target"])
               or (a["cond"] == "% increase (day)" and s["changePercent"] >= a["target"]) or (a["cond"] == "% decrease (day)" and s["changePercent"] <= -a["target"]))
        if hit and x("UPDATE alerts SET active=0, triggered=? WHERE id=? AND active=1", (now_iso(), a["id"])).rowcount:
            msg = f"{a['symbol']} alert ({a['cond']} {a['target']}) triggered at {inr(s['price'])}"
            x("INSERT INTO notifs(uid,msg,ts) VALUES(?,?,?)", (uid, msg, now_iso()))
            st.toast(msg, icon="🔔")


# ============================ UI helpers ============================
def banner():
    st.markdown('<div class="demo">📊 Connected to Live Yahoo Finance API. Real-time quotes and fundamentals active.</div>', unsafe_allow_html=True)


def source_line(asof=None):
    st.markdown(f'<div class="small">Source: {PROVIDER.name}' + (f" · as of {fmt_time(asof)}" if asof else "") + "</div>", unsafe_allow_html=True)


def symbols():
    return [s["symbol"] for s in get_quotes()]


def open_stock(sym):
    st.session_state.sel = sym
    st.session_state.page = "Stock Explorer"


def quote_table(rows, key):
    df = pd.DataFrame(rows)[["symbol", "name", "sector", "price", "changePercent", "volume"]]
    df.columns = ["Symbol", "Company", "Sector", "Price (₹)", "Change %", "Volume"]
    st.dataframe(df, use_container_width=True, hide_index=True, key=key)


# ============================ pages ============================
def page_dashboard():
    st.header("Dashboard"); banner()

    @st.fragment(run_every=10)
    def live():
        idx = safe_market(get_indices)
        if not idx:
            return
        cols = st.columns(len(idx))
        for c, i in zip(cols, idx):
            with c:
                st.metric(i["name"], f"{i['value']:,.2f}", f"{i['change']:+,.2f} ({i['changePercent']:+.2f}%)")
                st.caption(f"Prev close {i['prevClose']:,.2f}")
                f = go.Figure(go.Scatter(y=i["spark"], line=dict(color="#22C55E" if i["change"] >= 0 else "#EF4444", width=2)))
                f.update_layout(height=70, margin=dict(l=0, r=0, t=0, b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor="#0B0F19", plot_bgcolor="#0B0F19")
                st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False}, key=f"sp{i['name']}{dt.datetime.now().timestamp()}")
        source_line(idx[0]["asOf"])
    live()

    qs = safe_market(get_quotes) or []
    adv, dec = sum(s["change"] > 0 for s in qs), sum(s["change"] < 0 for s in qs)
    a, b, c, d = st.columns(4)
    a.metric("Advances", adv); b.metric("Declines", dec); c.metric("Unchanged", len(qs) - adv - dec); d.metric("Market", "Open" if market_open() else "Closed")
    
    if qs:
        t1, t2, t3 = st.columns(3)
        for col, title, rows in [(t1, "Top Gainers", sorted(qs, key=lambda s: -s["changePercent"])[:5]), 
                                 (t2, "Top Losers", sorted(qs, key=lambda s: s["changePercent"])[:5]),
                                 (t3, "Most Active", sorted(qs, key=lambda s: -s["volume"])[:5])]:
            with col:
                st.subheader(title)
                for s in rows:
                    st.button(f"{s['symbol']}  {inr(s['price'])}  ({pct(s['changePercent'])})", key=f"{title}{s['symbol']}", on_click=open_stock, args=(s["symbol"],), use_container_width=True)
        st.subheader("Sectors")
        sec = pd.DataFrame(qs).groupby("sector")["changePercent"].mean().round(2)
        cols = st.columns(4)
        for i, (n, v) in
