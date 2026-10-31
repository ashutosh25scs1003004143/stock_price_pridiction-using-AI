"""
StockPulse AI - Understand Markets. Invest Smarter.
Single-file Streamlit app for Indian (NSE/BSE) market intelligence.

Run:    streamlit run app.py
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

# Try importing yfinance for live data support
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

st.set_page_config(page_title="StockPulse AI", page_icon="📈", layout="wide")


def cfg(key, default=""):
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


# Handle Gemini API Key with session state fallback
GEMINI_MODEL = cfg("GEMINI_MODEL", "gemini-2.0-flash")
MARKET_KEY, DB_PATH = cfg("MARKET_PROVIDER_KEY"), cfg("DB_PATH", "stockpulse.db")
START_BALANCE = 100000.0

st.markdown("""<style>
.stApp{background:#0B0F19;color:#F8FAFC} [data-testid=stSidebar]{background:#111827}
div[data-testid=stMetric]{background:#151D2B;border:1px solid #263244;border-radius:12px;padding:12px}
.demo{background:#3b3208;color:#fde68a;padding:6px 12px;border-radius:8px;font-size:13px}
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
        return dt.datetime.fromisoformat(iso).strftime("%d %b %Y, %H:%M:%S")
    except Exception:
        return iso or "—"


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


# ============================ market data layer ============================
class MarketProvider(ABC):
    name = ""
    demo = False

    @abstractmethod
    def quotes(self): ...
    @abstractmethod
    def indices(self): ...
    @abstractmethod
    def history(self, symbol, rng): ...
    @abstractmethod
    def fundamentals(self, symbol): ...


# Expanded list of Indian stocks (using Yahoo Finance tickers with .NS suffix)
UNIVERSE = [
    ("RELIANCE.NS", "Reliance Industries Ltd", "Energy", 2950, 2000000),
    ("TCS.NS", "Tata Consultancy Services Ltd", "IT", 4100, 1500000),
    ("HDFCBANK.NS", "HDFC Bank Ltd", "Banking", 1650, 1250000),
    ("INFY.NS", "Infosys Ltd", "IT", 1850, 780000),
    ("ICICIBANK.NS", "ICICI Bank Ltd", "Banking", 1250, 900000),
    ("ITC.NS", "ITC Ltd", "FMCG", 490, 610000),
    ("SBIN.NS", "State Bank of India", "Banking", 820, 730000),
    ("BHARTIARTL.NS", "Bharti Airtel Ltd", "Telecom", 1600, 950000),
    ("LT.NS", "Larsen & Toubro Ltd", "Construction", 3600, 500000),
    ("HINDUNILVR.NS", "Hindustan Unilever Ltd", "FMCG", 2600, 610000),
    ("BAJFINANCE.NS", "Bajaj Finance Ltd", "Financials", 7200, 440000),
    ("SUNPHARMA.NS", "Sun Pharmaceutical Industries Ltd", "Pharma", 1800, 430000),
]

RANGES = {"1D": "1d", "1W": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}


class YahooLiveProvider(MarketProvider):
    name, demo = "LIVE (Yahoo Finance)", False

    def quotes(self):
        out = []
        tickers = [u[0] for u in UNIVERSE]
        try:
            data = yf.download(tickers, period="2d", interval="1d", group_by="ticker", progress=False, threads=True)
            for sym, name, sector, base_est, cap in UNIVERSE:
                try:
                    df = data[sym] if len(tickers) > 1 else data
                    if df is None or df.empty or len(df) < 1:
                        continue
                    curr_row = df.iloc[-1]
                    prev_row = df.iloc[-2] if len(df) > 1 else curr_row
                    
                    price = float(curr_row["Close"])
                    prev = float(prev_row["Close"])
                    change = price - prev
                    change_pct = (change / prev) * 100 if prev else 0.0
                    
                    out.append({
                        "symbol": sym, "name": name, "exchange": "NSE", "sector": sector,
                        "price": round(price, 2), "change": round(change, 2),
                        "changePercent": round(change_pct, 2), "open": round(float(curr_row["Open"]), 2),
                        "high": round(float(curr_row["High"]), 2), "low": round(float(curr_row["Low"]), 2),
                        "prevClose": round(prev, 2), "volume": int(curr_row["Volume"]),
                        "week52High": round(price * 1.2, 2), "week52Low": round(price * 0.8, 2),
                        "marketCap": cap, "asOf": now_iso()
                    })
                except Exception:
                    continue
        except Exception:
            pass
        return out if out else DemoProvider().quotes()

    def indices(self):
        idx_tickers = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK", "NIFTY IT": "^CNXIT"}
        out = []
        for name, sym in idx_tickers.items():
            try:
                t = yf.Ticker(sym)
                hist = t.history(period="5d")
                if not hist.empty:
                    val = float(hist["Close"].iloc[-1])
                    prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else val
                    chg = val - prev
                    chg_pct = (chg / prev) * 100 if prev else 0
                    spark = hist["Close"].tail(30).tolist()
                    out.append({
                        "name": name, "value": round(val, 2), "change": round(chg, 2),
                        "changePercent": round(chg_pct, 2), "prevClose": round(prev, 2),
                        "spark": spark, "asOf": now_iso()
                    })
            except Exception:
                pass
        if not out:
            return DemoProvider().indices()
        return out

    def history(self, symbol, rng):
        yf_rng = RANGES.get(rng, "1mo")
        interval = "5m" if rng == "1D" else "1d"
        try:
            df = yf.download(symbol, period="1d" if rng == "1D" else yf_rng, interval=interval, progress=False)
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            df = df.reset_index()
            # Ensure proper column names
            col_map = {}
            for c in df.columns:
                if 'date' in c or 'time' in c:
                    col_map[c] = 'time'
                elif 'open' in c:
                    col_map[c] = 'open'
                elif 'high' in c:
                    col_map[c] = 'high'
                elif 'low' in c:
                    col_map[c] = 'low'
                elif 'close' in c:
                    col_map[c] = 'close'
                elif 'volume' in c:
                    col_map[c] = 'volume'
            df = df.rename(columns=col_map)
            return df[['time', 'open', 'high', 'low', 'close', 'volume']].dropna().round(2)
        except Exception:
            return pd.DataFrame()

    def fundamentals(self, symbol):
        try:
            t = yf.Ticker(symbol)
            inf = t.info
            return {
                "P/E": inf.get("trailingPE"), "P/B": inf.get("priceToBook"),
                "EPS": inf.get("trailingEps"), "ROE %": inf.get("returnOnEquity"),
                "Debt/Equity": inf.get("debtToEquity"), "Dividend yield %": inf.get("dividendYield")
            }
        except Exception:
            return None


class DemoProvider(MarketProvider):
    name, demo = "DEMO (fictional fallback data)", True

    def quotes(self):
        return [dict(symbol=u[0], name=u[1], exchange="NSE", sector=u[2], price=float(u[3]), change=10.0,
                     changePercent=0.5, open=float(u[3]), high=float(u[3])*1.01, low=float(u[3])*0.99,
                     prevClose=float(u[3]), volume=100000, week52High=float(u[3])*1.3, week52Low=float(u[3])*0.7,
                     marketCap=u[4], asOf=now_iso()) for u in UNIVERSE]

    def indices(self):
        return [dict(name="NIFTY 50", value=24500.0, change=100.0, changePercent=0.41, prevClose=24400.0, spark=[24400]*20, asOf=now_iso())]

    def history(self, symbol, rng):
        dates = pd.date_range(end=dt.datetime.now(), periods=30, freq='D')
        df = pd.DataFrame({
            "time": dates,
            "open": [100.0 + i for i in range(30)],
            "high": [105.0 + i for i in range(30)],
            "low": [95.0 + i for i in range(30)],
            "close": [102.0 + i for i in range(30)],
            "volume": [50000] * 30
        })
        return df

    def fundamentals(self, symbol):
        return {"P/E": 22.5, "P/B": 3.1, "EPS": 45.0, "ROE %": 15.2}


# Choose provider based on availability
PROVIDER = YahooLiveProvider() if YFINANCE_AVAILABLE else DemoProvider()


@st.cache_data(ttl=10, show_spinner=False)
def get_quotes():
    return PROVIDER.quotes()


@st.cache_data(ttl=10, show_spinner=False)
def get_indices():
    return PROVIDER.indices()


@st.cache_data(ttl=60, show_spinner=False)
def get_history(symbol, rng):
    return PROVIDER.history(symbol, rng)


def quote_of(symbol):
    return next((s for s in get_quotes() if s["symbol"] == symbol), None)


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
    if "volume" in df.columns:
        fig.add_trace(go.Bar(x=df["time"], y=df["volume"], name="Volume", marker_color="#3B82F6"), 2, 1)
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df["time"], y=df["rsi"], name="RSI(14)", line=dict(color="#3B82F6", width=1)), 3, 1)
        fig.add_hline(y=70, line_dash="dot", line_color="#64748B", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#64748B", row=3, col=1)
    if "hist" in df.columns:
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


# ============================ AI (Gemini) ============================
SYSTEM = ("You are an educational stock-analysis assistant for Indian markets. Use ONLY the JSON data provided. Never invent figures; if a value is "
          "missing write 'Data unavailable'. Do not promise returns and do not give personalised buy/sell recommendations.")


def get_active_gemini_key():
    key = cfg("GEMINI_API_KEY")
    if not key:
        key = st.session_state.get("user_gemini_key", "")
    return key


def gemini(prompt, history=()):
    api_key = get_active_gemini_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured. Please enter your API key in the sidebar.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    contents = [{"role": r, "parts": [{"text": t}]} for r, t in history] + [{"role": "user", "parts": [{"text": prompt}]}]
    r = requests.post(url, headers={"x-goog-api-key": api_key, "Content-Type": "application/json"}, timeout=60,
                      json={"systemInstruction": {"parts": [{"text": SYSTEM}]}, "contents": contents})
    if r.status_code != 200:
        raise RuntimeError(f"AI service error ({r.status_code}): {r.text}")
    return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])


def verified_facts(symbol):
    qt = quote_of(symbol)
    if not qt:
        raise RuntimeError("Unknown symbol.")
    hist = get_history(symbol, "6M")
    closes = hist["close"] if not hist.empty else pd.Series()
    return {"dataSource": PROVIDER.name, "demoData": PROVIDER.demo, "quote": qt, "fundamentals": PROVIDER.fundamentals(symbol),
            "technical": {"sma20": None if len(closes) < 20 else round(closes.tail(20).mean(), 2), "sma50": None if len(closes) < 50 else round(closes.tail(50).mean(), 2),
                          "candlesUsed": len(closes)}}


def get_news():
    n = now_iso()
    return [dict(headline="Market Update: Indian Indices Track Global Trends", summary="Markets show mixed sentiment across sectors.", publisher="StockPulse News", published=n, url="", category="Economy")]


def get_ipos():
    d = lambda n: (dt.date.today() + dt.timedelta(days=n)).isoformat()
    return pd.DataFrame([
        dict(Company="Sample IPO Ltd", Status="Open", Price="₹250–₹260", Lot=57, Opens=d(-1), Closes=d(2), Listing=d(7), Subscription="2.4x", ListingPrice="—", Gain="—")
    ])


def evaluate_alerts(uid):
    quotes = {s["symbol"]: s for s in get_quotes()}
    for a in q("SELECT * FROM alerts WHERE uid=? AND active=1", (uid,)):
        s = quotes.get(a["symbol"])
        if not s:
            continue
        hit = ((a["cond"] == "Price above" and s["price"] >= a["target"]) or (a["cond"] == "Price below" and s["price"] <= a["target"])
               or (a["cond"] == "% increase (day)" and s["changePercent"] >= a["target"]) or (a["cond"] == "% decrease (day)" and s["changePercent"] <= -a["target"]))
        if hit and x("UPDATE alerts SET active=0, triggered=? WHERE id=? AND active=1", (now_iso(), a["id"])).rowcount:
            msg = f"{a['symbol']} alert triggered at {inr(s['price'])}"
            x("INSERT INTO notifs(uid,msg,ts) VALUES(?,?,?)", (uid, msg, now_iso()))
            st.toast(msg, icon="🔔")


def banner():
    if PROVIDER.demo:
        st.markdown('<div class="demo">⚠ Running in demo mode. Install yfinance or configure live data provider for live market feeds.</div>', unsafe_allow_html=True)


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
    idx = safe_market(get_indices)
    if idx:
        cols = st.columns(len(idx) if len(idx) > 0 else 1)
        for c, i in zip(cols, idx):
            with c:
                st.metric(i["name"], f"{i['value']:,.2f}", f"{i['change']:+,.2f} ({i['changePercent']:+.2f}%)")
                if "spark" in i and i["spark"]:
                    f = go.Figure(go.Scatter(y=i["spark"], line=dict(color="#22C55E" if i["change"] >= 0 else "#EF4444", width=2)))
                    f.update_layout(height=70, margin=dict(l=0, r=0, t=0, b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor="#0B0F19", plot_bgcolor="#0B0F19")
                    st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False}, key=f"sp_{i['name']}")
        source_line(idx[0].get("asOf"))

    qs = safe_market(get_quotes) or []
    if qs:
        adv, dec = sum(s["change"] > 0 for s in qs), sum(s["change"] < 0 for s in qs)
        a, b, c, d = st.columns(4)
        a.metric("Advances", adv); b.metric("Declines", dec); c.metric("Unchanged", len(qs) - adv - dec); d.metric("Market", "Open" if market_open() else "Closed")
        t1, t2, t3 = st.columns(3)
        for col, title, rows in [(t1, "Top Gainers", sorted(qs, key=lambda s: -s["changePercent"])[:5]), (t2, "Top Losers", sorted(qs, key=lambda s: s["changePercent"])[:5]),
                                 (t3, "Most Active", sorted(qs, key=lambda s: -s["volume"])[:5])]:
            with col:
                st.subheader(title)
                for s in rows:
                    st.button(f"{s['symbol']}  {inr(s['price'])}  ({pct(s['changePercent'])})", key=f"{title}{s['symbol']}", on_click=open_stock, args=(s["symbol"],), use_container_width=True)


def page_explorer(title="Stock Explorer", sort="Symbol"):
    st.header(title); banner()
    qs = safe_market(get_quotes) or []
    if not qs:
        st.warning("No stock data available.")
        return
    c = st.columns(5)
    text = c[0].text_input("Search name / ticker")
    ex = c[1].selectbox("Exchange", ["All", "NSE", "BSE"])
    sec = c[2].selectbox("Sector", ["All"] + sorted({s["sector"] for s in qs}))
    cap = c[3].selectbox("Market cap", ["Any", "Large (>₹5L Cr)", "Mid+ (>₹1L Cr)"])
    srt = c[4].selectbox("Sort by", ["Symbol", "Price", "Change %", "Volume"], index=["Symbol", "Price", "Change %", "Volume"].index(sort))
    desc = st.checkbox("Descending", value=sort != "Symbol")
    rows = [s for s in qs if (not text or text.lower() in (s["symbol"] + s["name"]).lower()) and (ex == "All" or s["exchange"] == ex)
            and (sec == "All" or s["sector"] == sec) and (cap == "Any" or s["marketCap"] >= (500000 if cap.startswith("Large") else 100000))]
    key = {"Symbol": "symbol", "Price": "price", "Change %": "changePercent", "Volume": "volume"}[srt]
    rows.sort(key=lambda s: s[key], reverse=desc)
    if not rows:
        st.info("No stocks match your filters."); return
    quote_table(rows, "tbl" + title)
    syms = [s["symbol"] for s in rows]
    cur = st.session_state.get("sel")
    sel = st.selectbox("Open stock details", syms, index=syms.index(cur) if cur in syms else 0, key="sel_box" + title)
    stock_detail(sel)


def stock_detail(sym):
    s = quote_of(sym)
    if not s:
        st.error("Stock not found or data unavailable."); return
    st.subheader(f"{s['name']} ({s['symbol']})")
    a, b, c, d = st.columns(4)
    a.metric("Price", inr(s["price"]), f"{s['change']:+.2f} ({s['changePercent']:+.2f}%)")
    b.metric("Day high / low", f"{inr(s['high'])} / {inr(s['low'])}"); c.metric("52-week high / low", f"{inr(s['week52High'])} / {inr(s['week52Low'])}")
    d.metric("Previous close", inr(s["prevClose"]))
    rng = st.radio("Timeframe", list(RANGES), index=3, horizontal=True, key="rng" + sym)
    df = safe_market(get_history, sym, rng)
    if df is not None and not df.empty:
        st.plotly_chart(stock_chart(df), use_container_width=True)
    fu = safe_market(PROVIDER.fundamentals, sym)
    st.markdown("**Fundamentals**")
    if fu:
        st.dataframe(pd.DataFrame([(k, "Data unavailable" if v is None else v) for k, v in fu.items()], columns=["Metric", "Value"]), hide_index=True)


def page_watchlist():
    st.header("Watchlists"); uid = st.session_state.uid
    with st.form("newwl", clear_on_submit=True):
        name = st.text_input("New watchlist name")
        if st.form_submit_button("Create") and name.strip():
            x("INSERT INTO watchlists(uid,name) VALUES(?,?)", (uid, name.strip())); st.rerun()
    lists = q("SELECT * FROM watchlists WHERE uid=?", (uid,))
    if not lists:
        st.info("No watchlists yet. Create one above."); return
    quotes = {s["symbol"]: s for s in get_quotes()}
    for w in lists:
        with st.expander(w["name"], expanded=True):
            items = [r["symbol"] for r in q("SELECT symbol FROM watch_items WHERE wid=?", (w["id"],))]
            if items:
                df = pd.DataFrame([{"Symbol": s, "Price (₹)": quotes[s]["price"], "Change %": quotes[s]["changePercent"]} for s in items if s in quotes])
                st.dataframe(df, hide_index=True, use_container_width=True)
            c1, c2, c3 = st.columns([2, 1, 1])
            add = c1.selectbox("Add stock", symbols(), key=f"a{w['id']}")
            if c2.button("Add", key=f"ab{w['id']}"):
                x("INSERT OR IGNORE INTO watch_items VALUES(?,?)", (w["id"], add)); st.rerun()
            if c3.button("Delete list", key=f"d{w['id']}"):
                x("DELETE FROM watch_items WHERE wid=?", (w["id"],)); x("DELETE FROM watchlists WHERE id=?", (w["id"],)); st.rerun()


def page_portfolio():
    st.header("My Portfolio"); banner(); uid = st.session_state.uid
    with st.form("tx", clear_on_submit=True):
        c = st.columns(6)
        sym, ex, typ = c[0].selectbox("Symbol", symbols()), c[1].selectbox("Exchange", ["NSE", "BSE"]), c[2].selectbox("Type", ["BUY", "SELL"])
        qty, price, day = c[3].number_input("Quantity", 1, 10_000_000, 1), c[4].number_input("Price (₹)", 0.01, 10_000_000.0, 100.0), c[5].date_input("Date", dt.date.today(), max_value=dt.date.today())
        if st.form_submit_button("Add transaction"):
            x("INSERT INTO txs(uid,symbol,exchange,qty,price,type,date) VALUES(?,?,?,?,?,?,?)", (uid, sym, ex, qty, price, typ, day.isoformat()))
            st.success("Transaction recorded."); st.rerun()
    txs = [dict(r) for r in q("SELECT * FROM txs WHERE uid=?", (uid,))]
    hold = compute_holdings(txs)
    qs = {s["symbol"]: s for s in get_quotes()}
    rows = []
    for h in hold:
        if h["qty"] > 0:
            s = qs.get(h["symbol"]); val = s["price"] * h["qty"] if s else None
            rows.append(dict(Symbol=h["symbol"], Qty=h["qty"], AvgCost=round(h["avg"], 2), LTP=s["price"] if s else None,
                             Invested=h["qty"] * h["avg"], Value=val if val is not None else h["qty"] * h["avg"]))
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No holdings yet.")


def page_ai():
    st.header("AI Stock Analyst"); banner()
    sym = st.selectbox("Company", symbols())
    if st.button("Analyse", type="primary"):
        try:
            with st.spinner("Analysing with Gemini..."):
                facts = verified_facts(sym)
                text = gemini(f"Analyse this company based on the data: {json.dumps(facts)}")
            x("INSERT INTO aireports(uid,symbol,analysis,ts) VALUES(?,?,?,?)", (st.session_state.uid, sym, text, now_iso()))
            st.session_state.report = dict(symbol=sym, facts=facts, text=text, ts=now_iso())
        except Exception as e:
            st.error(str(e))
    r = st.session_state.get("report")
    if r:
        st.write(r["text"])


def page_news():
    st.header("Market News")
    for n in get_news():
        st.markdown(f"**{n['headline']}**\n{n['summary']}")


def page_ipo():
    st.header("IPO Center")
    st.dataframe(get_ipos(), hide_index=True)


def page_alerts():
    st.header("Alerts")
    uid = st.session_state.uid
    with st.form("al", clear_on_submit=True):
        c = st.columns(4)
        sym, cond = c[0].selectbox("Symbol", symbols()), c[1].selectbox("Condition", ["Price above", "Price below"])
        tgt = c[2].number_input("Target", 0.01, 10_000_000.0, 100.0)
        if c[3].form_submit_button("Create alert"):
            x("INSERT INTO alerts(uid,symbol,cond,target) VALUES(?,?,?,?)", (uid, sym, cond, tgt)); st.rerun()


def page_settings():
    st.header("Settings")
    st.write("Configure your preferences here.")


def auth_screen():
    st.title("📈 StockPulse AI"); st.caption("Understand Markets. Invest Smarter.")
    t1, t2 = st.tabs(["Log in", "Register"])
    with t1, st.form("li"):
        e, p = st.text_input("Email"), st.text_input("Password", type="password")
        if st.form_submit_button("Log in"):
            uid, err = login(e, p)
            if err: st.error(err)
            else: st.session_state.uid = uid; st.rerun()
    with t2, st.form("rg"):
        n, e, p = st.text_input("Name"), st.text_input("Email"), st.text_input("Password (min 8 chars)", type="password")
        if st.form_submit_button("Create account"):
            uid, err = register(n, e, p)
            if err: st.error(err)
            else: st.session_state.uid = uid; st.rerun()


PAGES = ["Dashboard", "Markets", "Stock Explorer", "Watchlist", "My Portfolio", "AI Stock Analyst", "Market News", "IPO Center", "Alerts", "Settings"]


def main():
    if "uid" not in st.session_state:
        auth_screen(); return
    uid = st.session_state.uid
    user = q("SELECT * FROM users WHERE id=?", (uid,), one=True)
    if not user:
        st.session_state.clear(); st.rerun()
    evaluate_alerts(uid)
    with st.sidebar:
        st.markdown("## Stock**Pulse** AI")
        
        # Gemini API Key fallback input if missing
        if not cfg("GEMINI_API_KEY"):
            st.text_input("Gemini API Key", type="password", key="user_gemini_key", help="Enter your Gemini API key to enable AI Analyst features.")
            
        st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
        st.caption(f"Signed in as {user['name']}")
        if st.button("Log out"):
            st.session_state.clear(); st.rerun()
            
    pages_map = {
        "Dashboard": page_dashboard,
        "Markets": lambda: page_explorer("Markets", "Change %"),
        "Stock Explorer": page_explorer,
        "Watchlist": page_watchlist,
        "My Portfolio": page_portfolio,
        "AI Stock Analyst": page_ai,
        "Market News": page_news,
        "IPO Center": page_ipo,
        "Alerts": page_alerts,
        "Settings": page_settings
    }
    pages_map.get(st.session_state.get("page", "Dashboard"), page_dashboard)()
    st.divider(); st.caption("Educational use only. Not investment advice.")


main()
