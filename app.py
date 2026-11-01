"""
StockPulse AI - Understand Markets. Invest Smarter.
Single-file Streamlit app for Indian (NSE/BSE) market intelligence.

Run:    streamlit run app.py
Secrets (Streamlit Cloud -> App settings -> Secrets, or environment variables):
    GEMINI_API_KEY      Google Gemini key (AI Analyst)
    GEMINI_MODEL        optional, default gemini-2.0-flash
    MARKET_PROVIDER_KEY key of a LICENSED NSE/BSE data vendor (implement LiveProvider below)
    DB_PATH             optional SQLite path, default stockpulse.db

With no MARKET_PROVIDER_KEY the app runs in clearly labelled DEMO MODE using FICTIONAL data.
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

st.set_page_config(page_title="StockPulse AI", page_icon="📈", layout="wide")


def cfg(key, default=""):
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


GEMINI_KEY, GEMINI_MODEL = cfg("GEMINI_API_KEY"), cfg("GEMINI_MODEL", "gemini-2.0-flash")
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
    return dt.datetime.fromisoformat(iso).strftime("%d %b %Y, %H:%M:%S UTC") if iso else "—"


def now_iso():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


# ============================ database (SQLite) ============================
# NOTE: Streamlit Community Cloud's disk is ephemeral. For permanent data, point DB_PATH
# to a persistent volume or swap these helpers for MongoDB/Postgres/Supabase.
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


UNIVERSE = [  # FICTIONAL companies, synthetic numbers - NOT real market data
    ("DEMOBANK", "Demo Bank Ltd (fictional)", "Banking", 1500, 900000), ("DEMOFIN", "Demo Finserve Ltd (fictional)", "Banking", 780, 300000),
    ("DEMOTECH", "Demo Technologies Ltd (fictional)", "IT", 3400, 1200000), ("DEMOSOFT", "Demo Softworks Ltd (fictional)", "IT", 1650, 650000),
    ("DEMOMOTO", "Demo Motors Ltd (fictional)", "Automobile", 920, 340000), ("DEMOPHARMA", "Demo Pharma Ltd (fictional)", "Pharma", 1280, 310000),
    ("DEMOFMCG", "Demo Consumer Ltd (fictional)", "FMCG", 2450, 580000), ("DEMOPOWER", "Demo Power Ltd (fictional)", "Energy", 310, 420000),
    ("DEMOSTEEL", "Demo Steel Ltd (fictional)", "Metal", 145, 180000), ("DEMOREALTY", "Demo Realty Ltd (fictional)", "Realty", 2100, 90000),
    ("DEMOOIL", "Demo Petroleum Ltd (fictional)", "Energy", 2900, 1800000), ("DEMOCEMENT", "Demo Cement Ltd (fictional)", "Metal", 10500, 280000),
]
RANGES = {"1D": 1, "1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "5Y": 1825}


def _h(s):
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:4], "big")


def _noise(seed):  # deterministic -1..1
    m = 0xFFFFFFFF
    t = (seed + 0x6D2B79F5) & m
    t = ((t ^ (t >> 15)) * (t | 1)) & m
    t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & m)) & m
    return (((t ^ (t >> 14)) & m) / 4294967296) * 2 - 1


def _day_close(base, sym, d):
    return base * (1 + 0.12 * math.sin(d / 25 + _h(sym) % 100) + 0.02 * _noise(d * 7919 + _h(sym)))


def _today():
    return int(dt.datetime.now(dt.timezone.utc).timestamp() // 86400)


class DemoProvider(MarketProvider):
    name, demo = "DEMO (fictional data)", True

    def quotes(self):
        d, out, t = _today(), [], dt.datetime.now(dt.timezone.utc).timestamp()
        for sym, name, sector, base, cap in UNIVERSE:
            prev = _day_close(base, sym, d - 1)
            price = _day_close(base, sym, d) * (1 + 0.004 * math.sin(t / 60 + _h(sym) % 100))
            closes = [_day_close(base, sym, d - i) for i in range(0, 252, 3)]
            out.append(dict(symbol=sym, name=name, exchange="NSE", sector=sector, price=round(price, 2), change=round(price - prev, 2),
                            changePercent=round((price - prev) / prev * 100, 2), open=round(prev * 1.001, 2), high=round(max(price, prev) * 1.006, 2),
                            low=round(min(price, prev) * 0.994, 2), prevClose=round(prev, 2), volume=int(500000 + abs(_noise(d + _h(sym))) * 4_000_000),
                            week52High=round(max(closes), 2), week52Low=round(min(closes), 2), marketCap=cap, asOf=now_iso()))
        return out

    def indices(self):
        d, t, out = _today(), dt.datetime.now(dt.timezone.utc).timestamp(), []
        for sym, name, base in [("NIFTY 50", "NIFTY 50 (demo)", 24500), ("SENSEX", "SENSEX (demo)", 80500), ("BANK NIFTY", "BANK NIFTY (demo)", 52000), ("NIFTY IT", "NIFTY IT (demo)", 38000)]:
            h = _h(sym) % 100
            f = lambda i: base * (1 + 0.05 * math.sin(i / 30 + h))
            prev = f(d - 1)
            spark = [round(f(d) * (1 + 0.004 * math.sin(i / 5 + h) + 0.0015 * _noise(i * 13 + h + int(t // 3600))), 2) for i in range(40)]
            out.append(dict(name=name, value=spark[-1], change=round(spark[-1] - prev, 2), changePercent=round((spark[-1] - prev) / prev * 100, 2),
                            prevClose=round(prev, 2), spark=spark, asOf=now_iso()))
        return out

    def history(self, symbol, rng):
        u = next((u for u in UNIVERSE if u[0] == symbol), None)
        if not u:
            return pd.DataFrame()
        sym, base, h, rows = u[0], u[3], _h(u[0]) % 100, []
        if rng == "1D":
            start = dt.datetime.now(dt.timezone.utc).replace(hour=3, minute=45, second=0, microsecond=0, tzinfo=None)
            b = _day_close(base, sym, _today() - 1)
            for i in range(75):
                o = b * (1 + 0.004 * math.sin(i / 8 + h) + 0.001 * _noise(i + h)); c = b * (1 + 0.004 * math.sin((i + 1) / 8 + h) + 0.001 * _noise(i + 1 + h))
                rows.append((start + dt.timedelta(minutes=5 * i), o, max(o, c) * 1.001, min(o, c) * 0.999, c, int(20000 + abs(_noise(i + h)) * 90000)))
        else:
            d = _today()
            for i in range(RANGES[rng], -1, -1):
                idx = d - i; o, c = _day_close(base, sym, idx - 1), _day_close(base, sym, idx)
                rows.append((dt.datetime(1970, 1, 1) + dt.timedelta(days=idx), o, max(o, c) * (1 + 0.006 * abs(_noise(idx + h))),
                             min(o, c) * (1 - 0.006 * abs(_noise(idx * 3 + h))), c, int(500000 + abs(_noise(idx + h)) * 4_000_000)))
        return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"]).round(2)

    def fundamentals(self, symbol):
        if not any(u[0] == symbol for u in UNIVERSE):
            return None
        h = _h(symbol) % 1000
        n = lambda k, lo, hi: round(lo + (_noise(h + k) + 1) / 2 * (hi - lo), 2)
        return {"P/E": n(1, 12, 38), "P/B": n(2, 1.5, 8), "EPS": n(3, 10, 120), "ROE %": n(4, 8, 28), "ROCE %": n(5, 9, 30), "Debt/Equity": n(6, 0, 1.4),
                "Revenue growth %": n(7, -3, 22), "Profit growth %": n(8, -8, 30), "Dividend yield %": n(9, 0, 3)}


class LiveProvider(MarketProvider):
    """TEMPLATE: implement with a LICENSED NSE/BSE data vendor using MARKET_KEY. Do NOT scrape exchange websites."""
    name, demo = "LIVE (configure vendor)", False

    def _no(self):
        raise NotImplementedError("Implement LiveProvider with your licensed data vendor's API.")

    def quotes(self): self._no()
    def indices(self): self._no()
    def history(self, symbol, rng): self._no()
    def fundamentals(self, symbol): self._no()


PROVIDER = LiveProvider() if MARKET_KEY else DemoProvider()


@st.cache_data(ttl=3, show_spinner=False)
def get_quotes():
    return PROVIDER.quotes()


@st.cache_data(ttl=3, show_spinner=False)
def get_indices():
    return PROVIDER.indices()


@st.cache_data(ttl=30, show_spinner=False)
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
    """Weighted-average-cost method. BUY: avg=(q*avg+bq*p)/(q+bq). SELL: realized=(p-avg)*sq; avg unchanged.
    Overselling is rejected. Brokerage/taxes and corporate actions (splits/bonus) are NOT automatic - record adjusted transactions."""
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
          "missing write 'Data unavailable'. Do not promise returns and do not give personalised buy/sell recommendations. Clearly separate FACTS "
          "(from data) from INTERPRETATION (your reasoning). Be concise.")


def gemini(prompt, history=()):
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured (add it in Streamlit secrets).")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    contents = [{"role": r, "parts": [{"text": t}]} for r, t in history] + [{"role": "user", "parts": [{"text": prompt}]}]
    r = requests.post(url, headers={"x-goog-api-key": GEMINI_KEY, "Content-Type": "application/json"}, timeout=60,
                      json={"systemInstruction": {"parts": [{"text": SYSTEM}]}, "contents": contents})
    if r.status_code != 200:
        raise RuntimeError(f"AI service error ({r.status_code}).")
    return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])


def verified_facts(symbol):
    qt = quote_of(symbol)
    if not qt:
        raise RuntimeError("Unknown symbol.")
    hist = get_history(symbol, "6M")
    closes = hist["close"]
    return {"dataSource": PROVIDER.name, "demoData": PROVIDER.demo, "quote": qt, "fundamentals": PROVIDER.fundamentals(symbol),
            "technical": {"sma20": None if len(closes) < 20 else round(closes.tail(20).mean(), 2), "sma50": None if len(closes) < 50 else round(closes.tail(50).mean(), 2),
                          "candlesUsed": len(closes)}}


# ============================ news / IPO (demo placeholders) ============================
def get_news():
    if MARKET_KEY:
        return []  # plug a licensed news provider here
    n = now_iso()
    return [dict(headline="DEMO: Sample headline about the banking sector", summary="Placeholder text shown only in demo mode.", publisher="Demo Source", published=n, url="", category="Banking"),
            dict(headline="DEMO: Sample headline about technology stocks", summary="Placeholder text shown only in demo mode.", publisher="Demo Source", published=n, url="", category="Technology"),
            dict(headline="DEMO: Sample headline about the Indian economy", summary="Placeholder text shown only in demo mode.", publisher="Demo Source", published=n, url="", category="Economy")]


def get_ipos():
    if MARKET_KEY:
        return pd.DataFrame()  # plug a licensed IPO provider here
    d = lambda n: (dt.date.today() + dt.timedelta(days=n)).isoformat()
    return pd.DataFrame([
        dict(Company="Demo Upcoming Ltd (fictional)", Status="Upcoming", Price="₹100–₹105", Lot=140, Opens=d(5), Closes=d(8), Listing=d(13), Subscription="—", ListingPrice="—", Gain="—"),
        dict(Company="Demo Open Ltd (fictional)", Status="Open", Price="₹250–₹260", Lot=57, Opens=d(-1), Closes=d(2), Listing=d(7), Subscription="2.4x", ListingPrice="—", Gain="—"),
        dict(Company="Demo Listed Ltd (fictional)", Status="Listed", Price="₹80–₹85", Lot=175, Opens=d(-20), Closes=d(-17), Listing=d(-12), Subscription="35x", ListingPrice="₹98", Gain="+15.3%")])


# ============================ alerts ============================
def evaluate_alerts(uid):
    quotes = {s["symbol"]: s for s in get_quotes()}
    for a in q("SELECT * FROM alerts WHERE uid=? AND active=1", (uid,)):
        s = quotes.get(a["symbol"])
        if not s:
            continue
        hit = ((a["cond"] == "Price above" and s["price"] >= a["target"]) or (a["cond"] == "Price below" and s["price"] <= a["target"])
               or (a["cond"] == "% increase (day)" and s["changePercent"] >= a["target"]) or (a["cond"] == "% decrease (day)" and s["changePercent"] <= -a["target"]))
        if hit and x("UPDATE alerts SET active=0, triggered=? WHERE id=? AND active=1", (now_iso(), a["id"])).rowcount:  # fires once
            msg = f"{a['symbol']} alert ({a['cond']} {a['target']}) triggered at {inr(s['price'])}" + (" [DEMO data]" if PROVIDER.demo else "")
            x("INSERT INTO notifs(uid,msg,ts) VALUES(?,?,?)", (uid, msg, now_iso()))
            st.toast(msg, icon="🔔")


# ============================ UI helpers ============================
def banner():
    if PROVIDER.demo:
        st.markdown('<div class="demo">⚠ DEMO MODE – all prices, news and IPOs are fictional sample data, not real market prices. '
                    'Set MARKET_PROVIDER_KEY and implement LiveProvider for real data.</div>', unsafe_allow_html=True)


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

    @st.fragment(run_every=5)
    def live():
        idx = safe_market(get_indices)
        if not idx:
            return
        cols = st.columns(4)
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
    t1, t2, t3 = st.columns(3)
    for col, title, rows in [(t1, "Top Gainers", sorted(qs, key=lambda s: -s["changePercent"])[:5]), (t2, "Top Losers", sorted(qs, key=lambda s: s["changePercent"])[:5]),
                             (t3, "Most Active", sorted(qs, key=lambda s: -s["volume"])[:5])]:
        with col:
            st.subheader(title)
            for s in rows:
                st.button(f"{s['symbol']}  {inr(s['price'])}  ({pct(s['changePercent'])})", key=f"{title}{s['symbol']}", on_click=open_stock, args=(s["symbol"],), use_container_width=True)
    st.subheader("Sectors")
    sec = pd.DataFrame(qs).groupby("sector")["changePercent"].mean().round(2)
    cols = st.columns(4)
    for i, (n, v) in enumerate(sec.items()):
        cols[i % 4].metric(n, pct(v))
    st.subheader("Recent news")
    news = get_news()
    for n in news[:3]:
        st.markdown(f"**{n['headline']}**  \n<span class='small'>{n['publisher']} · {fmt_time(n['published'])}</span>", unsafe_allow_html=True)
    if not news:
        st.info("No news available.")


def page_explorer(title="Stock Explorer", sort="Symbol"):
    st.header(title); banner()
    qs = safe_market(get_quotes) or []
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
    e, f, g = st.columns(3)
    e.metric("Exchange / Sector", f"{s['exchange']} · {s['sector']}"); f.metric("Market cap (₹ Cr)", f"{s['marketCap']:,}"); g.metric("Volume", f"{s['volume']:,}")
    rng = st.radio("Timeframe", list(RANGES), index=3, horizontal=True, key="rng" + sym)
    df = safe_market(get_history, sym, rng)
    if df is not None and not df.empty:
        st.caption(f"Timeframe: {rng} · Data as of {fmt_time(s['asOf'])} · Source: {PROVIDER.name}")
        st.plotly_chart(stock_chart(df), use_container_width=True)
    fu = safe_market(PROVIDER.fundamentals, sym)
    st.markdown("**Fundamentals**")
    st.dataframe(pd.DataFrame([(k, "Data unavailable" if v is None else v) for k, v in (fu or {}).items()], columns=["Metric", "Value"]) if fu else pd.DataFrame(), hide_index=True)
    w = q("SELECT id,name FROM watchlists WHERE uid=?", (st.session_state.uid,))
    if w:
        pick = st.selectbox("Add to watchlist", [r["name"] for r in w], key="addw" + sym)
        if st.button("Add", key="addb" + sym):
            x("INSERT OR IGNORE INTO watch_items VALUES(?,?)", (next(r["id"] for r in w if r["name"] == pick), sym)); st.success("Added to watchlist.")


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
                df = pd.DataFrame([{"Symbol": s, "Price (₹)": quotes[s]["price"], "Change %": quotes[s]["changePercent"], "Updated": fmt_time(quotes[s]["asOf"])} for s in items if s in quotes])
                st.dataframe(df, hide_index=True, use_container_width=True)
            else:
                st.caption("Empty list.")
            c1, c2, c3 = st.columns([2, 1, 1])
            add = c1.selectbox("Add stock", symbols(), key=f"a{w['id']}")
            if c2.button("Add", key=f"ab{w['id']}"):
                x("INSERT OR IGNORE INTO watch_items VALUES(?,?)", (w["id"], add)); st.rerun()
            if items:
                rm = c1.selectbox("Remove stock", items, key=f"r{w['id']}")
                if c2.button("Remove", key=f"rb{w['id']}"):
                    x("DELETE FROM watch_items WHERE wid=? AND symbol=?", (w["id"], rm)); st.rerun()
            if c3.button("Delete list", key=f"d{w['id']}"):
                x("DELETE FROM watch_items WHERE wid=?", (w["id"],)); x("DELETE FROM watchlists WHERE id=?", (w["id"],)); st.rerun()


def page_portfolio():
    st.header("My Portfolio"); banner(); uid = st.session_state.uid
    with st.form("tx", clear_on_submit=True):
        c = st.columns(6)
        sym, ex, typ = c[0].selectbox("Symbol", symbols()), c[1].selectbox("Exchange", ["NSE", "BSE"]), c[2].selectbox("Type", ["BUY", "SELL"])
        qty, price, day = c[3].number_input("Quantity", 1, 10_000_000, 1), c[4].number_input("Price (₹)", 0.01, 10_000_000.0, 100.0), c[5].date_input("Date", dt.date.today(), max_value=dt.date.today())
        if st.form_submit_button("Add transaction"):
            txs = [dict(r) for r in q("SELECT * FROM txs WHERE uid=?", (uid,))] + [dict(symbol=sym, qty=qty, price=price, type=typ, date=day.isoformat())]
            try:
                compute_holdings(txs)
                x("INSERT INTO txs(uid,symbol,exchange,qty,price,type,date) VALUES(?,?,?,?,?,?,?)", (uid, sym, ex, qty, price, typ, day.isoformat())); st.success("Transaction recorded.")
            except ValueError as e:
                st.error(str(e))
    txs = [dict(r) for r in q("SELECT * FROM txs WHERE uid=?", (uid,))]
    hold = compute_holdings(txs)
    qs = {s["symbol"]: s for s in get_quotes()}
    rows = []
    for h in hold:
        if h["qty"] > 0:
            s = qs.get(h["symbol"]); val = s["price"] * h["qty"] if s else None
            rows.append(dict(Symbol=h["symbol"], Sector=s["sector"] if s else "Unknown", Qty=h["qty"], AvgCost=round(h["avg"], 2), LTP=s["price"] if s else None,
                             Invested=h["qty"] * h["avg"], Value=val if val is not None else h["qty"] * h["avg"], Unrealized=(val - h["qty"] * h["avg"]) if val is not None else None))
    realized = sum(h["realized"] for h in hold)
    inv, val = sum(r["Invested"] for r in rows), sum(r["Value"] for r in rows)
    c = st.columns(5)
    c[0].metric("Invested", inr(inv)); c[1].metric("Current value", inr(val)); c[2].metric("Realized P/L", inr(realized))
    c[3].metric("Unrealized P/L", inr(val - inv)); c[4].metric("Overall return", pct((val - inv + realized) / inv * 100 if inv else 0))
    st.caption("Cost basis: weighted average cost. Brokerage, taxes and corporate actions are not applied automatically.")
    if not rows:
        st.info("No holdings yet. Add your first transaction above."); return
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(bar_alloc([r["Symbol"] for r in rows], [r["Invested"] for r in rows], "Investment allocation"), use_container_width=True)
    sec = pd.DataFrame(rows).groupby("Sector")["Value"].sum()
    b.plotly_chart(bar_alloc(list(sec.index), list(sec.values), "Sector allocation"), use_container_width=True)
    pl = go.Figure(go.Bar(x=[r["Symbol"] for r in rows], y=[r["Unrealized"] or 0 for r in rows], marker_color=["#22C55E" if (r["Unrealized"] or 0) >= 0 else "#EF4444" for r in rows]))
    pl.update_layout(title="Profit/loss distribution", height=300, template="plotly_dark", paper_bgcolor="#151D2B")
    st.plotly_chart(pl, use_container_width=True)
    curve, run = [], []
    for t in sorted(txs, key=lambda t: t["date"]):
        run.append(t); curve.append((t["date"], sum(h["qty"] * h["avg"] for h in compute_holdings(run))))
    if len(curve) > 1:
        st.plotly_chart(go.Figure(go.Scatter(x=[c[0] for c in curve], y=[c[1] for c in curve], mode="lines+markers")).update_layout(title="Invested capital over time", height=300, template="plotly_dark", paper_bgcolor="#151D2B"), use_container_width=True)


def page_ai():
    st.header("AI Stock Analyst"); banner()
    sym = st.selectbox("Company", symbols())
    if st.button("Analyse", type="primary"):
        try:
            with st.spinner("Analysing with Gemini..."):
                facts = verified_facts(sym)
                text = gemini("Analyse this company with sections: Company Overview, Fundamental Analysis, Technical Analysis, Risk Analysis, AI Summary "
                              f"(positives, negatives, bullish scenario, bearish scenario, risks to monitor). DATA:\n{json.dumps(facts)}")
            x("INSERT INTO aireports(uid,symbol,analysis,ts) VALUES(?,?,?,?)", (st.session_state.uid, sym, text, now_iso()))
            st.session_state.report = dict(symbol=sym, facts=facts, text=text, ts=now_iso()); st.session_state.chat = []
        except Exception as e:
            st.error(str(e))
    r = st.session_state.get("report")
    if r:
        with st.expander("Verified data (FACTS)", expanded=False):
            st.caption(f"Source: {r['facts']['dataSource']}{' (DEMO fictional data)' if r['facts']['demoData'] else ''} · as of {fmt_time(r['facts']['quote']['asOf'])}")
            st.json({k: r["facts"][k] for k in ("quote", "fundamentals", "technical")})
        st.markdown(f"**AI-GENERATED INTERPRETATION** · {fmt_time(r['ts'])}"); st.write(r["text"])
        st.subheader(f"Ask about {r['symbol']}")
        for role, t in st.session_state.get("chat", []):
            st.chat_message("user" if role == "user" else "assistant").write(t)
        if msg := st.chat_input("e.g. What are the main risks?"):
            try:
                ans = gemini(f"DATA:\n{json.dumps(verified_facts(r['symbol']))}\n\nQUESTION: {msg}", st.session_state.get("chat", [])[-20:])
                st.session_state.chat = st.session_state.get("chat", []) + [("user", msg), ("model", ans)]; st.rerun()
            except Exception as e:
                st.error(str(e))
    st.caption("Educational use only. Not investment advice. AI can make mistakes; verify with official filings.")


def vholdings(uid):
    trades = [dict(symbol=t["symbol"], qty=t["qty"], price=t["price"], type=t["type"], date=t["ts"]) for t in q("SELECT * FROM vtrades WHERE uid=?", (uid,))]
    return [h for h in compute_holdings(trades) if h["qty"] > 0]


def equity(uid, bal, qs):
    return bal + sum(h["qty"] * (qs[h["symbol"]]["price"] if h["symbol"] in qs else h["avg"]) for h in vholdings(uid))


def page_virtual():
    st.header("Virtual Trading (Simulated)"); banner(); uid = st.session_state.uid
    st.warning("Paper trading only. No real money, no brokerage connection. Every transaction is simulated.")
    qs = {s["symbol"]: s for s in get_quotes()}
    bal = q("SELECT balance FROM vacct WHERE uid=?", (uid,), one=True)["balance"]
    eq = equity(uid, bal, qs)
    a, b, c, d = st.columns(4)
    a.metric("Virtual balance", inr(bal)); b.metric("Total equity", inr(eq)); c.metric("P/L", inr(eq - START_BALANCE)); d.metric("Return", pct((eq - START_BALANCE) / START_BALANCE * 100))
    with st.form("order"):
        c = st.columns(4)
        sym, typ, qty = c[0].selectbox("Symbol", symbols()), c[1].selectbox("Side", ["BUY", "SELL"]), c[2].number_input("Quantity", 1, 100000, 1)
        c[3].metric("Estimated value", inr(qs[sym]["price"] * qty) if sym in qs else "—")
        if st.form_submit_button("Confirm simulated order"):
            price = qs[sym]["price"]; cost = price * qty  # price is taken server-side, never from user input
            held = next((h["qty"] for h in vholdings(uid) if h["symbol"] == sym), 0)
            if typ == "BUY" and cost > bal:
                st.error("Insufficient virtual balance.")
            elif typ == "SELL" and qty > held:
                st.error("Insufficient shares to sell.")
            else:
                x("UPDATE vacct SET balance=balance+? WHERE uid=?", (-cost if typ == "BUY" else cost, uid))
                x("INSERT INTO vtrades(uid,symbol,qty,price,type,ts) VALUES(?,?,?,?,?,?)", (uid, sym, qty, price, typ, now_iso()))
                st.success(f"Simulated {typ} of {qty} {sym} at {inr(price)} executed."); st.rerun()
    l, r = st.columns(2)
    with l:
        st.subheader("Holdings")
        hs = vholdings(uid)
        st.dataframe(pd.DataFrame([dict(Symbol=h["symbol"], Qty=h["qty"], AvgCost=round(h["avg"], 2), LTP=qs.get(h["symbol"], {}).get("price"),
                                        PL=round((qs[h["symbol"]]["price"] - h["avg"]) * h["qty"], 2) if h["symbol"] in qs else None) for h in hs]), hide_index=True) if hs else st.caption("No holdings.")
    with r:
        st.subheader("Leaderboard")
        board = sorted([(u["name"], (equity(u["id"], u["balance"], qs) - START_BALANCE) / START_BALANCE * 100)
                        for u in q("SELECT users.id,users.name,vacct.balance FROM users JOIN vacct ON vacct.uid=users.id")], key=lambda t: -t[1])[:20]
        st.dataframe(pd.DataFrame([(i + 1, n, pct(v)) for i, (n, v) in enumerate(board)], columns=["#", "Trader", "Return"]), hide_index=True)
    st.subheader("Transaction history")
    hist = q("SELECT symbol,type,qty,price,ts FROM vtrades WHERE uid=? ORDER BY id DESC LIMIT 100", (uid,))
    st.dataframe(pd.DataFrame([dict(Time=fmt_time(h["ts"]), Side=h["type"], Symbol=h["symbol"], Qty=h["qty"], Price=h["price"], Note="SIMULATED") for h in hist]), hide_index=True) if hist else st.caption("No trades yet.")


def page_news():
    st.header("Market News"); banner()
    cat = st.radio("Category", ["All", "Indian stock market", "Company news", "Banking", "Technology", "Economy", "IPO", "Global markets"], horizontal=True)
    items = [n for n in get_news() if cat == "All" or n["category"] == cat]
    if not items:
        st.info("No news available for this category.")
    for n in items:
        with st.container(border=True):
            st.markdown(f"**{n['headline']}**"); st.write(n["summary"])
            st.caption(f"{n['publisher']} · {fmt_time(n['published'])}" + (f" · [Original article]({n['url']})" if n["url"] else ""))
    st.caption("Add a licensed news provider in get_news(); AI summaries/sentiment can then pass headlines to gemini().")


def page_ipo():
    st.header("IPO Center"); banner()
    df = get_ipos()
    if df.empty:
        st.info("No IPO data available."); return
    for status in ["Open", "Upcoming", "Listed", "Closed"]:
        sub = df[df["Status"] == status]
        if len(sub):
            st.subheader(status); st.dataframe(sub.drop(columns="Status"), hide_index=True, use_container_width=True)
    source_line(now_iso())


def page_alerts():
    st.header("Alerts"); uid = st.session_state.uid
    with st.form("al", clear_on_submit=True):
        c = st.columns(4)
        sym, cond = c[0].selectbox("Symbol", symbols()), c[1].selectbox("Condition", ["Price above", "Price below", "% increase (day)", "% decrease (day)"])
        tgt = c[2].number_input("Target (₹ or %)", 0.01, 10_000_000.0, 100.0)
        if c[3].form_submit_button("Create alert"):
            x("INSERT INTO alerts(uid,symbol,cond,target) VALUES(?,?,?,?)", (uid, sym, cond, tgt)); st.rerun()
    st.subheader("Your alerts")
    al = q("SELECT * FROM alerts WHERE uid=? ORDER BY id DESC", (uid,))
    if not al:
        st.info("No alerts.")
    for a in al:
        c = st.columns([4, 1, 1])
        c[0].write(f"{a['symbol']} · {a['cond']} {a['target']} — {'🟢 Active' if a['active'] else '✅ Triggered ' + fmt_time(a['triggered'])}")
        if c[2].button("Delete", key=f"da{a['id']}"):
            x("DELETE FROM alerts WHERE id=? AND uid=?", (a["id"], uid)); st.rerun()
    st.subheader("Notifications")
    for n in q("SELECT * FROM notifs WHERE uid=? ORDER BY id DESC LIMIT 30", (uid,)):
        st.write(f"🔔 {n['msg']}  ·  _{fmt_time(n['ts'])}_")
    st.caption("Alerts are evaluated whenever the app refreshes (page interaction or Dashboard auto-refresh).")


def page_settings():
    st.header("Settings"); uid = st.session_state.uid
    u = q("SELECT * FROM users WHERE id=?", (uid,), one=True)
    with st.form("prof"):
        name = st.text_input("Name", u["name"]); st.caption(u["email"])
        ex = st.selectbox("Default exchange", ["NSE", "BSE"], index=["NSE", "BSE"].index(u["exchange"]))
        if st.form_submit_button("Save profile") and name.strip():
            x("UPDATE users SET name=?, exchange=? WHERE id=?", (name.strip(), ex, uid)); st.success("Profile updated.")
    with st.form("pw", clear_on_submit=True):
        cur, new = st.text_input("Current password", type="password"), st.text_input("New password (min 8)", type="password")
        if st.form_submit_button("Change password"):
            if check_pw(cur, u["pw"]) and len(new) >= 8:
                x("UPDATE users SET pw=? WHERE id=?", (hash_pw(new), uid)); st.success("Password changed.")
            else:
                st.error("Current password incorrect or new password too short.")


# ============================ app shell ============================
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
        n, e, p = st.text_input("Name"), st.text_input("Email "), st.text_input("Password (min 8 chars)", type="password")
        if st.form_submit_button("Create account"):
            uid, err = register(n, e, p)
            if err: st.error(err)
            else: st.session_state.uid = uid; st.rerun()


PAGES = ["Dashboard", "Markets", "Stock Explorer", "Watchlist", "My Portfolio", "AI Stock Analyst", "Virtual Trading", "Market News", "IPO Center", "Alerts", "Settings"]


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
        st.selectbox("Global stock search", [""] + symbols(), key="gs", format_func=lambda s: s or "Search company / symbol…", on_change=lambda: open_stock(st.session_state.gs) if st.session_state.gs else None)
        st.selectbox("Exchange", ["NSE", "BSE"], index=["NSE", "BSE"].index(user["exchange"]), key="ex_sel")
        st.markdown("🟢 **Market open**" if market_open() else "🔴 **Market closed**")
        st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
        st.caption(f"Signed in as {user['name']}")
        if st.button("Log out"):
            st.session_state.clear(); st.rerun()
    {"Dashboard": page_dashboard, "Markets": lambda: page_explorer("Markets", "Change %"), "Stock Explorer": page_explorer, "Watchlist": page_watchlist,
     "My Portfolio": page_portfolio, "AI Stock Analyst": page_ai, "Virtual Trading": page_virtual, "Market News": page_news, "IPO Center": page_ipo,
     "Alerts": page_alerts, "Settings": page_settings}[st.session_state.get("page", "Dashboard")]()
    st.divider(); st.caption("Educational use only. Not investment advice. Market data may be delayed or unavailable.")


main()
