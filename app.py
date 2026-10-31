import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from stock_data import fetch_stock_data, get_stock_info
from indicators import add_all_indicators
from prediction import prepare_ml_data, train_and_evaluate_model, predict_future
from comparison import compare_stocks


# ---------------------------------------------------------
# PAGE CONFIGURATION — must be the first Streamlit command
# ---------------------------------------------------------
st.set_page_config(
    page_title="EquityAI | Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# DESIGN SYSTEM
# ---------------------------------------------------------
BG = "#080D18"
PANEL = "#111827"
PANEL_ALT = "#151F31"
BORDER = "rgba(148, 163, 184, 0.16)"
TEXT = "#F1F5F9"
MUTED = "#94A3B8"
BLUE = "#60A5FA"
GREEN = "#34D399"
RED = "#FB7185"
PURPLE = "#A78BFA"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root { color-scheme: dark; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
        background:
          radial-gradient(ellipse at 15% 0%, rgba(37,99,235,.13), transparent 34%),
          radial-gradient(ellipse at 90% 10%, rgba(124,58,237,.10), transparent 30%),
          #080D18;
        color: #F1F5F9;
    }
    [data-testid="stHeader"] { background: rgba(8,13,24,.82); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101827 0%, #0B1220 100%);
        border-right: 1px solid rgba(148,163,184,.13);
    }
    [data-testid="stSidebar"] > div { padding-top: 1.4rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #A8B5C7; }
    [data-testid="stSidebar"] .stRadio label { padding: 5px 0; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: #F8FAFC !important; }
    h1 { letter-spacing: -1.2px; font-weight: 800 !important; }
    h2, h3 { letter-spacing: -.4px; }
    p, label, span { color: inherit; }
    .block-container { padding-top: 2.0rem; padding-bottom: 3rem; max-width: 1500px; }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #172235, #101827);
        border: 1px solid rgba(148,163,184,.16);
        border-radius: 16px;
        padding: 17px 18px;
        box-shadow: 0 12px 28px rgba(0,0,0,.14);
    }
    [data-testid="stMetricLabel"] { color: #9CAEC4 !important; }
    [data-testid="stMetricValue"] { color: #F8FAFC !important; font-family: 'Manrope', sans-serif; }
    [data-testid="stMetricDelta"] { font-weight: 700; }
    div[data-testid="stPlotlyChart"] {
        background: #101827;
        border: 1px solid rgba(148,163,184,.13);
        border-radius: 17px;
        padding: 8px 8px 0 8px;
    }
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        border: 1px solid rgba(148,163,184,.15);
        border-radius: 14px;
        overflow: hidden;
    }
    .hero {
        background: linear-gradient(115deg, rgba(30,64,175,.34), rgba(17,24,39,.94) 54%, rgba(91,33,182,.22));
        border: 1px solid rgba(96,165,250,.25);
        border-radius: 22px;
        padding: 25px 28px;
        margin: 5px 0 24px 0;
        box-shadow: 0 18px 45px rgba(0,0,0,.20);
    }
    .eyebrow {
        color: #93C5FD;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 11px;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .hero-title {
        font-family: 'Manrope', sans-serif;
        color: #F8FAFC;
        font-size: clamp(25px, 3vw, 38px);
        font-weight: 800;
        line-height: 1.15;
        margin: 0 0 9px 0;
    }
    .hero-copy { color: #B8C6D9; font-size: 14px; margin: 0; }
    .section-kicker {
        color: #93C5FD;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.7px;
        text-transform: uppercase;
        margin: 18px 0 6px 0;
    }
    .subtle { color: #94A3B8; font-size: 13px; }
    .info-card {
        background: linear-gradient(145deg, #172235, #101827);
        border: 1px solid rgba(148,163,184,.16);
        border-radius: 16px;
        padding: 17px 18px;
        min-height: 116px;
    }
    .info-label { color: #9CAEC4; font-size: 12px; margin-bottom: 9px; }
    .info-value { color: #F8FAFC; font-family: 'Manrope',sans-serif; font-size: 23px; font-weight: 800; }
    .info-note { color: #9CAEC4; font-size: 12px; margin-top: 5px; }
    .positive { color: #34D399 !important; }
    .negative { color: #FB7185 !important; }
    .tag {
        display: inline-block; padding: 5px 10px; border-radius: 999px;
        background: rgba(96,165,250,.12); color: #93C5FD;
        border: 1px solid rgba(96,165,250,.22); font-size: 11px; font-weight: 700;
    }
    .disclaimer {
        background: rgba(245,158,11,.08);
        border: 1px solid rgba(245,158,11,.22);
        border-radius: 12px; padding: 13px 15px;
        color: #FCDFA0; font-size: 12px; margin-top: 18px;
    }
    .stButton > button {
        border-radius: 10px; border: 1px solid rgba(96,165,250,.35);
        font-weight: 700; min-height: 42px;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #2563EB, #4F46E5);
        color: white; border: none;
    }
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
        background: #111C2D !important; border-color: rgba(148,163,184,.22) !important;
        border-radius: 10px !important; color: #F8FAFC !important;
    }
    hr { border-color: rgba(148,163,184,.14); }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# REUSABLE HELPERS
# ---------------------------------------------------------
def page_header(kicker, title, description):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{kicker}</div>
            <div class="hero-title">{title}</div>
            <p class="hero-copy">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title, note=None):
    st.markdown(f'<div class="section-kicker">{title}</div>', unsafe_allow_html=True)
    if note:
        st.caption(note)


def make_chart_layout(fig, height=420, y_title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,.35)",
        font=dict(family="DM Sans, sans-serif", color="#CBD5E1", size=12),
        height=height,
        margin=dict(l=18, r=18, t=28, b=18),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(showgrid=False, zeroline=False, color="#94A3B8"),
        yaxis=dict(
            title=y_title, showgrid=True, gridcolor="rgba(148,163,184,.12)",
            zeroline=False, color="#94A3B8"
        ),
    )
    return fig


def show_data_error(message, fallback="Please check the ticker symbol and try again."):
    st.error(f"{message or 'The data provider did not return usable data.'} {fallback}")


def get_valid_stock_data(symbol, period):
    if not symbol:
        return None, "Please enter a stock symbol."
    df, error = fetch_stock_data(symbol, period=period)
    if error or df is None or df.empty:
        return None, error or "No historical data was returned."
    return df.copy(), None


def currency_price(info):
    return f"{info.get('currency', '')} {float(info.get('current_price', 0)):,.2f}"


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding:4px 0 15px 0;">
            <div style="font-size:30px;">📈</div>
            <div style="font-family:Manrope,sans-serif;font-size:20px;font-weight:800;color:#F8FAFC;">
                Equity<span style="color:#60A5FA;">AI</span>
            </div>
            <div style="font-size:11px;color:#8191A8;letter-spacing:1.4px;">STOCK INTELLIGENCE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    menu = st.radio(
        "WORKSPACE",
        [
            "Dashboard",
            "Stock Analysis",
            "AI Stock Prediction",
            "Stock Comparison",
            "Market Watchlist",
            "Prediction History",
            "About Project",
        ],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**Market shortcuts**")
    quick_symbol = st.selectbox(
        "Popular symbols",
        ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "AAPL", "MSFT", "NVDA"],
        label_visibility="collapsed",
    )
    st.caption("Indian NSE symbols use the `.NS` suffix. Example: TCS.NS.")
    st.markdown("---")
    st.markdown(
        '<span class="tag">EDUCATIONAL PROJECT</span>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# 1. DASHBOARD
# ---------------------------------------------------------
if menu == "Dashboard":
    page_header(
        "MARKET PULSE  /  OVERVIEW",
        "Your market, in focus.",
        "Explore historical market movement, key trading metrics and stock performance from one clean workspace.",
    )

    col1, col2, col3 = st.columns([2.2, 1, 1])
    with col1:
        symbol = st.text_input(
            "Search by ticker",
            value=quick_symbol,
            placeholder="e.g. RELIANCE.NS, TCS.NS, AAPL",
        ).strip().upper()
    with col2:
        period_choice = st.selectbox("Chart range", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    with col3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        refresh = st.button("↻  Refresh data", use_container_width=True)

    if refresh:
        st.cache_data.clear()

    with st.spinner(f"Loading market data for {symbol}..."):
        info_data, info_error = get_stock_info(symbol)
        df, hist_error = get_valid_stock_data(symbol, period_choice)

    if info_error or not info_data or df is None:
        show_data_error(info_error or hist_error)
    else:
        change = float(info_data.get("change", 0) or 0)
        pct_change = float(info_data.get("pct_change", 0) or 0)
        sign = "+" if change >= 0 else ""

        section_title("Market snapshot", f"Latest data returned by the connected market-data source for {symbol}.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last price", currency_price(info_data), f"{sign}{change:,.2f} ({sign}{pct_change:.2f}%)")
        c2.metric("Day high", f"{float(info_data.get('high', 0)):,.2f}")
        c3.metric("Day low", f"{float(info_data.get('low', 0)):,.2f}")
        c4.metric("Volume", f"{int(info_data.get('volume', 0) or 0):,}")

        st.markdown("")
        section_title("Price performance", f"{symbol} · {period_choice} historical closing prices")
        line_color = GREEN if change >= 0 else RED
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["Date"], y=df["Close"], name="Closing price",
                mode="lines", line=dict(color=line_color, width=2.6),
                fill="tozeroy", fillcolor="rgba(52,211,153,.08)",
                hovertemplate="%{x|%d %b %Y}<br>Close: %{y:,.2f}<extra></extra>",
            )
        )
        make_chart_layout(fig, 450, f"Price ({info_data.get('currency', '')})")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        left, right = st.columns([1.4, 1])
        with left:
            section_title("Latest available data")
            st.caption(f"Data timestamp: {info_data.get('timestamp', 'Not provided by data source')}")
        with right:
            st.markdown(
                '<div class="disclaimer">Market data may be delayed or subject to provider limitations. '
                'This dashboard is for educational purposes and is not investment advice.</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------
# 2. TECHNICAL STOCK ANALYSIS
# ---------------------------------------------------------
elif menu == "Stock Analysis":
    page_header(
        "TECHNICAL RESEARCH",
        "Understand the trend.",
        "Study moving averages, momentum and trend strength using the indicators already implemented in your project.",
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        symbol = st.text_input("Stock ticker", value=quick_symbol).strip().upper()
    with c2:
        period = st.selectbox("Analysis period", ["6mo", "1y", "2y", "5y"], index=1)

    with st.spinner("Loading price history and calculating indicators..."):
        df, error = get_valid_stock_data(symbol, period)

    if error:
        show_data_error(error)
    else:
        try:
            df = add_all_indicators(df)
            section_title("Indicator workspace", f"Technical view for {symbol}")
            tab1, tab2, tab3 = st.tabs(["Moving averages", "RSI momentum", "MACD trend"])

            with tab1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close", line=dict(color="#E2E8F0", width=1.8)))
                if "SMA_20" in df:
                    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_20"], name="20-day SMA", line=dict(color=BLUE, width=2)))
                if "SMA_50" in df:
                    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_50"], name="50-day SMA", line=dict(color=PURPLE, width=2)))
                make_chart_layout(fig, 420, "Price")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("A moving average smooths price data to help visualize the broader trend.")

            with tab2:
                if "RSI" not in df:
                    st.warning("RSI data is not available from the indicator function.")
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI", line=dict(color=PURPLE, width=2)))
                    fig.add_hline(y=70, line_dash="dash", line_color=RED, annotation_text="Upper guide · 70")
                    fig.add_hline(y=30, line_dash="dash", line_color=GREEN, annotation_text="Lower guide · 30")
                    make_chart_layout(fig, 360, "RSI")
                    fig.update_yaxes(range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("RSI is a momentum indicator. The 70 and 30 lines are common reference levels, not automatic buy or sell signals.")

            with tab3:
                needed = {"MACD", "MACD_Signal", "MACD_Hist"}
                if not needed.issubset(df.columns):
                    st.warning("MACD columns are not available from the indicator function.")
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_Hist"], name="Histogram", marker_color="#64748B", opacity=.55))
                    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD", line=dict(color=BLUE, width=2)))
                    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"], name="Signal", line=dict(color=RED, width=2)))
                    make_chart_layout(fig, 380, "MACD")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("MACD compares moving averages to help examine momentum and trend changes.")
        except Exception as exc:
            st.error(f"Could not calculate or display the technical indicators: {exc}")


# ---------------------------------------------------------
# 3. AI STOCK PREDICTION
# ---------------------------------------------------------
elif menu == "AI Stock Prediction":
    page_header(
        "MACHINE LEARNING LAB",
        "Forecast with your AI model.",
        "Train the selected model on historical data, review test-set metrics and inspect the next seven predicted sessions.",
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        symbol = st.text_input("Stock ticker", value="TCS.NS").strip().upper()
    with c2:
        model_choice = st.selectbox("Prediction model", ["Random Forest Regressor", "Linear Regression"])

    run_prediction = st.button("⚡  Run AI prediction", type="primary", use_container_width=True)

    if run_prediction:
        if not symbol:
            st.warning("Enter a stock ticker before running the model.")
        else:
            try:
                with st.spinner(f"Training {model_choice} for {symbol}..."):
                    df, error = get_valid_stock_data(symbol, "2y")
                    if error:
                        show_data_error(error)
                    else:
                        X, y, dates, df_ml = prepare_ml_data(df)
                        model, metrics, X_test, y_test, y_pred = train_and_evaluate_model(X, y, model_choice)
                        future_prices = predict_future(model, df_ml, days=7)

                if not error:
                    section_title("Model evaluation", "Metrics below are from the model's out-of-sample test set.")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Mean absolute error", f"{float(metrics['MAE']):,.3f}")
                    m2.metric("Root mean squared error", f"{float(metrics['RMSE']):,.3f}")
                    m3.metric("R² score", f"{float(metrics['R2']):.4f}")

                    last_date = pd.to_datetime(df["Date"].iloc[-1])
                    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=7)
                    pred_df = pd.DataFrame({
                        "Date": future_dates.strftime("%Y-%m-%d"),
                        "Predicted close price": [round(float(p), 2) for p in future_prices],
                    })

                    first_prediction = float(future_prices[0])
                    last_close = float(df["Close"].iloc[-1])
                    expected_change = ((first_prediction - last_close) / last_close * 100) if last_close else 0

                    section_title("Forecast highlights", "Model-generated estimates; not guaranteed market outcomes.")
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Latest historical close", f"{last_close:,.2f}")
                    p2.metric("Next session estimate", f"{first_prediction:,.2f}", f"{expected_change:+.2f}%")
                    p3.metric("Forecast horizon", "7 sessions")

                    left, right = st.columns([1.35, 1])
                    with left:
                        section_title("Historical vs forecast")
                        fig = go.Figure()
                        recent = df.tail(60)
                        fig.add_trace(go.Scatter(
                            x=recent["Date"], y=recent["Close"], name="Historical close",
                            line=dict(color=BLUE, width=2.4),
                        ))
                        fig.add_trace(go.Scatter(
                            x=future_dates, y=future_prices, name="AI forecast",
                            line=dict(color=GREEN, width=2.5, dash="dash"),
                            mode="lines+markers",
                        ))
                        make_chart_layout(fig, 410, "Price")
                        st.plotly_chart(fig, use_container_width=True)
                    with right:
                        section_title("Seven-session forecast")
                        st.dataframe(pred_df, use_container_width=True, hide_index=True, height=310)

                    st.session_state.history.append({
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Symbol": symbol,
                        "Model": model_choice,
                        "Next Session Estimate": round(first_prediction, 2),
                        "R2 Score": round(float(metrics["R2"]), 4),
                    })
                    st.markdown(
                        '<div class="disclaimer"><b>Important:</b> Predictions are model estimates based on historical patterns. '
                        'They can be inaccurate and should not be used as financial advice or as the sole basis for investment decisions.</div>',
                        unsafe_allow_html=True,
                    )
            except Exception as exc:
                st.error(f"The prediction could not be completed: {exc}")
                st.info("Check that the prediction module, its dependencies and the selected ticker are working correctly.")


# ---------------------------------------------------------
# 4. STOCK COMPARISON
# ---------------------------------------------------------
elif menu == "Stock Comparison":
    page_header(
        "RELATIVE PERFORMANCE",
        "Compare stocks side by side.",
        "Compare normalized percentage movement so stocks with different price levels can be viewed on a common scale.",
    )
    symbols_input = st.text_input(
        "Ticker symbols — separate with commas",
        value="RELIANCE.NS, TCS.NS, INFY.NS",
    )
    symbols_list = list(dict.fromkeys(s.strip().upper() for s in symbols_input.split(",") if s.strip()))
    period = st.selectbox("Comparison range", ["6mo", "1y", "2y", "5y"], index=1)

    if st.button("Compare selected stocks", type="primary"):
        if len(symbols_list) < 2:
            st.warning("Enter at least two stock symbols to compare.")
        else:
            with st.spinner("Fetching and comparing stock performance..."):
                try:
                    comp_df, norm_df = compare_stocks(symbols_list, period=period)
                    if comp_df is None or comp_df.empty or norm_df is None or norm_df.empty:
                        st.error("No comparison data was returned. Check the symbols and try again.")
                    else:
                        section_title("Normalized return", "Each series is rebased to 0% at the beginning of the selected period.")
                        fig = go.Figure()
                        for column in norm_df.columns:
                            fig.add_trace(go.Scatter(
                                x=norm_df.index, y=norm_df[column],
                                name=str(column), mode="lines",
                            ))
                        make_chart_layout(fig, 460, "Return (%)")
                        fig.add_hline(y=0, line_dash="dot", line_color="#64748B")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as exc:
                    st.error(f"Comparison failed: {exc}")


# ---------------------------------------------------------
# 5. MARKET WATCHLIST
# ---------------------------------------------------------
elif menu == "Market Watchlist":
    page_header(
        "MARKET MONITOR",
        "Your watchlist, at a glance.",
        "A quick view of selected Indian and US stocks using the same market-data connection as the rest of the application.",
    )
    watchlist_symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "AAPL", "MSFT", "NVDA"]
    rows = []

    with st.spinner("Loading watchlist quotes..."):
        for ticker in watchlist_symbols:
            try:
                info, error = get_stock_info(ticker)
                if not error and info:
                    rows.append({
                        "Symbol": info.get("symbol", ticker),
                        "Company": info.get("shortName", ticker),
                        "Last price": round(float(info.get("current_price", 0)), 2),
                        "Change": round(float(info.get("change", 0)), 2),
                        "Change %": round(float(info.get("pct_change", 0)), 2),
                        "Currency": info.get("currency", ""),
                    })
            except Exception:
                continue

    if rows:
        watch_df = pd.DataFrame(rows)
        section_title("Tracked instruments", f"{len(rows)} of {len(watchlist_symbols)} symbols returned data.")
        st.dataframe(
            watch_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Last price": st.column_config.NumberColumn(format="%.2f"),
                "Change": st.column_config.NumberColumn(format="%.2f"),
                "Change %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
    else:
        st.error("The data provider did not return watchlist quotes. Please try again later.")


# ---------------------------------------------------------
# 6. PREDICTION HISTORY
# ---------------------------------------------------------
elif menu == "Prediction History":
    page_header(
        "YOUR MODEL RUNS",
        "Prediction history.",
        "Review predictions generated during this active Streamlit session. This history resets when the session is cleared.",
    )
    if not st.session_state.history:
        st.info("No predictions have been generated in this session yet. Open AI Stock Prediction to run your first model.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        section_title("Recent model runs", f"{len(hist_df)} prediction run(s)")
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        csv_data = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇  Download history as CSV",
            data=csv_data,
            file_name="equityai_prediction_history.csv",
            mime="text/csv",
        )
        if st.button("Clear session history"):
            st.session_state.history = []
            st.rerun()


# ---------------------------------------------------------
# 7. ABOUT PROJECT
# ---------------------------------------------------------
elif menu == "About Project":
    page_header(
        "ABOUT EQUITYAI",
        "Making market data easier to explore.",
        "An educational stock analytics and machine-learning project built with Python and Streamlit.",
    )
    st.markdown("### What this application does")
    a, b = st.columns(2)
    with a:
        st.markdown(
            """
            **Market data**
            - Retrieves historical OHLCV stock data through the existing data module.
            - Displays price history and selected market metrics.
            - Supports ticker symbols such as `RELIANCE.NS`, `TCS.NS` and `AAPL`.

            **Technical analysis**
            - Simple Moving Averages (SMA)
            - Relative Strength Index (RSI)
            - Moving Average Convergence Divergence (MACD)
            """
        )
    with b:
        st.markdown(
            """
            **Machine learning**
            - Random Forest Regressor
            - Linear Regression
            - Out-of-sample evaluation using MAE, RMSE and R²
            - Seven-session model forecast

            **Technology**
            - Python and Streamlit
            - Pandas for data handling
            - Plotly for interactive charts
            - Existing project modules for data, indicators and predictions
            """
        )
    st.markdown("---")
    st.markdown("### Important limitations")
    st.warning(
        "Stock prices are volatile and influenced by many factors that are not represented in historical-price models. "
        "Predictions are estimates for learning and demonstration only, not financial advice."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#718096;font-size:11px;">
      <span>© EquityAI · AI Stock Intelligence Dashboard</span>
      <span>Educational use only · Market data may be delayed</span>
    </div>
    """,
    unsafe_allow_html=True,
)
