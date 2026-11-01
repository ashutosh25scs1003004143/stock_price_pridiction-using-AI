import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from stock_data import fetch_stock_data, get_stock_info
from indicators import add_all_indicators
from prediction import prepare_ml_data, train_and_evaluate_model, predict_future
from comparison import compare_stocks

from stock_data import get_stock_categories
from analysis_helper import generate_ai_analysis

# Inside your sidebar or search section:
categories = get_stock_categories()
selected_sector = st.sidebar.selectbox("Select Sector", list(categories.keys()))
selected_company = st.sidebar.selectbox("Select Company", list(categories[selected_sector].keys()))
symbol = categories[selected_sector][selected_company]
# Page Configuration
st.set_page_config(
    page_title="AI Stock Predictor | Professional Financial Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Financial Dashboard Theme
st.markdown("""
<style>
    /* Main Background & Font */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #f0f6fc;
    }
    .metric-positive {
        color: #2ea043;
        font-weight: 600;
    }
    .metric-negative {
        color: #da3633;
        font-weight: 600;
    }
    
    /* AI Insight Card Styling */
    .ai-insight-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f0f6fc !important;
    }
    
    /* Tables and Dataframes */
    .dataframe {
        background-color: #161b22 !important;
        color: #c9d1d9 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State for Prediction History
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/stocks.png", width=60)
st.sidebar.title("AI Stock Predictor")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Stock Analysis", "AI Stock Prediction", "Stock Comparison", "Market Watchlist", "Prediction History", "About Project"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use `.NS` suffix for Indian NSE stocks (e.g., `RELIANCE.NS`) and standard tickers for US stocks (e.g., `AAPL`).")

# ==========================================
# 1. DASHBOARD
# ==========================================
if menu == "Dashboard":
    st.title("📊 Market Overview & Dashboard")
    st.markdown("Welcome to the AI Stock Predictor professional financial dashboard. Get real-time data insights and market overviews.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        default_symbol = st.text_input("Search Stock Symbol", value="AAPL", help="Enter ticker symbol e.g., AAPL, RELIANCE.NS, TSLA, TCS.NS")
    with col2:
        period_choice = st.selectbox("Time Period", ["1mo", "6mo", "1y", "5y"], index=2)
        
    symbol = default_symbol.strip().upper()
    
    with st.spinner(f"Fetching market data for {symbol}..."):
        info_data, err = get_stock_info(symbol)
        df, err_hist = fetch_stock_data(symbol, period=period_choice)
        
    if err or df is None:
        st.error(f"Error fetching data for {symbol}: {err or err_hist}. Please check the symbol and try again.")
    else:
        # Display Key Metrics Cards
        c1, c2, c3, c4 = st.columns(4)
        
        change_color = "metric-positive" if info_data['change'] >= 0 else "metric-negative"
        sign = "+" if info_data['change'] >= 0 else ""
        
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #8b949e; margin-bottom: 5px;">Current Price</p>
                <div class="metric-value">{info_data['currency']} {info_data['current_price']:.2f}</div>
                <p class="{change_color}" style="margin-top: 5px;">{sign}{info_data['change']:.2f} ({sign}{info_data['pct_change']:.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #8b949e; margin-bottom: 5px;">Day High / Low</p>
                <div class="metric-value" style="font-size: 20px;">{info_data['high']:.2f}</div>
                <p style="color: #8b949e; margin-top: 5px;">Low: {info_data['low']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #8b949e; margin-bottom: 5px;">Trading Volume</p>
                <div class="metric-value" style="font-size: 20px;">{info_data['volume']:,}</div>
                <p style="color: #8b949e; margin-top: 5px;">Latest Available</p>
            </div>
            """, unsafe_allow_html=True)
            
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #8b949e; margin-bottom: 5px;">Data Timestamp</p>
                <div class="metric-value" style="font-size: 18px;">{info_data['timestamp']}</div>
                <p style="color: #2ea043; margin-top: 5px;">● Verified Data</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### 📈 Historical Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Close'],
            mode='lines', name='Close Price',
            line=dict(color='#2ea043' if info_data['change'] >= 0 else '#da3633', width=2),
            fill='tozeroy', fillcolor='rgba(46, 160, 67, 0.1)'
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            margin=dict(l=20, r=20, t=20, b=20),
            height=450,
            xaxis_title="Date",
            yaxis_title=f"Price ({info_data['currency']})"
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 2. STOCK ANALYSIS
# ==========================================
elif menu == "Stock Analysis":
    st.title("🔍 Technical Stock Analysis")
    st.markdown("Analyze stock movements with integrated technical indicators including Moving Averages, RSI, and MACD.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol for Analysis", value="RELIANCE.NS").strip().upper()
    with col2:
        period = st.selectbox("Select Duration", ["6mo", "1y", "2y", "5y"], index=1)
        
    with st.spinner("Computing technical indicators..."):
        df, err = fetch_stock_data(symbol, period=period)
        
    if err or df is None:
        st.error(f"Could not retrieve data: {err}")
    else:
        df = add_all_indicators(df)
        
        st.subheader(f"Technical Indicators for {symbol}")
        
        # Tabs for charts
        tab1, tab2, tab3 = st.tabs(["Moving Averages (SMA)", "RSI Indicator", "MACD Indicator"])
        
        with tab1:
            fig_sma = go.Figure()
            fig_sma.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', line=dict(color='#c9d1d9', width=1)))
            fig_sma.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name='20-Day SMA', line=dict(color='#58a6ff', width=2)))
            fig_sma.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name='50-Day SMA', line=dict(color='#f0883e', width=2)))
            fig_sma.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#161b22", height=400)
            st.plotly_chart(fig_sma, use_container_width=True)
            
        with tab2:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='#bc8cff', width=2)))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="#da3633", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="#2ea043", annotation_text="Oversold (30)")
            fig_rsi.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#161b22", height=350, yaxis_range=[0, 100])
            st.plotly_chart(fig_rsi, use_container_width=True)
            
        with tab3:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='#58a6ff', width=1.5)))
            fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], name='Signal', line=dict(color='#f85149', width=1.5)))
            fig_macd.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram', marker_color='#3fb950'))
            fig_macd.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#161b22", height=350)
            st.plotly_chart(fig_macd, use_container_width=True)

# ==========================================
# 3. AI STOCK PRICE PREDICTION
# ==========================================
elif menu == "AI Stock Prediction":
    st.title("🤖 AI Stock Price Prediction")
    st.markdown("Train machine learning models using historical data, technical indicators, and chronological splitting to forecast the next 7 trading sessions.")
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Stock Symbol for Prediction", value="TCS.NS").strip().upper()
    with col2:
        model_choice = st.selectbox("Select ML Model", ["Random Forest Regressor", "Linear Regression"])
        
    if st.button("Run AI Prediction Model", type="primary"):
        with st.spinner(f"Training {model_choice} model and evaluating out-of-sample test performance..."):
            df, err = fetch_stock_data(symbol, period="2y")
            if err or df is None:
                st.error(f"Error loading data: {err}")
            else:
                X, y, dates, df_ml = prepare_ml_data(df)
                model, metrics, X_test, y_test, y_pred = train_and_evaluate_model(X, y, model_choice)
                
                # Display Metrics
                st.subheader("📊 Model Performance Metrics (Out-of-Sample Test Set)")
                m1, m2, m3 = st.columns(3)
                m1.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']}")
                m2.metric("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']}")
                m3.metric("R² Score", f"{metrics['R2']}")
                
                # Future 7-day prediction
                future_prices = predict_future(model, df_ml, days=7)
                last_date = df['Date'].iloc[-1]
                future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=7)
                
                pred_df = pd.DataFrame({
                    "Date": [d.strftime('%Y-%m-%d') for d in future_dates],
                    "Predicted Close Price": [round(p, 2) for p in future_prices]
                })
                
                st.subheader("📅 Next 7 Trading Sessions Forecast")
                st.dataframe(pred_df, use_container_width=True)
                
                # Save to History State
                st.session_state.history.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Symbol": symbol,
                    "Model": model_choice,
                    "Next Day Prediction": f"{future_prices[0]:.2f}",
                    "R2 Score": metrics['R2']
                })
                
                # Plot Predictions vs Historical
                st.subheader("📈 Historical & 7-Day Forecast Chart")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'][-60:], y=df['Close'][-60:], name='Recent Historical', line=dict(color='#58a6ff', width=2)))
                fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted Close Price'], name='AI Forecast (7 Days)', line=dict(color='#2ea043', width=2, dash='dash'), mode='lines+markers'))
                fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#161b22", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.warning("⚠️ **Disclaimer:** Stock market predictions are generated for educational and project purposes only and should not be used as financial advice.")

# ==========================================
# 4. STOCK COMPARISON
# ==========================================
elif menu == "Stock Comparison":
    st.title("⚖️ Comparative Stock Performance")
    st.markdown("Compare price performance and percentage returns across multiple stocks.")
    
    default_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    selected_symbols = st.text_input("Enter Symbols (comma separated)", value="RELIANCE.NS, TCS.NS, INFY.NS").strip().upper()
    symbols_list = [s.strip() for s in selected_symbols.split(",") if s.strip()]
    
    period = st.selectbox("Comparison Period", ["6mo", "1y", "2y", "5y"], index=1)
    
    if st.button("Compare Stocks", type="primary"):
        with st.spinner("Fetching and normalizing comparative data..."):
            comp_df, norm_df = compare_stocks(symbols_list, period=period)
            
        if comp_df is None or comp_df.empty:
            st.error("Could not fetch comparison data for the given symbols.")
        else:
            st.subheader("Normalized Percentage Return (%)")
            fig = go.Figure()
            for col in norm_df.columns:
                fig.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], name=col, mode='lines'))
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#161b22", height=450, yaxis_title="Growth (%) Relative to Start")
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 5. MARKET WATCHLIST
# ==========================================
elif menu == "Market Watchlist":
    st.title("⭐ Market Watchlist")
    st.markdown("Track key Indian and US benchmark stocks in real time.")
    
    watchlist_symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "AAPL", "MSFT", "TSLA"]
    
    watchlist_data = []
    with st.spinner("Loading watchlist..."):
        for sym in watchlist_symbols:
            info, err = get_stock_info(sym)
            if info:
                watchlist_data.append({
                    "Symbol": info['symbol'],
                    "Name": info['shortName'],
                    "Price": round(info['current_price'], 2),
                    "Change": round(info['change'], 2),
                    "Change %": f"{info['pct_change']:.2f}%",
                    "Currency": info['currency']
                })
                
    if watchlist_data:
        watch_df = pd.DataFrame(watchlist_data)
        st.dataframe(watch_df, use_container_width=True)
    else:
        st.error("Unable to load watchlist data.")

# ==========================================
# 6. PREDICTION HISTORY
# ==========================================
elif menu == "Prediction History":
    st.title("📜 Session Prediction History")
    st.markdown("Review all AI predictions generated during your current active session.")
    
    if not st.session_state.history:
        st.info("No predictions generated yet in this session. Visit the 'AI Stock Prediction' tab to run a model.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
        
        csv = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Prediction History as CSV",
            data=csv,
            file_name='ai_stock_predictions_history.csv',
            mime='text/csv',
        )

# ==========================================
# 7. ABOUT PROJECT
# ==========================================
elif menu == "About Project":
    st.title("ℹ️ About AI Stock Predictor")
    st.markdown("""
    ### 🎓 College Major Project Overview
    **AI Stock Predictor** is a modern, full-stack financial analysis and machine learning platform designed to bridge the gap between complex quantitative finance and accessible web applications.
    
    ### ⚙️ How It Works
    1. **Data Acquisition:** Historical open, high, low, close, and volume data are retrieved securely via `yfinance`.
    2. **Feature Engineering:** Technical indicators like Simple Moving Average (SMA), Relative Strength Index (RSI), and MACD are computed.
    3. **Machine Learning Modeling:** 
       - **Linear Regression:** Establishes linear relationships between technical indicators and future price movement.
       - **Random Forest Regressor:** Captures non-linear feature interactions through ensemble decision trees.
    4. **Rigorous Evaluation:** Chronological train/test splits are utilized to prevent data leakage, evaluated using MAE, RMSE, and $R^2$ scores.
    
    ### ⚠️ Project Limitations
    - Stock markets are highly volatile and influenced by unpredictable macroeconomic factors, geopolitical news, and investor sentiment.
    - Historical patterns do not guarantee future performance. This platform is built solely for educational and portfolio demonstration purposes.
    """)
