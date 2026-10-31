import pandas as pd

def generate_ai_analysis(df, info_data, prediction_prices=None):
    """
    Analyzes technical indicators and stock metrics to generate
    professional automated AI commentary and insights.
    """
    if df is None or df.empty:
        return ["Insufficient data available for generating AI insights."]
        
    latest = df.iloc[-1]
    close = latest['Close']
    sma_20 = latest.get('SMA_20', close)
    sma_50 = latest.get('SMA_50', close)
    rsi = latest.get('RSI', 50)
    macd = latest.get('MACD', 0)
    macd_signal = latest.get('MACD_Signal', 0)
    
    insights = []
    
    # 1. Trend Analysis (Moving Averages)
    if close > sma_20 and sma_20 > sma_50:
        insights.append("🟢 **Primary Trend (Bullish):** The stock is currently trading above both its 20-day and 50-day Moving Averages, indicating robust upward momentum and buyer strength.")
    elif close < sma_20 and sma_20 < sma_50:
        insights.append("🔴 **Primary Trend (Bearish):** The stock price is currently suppressed below key moving averages, signaling persistent downward pressure and caution for long positions.")
    else:
        insights.append("🟡 **Primary Trend (Consolidation):** Moving averages reflect mixed signals, suggesting the stock is in a consolidation or transitional phase.")
        
    # 2. Momentum Analysis (RSI)
    if rsi > 70:
        insights.append(f"⚠️ **Momentum Risk (Overbought):** The RSI stands at **{rsi:.1f}** (>70). This indicates the asset may be overbought in the short term, increasing the probability of a minor pullback or profit-booking.")
    elif rsi < 30:
        insights.append(f"💡 **Momentum Opportunity (Oversold):** The RSI stands at **{rsi:.1f}** (<30). Historically, levels near or below 30 indicate oversold conditions, which value buyers frequently monitor.")
    else:
        insights.append(f"⚖️ **Momentum Balance (Neutral):** The RSI is at **{rsi:.1f}**, showing balanced market momentum without extreme buying or selling exhaustion.")
        
    # 3. MACD Crossover Analysis
    if macd > macd_signal:
        insights.append("📈 **MACD Indicator:** The MACD line is positioned above the signal line, confirming short-term bullish momentum continuation.")
    else:
        insights.append("📉 **MACD Indicator:** The MACD line is below the signal line, indicating short-term bearish or weakening momentum.")
        
    # 4. Machine Learning Forecast Integration
    if prediction_prices and len(prediction_prices) > 0:
        start_price = close
        end_pred = prediction_prices[-1]
        diff = end_pred - start_price
        pct_diff = (diff / start_price) * 100
        direction = "gain" if diff >= 0 else "decline"
        emoji = "🚀" if diff >= 0 else "🔻"
        
        insights.append(f"{emoji} **AI 7-Day Forecast Outlook:** The machine learning prediction model projects a potential {direction} of **{pct_diff:.2f}%** over the next 7 trading sessions, shifting from the current price of {start_price:.2f} to an estimated target of **{end_pred:.2f}**.")
        
    return insights


def generate_detailed_trade_analysis(df, symbol):
    """
    Generates a comprehensive trade analysis report including executive summary,
    indicator breakdowns, and step-by-step trading evaluation steps.
    """
    if df is None or df.empty:
        return {"error": "Insufficient data for detailed analysis."}
        
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    close = latest['Close']
    prev_close = prev['Close']
    price_change = close - prev_close
    pct_change = (price_change / prev_close) * 100
    
    sma_20 = latest.get('SMA_20', close)
    sma_50 = latest.get('SMA_50', close)
    rsi = latest.get('RSI', 50)
    macd = latest.get('MACD', 0)
    macd_signal = latest.get('MACD_Signal', 0)
    
    if close > sma_20 and sma_20 > sma_50 and rsi < 70:
        market_bias = "Strong Bullish Continuation"
        bias_color = "🟢"
    elif close < sma_20 and sma_20 < sma_50 and rsi > 30:
        market_bias = "Persistent Bearish Pressure"
        bias_color = "🔴"
    elif rsi >= 70:
        market_bias = "Overbought / Correction Risk"
        bias_color = "⚠️"
    elif rsi <= 30:
        market_bias = "Oversold / Potential Reversal Zone"
        bias_color = "💡"
    else:
        market_bias = "Neutral / Sideways Consolidation"
        bias_color = "⚖️"

    steps = [
        {
            "step": "Step 1: Trend Confirmation (Moving Averages)",
            "status": "Bullish" if close > sma_20 else "Bearish",
            "detail": f"Current price ({close:.2f}) is {'above' if close > sma_20 else 'below'} the 20-day SMA ({sma_20:.2f}) and 50-day SMA ({sma_50:.2f}). This confirms a {'primary uptrend' if close > sma_20 else 'downtrend'}."
        },
        {
            "step": "Step 2: Momentum & Exhaustion Check (RSI)",
            "status": "Overbought (>70)" if rsi > 70 else ("Oversold (<30)" if rsi < 30 else "Balanced (30-70)"),
            "detail": f"The 14-period RSI reads **{rsi:.1f}**. {'Buyers are in control, but watch for overextension.' if rsi > 50 else 'Sellers hold short-term dominance; monitor for accumulation zones.'}"
        },
        {
            "step": "Step 3: Trend Momentum Divergence (MACD)",
            "status": "Bullish Crossover" if macd > macd_signal else "Bearish Crossover",
            "detail": f"The MACD line ({macd:.2f}) is {'above' if macd > macd_signal else 'below'} the signal line ({macd_signal:.2f}), confirming {'positive' if macd > macd_signal else 'negative'} momentum velocity."
        },
        {
            "step": "Step 4: Risk Management & Actionable Strategy",
            "status": "Execution Guidance",
            "detail": f"For risk-managed positioning on **{symbol}**, utilize trailing stop-losses near the 20-day moving average ({sma_20:.2f}). Avoid aggressive entries during high volatility windows."
        }
    ]

    return {
        "symbol": symbol,
        "bias": market_bias,
        "bias_icon": bias_color,
        "current_price": close,
        "price_change": price_change,
        "pct_change": pct_change,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "rsi": rsi,
        "macd": macd,
        "steps": steps
    }
