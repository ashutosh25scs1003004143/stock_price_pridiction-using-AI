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
