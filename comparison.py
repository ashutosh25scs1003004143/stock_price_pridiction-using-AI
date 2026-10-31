import pandas as pd
import yfinance as yf

def compare_stocks(symbols, period="1y"):
    data = {}
    for sym in symbols:
        sym = sym.strip().upper()
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=period)
            if not df.empty:
                data[sym] = df['Close']
        except Exception:
            continue
            
    if not data:
        return None, None
        
    comparison_df = pd.DataFrame(data).dropna()
    normalized_df = (comparison_df / comparison_df.iloc[0]) * 100
    
    return comparison_df, normalized_df
