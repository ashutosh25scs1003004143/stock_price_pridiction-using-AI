import pandas as pd
import yfinance as yf

# Pre-populated Categorized Stock Directory for Indian and US Markets
STOCK_CATEGORIES = {
    "Indian IT": {
        "TCS (Tata Consultancy Services)": "TCS.NS",
        "Infosys": "INFY.NS",
        "Wipro": "WIPRO.NS",
        "HCL Technologies": "HCLTECH.NS",
        "Tech Mahindra": "TECHM.NS"
    },
    "Indian Banking & Finance": {
        "HDFC Bank": "HDFCBANK.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "State Bank of India (SBI)": "SBIN.NS",
        "Axis Bank": "AXISBANK.NS",
        "Kotak Mahindra Bank": "KOTAKBANK.NS"
    },
    "Indian Energy & Auto": {
        "Reliance Industries": "RELIANCE.NS",
        "Tata Motors": "TATAMOTORS.NS",
        "NTPC Limited": "NTPC.NS",
        "Power Grid Corporation": "POWERGRID.NS",
        "ONGC": "ONGC.NS"
    },
    "Indian FMCG": {
        "Hindustan Unilever": "HINDUNILVR.NS",
        "ITC Limited": "ITC.NS",
        "Nestlé India": "NESTLEIND.NS",
        "Britannia Industries": "BRITANNIA.NS"
    },
    "US Big Tech": {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Alphabet (Google)": "GOOGL",
        "Amazon": "AMZN",
        "Meta Platforms": "META",
        "NVIDIA": "NVDA",
        "Netflix": "NFLX"
    },
    "US EV & Finance": {
        "Tesla": "TSLA",
        "JPMorgan Chase": "JPM",
        "Visa": "V",
        "Mastercard": "MA",
        "Bank of America": "BAC"
    }
}

def get_stock_categories():
    """
    Returns the dictionary of categorized stock symbols for UI dropdowns.
    """
    return STOCK_CATEGORIES


def fetch_stock_data(symbol, period="1y"):
    """
    Fetches historical stock data using yfinance.
    Handles missing values and ticker errors.
    """
    try:
        ticker = yf.Ticker(symbol.strip().upper())
        df = ticker.history(period=period)
        if df.empty:
            return None, "No data found for the given symbol."
        
        df = df.dropna()
        if df.empty:
            return None, "Data is empty after cleaning missing values."
        
        df.reset_index(inplace=True)
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
            
        return df, None
    except Exception as e:
        return None, str(e)


def get_stock_info(symbol):
    """
    Fetches latest quote and company info with robust fallbacks.
    """
    try:
        ticker = yf.Ticker(symbol.strip().upper())
        info = ticker.info
        hist = ticker.history(period="5d")
        if hist.empty:
            return None, "No historical info found."
            
        latest_row = hist.iloc[-1]
        prev_row = hist.iloc[-2] if len(hist) > 1 else latest_row
        
        close_price = latest_row['Close']
        prev_close = prev_row['Close']
        change = close_price - prev_close
        pct_change = (change / prev_close) * 100 if prev_close != 0 else 0
        
        data = {
            "symbol": symbol.upper(),
            "shortName": info.get("shortName", symbol.upper()),
            "current_price": close_price,
            "previous_close": prev_close,
            "change": change,
            "pct_change": pct_change,
            "open": latest_row['Open'],
            "high": latest_row['High'],
            "low": latest_row['Low'],
            "volume": latest_row['Volume'],
            "currency": info.get("currency", "USD"),
            "timestamp": str(hist.index[-1].date())
        }
        return data, None
    except Exception as e:
        try:
            ticker = yf.Ticker(symbol.strip().upper())
            hist = ticker.history(period="2d")
            if hist.empty:
                return None, str(e)
            latest = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else latest
            change = latest['Close'] - prev['Close']
            pct_change = (change / prev['Close']) * 100
            data = {
                "symbol": symbol.upper(),
                "shortName": symbol.upper(),
                "current_price": latest['Close'],
                "previous_close": prev['Close'],
                "change": change,
                "pct_change": pct_change,
                "open": latest['Open'],
                "high": latest['High'],
                "low": latest['Low'],
                "volume": latest['Volume'],
                "currency": "USD",
                "timestamp": str(hist.index[-1].date())
            }
            return data, None
        except Exception as inner_e:
            return None, str(inner_e)
