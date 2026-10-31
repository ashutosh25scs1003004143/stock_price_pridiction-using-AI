import pandas as pd

def calculate_sma(df, window=20):
    return df['Close'].rolling(window=window).mean()

def calculate_ema(df, window=20):
    return df['Close'].ewm(span=window, adjust=False).mean()

def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_macd(df, slow=26, fast=12, signal=9):
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def add_all_indicators(df):
    df = df.copy()
    df['SMA_20'] = calculate_sma(df, 20)
    df['SMA_50'] = calculate_sma(df, 50)
    df['EMA_20'] = calculate_ema(df, 20)
    df['RSI'] = calculate_rsi(df, 14)
    macd, signal, hist = calculate_macd(df)
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist
    df['Daily_Return'] = df['Close'].pct_change()
    return df
