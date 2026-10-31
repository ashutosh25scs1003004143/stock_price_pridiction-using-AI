import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from indicators import add_all_indicators

def prepare_ml_data(df):
    """
    Prepares features and target variable without data leakage.
    """
    df = add_all_indicators(df)
    df['Target'] = df['Close'].shift(-1)
    
    features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Daily_Return']
    df_ml = df[features + ['Target', 'Date']].dropna()
    
    X = df_ml[features]
    y = df_ml['Target']
    dates = df_ml['Date']
    
    return X, y, dates, df_ml

def train_and_evaluate_model(X, y, model_type="Random Forest Regressor"):
    """
    Chronological train/test split (80% train, 20% test).
    """
    split_idx = int(len(X) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    if model_type == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }
    
    return model, metrics, X_test, y_test, y_pred

def predict_future(model, df_ml, days=7):
    """
    Recursive forecasting for future trading sessions.
    """
    features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Daily_Return']
    last_row = df_ml.iloc[-1].copy()
    
    predictions = []
    # Explicitly cast to float array to prevent assignment errors
    current_features = last_row[features].values.astype(float).reshape(1, -1)
    last_close = last_row['Close']
    
    for i in range(days):
        pred_price = model.predict(current_features)[0]
        predictions.append(pred_price)
        
        daily_ret = (pred_price - last_close) / last_close if last_close != 0 else 0
        last_close = pred_price
        
        current_features[0, 0] = float(pred_price)
        current_features[0, 6] = float(daily_ret)
        
    return predictions
