
import sys
import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

# Import the wrapper directly to test logic
from src.models.model_registry_init import PullerXGBoostWrapper

def fetch_yfinance_data(ticker="MES=F", days=7):
    """Fetch 1m data exactly like the backend."""
    print(f"Fetching {days} days of 1m data for {ticker}...")
    end = datetime.now()
    start = end - timedelta(days=days)
    
    # Using the same call structure as src/server/main.py
    yf_ticker = yf.Ticker(ticker)
    df = yf_ticker.history(start=start, end=end, interval="1m")
    
    if df.empty:
        print("No data found!")
        return []
        
    # Standardize columns
    df.columns = [c.lower() for c in df.columns]
    df = df.reset_index()
    
    # Handle time column
    time_col = None
    for col in ['Datetime', 'datetime', 'Date', 'date', 'time']:
        if col in df.columns:
            time_col = col
            break
            
    if not time_col:
        print("No time column found!")
        return []
        
    bars = []
    for _, row in df.iterrows():
        ts = row[time_col]
        # Convert to ISO format string to match UI
        if hasattr(ts, 'isoformat'):
            ts_str = ts.isoformat()
        else:
            ts_str = str(ts)

        bars.append({
            'time': ts_str,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row.get('volume', 0))
        })
        
    print(f"Loaded {len(bars)} bars")
    return bars

def run_simulation(bars, model_path='models/puller_xgb_4class.json'):
    """Run inference bar-by-bar."""
    print(f"Loading model from {model_path}...")
    try:
        model = PullerXGBoostWrapper(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Running simulation...")
    triggers = []
    
    # Need at least 30 bars for features
    for i in range(30, len(bars)):
        # Current window of 30 bars (ending at i)
        window = bars[i-29:i+1] 
        current_bar = bars[i]
        
        # Predict
        # Wrapper expects 'bars' key in features
        try:
            result = model.predict({'bars': window})
        except Exception as e:
            print(f"Prediction error at {i}: {e}")
            continue
        
        if result['triggered']:
            triggers.append({
                'time': current_bar['time'],
                'price': current_bar['close'],
                'direction': result['direction'],
                'probs': result['probs']
            })
            probs_str = ", ".join([f"{p:.2f}" for p in result['probs']])
            print(f"TRIGGER: {current_bar['time']} {result['direction']} @ {current_bar['close']:.2f} (Probs: [{probs_str}])")
            
    print(f"\nTotal Triggers: {len(triggers)}")
    return triggers

if __name__ == "__main__":
    bars = fetch_yfinance_data()
    if bars:
        run_simulation(bars)
