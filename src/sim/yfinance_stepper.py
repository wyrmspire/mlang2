"""
YFinance Stepper

A MarketStepper implementation that uses yfinance for data.
- Starts with N days of history (backfill).
- Simulates through history at requested speed.
- When history catches up to now, switches to LIVE mode:
  - Polls yfinance every minute for the newest closed bar.
  - Yields new bars in real-time.
"""

import time
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, List
from zoneinfo import ZoneInfo

from src.sim.stepper import StepResult

EST = ZoneInfo("America/New_York")


class YFinanceStepper:
    """
    Market simulation using yfinance data.
    Seamless transition from historical backfill to live polling.
    """
    
    def __init__(
        self,
        ticker: str = "MES=F",
        days_back: int = 7,
        lookback_padding: int = 60,
    ):
        """
        Args:
            ticker: Symbol to trade (default MES=F)
            days_back: Number of days of history to load (max 7 for 1m)
            lookback_padding: Extra bars to keep for indicator calculation
        """
        self.ticker_symbol = ticker
        self.interval = "1m"
        self.ticker = yf.Ticker(ticker)
        
        # Load initial history
        print(f"[YF] Loading {days_back} days history for {ticker}...", file=sys.stderr)
        self.df = self._fetch_initial_history(days_back)
        
        if len(self.df) == 0:
            raise ValueError(f"No data found for {ticker}")
        
        # State
        self.current_idx = lookback_padding  # Start after padding
        self.live_mode = False
        self.last_poll_time = time.time()
        
        print(f"[YF] Loaded {len(self.df)} bars. Starting at index {self.current_idx}", file=sys.stderr)
        print(f"[YF] Range: {self.df['time'].iloc[0]} -> {self.df['time'].iloc[-1]}", file=sys.stderr)

    def _fetch_initial_history(self, days: int) -> pd.DataFrame:
        """Fetch historical data."""
        # YFinance 1m is max 7 days
        days = min(days, 7)
        end = datetime.now()
        start = end - timedelta(days=days)
        
        df = self.ticker.history(start=start, end=end, interval="1m")
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        # Normalize columns
        df.columns = [c.lower() for c in df.columns]
        df = df.reset_index()
        
        # Handle timezone
        if 'Datetime' in df.columns:
            df['time'] = df['Datetime']
        elif 'datetime' in df.columns:
            df['time'] = df['datetime']
        
        # Ensure TZ aware (NY)
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize(EST)
        else:
            df['time'] = df['time'].dt.tz_convert(EST)
            
        return df[['time', 'open', 'high', 'low', 'close', 'volume']]

    def step(self) -> StepResult:
        """
        Advance one step.
        If in history: returns next bar immediately.
        If at live edge: POLLS until new bar appears.
        """
        # Check if we are at the end of known data
        if self.current_idx >= len(self.df):
            self.live_mode = True
            self._poll_for_new_bar()
        
        # Return current bar
        bar = self.df.iloc[self.current_idx]
        idx = self.current_idx
        self.current_idx += 1
        
        return StepResult(bar=bar, bar_idx=idx, is_done=False)

    def _poll_for_new_bar(self):
        """Block and poll until a NEW bar appears."""
        print(f"[YF] Live mode: Waiting for next {self.interval} candle...", file=sys.stderr)
        
        last_timestamp = self.df['time'].iloc[-1]
        
        while True:
            # Respect API limits - poll every 30s
            time.sleep(30)
            
            try:
                # Fetch just the last day or hour to get latest
                latest = self.ticker.history(period="1d", interval="1m")
                if len(latest) == 0:
                    continue
                
                # Normalize
                latest.columns = [c.lower() for c in latest.columns]
                latest = latest.reset_index()
                if 'Datetime' in latest.columns:
                    latest['time'] = latest['Datetime']
                if latest['time'].dt.tz is None:
                    latest['time'] = latest['time'].dt.tz_localize(EST)
                else:
                    latest['time'] = latest['time'].dt.tz_convert(EST)
                
                latest = latest[['time', 'open', 'high', 'low', 'close', 'volume']]
                
                # Check for new bar
                # The last bar in 'latest' might be the currently forming bar.
                # We usually want completed bars. YFinance '1m' often returns the latest minute incomplete?
                # Actually for 1m, yfinance returns up to the last minute.
                
                # Latest closed candidate
                candidate = latest.iloc[-1]
                
                if candidate['time'] > last_timestamp:
                    # New bar confirmed
                    print(f"[YF] New bar: {candidate['time']}", file=sys.stderr)
                    
                    # Append to internal dataframe
                    # Use pandas concat to avoid SettingWithCopy warning/fragmentation
                    self.df = pd.concat([self.df, latest.iloc[[-1]]], ignore_index=True)
                    return
                
            except Exception as e:
                print(f"[YF] Poll error: {e}", file=sys.stderr)
                time.sleep(5)

    def get_history(self, lookback: int) -> pd.DataFrame:
        """Get CAUSAL history from current point."""
        # Calculate END index (exclusive)
        # step() increments current_idx AFTER returning.
        # So current_idx points to the *next* bar (future).
        # We want history UP TO the bar just returned.
        end_idx = self.current_idx
        start_idx = max(0, end_idx - lookback)
        return self.df.iloc[start_idx:end_idx].copy()

