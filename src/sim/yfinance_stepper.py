"""
YFinance Stepper

A MarketStepper implementation that uses yfinance for data.
- Starts with N days of history (backfill).
- Simulates through history at requested speed.
- When history catches up to now, switches to LIVE mode:
  - Polls yfinance periodically for the newest closed bar.
  - Yields new bars in real-time or None if waiting.
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
            # Empty init fallback
            self.df = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        # State
        self.current_idx = max(0, lookback_padding) if len(self.df) > lookback_padding else 0
        self.live_mode = False
        self.last_poll_time = 0
        self.poll_interval = 20  # Seconds between API calls
        
        print(f"[YF] Loaded {len(self.df)} bars. Starting at index {self.current_idx}", file=sys.stderr)
        if len(self.df) > 0:
            print(f"[YF] Range: {self.df['time'].iloc[0]} -> {self.df['time'].iloc[-1]}", file=sys.stderr)

    def _fetch_initial_history(self, days: int) -> pd.DataFrame:
        """Fetch historical data."""
        # YFinance 1m is max 7 days
        days = min(days, 7)
        end = datetime.now()
        start = end - timedelta(days=days)
        
        try:
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
        except Exception as e:
            print(f"[YF] Init Error: {e}", file=sys.stderr)
            return pd.DataFrame()

    def step(self) -> Optional[StepResult]:
        """
        Advance one step. NON-BLOCKING.
        If in history: returns next bar immediately.
        If at live edge: Checks API once. If new bar, returns it. Else returns None.
        """
        # Check if we are at the end of known data
        if self.current_idx >= len(self.df):
            self.live_mode = True
            self._poll_for_new_bar_once()
            
            # Check again if data arrived
            if self.current_idx >= len(self.df):
                return None  # Still waiting
        
        # Return current bar
        bar = self.df.iloc[self.current_idx]
        idx = self.current_idx
        self.current_idx += 1
        
        return StepResult(bar=bar, bar_idx=idx, is_done=False)

    def _poll_for_new_bar_once(self):
        """Poll YFinance API once if interval has passed. NON-BLOCKING."""
        now = time.time()
        if now - self.last_poll_time < self.poll_interval:
            return

        self.last_poll_time = now
        print(f"[YF] Checking for new candle...", file=sys.stderr)
        
        try:
            # Fetch just the last day to get latest
            latest = self.ticker.history(period="1d", interval="1m")
            if len(latest) == 0:
                return
            
            # Normalize
            latest.columns = [c.lower() for c in latest.columns]
            latest = latest.reset_index()
            if 'Datetime' in latest.columns:
                latest['time'] = latest['Datetime']
            elif 'datetime' in latest.columns:
                latest['time'] = latest['datetime']
            if latest['time'].dt.tz is None:
                latest['time'] = latest['time'].dt.tz_localize(EST)
            else:
                latest['time'] = latest['time'].dt.tz_convert(EST)
            
            latest = latest[['time', 'open', 'high', 'low', 'close', 'volume']]
            
            # Filter for strictly new bars
            if len(self.df) > 0:
                last_timestamp = self.df['time'].iloc[-1]
                new_bars = latest[latest['time'] > last_timestamp]
            else:
                new_bars = latest

            if not new_bars.empty:
                print(f"[YF] Found {len(new_bars)} new bars. Latest: {new_bars['time'].iloc[-1]}", file=sys.stderr)
                # Append to internal dataframe
                self.df = pd.concat([self.df, new_bars], ignore_index=True)
            
        except Exception as e:
            print(f"[YF] Poll error: {e}", file=sys.stderr)

    def get_history(self, lookback: int) -> pd.DataFrame:
        """Get CAUSAL history from current point."""
        end_idx = self.current_idx
        start_idx = max(0, end_idx - lookback)
        return self.df.iloc[start_idx:end_idx].copy()
