import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class EventScheduler:
    """
    Identifies 'interesting' bars where a decision might be needed.
    Used to skip empty periods in bar-by-bar simulation.
    """
    
    @staticmethod
    def get_events(df: pd.DataFrame, recipe: Dict[str, Any]) -> Optional[List[int]]:
        """
        Get sorted list of indices where signals MIGHT occur.
        Returns None if optimization is not possible (must check all bars).
        
        Args:
            df: DataFrame with OHLCV data
            recipe: Strategy recipe dictionary
        """
        try:
            trigger = recipe.get("entry_trigger", {})
            t_type = trigger.get("type", "").lower()
            
            # 1. EMA Cross
            if "ema_cross" in t_type:
                return EventScheduler._scan_ema_cross(df, trigger)
            
            # 2. RSI Extreme
            if "rsi" in t_type:
                return EventScheduler._scan_rsi(df, trigger)
                
            # 3. Composite (AND/OR) - specific case for common combos?
            # For now, default to None (safe mode)
            
            print(f"[FastForward] Unknown trigger type '{t_type}', running full simulation.")
            return None
            
        except Exception as e:
            print(f"[FastForward] Error predicting events: {e}. Running full simulation.")
            return None

    @staticmethod
    def _scan_ema_cross(df: pd.DataFrame, config: Dict[str, Any]) -> List[int]:
        """Vectorized scan for EMA Cross."""
        fast_len = config.get("fast", 9)
        slow_len = config.get("slow", 21)
        
        close = df['close']
        fast_ema = close.ewm(span=fast_len, adjust=False).mean()
        slow_ema = close.ewm(span=slow_len, adjust=False).mean()
        
        # Find where relation changes
        # fast > slow
        above = fast_ema > slow_ema
        
        # Cross occurred where 'above' value changes
        # shift(1) compares current to prev
        # fillna(False) to handle first bar
        crosses = (above != above.shift(1)).fillna(False)
        
        # Get indices
        # We need to ensure we return the index of the BAR that completes the cross
        # indices here are from df.index.
        # But we need INTEGER indices for the stepper.
        # df.index might be DatetimeIndex.
        # We need iloc positions.
        
        indices = np.where(crosses)[0].tolist()
        
        print(f"[FastForward] Found {len(indices)} potential EMA Check points")
        return indices

    @staticmethod
    def _scan_rsi(df: pd.DataFrame, config: Dict[str, Any]) -> List[int]:
        """Vectorized scan for RSI Extreme."""
        length = config.get("length", 14)
        oversold = config.get("oversold", 30)
        overbought = config.get("overbought", 70)
        
        close = df['close']
        delta = close.diff()
        
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan) # Handle div by zero
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
        
        # Check condition
        mask = (rsi <= oversold) | (rsi >= overbought)
        
        indices = np.where(mask)[0].tolist()
        print(f"[FastForward] Found {len(indices)} potential RSI Check points")
        return indices
