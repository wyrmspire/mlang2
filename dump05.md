
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class PatternResult:
    pattern_type: str  # "FLAG", "WEDGE", "PULLBACK"
    direction: str     # "BULLISH", "BEARISH"
    start_idx: int     # Index in the passed dataframe
    end_idx: int
    entry_price: float
    stop_loss: float
    target_price: float
    confidence: float  # 0.0 to 1.0

def _get_slope(y: np.ndarray) -> float:
    """Calculate linear regression slope of an array."""
    x = np.arange(len(y))
    if len(x) < 2:
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return slope

def detect_flags(
    df: pd.DataFrame,
    lookback: int = 30,
    pole_bars: int = 15,
    flag_bars: int = 15,
    atr_period: int = 14
) -> List[PatternResult]:
    """
    Detect flag patterns (Bull/Bear Flags).

    Args:
        df: DataFrame with open, high, low, close
        lookback: Total window size to analyze
        pole_bars: Number of bars for the flagpole
        flag_bars: Number of bars for the flag/consolidation
        atr_period: ATR period for volatility normalization

    Returns:
        List of detected patterns.
    """
    results = []
    if len(df) < lookback + atr_period:
        return results

    # Calculate ATR for normalization
    # Simple TR calculation
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    atr = tr.rolling(window=atr_period).mean()

    # Iterate through the dataframe
    # We only check the most recent window for efficiency if used in real-time
    # But if scanning history, we might want to slide.
    # For now, let's just check the *current* window (tail) to act as a signal generator.
    # If the user wants historical scan, they can loop outside or we can loop here.
    # Given the request is "analyze price historically", I'll loop.

    for i in range(lookback, len(df) + 1):
        window = df.iloc[i-lookback:i]
        curr_atr = atr.iloc[i-1]

        if pd.isna(curr_atr) or curr_atr == 0:
            continue

        pole_data = window.iloc[:pole_bars]
        flag_data = window.iloc[pole_bars:]

        # --- BULL FLAG CHECK ---
        # 1. Pole: Strong move up
        pole_move = pole_data['close'].iloc[-1] - pole_data['open'].iloc[0]
        if pole_move > 2.0 * curr_atr: # Significant move
            # 2. Flag: Consolidation sloping down
            flag_highs = flag_data['high'].values
            flag_lows = flag_data['low'].values

            slope_highs = _get_slope(flag_highs)
            slope_lows = _get_slope(flag_lows)

            # Check for downward slope
            if slope_highs < 0 and slope_lows < 0:
                # Check for parallel (roughly)
                if abs(slope_highs - slope_lows) < 0.5 * (curr_atr / flag_bars):
                    # Found Bull Flag
                    results.append(PatternResult(
                        pattern_type="FLAG",
                        direction="BULLISH",
                        start_idx=window.index[0],
                        end_idx=window.index[-1],
                        entry_price=flag_data['high'].max(), # Breakout entry
                        stop_loss=flag_data['low'].min(),
                        target_price=flag_data['high'].max() + pole_move, # Measured move
                        confidence=0.8
                    ))

        # --- BEAR FLAG CHECK ---
        # 1. Pole: Strong move down
        pole_move = pole_data['close'].iloc[-1] - pole_data['open'].iloc[0]
        if pole_move < -2.0 * curr_atr:
            # 2. Flag: Consolidation sloping up
            flag_highs = flag_data['high'].values
            flag_lows = flag_data['low'].values

            slope_highs = _get_slope(flag_highs)
            slope_lows = _get_slope(flag_lows)

            # Check for upward slope
            if slope_highs > 0 and slope_lows > 0:
                # Check for parallel
                if abs(slope_highs - slope_lows) < 0.5 * (curr_atr / flag_bars):
                    # Found Bear Flag
                    results.append(PatternResult(
                        pattern_type="FLAG",
                        direction="BEARISH",
                        start_idx=window.index[0],
                        end_idx=window.index[-1],
                        entry_price=flag_data['low'].min(),
                        stop_loss=flag_data['high'].max(),
                        target_price=flag_data['low'].min() + pole_move,
                        confidence=0.8
                    ))

    return results

def detect_wedges(
    df: pd.DataFrame,
    lookback: int = 30,
    atr_period: int = 14
) -> List[PatternResult]:
    """
    Detect wedge patterns (Falling/Rising Wedges).
    """
    results = []
    if len(df) < lookback + atr_period:
        return results

    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    atr = tr.rolling(window=atr_period).mean()

    for i in range(lookback, len(df) + 1):
        window = df.iloc[i-lookback:i]
        curr_atr = atr.iloc[i-1]

        if pd.isna(curr_atr) or curr_atr == 0:
            continue

        highs = window['high'].values
        lows = window['low'].values

        slope_highs = _get_slope(highs)
        slope_lows = _get_slope(lows)

        # --- FALLING WEDGE (BULLISH) ---
        # Both slopes negative, but lows steeper than highs (converging)
        # OR Highs negative, Lows flat/slightly negative
        # slope_lows > slope_highs means lows are LESS NEGATIVE (flatter) than highs?
        # Wait, if both negative: -1 (highs) vs -0.5 (lows). -0.5 > -1. Correct.
        # But for falling wedge, usually lows are steeper? No, highs are steeper?
        # Falling wedge: Price narrows. Highs drop fast. Lows drop slow.
        # So slope_highs is more negative (smaller). slope_lows is less negative (larger).
        # So slope_lows > slope_highs.

        if slope_highs < 0 and slope_lows < 0:
            if slope_lows > slope_highs: # Converging
                 results.append(PatternResult(
                    pattern_type="WEDGE",
                    direction="BULLISH",
                    start_idx=window.index[0],
                    end_idx=window.index[-1],
                    entry_price=window['high'].iloc[-1],
                    stop_loss=window['low'].min(),
                    target_price=window['high'].iloc[-1] + (window['high'].max() - window['low'].min()),
                    confidence=0.7
                ))

        # --- RISING WEDGE (BEARISH) ---
        # Both slopes positive, but lows steeper (converging)
        if slope_highs > 0 and slope_lows > 0:
             if slope_lows > slope_highs: # Converging (Lows rising faster than Highs? No, Highs rising slower than Lows)
                # Wait, for rising wedge:
                # Highs slope positive. Lows slope positive.
                # Lines converge. So Highs slope < Lows slope.
                results.append(PatternResult(
                    pattern_type="WEDGE",
                    direction="BEARISH",
                    start_idx=window.index[0],
                    end_idx=window.index[-1],
                    entry_price=window['low'].iloc[-1],
                    stop_loss=window['high'].max(),
                    target_price=window['low'].iloc[-1] - (window['high'].max() - window['low'].min()),
                    confidence=0.7
                ))

    return results

def detect_pullback(
    df: pd.DataFrame,
    ema_period: int = 20,
    lookback: int = 5
) -> List[PatternResult]:
    """
    Detect pullbacks to EMA.
    """
    results = []
    if len(df) < ema_period + lookback:
        return results

    ema = df['close'].ewm(span=ema_period, adjust=False).mean()

    # Check just the last bar for efficiency (this is more of a signal check)
    # But to be consistent, we scan

    for i in range(ema_period + lookback, len(df) + 1):
        curr_idx = i - 1
        curr_close = df['close'].iloc[curr_idx]
        curr_ema = ema.iloc[curr_idx]
        prev_close = df['close'].iloc[curr_idx-1]

        # Determine trend from EMA slope
        # Need to ensure we don't go out of bounds
        if curr_idx < 5:
            continue

        ema_slope = ema.iloc[curr_idx] - ema.iloc[curr_idx-5]

        # Bullish Pullback
        # Trend is up, Price pulled back to near EMA
        if ema_slope > 0:
            # Check if we touched EMA recently or are close
            dist = (curr_close - curr_ema) / curr_ema
            # Relax tolerance for testing
            if abs(dist) < 0.005: # 0.5%
                results.append(PatternResult(
                    pattern_type="PULLBACK",
                    direction="BULLISH",
                    start_idx=df.index[curr_idx],
                    end_idx=df.index[curr_idx],
                    entry_price=curr_close,
                    stop_loss=curr_ema * 0.995,
                    target_price=curr_close * 1.02,
                    confidence=0.6
                ))

        # Bearish Pullback
        if ema_slope < 0:
            dist = (curr_close - curr_ema) / curr_ema
            if abs(dist) < 0.005:
                results.append(PatternResult(
                    pattern_type="PULLBACK",
                    direction="BEARISH",
                    start_idx=df.index[curr_idx],
                    end_idx=df.index[curr_idx],
                    entry_price=curr_close,
                    stop_loss=curr_ema * 1.005,
                    target_price=curr_close * 0.98,
                    confidence=0.6
                ))

    return results

```

### src/features/pipeline.py

```python
"""
Feature Pipeline
Compose all features into a single bundle for model input.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from src.sim.stepper import MarketStepper
from src.features.state import MarketState, get_market_state, normalize_ohlcv
from src.features.indicators import IndicatorValues
from src.features.levels import LevelValues
from src.features.time_features import TimeFeatures, compute_time_features
from src.features.context import ContextFeatures, compute_context_features
from src.config import DEFAULT_LOOKBACK_1M, DEFAULT_LOOKBACK_5M, DEFAULT_LOOKBACK_15M


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""
    # Lookback windows (in bars)
    lookback_1m: int = DEFAULT_LOOKBACK_1M    # 120 bars = 2 hours
    lookback_5m: int = DEFAULT_LOOKBACK_5M    # 24 bars = 2 hours
    lookback_15m: int = DEFAULT_LOOKBACK_15M  # 8 bars = 2 hours
    
    # What to include
    include_ohlcv: bool = True
    include_indicators: bool = True
    include_levels: bool = True
    include_time: bool = True
    
    # Normalization
    price_norm: str = "zscore"  # 'zscore', 'minmax', 'none'
    
    def to_dict(self) -> dict:
        return {
            'lookback_1m': self.lookback_1m,
            'lookback_5m': self.lookback_5m,
            'lookback_15m': self.lookback_15m,
            'include_ohlcv': self.include_ohlcv,
            'include_indicators': self.include_indicators,
            'include_levels': self.include_levels,
            'include_time': self.include_time,
            'price_norm': self.price_norm,
        }


@dataclass
class FeatureBundle:
    """
    Complete feature bundle for a single decision point.
    
    Separates price windows (for CNN) from context vector (for MLP).
    """
    # Price windows - (lookback, channels) for CNN
    x_price_1m: np.ndarray      # (120, 5) default
    x_price_5m: np.ndarray      # (24, 5) default  
    x_price_15m: np.ndarray     # (8, 5) default
    
    # Context vector - (dim,) for MLP
    x_context: np.ndarray       # (20,) default
    
    # Raw components (for debugging/logging)
    market_state: Optional[MarketState] = None
    indicators: Optional[IndicatorValues] = None
    levels: Optional[LevelValues] = None
    time_features: Optional[TimeFeatures] = None
    context_features: Optional[ContextFeatures] = None
    
    # Metadata
    bar_idx: int = 0
    timestamp: Optional[pd.Timestamp] = None
    current_price: float = 0.0
    atr: float = 0.0


def compute_features(
    stepper: MarketStepper,
    config: FeatureConfig,
    df_5m: pd.DataFrame = None,
    df_15m: pd.DataFrame = None,
    df_1h: pd.DataFrame = None,
    df_4h: pd.DataFrame = None,
    precomputed_indicators: Dict[int, IndicatorValues] = None,
    precomputed_levels: Dict[int, LevelValues] = None,
) -> FeatureBundle:
    """
    Compute all features at current stepper position.
    
    All data access is CAUSAL (via stepper.get_history only).
    
    Args:
        stepper: MarketStepper at current position
        config: Feature configuration
        df_5m, df_15m, df_1h, df_4h: Pre-resampled higher timeframe data
        precomputed_indicators: Optional pre-computed indicator values by bar_idx
        precomputed_levels: Optional pre-computed level values by bar_idx
        
    Returns:
        FeatureBundle with all features
    """
    # Get market state
    state = get_market_state(
        stepper,
        df_5m=df_5m,
        df_15m=df_15m,
        lookback_1m=config.lookback_1m,
        lookback_5m=config.lookback_5m,
        lookback_15m=config.lookback_15m,
    )
    
    # Normalize price windows
    x_price_1m = normalize_ohlcv(state.ohlcv_1m, config.price_norm)
    x_price_5m = normalize_ohlcv(state.ohlcv_5m, config.price_norm)
    x_price_15m = normalize_ohlcv(state.ohlcv_15m, config.price_norm)
    
    # Get indicator values (from precomputed or compute)
    bar_idx = state.current_bar_idx
    if precomputed_indicators and bar_idx in precomputed_indicators:
        indicators = precomputed_indicators[bar_idx]
    else:
        indicators = IndicatorValues()  # Empty if not provided
    
    # Get level values
    if precomputed_levels and bar_idx in precomputed_levels:
        levels = precomputed_levels[bar_idx]
    else:
        levels = LevelValues()
    
    # Time features
    time_feats = TimeFeatures()
    if state.current_time:
        time_feats = compute_time_features(state.current_time)
    
    # Context features (combines everything)
    atr = indicators.atr_5m_14 if indicators.atr_5m_14 > 0 else 1.0
    context_feats = compute_context_features(
        state.current_price,
        indicators,
        levels,
        time_feats,
        atr=atr
    )
    
    x_context = context_feats.to_array()
    
    return FeatureBundle(
        x_price_1m=x_price_1m,
        x_price_5m=x_price_5m,
        x_price_15m=x_price_15m,
        x_context=x_context,
        market_state=state,
        indicators=indicators,
        levels=levels,
        time_features=time_feats,
        context_features=context_feats,
        bar_idx=bar_idx,
        timestamp=state.current_time,
        current_price=state.current_price,
        atr=atr,
    )


def precompute_indicators(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
) -> Dict[int, IndicatorValues]:
    """
    Pre-compute all indicators for efficiency.
    
    Returns dict mapping bar_idx to IndicatorValues.
    This should be cached via FeatureStore.
    """
    from src.features.indicators import calculate_ema, calculate_rsi, calculate_atr, calculate_vwap, calculate_adr
    
    # Compute on each timeframe
    df_5m = df_5m.copy()
    df_5m['ema_20'] = calculate_ema(df_5m['close'], 20)
    df_5m['ema_200'] = calculate_ema(df_5m['close'], 200)
    df_5m['rsi_14'] = calculate_rsi(df_5m['close'], 14)
    df_5m['atr_14'] = calculate_atr(df_5m, 14)
    
    df_15m = df_15m.copy()
    df_15m['ema_20'] = calculate_ema(df_15m['close'], 20)
    df_15m['ema_200'] = calculate_ema(df_15m['close'], 200)
    df_15m['rsi_14'] = calculate_rsi(df_15m['close'], 14)
    df_15m['atr_14'] = calculate_atr(df_15m, 14)
    
    # VWAP on 1m
    df_1m = df_1m.copy()
    df_1m['vwap_session'] = calculate_vwap(df_1m, period='session')
    df_1m['vwap_weekly'] = calculate_vwap(df_1m, period='weekly')
    
    # ADR
    df_1m['adr'] = calculate_adr(df_1m, 14)
    
    # Map back to 1m bar indices
    # Forward fill 5m and 15m data to align with 1m timestamps
    # We assume df_1m, df_5m, df_15m are indexed by time or have 'time' column
    
    # Ensure all have time index for alignment
    def ensure_time_index(df):
        if 'time' in df.columns:
            return df.set_index('time')
        return df
        
    df_1m_idx = ensure_time_index(df_1m)
    df_5m_idx = ensure_time_index(df_5m)
    df_15m_idx = ensure_time_index(df_15m)
    
    # Reindex 5m/15m to 1m (forward fill)
    df_5m_aligned = df_5m_idx.reindex(df_1m_idx.index, method='ffill')
    df_15m_aligned = df_15m_idx.reindex(df_1m_idx.index, method='ffill')
    
    result = {}
    
    for idx, (time, row) in enumerate(df_1m_idx.iterrows()):
        row_5m = df_5m_aligned.iloc[idx]
        row_15m = df_15m_aligned.iloc[idx]
        
        result[idx] = IndicatorValues(
            ema_5m_20=float(row_5m.get('ema_20', 0.0)),
            ema_15m_20=float(row_15m.get('ema_20', 0.0)),
            ema_5m_200=float(row_5m.get('ema_200', 0.0)),
            ema_15m_200=float(row_15m.get('ema_200', 0.0)),
            rsi_5m_14=float(row_5m.get('rsi_14', 50.0)),
            rsi_15m_14=float(row_15m.get('rsi_14', 50.0)),
            atr_5m_14=float(row_5m.get('atr_14', 0.0)),
            atr_15m_14=float(row_15m.get('atr_14', 0.0)),
            adr_14=float(row.get('adr', 0.0)),
            vwap_session=float(row.get('vwap_session', 0.0)),
            vwap_weekly=float(row.get('vwap_weekly', 0.0)),
        )
    
    return result

```

### src/features/session_levels.py

```python
"""
Session Levels
Compute session-specific high/low levels (Asian, London).

These levels are used for ICT-style strategies that trade overnight level breaks.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional
from datetime import time
from zoneinfo import ZoneInfo

from src.config import NY_TZ


@dataclass
class SessionLevels:
    """Bundle of session high/low levels."""
    # Asian session (NY evening prior day)
    asian_high: float = 0.0
    asian_low: float = 0.0
    asian_range: float = 0.0
    
    # London session
    london_high: float = 0.0
    london_low: float = 0.0
    london_range: float = 0.0
    
    # Combined overnight range
    overnight_high: float = 0.0
    overnight_low: float = 0.0


# Session time boundaries (NY timezone)
ASIAN_START = time(19, 0)   # 7:00 PM previous day
ASIAN_END = time(0, 0)      # Midnight

LONDON_START = time(2, 0)    # 2:00 AM
LONDON_END = time(8, 30)     # 8:30 AM (cutoff for level establishment)

TRADE_WINDOW_START = time(9, 30)   # NY Open
TRADE_WINDOW_END = time(11, 30)    # Mid-morning cutoff

# Cache for session levels (keyed by date)
_session_cache = {}


def clear_session_cache():
    """Clear the session level cache. Call at start of new backtests."""
    global _session_cache
    _session_cache = {}


def compute_session_levels(
    df_1m: pd.DataFrame,
    current_time: pd.Timestamp,
    tz: ZoneInfo = NY_TZ
) -> SessionLevels:
    """
    Compute Asian and London session high/low levels.
    
    Asian Session: 19:00 - 00:00 NY (previous evening)
    London Session: 02:00 - 08:30 NY (cutoff before NY open)
    
    Uses caching to avoid recomputing for same trading day.
    
    Args:
        df_1m: 1-minute OHLCV data with 'time' column
        current_time: Current timestamp
        tz: Timezone for session boundaries
        
    Returns:
        SessionLevels with Asian and London high/low values
    """
    global _session_cache
    
    levels = SessionLevels()
    
    if df_1m is None or df_1m.empty:
        return levels
    
    # Get current date for caching
    current_ny = current_time.astimezone(tz)
    current_date = current_ny.date()
    
    # Check cache first
    if current_date in _session_cache:
        return _session_cache[current_date]
    
    df = df_1m.copy()
    
    # Ensure time column is datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC')
        df['time_ny'] = df['time'].dt.tz_convert(tz)
    else:
        return levels
    
    current_ny = current_time.astimezone(tz)
    current_date = current_ny.date()
    
    # Asian session: 19:00 previous day to 00:00 current day
    # We need to look at yesterday evening
    from datetime import datetime, timedelta
    
    prev_day = current_date - timedelta(days=1)
    
    # Asian start: yesterday at 19:00
    asian_start = datetime.combine(prev_day, ASIAN_START)
    asian_start = tz.localize(asian_start) if hasattr(tz, 'localize') else asian_start.replace(tzinfo=tz)
    
    # Asian end: today at 00:00 (midnight)
    asian_end = datetime.combine(current_date, time(0, 0))
    asian_end = tz.localize(asian_end) if hasattr(tz, 'localize') else asian_end.replace(tzinfo=tz)
    
    # Filter Asian session bars
    asian_mask = (df['time_ny'] >= asian_start) & (df['time_ny'] < asian_end)
    asian_bars = df.loc[asian_mask]
    
    if not asian_bars.empty:
        levels.asian_high = asian_bars['high'].max()
        levels.asian_low = asian_bars['low'].min()
        levels.asian_range = levels.asian_high - levels.asian_low
    
    # London session: 02:00 to 08:30 today
    london_start = datetime.combine(current_date, LONDON_START)
    london_start = tz.localize(london_start) if hasattr(tz, 'localize') else london_start.replace(tzinfo=tz)
    
    london_end = datetime.combine(current_date, LONDON_END)
    london_end = tz.localize(london_end) if hasattr(tz, 'localize') else london_end.replace(tzinfo=tz)
    
    # Only include bars up to current time and within session
    london_end_actual = min(london_end, current_ny)
    
    london_mask = (df['time_ny'] >= london_start) & (df['time_ny'] < london_end_actual)
    london_bars = df.loc[london_mask]
    
    if not london_bars.empty:
        levels.london_high = london_bars['high'].max()
        levels.london_low = london_bars['low'].min()
        levels.london_range = levels.london_high - levels.london_low
    
    # Combined overnight (Asian + London)
    if levels.asian_high > 0 or levels.london_high > 0:
        levels.overnight_high = max(
            levels.asian_high if levels.asian_high > 0 else 0,
            levels.london_high if levels.london_high > 0 else 0
        )
        levels.overnight_low = min(
            levels.asian_low if levels.asian_low > 0 else float('inf'),
            levels.london_low if levels.london_low > 0 else float('inf')
        )
        if levels.overnight_low == float('inf'):
            levels.overnight_low = 0.0
    
    # Cache the result only if London is complete (levels are final for the day)
    current_t = current_ny.time()
    if current_t >= LONDON_END:
        _session_cache[current_date] = levels
    
    return levels


def is_in_trade_window(current_time: pd.Timestamp, tz: ZoneInfo = NY_TZ) -> bool:
    """
    Check if current time is within the trade window (9:30 - 11:30 NY).
    
    Args:
        current_time: Current timestamp
        tz: Timezone
        
    Returns:
        True if within trade window
    """
    current_ny = current_time.astimezone(tz)
    current_t = current_ny.time()
    
    return TRADE_WINDOW_START <= current_t <= TRADE_WINDOW_END


def is_london_complete(current_time: pd.Timestamp, tz: ZoneInfo = NY_TZ) -> bool:
    """
    Check if London session is complete (past 8:30 AM NY).
    
    The London low/high must be established before trading.
    
    Args:
        current_time: Current timestamp
        tz: Timezone
        
    Returns:
        True if London session is complete
    """
    current_ny = current_time.astimezone(tz)
    current_t = current_ny.time()
    
    return current_t >= LONDON_END


def get_overnight_levels_for_trade(
    df_1m: pd.DataFrame,
    current_time: pd.Timestamp,
    tz: ZoneInfo = NY_TZ
) -> Optional[SessionLevels]:
    """
    Get overnight levels only if:
    1. We are in the trade window (9:30-11:30)
    2. London session is complete (past 8:30)
    
    Returns None if conditions not met.
    """
    if not is_in_trade_window(current_time, tz):
        return None
    
    if not is_london_complete(current_time, tz):
        return None
    
    return compute_session_levels(df_1m, current_time, tz)

```

### src/features/state.py

```python
"""
Market State
Raw OHLCV windows from the stepper.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict

from src.sim.stepper import MarketStepper
from src.config import DEFAULT_LOOKBACK_1M, DEFAULT_LOOKBACK_5M, DEFAULT_LOOKBACK_15M


@dataclass
class MarketState:
    """
    Raw market state at a point in time.
    All data is CAUSAL (from past only).
    """
    # Price windows (lookback, 5) for OHLCV
    ohlcv_1m: np.ndarray     # (120, 5) default - 2 hours
    ohlcv_5m: np.ndarray     # (24, 5) default - 2 hours
    ohlcv_15m: np.ndarray    # (8, 5) default - 2 hours
    
    # Current bar info
    current_price: float
    current_time: pd.Timestamp
    current_bar_idx: int
    
    # Additional context
    current_bar: Optional[pd.Series] = None


def get_market_state(
    stepper: MarketStepper,
    df_5m: pd.DataFrame = None,
    df_15m: pd.DataFrame = None,
    lookback_1m: int = DEFAULT_LOOKBACK_1M,
    lookback_5m: int = DEFAULT_LOOKBACK_5M,
    lookback_15m: int = DEFAULT_LOOKBACK_15M,
) -> MarketState:
    """
    Extract market state from stepper.
    
    CAUSAL ONLY - uses stepper.get_history().
    
    Args:
        stepper: MarketStepper positioned at current bar
        df_5m: Pre-resampled 5m data (optional, for efficiency)
        df_15m: Pre-resampled 15m data (optional)
        lookback_*: Number of bars for each timeframe
        
    Returns:
        MarketState with all price windows
    """
    # Get 1m history
    ohlcv_1m = stepper.get_history_array(
        lookback_1m, 
        columns=['open', 'high', 'low', 'close', 'volume']
    )
    
    # Get current bar info
    current_bar = stepper.get_current_bar()
    current_time = stepper.get_current_time()
    current_bar_idx = stepper.get_current_idx()
    
    if current_bar is not None:
        current_price = current_bar['close']
    else:
        current_price = 0.0
    
    # Get higher timeframe windows if data provided
    if df_5m is not None and current_time is not None:
        ohlcv_5m = _get_htf_window(df_5m, current_time, lookback_5m)
    else:
        ohlcv_5m = np.zeros((lookback_5m, 5), dtype=np.float32)
    
    if df_15m is not None and current_time is not None:
        ohlcv_15m = _get_htf_window(df_15m, current_time, lookback_15m)
    else:
        ohlcv_15m = np.zeros((lookback_15m, 5), dtype=np.float32)
    
    return MarketState(
        ohlcv_1m=ohlcv_1m,
        ohlcv_5m=ohlcv_5m,
        ohlcv_15m=ohlcv_15m,
        current_price=current_price,
        current_time=current_time,
        current_bar_idx=current_bar_idx,
        current_bar=current_bar,
    )


def _get_htf_window(
    df_htf: pd.DataFrame,
    current_time: pd.Timestamp,
    lookback: int
) -> np.ndarray:
    """Get higher timeframe window as numpy array."""
    from src.data.resample import get_htf_window
    
    window = get_htf_window(df_htf, current_time, lookback)
    
    if len(window) == 0:
        return np.zeros((lookback, 5), dtype=np.float32)
    
    values = window[['open', 'high', 'low', 'close', 'volume']].values
    
    # Pad if insufficient
    if len(values) < lookback:
        padding = np.zeros((lookback - len(values), 5), dtype=np.float32)
        values = np.vstack([padding, values])
    
    return values.astype(np.float32)


def normalize_ohlcv(
    ohlcv: np.ndarray,
    method: str = 'zscore'
) -> np.ndarray:
    """
    Normalize OHLCV window for model input.
    
    Args:
        ohlcv: (lookback, 5) array
        method: 'zscore', 'minmax', or 'none'
        
    Returns:
        Normalized array
    """
    if method == 'none':
        return ohlcv
    
    # Normalize price columns (0-3), leave volume separate
    prices = ohlcv[:, :4]
    volume = ohlcv[:, 4:5]
    
    if method == 'zscore':
        mean = np.mean(prices)
        std = np.std(prices)
        if std < 1e-8:
            std = 1.0
        prices_norm = (prices - mean) / std
        
        vol_mean = np.mean(volume)
        vol_std = np.std(volume)
        if vol_std < 1e-8:
            vol_std = 1.0
        vol_norm = (volume - vol_mean) / vol_std
        
    elif method == 'minmax':
        p_min, p_max = np.min(prices), np.max(prices)
        if p_max - p_min < 1e-8:
            prices_norm = np.zeros_like(prices)
        else:
            prices_norm = (prices - p_min) / (p_max - p_min)
        
        v_min, v_max = np.min(volume), np.max(volume)
        if v_max - v_min < 1e-8:
            vol_norm = np.zeros_like(volume)
        else:
            vol_norm = (volume - v_min) / (v_max - v_min)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return np.hstack([prices_norm, vol_norm]).astype(np.float32)

```

### src/features/swings.py

```python
"""
Swing Point Detection

Identifies swing highs and lows in price data for structure analysis.
A swing high is a bar whose high is higher than N bars before and after.
A swing low is a bar whose low is lower than N bars before and after.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SwingPoint:
    """Represents a swing high or low."""
    price: float
    bar_idx: int
    time: pd.Timestamp
    is_high: bool  # True for swing high, False for swing low


def find_swings(
    df: pd.DataFrame,
    lookback: int = 5,
    max_points: int = 20
) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    Find swing highs and lows in OHLCV data.
    
    A swing high is a bar whose high is higher than `lookback` bars before and after.
    A swing low is a bar whose low is lower than `lookback` bars before and after.
    
    Args:
        df: OHLCV DataFrame with 'high', 'low', 'time' columns
        lookback: Number of bars on each side to confirm swing
        max_points: Maximum number of swing points to return per type
        
    Returns:
        Tuple of (swing_highs, swing_lows) as lists of SwingPoint
    """
    if df is None or len(df) < lookback * 2 + 1:
        return [], []
    
    # Use recent data for efficiency
    recent = df.tail(100).copy()
    if len(recent) < lookback * 2 + 1:
        return [], []
    
    highs = recent['high'].values
    lows = recent['low'].values
    times = recent['time'].values if 'time' in recent.columns else [pd.Timestamp.now()] * len(recent)
    indices = recent.index.tolist()
    
    swing_highs: List[SwingPoint] = []
    swing_lows: List[SwingPoint] = []
    
    # Iterate through bars that can be confirmed as swings
    for i in range(lookback, len(recent) - lookback):
        # Check for swing high
        is_swing_high = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        
        if is_swing_high:
            swing_highs.append(SwingPoint(
                price=highs[i],
                bar_idx=indices[i],
                time=pd.Timestamp(times[i]),
                is_high=True
            ))
        
        # Check for swing low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        
        if is_swing_low:
            swing_lows.append(SwingPoint(
                price=lows[i],
                bar_idx=indices[i],
                time=pd.Timestamp(times[i]),
                is_high=False
            ))
    
    # Return most recent first, limited
    return swing_highs[-max_points:], swing_lows[-max_points:]


def count_levels_swept(
    swings: List[SwingPoint],
    price_from: float,
    price_to: float
) -> int:
    """
    Count how many swing levels were swept by a price move.
    
    Args:
        swings: List of swing points to check
        price_from: Starting price of the move
        price_to: Ending price of the move
        
    Returns:
        Number of swing levels between price_from and price_to
    """
    count = 0
    low_price = min(price_from, price_to)
    high_price = max(price_from, price_to)
    
    for swing in swings:
        if low_price < swing.price < high_price:
            count += 1
    
    return count

```

### src/features/time_features.py

```python
"""
Time Features
Session, time of day, and calendar features.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from zoneinfo import ZoneInfo

from src.config import NY_TZ, SESSION_RTH_START, SESSION_RTH_END


class Session(Enum):
    """Trading session."""
    RTH = "RTH"          # Regular Trading Hours (9:30-16:00 NY)
    GLOBEX = "GLOBEX"    # Overnight session (18:00-9:30 NY)
    PRE = "PRE"          # Pre-market (could be 4:00-9:30)
    POST = "POST"        # After-hours
    CLOSED = "CLOSED"    # Market closed


@dataclass
class TimeFeatures:
    """Bundle of time-based features."""
    # Raw values
    hour_ny: int = 0              # Hour in NY time (0-23)
    minute: int = 0               # Minute (0-59)
    day_of_week: int = 0          # 0=Monday, 4=Friday
    
    # Session info
    session: str = "UNKNOWN"      # RTH, GLOBEX, etc.
    mins_into_session: int = 0    # Minutes since session start
    mins_to_session_end: int = 0  # Minutes until session end
    
    # Flags
    is_rth: bool = False          # In regular trading hours
    is_first_hour: bool = False   # First hour of RTH (9:30-10:30)
    is_last_hour: bool = False    # Last hour of RTH (15:00-16:00)
    is_lunch: bool = False        # Lunch (12:00-13:00 NY)
    
    # Cyclical encodings (for neural nets)
    hour_sin: float = 0.0
    hour_cos: float = 0.0
    dow_sin: float = 0.0
    dow_cos: float = 0.0


def get_session(
    timestamp: pd.Timestamp,
    tz: ZoneInfo = NY_TZ
) -> Session:
    """
    Determine which trading session a timestamp falls in.
    
    Args:
        timestamp: Timezone-aware timestamp
        tz: Reference timezone (NY for CME)
        
    Returns:
        Session enum value
    """
    # Convert to NY time
    t = timestamp.astimezone(tz)
    
    # Check if weekend
    if t.weekday() >= 5:  # Saturday=5, Sunday=6
        return Session.CLOSED
    
    hour = t.hour
    minute = t.minute
    time_mins = hour * 60 + minute
    
    # RTH: 9:30 - 16:00 (570 - 960 mins)
    rth_start_mins = 9 * 60 + 30
    rth_end_mins = 16 * 60
    
    # Globex: 18:00 - 9:30 (next day)
    globex_start_mins = 18 * 60
    
    if rth_start_mins <= time_mins < rth_end_mins:
        return Session.RTH
    elif time_mins >= globex_start_mins or time_mins < rth_start_mins:
        return Session.GLOBEX
    else:
        return Session.CLOSED


def compute_time_features(
    timestamp: pd.Timestamp,
    tz: ZoneInfo = NY_TZ
) -> TimeFeatures:
    """
    Compute all time-based features for a timestamp.
    """
    t = timestamp.astimezone(tz)
    
    hour = t.hour
    minute = t.minute
    dow = t.weekday()
    
    session = get_session(timestamp, tz)
    
    # RTH boundaries
    rth_start_mins = 9 * 60 + 30
    rth_end_mins = 16 * 60
    current_mins = hour * 60 + minute
    
    # Session timing
    is_rth = session == Session.RTH
    if is_rth:
        mins_into = current_mins - rth_start_mins
        mins_to_end = rth_end_mins - current_mins
    else:
        mins_into = 0
        mins_to_end = 0
    
    # Cyclical encodings
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 5)  # 5 trading days
    dow_cos = np.cos(2 * np.pi * dow / 5)
    
    return TimeFeatures(
        hour_ny=hour,
        minute=minute,
        day_of_week=dow,
        session=session.value,
        mins_into_session=mins_into,
        mins_to_session_end=mins_to_end,
        is_rth=is_rth,
        is_first_hour=(is_rth and hour == 9) or (is_rth and hour == 10 and minute < 30),
        is_last_hour=is_rth and hour == 15,
        is_lunch=is_rth and hour == 12,
        hour_sin=hour_sin,
        hour_cos=hour_cos,
        dow_sin=dow_sin,
        dow_cos=dow_cos,
    )


def add_time_features(
    df: pd.DataFrame,
    time_col: str = 'time',
    tz: ZoneInfo = NY_TZ
) -> pd.DataFrame:
    """
    Add time features as columns to a DataFrame.
    """
    df = df.copy()
    
    times = pd.to_datetime(df[time_col])
    if times.dt.tz is None:
        times = times.tz_localize('UTC').tz_convert(tz)
    else:
        times = times.dt.tz_convert(tz)
    
    df['hour_ny'] = times.dt.hour
    df['minute'] = times.dt.minute
    df['day_of_week'] = times.dt.weekday
    
    # Cyclical
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_ny'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_ny'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
    
    # Session
    df['is_rth'] = ((df['hour_ny'] > 9) | ((df['hour_ny'] == 9) & (df['minute'] >= 30))) & (df['hour_ny'] < 16)
    df['is_first_hour'] = df['is_rth'] & ((df['hour_ny'] == 9) | ((df['hour_ny'] == 10) & (df['minute'] < 30)))
    df['is_last_hour'] = df['is_rth'] & (df['hour_ny'] == 15)
    df['is_lunch'] = df['is_rth'] & (df['hour_ny'] == 12)
    
    return df

```

### src/hooks/useIndicators.ts

```typescript
/**
 * useIndicators Hook
 * 
 * Calculates all enabled indicators from candle data.
 * Memoized for performance - only recalculates when data or settings change.
 */

import { useMemo } from 'react';
import type { OHLCV, IndicatorSettings, IndicatorPoint, BandPoint, AdrZones } from '../features/chart_indicators';
import {
    calculateEMA,
    calculateSMA,
    calculateVWAP,
    calculateATRBands,
    calculateBollingerBands,
    calculateDonchianChannels,
    calculateADR,
} from '../features/chart_indicators';

export interface IndicatorData {
    ema9: IndicatorPoint[];
    ema21: IndicatorPoint[];
    ema200: IndicatorPoint[];
    vwap: IndicatorPoint[];
    atrBands: BandPoint[];
    bollingerBands: BandPoint[];
    donchianChannels: BandPoint[];
    adr: AdrZones[];
    customIndicators: Map<string, IndicatorPoint[]>;  // id -> data
}

/**
 * Hook to calculate indicators based on settings.
 * Only calculates indicators that are enabled.
 */
export function useIndicators(candles: OHLCV[], settings: IndicatorSettings): IndicatorData {
    return useMemo(() => {
        const data: IndicatorData = {
            ema9: [],
            ema21: [],
            ema200: [],
            vwap: [],
            atrBands: [],
            bollingerBands: [],
            donchianChannels: [],
            adr: [],
            customIndicators: new Map(),
        };

        if (!candles || candles.length < 3) {
            return data;
        }

        // EMAs
        if (settings.ema9) {
            data.ema9 = calculateEMA(candles, 9);
        }
        if (settings.ema21) {
            data.ema21 = calculateEMA(candles, 21);
        }
        if (settings.ema200) {
            data.ema200 = calculateEMA(candles, 200);
        }

        // VWAP
        if (settings.vwap) {
            data.vwap = calculateVWAP(candles);
        }

        // Bands
        if (settings.atrBands) {
            data.atrBands = calculateATRBands(candles, 14, 2);
        }
        if (settings.bollingerBands) {
            data.bollingerBands = calculateBollingerBands(candles, 20, 2);
        }
        if (settings.donchianChannels) {
            data.donchianChannels = calculateDonchianChannels(candles, 20);
        }

        // ADR
        if (settings.adr) {
            data.adr = calculateADR(candles, 14);
        }

        // Custom Indicators
        if (settings.customIndicators) {
            for (const custom of settings.customIndicators) {
                if (custom.type === 'ema') {
                    data.customIndicators.set(custom.id, calculateEMA(candles, custom.period));
                } else if (custom.type === 'sma') {
                    data.customIndicators.set(custom.id, calculateSMA(candles, custom.period));
                }
            }
        }

        return data;
    }, [candles, settings]);
}

```

### src/index.css

```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --color-bg-primary: #0b0e14;
  --color-bg-secondary: #131720;
  --color-bg-tertiary: #1b212d;
  --color-accent-primary: #3b82f6; /* Blue 500 */
  --color-accent-secondary: #10b981; /* Green 500 */
  --color-text-primary: #f1f5f9;
  --color-text-secondary: #94a3b8;
  --font-sans: 'Inter', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

body {
  font-family: var(--font-sans);
  background-color: var(--color-bg-primary);
  color: var(--color-text-primary);
}

.font-mono {
  font-family: var(--font-mono);
}

/* Custom Scrollbar */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: var(--color-bg-primary);
}

::-webkit-scrollbar-thumb {
  background: #334155;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #475569;
}

/* Glassmorphism Utilities */
.glass {
  background: rgba(19, 23, 32, 0.7);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.05);
}

.glass-panel {
  background: rgba(27, 33, 45, 0.6);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.08);
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

/* Button Styles */
.btn-primary {
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  color: white;
  transition: all 0.2s ease;
  box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
}

.btn-primary:hover {
  background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
  box-shadow: 0 4px 8px rgba(59, 130, 246, 0.4);
  transform: translateY(-1px);
}

.btn-secondary {
  background: rgba(51, 65, 85, 0.5);
  color: #e2e8f0;
  border: 1px solid rgba(148, 163, 184, 0.2);
  transition: all 0.2s ease;
}

.btn-secondary:hover {
  background: rgba(51, 65, 85, 0.8);
  border-color: rgba(148, 163, 184, 0.4);
}

/* Card Styles */
.card {
  background: var(--color-bg-secondary);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 0.5rem;
}

.card:hover {
  border-color: rgba(255, 255, 255, 0.1);
}

/* Animations */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(5px); }
  to { opacity: 1; transform: translateY(0); }
}

.animate-fade-in {
  animation: fadeIn 0.3s ease-out forwards;
}

```

### src/labels/__init__.py

```python
# Labels module
"""Future-aware outcome computation - QUARANTINED from features/policy."""

```

### src/labels/counterfactual.py

```python
"""
Counterfactual Labeler
Label ALL decision points with "what would have happened".
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from src.labels.future_window import FutureWindowProvider
from src.labels.trade_outcome import TradeOutcome, compute_trade_outcome
from src.sim.oco_engine import OCOConfig, create_oco_bracket
from src.sim.bar_fill_model import BarFillConfig, BarFillEngine
from src.sim.costs import CostModel, DEFAULT_COSTS


def find_bar_idx_by_time(df: pd.DataFrame, timestamp: pd.Timestamp) -> int:
    """
    Find the bar index in df that matches or is closest to the given timestamp.
    
    This is more accurate than multiplying by timeframe ratio, especially
    when there are gaps in the data (weekends, holidays).
    
    Args:
        df: DataFrame with 'time' column
        timestamp: Target timestamp
        
    Returns:
        Index of the matching or closest bar
    """
    if 'time' not in df.columns:
        raise ValueError("DataFrame must have 'time' column")
    
    # Find first bar at or after target time
    mask = df['time'] >= timestamp
    matches = df[mask]
    
    if len(matches) > 0:
        return matches.index[0]
    
    # If no match, return last bar
    return len(df) - 1


@dataclass
class CounterfactualLabel:
    """
    Counterfactual outcome label.
    
    "What WOULD have happened if we traded here?"
    """
    # Primary outcome
    outcome: str          # WIN, LOSS, TIMEOUT
    pnl: float           # Points
    pnl_dollars: float   # Actual dollars (with costs)
    
    # Excursions
    mae: float           # Max Adverse Excursion (points)
    mfe: float           # Max Favorable Excursion (points)
    mae_atr: float       # MAE normalized by ATR
    mfe_atr: float       # MFE normalized by ATR
    
    # Timing
    bars_held: int
    
    # Prices
    entry_price: float
    exit_price: float
    stop_price: float
    tp_price: float
    
    # OCO config used
    oco_config: OCOConfig = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cf_outcome': self.outcome,
            'cf_pnl': self.pnl,
            'cf_pnl_dollars': self.pnl_dollars,
            'cf_mae': self.mae,
            'cf_mfe': self.mfe,
            'cf_mae_atr': self.mae_atr,
            'cf_mfe_atr': self.mfe_atr,
            'cf_bars_held': self.bars_held,
            'cf_entry_price': self.entry_price,
            'cf_exit_price': self.exit_price,
        }


def compute_counterfactual(
    df: pd.DataFrame,
    entry_idx: int,
    oco_config: OCOConfig,
    atr: float,
    fill_config: BarFillConfig = None,
    costs: CostModel = None,
    max_bars: int = 200
) -> CounterfactualLabel:
    """
    Compute counterfactual outcome for a decision point.
    
    "If we entered here with this OCO, what would happen?"
    
    Args:
        df: Full dataframe
        entry_idx: Bar index of decision point
        oco_config: OCO configuration to simulate
        atr: ATR at decision point (for bracket calculation)
        fill_config: Bar fill configuration
        costs: Cost model
        max_bars: Max bars to simulate
        
    Returns:
        CounterfactualLabel with complete outcome info
    """
    costs = costs or DEFAULT_COSTS
    fill_config = fill_config or BarFillConfig()
    
    # Create OCO bracket
    entry_bar = df.iloc[entry_idx]
    base_price = entry_bar['close']
    
    bracket = create_oco_bracket(
        config=oco_config,
        base_price=base_price,
        atr=atr,
        costs=costs
    )
    
    # Create future provider
    future_provider = FutureWindowProvider(df, entry_idx)
    
    # Compute outcome
    outcome = compute_trade_outcome(
        future_provider=future_provider,
        entry_price=bracket.entry_price,
        direction=oco_config.direction,
        stop_loss=bracket.stop_price,
        take_profit=bracket.tp_price,
        max_bars=max_bars,
        fill_config=fill_config
    )
    
    # Calculate dollar PnL
    pnl_dollars = costs.calculate_pnl(
        bracket.entry_price,
        outcome.exit_price,
        oco_config.direction,
        contracts=1,
        include_commission=True
    )
    
    return CounterfactualLabel(
        outcome=outcome.outcome,
        pnl=outcome.pnl,
        pnl_dollars=pnl_dollars,
        mae=outcome.mae,
        mfe=outcome.mfe,
        mae_atr=outcome.mae / atr if atr > 0 else 0,
        mfe_atr=outcome.mfe / atr if atr > 0 else 0,
        bars_held=outcome.bars_held,
        entry_price=bracket.entry_price,
        exit_price=outcome.exit_price,
        stop_price=bracket.stop_price,
        tp_price=bracket.tp_price,
        oco_config=oco_config,
    )


def compute_multi_oco_counterfactuals(
    df: pd.DataFrame,
    entry_idx: int,
    oco_configs: List[OCOConfig],
    atr: float,
    fill_config: BarFillConfig = None,
    costs: CostModel = None
) -> Dict[str, CounterfactualLabel]:
    """
    Compute counterfactual outcomes for multiple OCO variants.
    
    Enables "which OCO would have worked best?" analysis.
    
    Returns:
        Dict mapping oco_config.name to CounterfactualLabel
    """
    results = {}
    
    for oco in oco_configs:
        label = compute_counterfactual(
            df=df,
            entry_idx=entry_idx,
            oco_config=oco,
            atr=atr,
            fill_config=fill_config,
            costs=costs
        )
        name = oco.name or f"{oco.direction}_{oco.tp_multiple}R"
        results[name] = label
    
    return results


def label_is_good_skip(cf_label: CounterfactualLabel) -> bool:
    """
    Determine if skipping this trade was a good decision.
    
    "Skipped good" = would have lost
    "Skipped bad" = would have won
    
    From the perspective of improving the model, we want to
    learn to skip the losses.
    """
    return cf_label.outcome == 'LOSS'


def label_is_bad_skip(cf_label: CounterfactualLabel) -> bool:
    """Trade we skipped but should have taken (would have won)."""
    return cf_label.outcome == 'WIN'


def compute_smart_stop_counterfactual(
    df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    stop_price: float,
    tp_multiple: float,
    atr: float,
    fill_config: BarFillConfig = None,
    costs: CostModel = None,
    max_bars: int = 200,
    oco_name: str = ""
) -> CounterfactualLabel:
    """
    Compute counterfactual with pre-calculated stop price.
    
    Use this with stop_calculator for smart stops based on
    candle levels, ranges, swings, etc.
    
    Args:
        df: Full dataframe
        entry_idx: Bar index of decision point
        direction: 'LONG' or 'SHORT'
        stop_price: Pre-calculated stop price (from stop_calculator)
        tp_multiple: R multiple for take profit
        atr: ATR for reference (not used for stop)
        fill_config: Bar fill configuration
        costs: Cost model
        max_bars: Max bars to simulate
        oco_name: Name for this configuration
        
    Returns:
        CounterfactualLabel with complete outcome info
    """
    costs = costs or DEFAULT_COSTS
    fill_config = fill_config or BarFillConfig()
    
    entry_bar = df.iloc[entry_idx]
    entry_price = entry_bar['close']
    
    # Calculate risk and TP
    if direction == "LONG":
        risk = entry_price - stop_price
        tp_price = entry_price + (risk * tp_multiple)
    else:
        risk = stop_price - entry_price
        tp_price = entry_price - (risk * tp_multiple)
    
    # Round to tick
    stop_price = costs.round_to_tick(stop_price, 'down' if direction == 'LONG' else 'up')
    tp_price = costs.round_to_tick(tp_price, 'up' if direction == 'LONG' else 'down')
    
    # Create future provider
    future_provider = FutureWindowProvider(df, entry_idx)
    
    # Compute outcome
    outcome = compute_trade_outcome(
        future_provider=future_provider,
        entry_price=entry_price,
        direction=direction,
        stop_loss=stop_price,
        take_profit=tp_price,
        max_bars=max_bars,
        fill_config=fill_config
    )
    
    # Calculate dollar PnL
    pnl_dollars = costs.calculate_pnl(
        entry_price,
        outcome.exit_price,
        direction,
        contracts=1,
        include_commission=True
    )
    
    # Create a minimal OCOConfig for storage
    oco_config = OCOConfig(
        direction=direction,
        stop_atr=0,  # Not used
        tp_multiple=tp_multiple,
        name=oco_name
    )
    
    return CounterfactualLabel(
        outcome=outcome.outcome,
        pnl=outcome.pnl,
        pnl_dollars=pnl_dollars,
        mae=outcome.mae,
        mfe=outcome.mfe,
        mae_atr=outcome.mae / atr if atr > 0 else 0,
        mfe_atr=outcome.mfe / atr if atr > 0 else 0,
        bars_held=outcome.bars_held,
        entry_price=entry_price,
        exit_price=outcome.exit_price,
        stop_price=stop_price,
        tp_price=tp_price,
        oco_config=oco_config,
    )

```

### src/labels/future_window.py

```python
"""
Future Window Provider
QUARANTINED future access - only used in labels/ module.
"""

import pandas as pd
import numpy as np
from typing import Optional


class FutureWindowProvider:
    """
    Provides access to future data for labeling.
    
    This class is ONLY instantiated inside the labels/ module.
    It takes the full dataframe and an entry index, then provides
    controlled access to future bars.
    
    Physical separation ensures features/ and policy/ cannot
    accidentally access future data.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        time_col: str = 'time'
    ):
        """
        Initialize future window provider.
        
        Args:
            df: Full dataframe with all bars
            entry_idx: Index of the entry bar (decision point)
            time_col: Name of time column
        """
        self.df = df
        self.entry_idx = entry_idx
        self.time_col = time_col
        self._validate()
    
    def _validate(self):
        """Validate inputs."""
        if self.entry_idx < 0:
            raise ValueError("entry_idx must be >= 0")
        if self.entry_idx >= len(self.df):
            raise ValueError(f"entry_idx {self.entry_idx} >= data length {len(self.df)}")
    
    def get_future(self, lookahead: int) -> pd.DataFrame:
        """
        Get next N bars AFTER entry.
        
        Does NOT include the entry bar itself.
        """
        start = self.entry_idx + 1
        end = min(start + lookahead, len(self.df))
        return self.df.iloc[start:end].copy()
    
    def get_future_array(
        self,
        lookahead: int,
        columns: list = None
    ) -> np.ndarray:
        """Get future as numpy array."""
        columns = columns or ['open', 'high', 'low', 'close']
        future = self.get_future(lookahead)
        
        if len(future) == 0:
            return np.array([]).reshape(0, len(columns))
        
        return future[columns].values
    
    def get_entry_bar(self) -> pd.Series:
        """Get the entry bar."""
        return self.df.iloc[self.entry_idx]
    
    def get_entry_price(self, price_col: str = 'close') -> float:
        """Get entry price (default: close of entry bar)."""
        return self.df.iloc[self.entry_idx][price_col]
    
    def get_entry_time(self) -> Optional[pd.Timestamp]:
        """Get entry timestamp."""
        bar = self.get_entry_bar()
        if self.time_col in bar:
            return bar[self.time_col]
        return None
    
    def bars_available(self) -> int:
        """Number of future bars available."""
        return len(self.df) - self.entry_idx - 1
    
    def has_sufficient_future(self, required: int) -> bool:
        """Check if enough future bars exist."""
        return self.bars_available() >= required

```

### src/labels/labeler.py

```python
"""
Labeler
Main labeling pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import pandas as pd

from src.labels.counterfactual import (
    CounterfactualLabel, 
    compute_counterfactual,
    compute_multi_oco_counterfactuals
)
from src.sim.oco_engine import OCOConfig
from src.sim.bar_fill_model import BarFillConfig
from src.sim.costs import CostModel, DEFAULT_COSTS


@dataclass
class LabelConfig:
    """Configuration for labeling."""
    # Primary OCO for counterfactual
    oco_config: OCOConfig = field(default_factory=OCOConfig)
    
    # Optional: additional OCO variants for multi-armed bandit
    oco_variants: List[OCOConfig] = field(default_factory=list)
    
    # Fill model
    fill_config: BarFillConfig = field(default_factory=BarFillConfig)
    
    # Cost model
    cost_model: CostModel = field(default_factory=lambda: DEFAULT_COSTS)
    
    # Simulation
    max_bars: int = 200
    
    def to_dict(self) -> dict:
        return {
            'oco_config': self.oco_config.to_dict(),
            'oco_variants': [o.to_dict() for o in self.oco_variants],
            'fill_config': self.fill_config.to_dict(),
            'max_bars': self.max_bars,
        }


class Labeler:
    """
    Main labeling class.
    
    Takes decision points and adds counterfactual labels.
    """
    
    def __init__(self, config: LabelConfig):
        self.config = config
    
    def label_decision_point(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        atr: float
    ) -> CounterfactualLabel:
        """
        Label a single decision point.
        
        Args:
            df: Full dataframe
            entry_idx: Index of decision point
            atr: ATR at decision point
            
        Returns:
            CounterfactualLabel
        """
        return compute_counterfactual(
            df=df,
            entry_idx=entry_idx,
            oco_config=self.config.oco_config,
            atr=atr,
            fill_config=self.config.fill_config,
            costs=self.config.cost_model,
            max_bars=self.config.max_bars
        )
    
    def label_decision_point_multi(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        atr: float
    ) -> Dict[str, CounterfactualLabel]:
        """
        Label with multiple OCO variants.
        """
        all_ocos = [self.config.oco_config] + self.config.oco_variants
        
        return compute_multi_oco_counterfactuals(
            df=df,
            entry_idx=entry_idx,
            oco_configs=all_ocos,
            atr=atr,
            fill_config=self.config.fill_config,
            costs=self.config.cost_model
        )
    
    def label_batch(
        self,
        df: pd.DataFrame,
        entry_indices: List[int],
        atrs: List[float]
    ) -> List[CounterfactualLabel]:
        """
        Label a batch of decision points.
        """
        results = []
        for idx, atr in zip(entry_indices, atrs):
            label = self.label_decision_point(df, idx, atr)
            results.append(label)
        return results


# Convenience function
def create_default_labeler(
    direction: str = "LONG",
    tp_multiple: float = 1.4,
    stop_atr: float = 1.0
) -> Labeler:
    """Create labeler with common defaults."""
    oco = OCOConfig(
        direction=direction,
        tp_multiple=tp_multiple,
        stop_atr=stop_atr,
        name=f"{direction}_{tp_multiple}R"
    )
    config = LabelConfig(oco_config=oco)
    return Labeler(config)

```

### src/labels/trade_outcome.py

```python
"""
Trade Outcome
Compute trade outcomes from future data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

from src.labels.future_window import FutureWindowProvider
from src.sim.bar_fill_model import BarFillEngine, BarFillConfig


@dataclass
class TradeOutcome:
    """Outcome of a simulated trade."""
    outcome: str          # 'WIN', 'LOSS', 'TIMEOUT'
    pnl: float           # Points of profit/loss
    exit_bar_offset: int  # Bars from entry to exit
    exit_price: float
    mae: float           # Max Adverse Excursion (points)
    mfe: float           # Max Favorable Excursion (points)
    bars_held: int


def compute_trade_outcome(
    future_provider: FutureWindowProvider,
    entry_price: float,
    direction: str,
    stop_loss: float,
    take_profit: float,
    max_bars: int = 200,
    fill_config: BarFillConfig = None
) -> TradeOutcome:
    """
    Simulate a trade to completion using future data.
    
    This is the core labeling function - uses future data
    to determine what WOULD have happened.
    
    Args:
        future_provider: Provider for future bars
        entry_price: Entry price
        direction: 'LONG' or 'SHORT'
        stop_loss: Stop loss price
        take_profit: Take profit price
        max_bars: Maximum bars before timeout
        fill_config: Bar fill configuration
        
    Returns:
        TradeOutcome with all metrics
    """
    fill_config = fill_config or BarFillConfig()
    future = future_provider.get_future(max_bars)
    
    if len(future) == 0:
        return TradeOutcome(
            outcome='TIMEOUT',
            pnl=0.0,
            exit_bar_offset=0,
            exit_price=entry_price,
            mae=0.0,
            mfe=0.0,
            bars_held=0
        )
    
    # Track excursions
    highs = future['high'].values
    lows = future['low'].values
    
    if direction == 'LONG':
        # LONG: adverse = low below entry, favorable = high above entry
        mae = max(0, entry_price - lows.min())
        mfe = max(0, highs.max() - entry_price)
        
        # Find first SL or TP hit
        sl_hits = np.where(lows <= stop_loss)[0]
        tp_hits = np.where(highs >= take_profit)[0]
        
    else:  # SHORT
        # SHORT: adverse = high above entry, favorable = low below entry
        mae = max(0, highs.max() - entry_price)
        mfe = max(0, entry_price - lows.min())
        
        sl_hits = np.where(highs >= stop_loss)[0]
        tp_hits = np.where(lows <= take_profit)[0]
    
    sl_bar = sl_hits[0] if len(sl_hits) > 0 else max_bars + 1
    tp_bar = tp_hits[0] if len(tp_hits) > 0 else max_bars + 1
    
    # Determine outcome
    if tp_bar < sl_bar:
        outcome = 'WIN'
        exit_bar = tp_bar
        exit_price = take_profit
    elif sl_bar < max_bars:
        outcome = 'LOSS'
        exit_bar = sl_bar
        exit_price = stop_loss
    else:
        outcome = 'TIMEOUT'
        exit_bar = min(len(future) - 1, max_bars - 1)
        exit_price = future.iloc[exit_bar]['close']
    
    # Handle same-bar hits (both SL and TP)
    if sl_bar == tp_bar and sl_bar < max_bars:
        # Apply tie-break rule from fill config
        from src.sim.bar_fill_model import SLTPTieBreak
        
        if fill_config.sl_tp_tiebreak == SLTPTieBreak.CONSERVATIVE:
            outcome = 'LOSS'
            exit_price = stop_loss
        elif fill_config.sl_tp_tiebreak == SLTPTieBreak.OPTIMISTIC:
            outcome = 'WIN'
            exit_price = take_profit
        else:
            # Open proximity
            bar = future.iloc[sl_bar]
            sl_dist = abs(bar['open'] - stop_loss)
            tp_dist = abs(bar['open'] - take_profit)
            if sl_dist <= tp_dist:
                outcome = 'LOSS'
                exit_price = stop_loss
            else:
                outcome = 'WIN'
                exit_price = take_profit
    
    # Calculate PnL
    if direction == 'LONG':
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price
    
    return TradeOutcome(
        outcome=outcome,
        pnl=pnl,
        exit_bar_offset=exit_bar + 1,  # +1 because futures start at entry+1
        exit_price=exit_price,
        mae=mae,
        mfe=mfe,
        bars_held=exit_bar + 1
    )


def compute_price_target_reached(
    future_provider: FutureWindowProvider,
    target_price: float,
    direction: str,  # 'UP' or 'DOWN'
    within_bars: int
) -> Tuple[bool, int]:
    """
    Check if price reaches target within N bars.
    
    Returns:
        (reached: bool, bars_to_reach: int or -1 if not reached)
    """
    future = future_provider.get_future(within_bars)
    
    if len(future) == 0:
        return (False, -1)
    
    if direction == 'UP':
        hits = np.where(future['high'].values >= target_price)[0]
    else:
        hits = np.where(future['low'].values <= target_price)[0]
    
    if len(hits) > 0:
        return (True, int(hits[0]) + 1)
    
    return (False, -1)

```

### src/models/__init__.py

```python
# Models package
from src.core.enums import ModelRole

"""Neural network architectures and training."""

```

### src/models/context_mlp.py

```python
"""
Context MLP
MLP encoder for context feature vector.
"""

import torch
import torch.nn as nn


class ContextMLP(nn.Module):
    """
    MLP for encoding context features.
    
    Input: (batch, context_dim) e.g., (64, 20)
    Output: (batch, embedding_dim)
    """
    
    def __init__(
        self,
        input_dim: int = 20,
        embedding_dim: int = 32,
        hidden_dims: list = None,
        dropout: float = 0.3
    ):
        super().__init__()
        
        hidden_dims = hidden_dims or [64, 64]
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, embedding_dim))
        
        self.net = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, input_dim)
            
        Returns:
            (batch, embedding_dim)
        """
        return self.net(x)

```

### src/models/encoders.py

```python
"""
CNN Encoders
Price window encoders for pattern recognition.
"""

import torch
import torch.nn as nn
from typing import Tuple


class CNNEncoder(nn.Module):
    """
    1D CNN for encoding price windows.
    
    Input: (batch, channels, length) e.g., (64, 5, 120)
    Output: (batch, embedding_dim)
    """
    
    def __init__(
        self,
        input_channels: int = 5,      # OHLCV
        seq_length: int = 120,        # 2 hours of 1m
        embedding_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # length / 2
            
            # Conv block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # length / 4
            
            # Conv block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # length / 8
            
            # Conv block 4
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (batch, 128, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, channels, length)
            
        Returns:
            (batch, embedding_dim)
        """
        x = self.features(x)
        x = self.fc(x)
        return x


class MultiTFEncoder(nn.Module):
    """
    Encode multiple timeframe price windows.
    
    Separate CNN for each timeframe, then concatenate.
    """
    
    def __init__(
        self,
        tf_configs: dict = None,
        embedding_dim_per_tf: int = 32,
        dropout: float = 0.3
    ):
        """
        Args:
            tf_configs: Dict of {name: (length, channels)}
                Default: {'1m': (120, 5), '5m': (24, 5), '15m': (8, 5)}
            embedding_dim_per_tf: Embedding size per timeframe
        """
        super().__init__()
        
        self.tf_configs = tf_configs or {
            '1m': (120, 5),
            '5m': (24, 5),
            '15m': (8, 5),
        }
        
        self.encoders = nn.ModuleDict()
        for name, (length, channels) in self.tf_configs.items():
            self.encoders[name] = CNNEncoder(
                input_channels=channels,
                seq_length=length,
                embedding_dim=embedding_dim_per_tf,
                dropout=dropout
            )
        
        self.total_dim = embedding_dim_per_tf * len(self.tf_configs)
    
    def forward(
        self,
        x_1m: torch.Tensor,
        x_5m: torch.Tensor,
        x_15m: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode all timeframes and concatenate.
        
        Returns:
            (batch, total_dim)
        """
        embeddings = []
        
        if '1m' in self.encoders:
            embeddings.append(self.encoders['1m'](x_1m))
        if '5m' in self.encoders:
            embeddings.append(self.encoders['5m'](x_5m))
        if '15m' in self.encoders:
            embeddings.append(self.encoders['15m'](x_15m))
        
        return torch.cat(embeddings, dim=-1)

```

### src/models/fusion.py

```python
"""
Fusion Model
Combine CNN price encoders with context MLP.
"""

import torch
import torch.nn as nn
from typing import Optional

from src.core.enums import ModelRole, RunMode
from src.models.encoders import MultiTFEncoder
from src.models.context_mlp import ContextMLP


class FusionModel(nn.Module):
    """
    CNN + MLP fusion for decision classification.
    
    Architecture:
    - MultiTFEncoder processes price windows
    - ContextMLP processes context features
    - Concatenate and pass through classification head
    """
    
    def __init__(
        self,
        context_dim: int = 20,
        price_embedding_per_tf: int = 32,
        context_embedding: int = 32,
        num_classes: int = 2,  # WIN/LOSS
        dropout: float = 0.3,
        role: ModelRole = ModelRole.TRAINING_ONLY
    ):
        super().__init__()
        self.role = role
        
        # Price encoder
        self.price_encoder = MultiTFEncoder(
            embedding_dim_per_tf=price_embedding_per_tf,
            dropout=dropout
        )
        
        # Context encoder
        self.context_encoder = ContextMLP(
            input_dim=context_dim,
            embedding_dim=context_embedding,
            dropout=dropout
        )
        
        # Combined dimension
        combined_dim = self.price_encoder.total_dim + context_embedding
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        
        self.num_classes = num_classes
    
    def check_can_run(self, run_mode: RunMode):
        """Verify model is allowed to run in the current mode."""
        if run_mode == RunMode.REPLAY and self.role == ModelRole.TRAINING_ONLY:
            raise RuntimeError(f"Model with role {self.role} is barred from REPLAY mode to prevent future leakage.")
        if run_mode == RunMode.TRAIN and self.role == ModelRole.REPLAY_ONLY:
             raise RuntimeError(f"Model with role {self.role} is for REPLAY only, not training.")

    def forward(
        self,
        x_price_1m: torch.Tensor,
        x_price_5m: torch.Tensor,
        x_price_15m: torch.Tensor,
        x_context: torch.Tensor,
        run_mode: Optional[RunMode] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_price_*: Price windows (batch, channels, length)
            x_context: Context vector (batch, context_dim)
            run_mode: Optional current run mode for enforcement
            
        Returns:
            Logits (batch, num_classes)
        """
        if run_mode:
            self.check_can_run(run_mode)
            
        # Encode price
        price_emb = self.price_encoder(x_price_1m, x_price_5m, x_price_15m)
        
        # Encode context
        context_emb = self.context_encoder(x_context)
        
        # Fuse
        combined = torch.cat([price_emb, context_emb], dim=-1)
        
        # Classify
        logits = self.classifier(combined)
        
        return logits
    
    def predict_proba(
        self,
        x_price_1m: torch.Tensor,
        x_price_5m: torch.Tensor,
        x_price_15m: torch.Tensor,
        x_context: torch.Tensor,
        run_mode: Optional[RunMode] = None
    ) -> torch.Tensor:
        """Get probability of WIN class."""
        logits = self.forward(x_price_1m, x_price_5m, x_price_15m, x_context, run_mode=run_mode)
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1] if self.num_classes == 2 else probs  # P(WIN)


class SimpleCNN(nn.Module):
    """
    Simple CNN model using only 1m price data.
    Good for baseline comparisons.
    """
    
    def __init__(
        self,
        input_channels: int = 5,
        seq_length: int = 120,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, length)
            
        Returns:
            Logits (batch, num_classes)
        """
        x = self.features(x)
        return self.classifier(x)

```

### src/models/heads.py

```python
"""
Model Heads
Classification and regression heads.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Classification head for binary or multi-class.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits."""
        return self.net(x)


class RegressionHead(nn.Module):
    """
    Regression head for continuous outputs (PnL, MAE, MFE).
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskHead(nn.Module):
    """
    Multi-task head for joint classification + regression.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        num_regression: int = 4,  # pnl, mae, mfe, bars_held
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Task-specific heads
        self.classification_head = nn.Linear(hidden_dim, num_classes)
        self.regression_head = nn.Linear(hidden_dim, num_regression)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Returns:
            Dict with 'logits' and 'regression' tensors
        """
        shared = self.shared(x)
        
        return {
            'logits': self.classification_head(shared),
            'regression': self.regression_head(shared),
        }

```

### src/models/model_registry_init.py

```python
"""
Model Registration
Wire existing models into the ModelRegistry.
"""

from src.core.registries import ModelRegistry


# =============================================================================
# Register built-in models
# =============================================================================

@ModelRegistry.register(
    model_id="fusion_cnn",
    name="Fusion CNN Model",
    description="Multi-timeframe CNN with MLP context fusion",
    input_schema={
        "x_price_1m": {"type": "array", "shape": [None, 5]},
        "x_price_5m": {"type": "array", "shape": [None, 5]},
        "x_price_15m": {"type": "array", "shape": [None, 5]},
        "x_context": {"type": "array", "shape": [None]},
    },
    output_schema={
        "logits": {"type": "array", "shape": [3]},
        "probs": {"type": "array", "shape": [3]},
    }
)
class FusionCNNWrapper:
    """Wrapper for FusionModel."""
    def __init__(self, model_path: str):
        from src.models.fusion import FusionModel
        from src.core.enums import ModelRole
        import torch
        
        # Load model
        self.model = FusionModel(role=ModelRole.REPLAY_ONLY)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
    
    def predict(self, features):
        import torch
        from src.core.enums import RunMode
        
        # Extract features
        x_1m = torch.tensor(features['x_price_1m'], dtype=torch.float32).unsqueeze(0)
        x_5m = torch.tensor(features['x_price_5m'], dtype=torch.float32).unsqueeze(0)
        x_15m = torch.tensor(features['x_price_15m'], dtype=torch.float32).unsqueeze(0)
        x_context = torch.tensor(features['x_context'], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(x_1m, x_5m, x_15m, x_context, run_mode=RunMode.REPLAY)
            probs = torch.softmax(logits, dim=1)
        
        return {
            'logits': logits[0].numpy().tolist(),
            'probs': probs[0].numpy().tolist(),
        }


# Auto-register on import
def register_all_models():
    """
    Register all available models.
    Call this at startup to populate the registry.
    """
    pass


@ModelRegistry.register(
    model_id="ifvg_4class",
    name="IFVG 4-Class CNN",
    description="4-class CNN for IFVG pattern detection (LONG_WIN, LONG_LOSS, SHORT_WIN, SHORT_LOSS)",
    input_schema={
        "ohlcv": {"type": "array", "shape": [5, 30], "normalization": "percent_change"},
    },
    output_schema={
        "probs": {"type": "array", "shape": [4]},  # [LONG_WIN, LONG_LOSS, SHORT_WIN, SHORT_LOSS]
    }
)
class IFVG4ClassWrapper:
    """Wrapper for IFVG4ClassCNN."""
    
    def __init__(self, model_path: str = None, **kwargs):
        import torch
        import torch.nn as nn
        
        # Define architecture inline (same as train_ifvg_4class.py)
        class IFVG4ClassCNN(nn.Module):
            def __init__(self, input_channels=5, seq_length=30, num_classes=4):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(32, num_classes),
                )
            
            def forward(self, x):
                x = self.features(x)
                return self.classifier(x)
        
        self.model = IFVG4ClassCNN(**kwargs)
        if model_path:
            state = torch.load(model_path, map_location='cpu', weights_only=False)
            # Handle both raw state_dict and checkpoint bundles
            if 'model_state_dict' in state:
                self.model.load_state_dict(state['model_state_dict'])
            else:
                self.model.load_state_dict(state)
        self.model.eval()
    
    def predict(self, features):
        import torch
        
        # Expects features['ohlcv'] as (5, 30) normalized array
        x = torch.tensor(features['ohlcv'], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
        
        probs_list = probs[0].numpy().tolist()
        
        # Determine triggered direction
        long_win, long_loss, short_win, short_loss = probs_list
        
        return {
            'probs': probs_list,
            'long_win_prob': long_win,
            'short_win_prob': short_win,
            'direction': 'LONG' if long_win > short_win else 'SHORT',
            'triggered': True,  # Always true - let caller apply threshold
        }


@ModelRegistry.register(
    model_id="puller_xgb_4class",
    name="Puller XGBoost 4-Class",
    description="XGBoost model for Puller pattern (LONG_WIN, LONG_LOSS, SHORT_WIN, SHORT_LOSS)",
    input_schema={
        "bars": {"type": "array", "description": "OHLCV bars"},
        "ohlcv": {"type": "array", "shape": [5, 30], "description": "Normalized OHLCV"},
    },
    output_schema={
        "probs": {"type": "array", "shape": [4]},
        "direction": {"type": "string"},
        "triggered": {"type": "boolean"},
    }
)
class PullerXGBoostWrapper:
    """Wrapper for Puller XGBoost 4-class model.
    
    Computes features from raw OHLCV bars for inference.
    """
    
    def __init__(self, model_path: str = None, **kwargs):
        import xgboost as xgb
        
        self.model = xgb.XGBClassifier()
        if model_path:
            self.model.load_model(model_path)
        else:
            self.model.load_model('models/puller_xgb_4class.json')
    
    def predict(self, features):
        import numpy as np
        
        # Extract bars or use pre-computed ohlcv
        bars = features.get('bars', [])
        
        # Compute features from bars (pattern indicators that predict win/loss)
        if len(bars) >= 10:
            # Compute pattern-based features from price action
            closes = np.array([b['close'] for b in bars[-30:]])
            highs = np.array([b['high'] for b in bars[-30:]])
            lows = np.array([b['low'] for b in bars[-30:]])
            
            # Feature 1: Recent volatility (normalized range)
            atr = np.mean(highs - lows)
            volatility = atr / closes[-1] if closes[-1] > 0 else 0
            
            # Feature 2: Momentum (close change over last 10 bars)
            momentum = (closes[-1] - closes[-10]) / atr if atr > 0 else 0
            
            # Feature 3: Range position (where close is in recent range)
            range_high = np.max(highs[-20:])
            range_low = np.min(lows[-20:])
            range_pos = (closes[-1] - range_low) / (range_high - range_low) if range_high > range_low else 0.5
            
            x = np.array([volatility * 100, momentum, range_pos], dtype=np.float32).reshape(1, -1)
        else:
            # Fallback: default features
            x = np.array([50, 0, 0.5], dtype=np.float32).reshape(1, -1)
        
        probs = self.model.predict_proba(x)[0].tolist()
        
        # 0=LONG_WIN, 1=LONG_LOSS, 2=SHORT_WIN, 3=SHORT_LOSS
        long_win_prob = probs[0]
        short_win_prob = probs[2]
        
        # Determine direction based on which WIN class has higher prob
        if long_win_prob > short_win_prob:
            direction = 'LONG'
            prob = long_win_prob
        else:
            direction = 'SHORT'
            prob = short_win_prob
        
        return {
            'probs': probs,
            'long_win_prob': long_win_prob,
            'short_win_prob': short_win_prob,
            'direction': direction,
            'triggered': prob >= 0.35,
        }

```

### src/models/train.py

```python
"""
Training
Training loop and configuration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum
import numpy as np

from src.config import MODELS_DIR
from src.core.enums import RunMode


class ImbalanceStrategy(Enum):
    """Strategy for handling class imbalance."""
    NONE = "none"
    WEIGHTED_LOSS = "weighted"
    FOCAL_LOSS = "focal"
    BALANCED_SAMPLING = "balanced"


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    dropout: float = 0.3
    
    # Imbalance handling
    imbalance_strategy: ImbalanceStrategy = ImbalanceStrategy.WEIGHTED_LOSS
    focal_gamma: float = 2.0
    class_weights: Optional[Dict[int, float]] = None
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.001
    
    # Checkpointing
    save_best: bool = True
    save_path: Path = None
    
    def to_dict(self) -> dict:
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'imbalance_strategy': self.imbalance_strategy.value,
            'patience': self.patience,
        }


@dataclass
class TrainResult:
    """Training result."""
    best_val_loss: float
    best_epoch: int
    train_losses: List[float]
    val_losses: List[float]
    val_accuracies: List[float]
    model_path: Optional[Path] = None


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def compute_class_weights(labels: List[int], num_classes: int = 2) -> torch.Tensor:
    """Compute inverse frequency class weights."""
    counts = np.bincount(labels, minlength=num_classes)
    total = sum(counts)
    weights = total / (num_classes * counts + 1e-6)
    return torch.FloatTensor(weights)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        x_1m = batch['x_price_1m'].to(device)
        x_5m = batch['x_price_5m'].to(device)
        x_15m = batch['x_price_15m'].to(device)
        x_context = batch['x_context'].to(device)
        y = batch['y'].squeeze().to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(x_1m, x_5m, x_15m, x_context, run_mode=RunMode.TRAIN)
        loss = criterion(logits, y)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Validate and return loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            x_1m = batch['x_price_1m'].to(device)
            x_5m = batch['x_price_5m'].to(device)
            x_15m = batch['x_price_15m'].to(device)
            x_context = batch['x_context'].to(device)
            y = batch['y'].squeeze().to(device)
            
            logits = model(x_1m, x_5m, x_15m, x_context, run_mode=RunMode.TRAIN)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig
) -> TrainResult:
    """
    Full training loop.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup criterion based on imbalance strategy
    if config.imbalance_strategy == ImbalanceStrategy.WEIGHTED_LOSS:
        # Compute weights from training data
        labels = [batch['y'].squeeze().tolist() for batch in train_loader]
        labels = [l for batch_labels in labels for l in (batch_labels if isinstance(batch_labels, list) else [batch_labels])]
        weights = compute_class_weights(labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        
    elif config.imbalance_strategy == ImbalanceStrategy.FOCAL_LOSS:
        criterion = FocalLoss(gamma=config.focal_gamma)
        
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Training state
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    save_path = config.save_path or MODELS_DIR / "best_model.pth"
    
    for epoch in range(config.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{config.epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")
        
        # Check improvement
        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            if config.save_best:
                torch.save(model.state_dict(), save_path)
                print(f"  [Saved best model]")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return TrainResult(
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        train_losses=train_losses,
        val_losses=val_losses,
        val_accuracies=val_accuracies,
        model_path=save_path if config.save_best else None
    )

```

### src/policy/__init__.py

```python
# Policy module
"""Decision logic - scanners, filters, and actions."""

```

### src/policy/actions.py

```python
"""
Actions
Decision action types and policy decision structure.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from src.sim.oco_engine import OCOConfig


class Action(Enum):
    """What action to take at a decision point."""
    NO_TRADE = "NO_TRADE"         # Skip this opportunity
    PLACE_ORDER = "PLACE_ORDER"   # Enter new position
    MANAGE = "MANAGE"             # Adjust existing position
    EXIT = "EXIT"                 # Close position


class SkipReason(Enum):
    """
    Why a decision point was skipped.
    
    This is crucial for understanding dataset composition:
    - FILTER_BLOCK: Filtered out before reaching policy
    - COOLDOWN: Too soon after last trade
    - IN_POSITION: Already have open position
    - POLICY_NO: Policy decided not to trade
    - OTHER: Other reason
    """
    NOT_SKIPPED = "NOT_SKIPPED"   # Trade was taken
    FILTER_BLOCK = "FILTER_BLOCK"
    COOLDOWN = "COOLDOWN"
    IN_POSITION = "IN_POSITION"
    POLICY_NO = "POLICY_NO"
    OTHER = "OTHER"


@dataclass
class PolicyDecision:
    """
    Complete decision at a decision point.
    """
    action: Action
    skip_reason: SkipReason = SkipReason.NOT_SKIPPED
    reason_detail: str = ""       # Human-readable explanation
    
    # If PLACE_ORDER
    order_config: Optional[OCOConfig] = None
    
    # Scanner context that led to this decision
    scanner_id: str = ""
    scanner_context: Dict[str, Any] = field(default_factory=dict)
    
    # Confidence/score (for ML-based policy)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action.value,
            'skip_reason': self.skip_reason.value,
            'reason_detail': self.reason_detail,
            'scanner_id': self.scanner_id,
            'confidence': self.confidence,
            'order_config': self.order_config.to_dict() if self.order_config else None,
        }


def make_no_trade(
    reason: SkipReason,
    detail: str = "",
    scanner_id: str = ""
) -> PolicyDecision:
    """Helper to create NO_TRADE decision."""
    return PolicyDecision(
        action=Action.NO_TRADE,
        skip_reason=reason,
        reason_detail=detail,
        scanner_id=scanner_id,
    )


def make_trade(
    order_config: OCOConfig,
    scanner_id: str = "",
    confidence: float = 1.0,
    context: Dict[str, Any] = None
) -> PolicyDecision:
    """Helper to create PLACE_ORDER decision."""
    return PolicyDecision(
        action=Action.PLACE_ORDER,
        skip_reason=SkipReason.NOT_SKIPPED,
        order_config=order_config,
        scanner_id=scanner_id,
        confidence=confidence,
        scanner_context=context or {},
    )

```

### src/policy/brackets.py

```python
"""
Bracket Components (Exit Strategy)

Define stop-loss and take-profit levels for OCO orders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from enum import Enum


class BracketType(Enum):
    ATR = "atr"
    PERCENT = "percent"
    FIXED = "fixed"
    LEVEL = "level"


@dataclass
class BracketLevels:
    """Computed stop and TP prices."""
    stop_price: float
    tp_price: float
    risk_points: float  # Distance from entry to stop
    reward_points: float  # Distance from entry to TP
    r_multiple: float  # reward / risk


class Bracket(ABC):
    """
    Base class for exit bracket strategies.
    
    Brackets compute stop-loss and take-profit prices
    given an entry price, direction, and market context.
    """
    
    @property
    @abstractmethod
    def bracket_type(self) -> BracketType:
        pass
    
    @property
    def params(self) -> Dict[str, Any]:
        """Serializable parameters."""
        return {}
    
    @abstractmethod
    def compute(
        self,
        entry_price: float,
        direction: str,  # "LONG" or "SHORT"
        atr: float,
        **kwargs
    ) -> BracketLevels:
        """
        Compute stop and TP levels.
        
        Args:
            entry_price: Entry price
            direction: Trade direction
            atr: Current ATR value
            **kwargs: Additional context (levels, etc.)
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.bracket_type.value,
            **self.params
        }


class ATRBracket(Bracket):
    """
    Stop and TP as ATR multiples.
    
    Agent config:
        {"type": "atr", "stop_atr": 2.0, "tp_atr": 3.0}
    """
    
    def __init__(self, stop_atr: float = 2.0, tp_atr: float = 3.0):
        self._stop_atr = stop_atr
        self._tp_atr = tp_atr
    
    @property
    def bracket_type(self) -> BracketType:
        return BracketType.ATR
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "stop_atr": self._stop_atr,
            "tp_atr": self._tp_atr,
        }
    
    def compute(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        **kwargs
    ) -> BracketLevels:
        risk_points = self._stop_atr * atr
        reward_points = self._tp_atr * atr
        
        if direction.upper() == "LONG":
            stop_price = entry_price - risk_points
            tp_price = entry_price + reward_points
        else:  # SHORT
            stop_price = entry_price + risk_points
            tp_price = entry_price - reward_points
        
        return BracketLevels(
            stop_price=stop_price,
            tp_price=tp_price,
            risk_points=risk_points,
            reward_points=reward_points,
            r_multiple=reward_points / risk_points if risk_points > 0 else 0
        )


class PercentBracket(Bracket):
    """
    Stop and TP as percentage of entry price.
    
    Agent config:
        {"type": "percent", "stop_pct": 0.5, "tp_pct": 1.0}
    """
    
    def __init__(self, stop_pct: float = 0.5, tp_pct: float = 1.0):
        self._stop_pct = stop_pct / 100  # Convert to decimal
        self._tp_pct = tp_pct / 100
    
    @property
    def bracket_type(self) -> BracketType:
        return BracketType.PERCENT
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "stop_pct": self._stop_pct * 100,
            "tp_pct": self._tp_pct * 100,
        }
    
    def compute(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        **kwargs
    ) -> BracketLevels:
        risk_points = entry_price * self._stop_pct
        reward_points = entry_price * self._tp_pct
        
        if direction.upper() == "LONG":
            stop_price = entry_price - risk_points
            tp_price = entry_price + reward_points
        else:
            stop_price = entry_price + risk_points
            tp_price = entry_price - reward_points
        
        return BracketLevels(
            stop_price=stop_price,
            tp_price=tp_price,
            risk_points=risk_points,
            reward_points=reward_points,
            r_multiple=reward_points / risk_points if risk_points > 0 else 0
        )


class FixedBracket(Bracket):
    """
    Fixed point stop and TP.
    
    Agent config:
        {"type": "fixed", "stop_points": 5.0, "tp_points": 10.0}
    """
    
    def __init__(self, stop_points: float = 5.0, tp_points: float = 10.0):
        self._stop_points = stop_points
        self._tp_points = tp_points
    
    @property
    def bracket_type(self) -> BracketType:
        return BracketType.FIXED
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "stop_points": self._stop_points,
            "tp_points": self._tp_points,
        }
    
    def compute(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        **kwargs
    ) -> BracketLevels:
        if direction.upper() == "LONG":
            stop_price = entry_price - self._stop_points
            tp_price = entry_price + self._tp_points
        else:
            stop_price = entry_price + self._stop_points
            tp_price = entry_price - self._tp_points
        
        return BracketLevels(
            stop_price=stop_price,
            tp_price=tp_price,
            risk_points=self._stop_points,
            reward_points=self._tp_points,
            r_multiple=self._tp_points / self._stop_points if self._stop_points > 0 else 0
        )


class ICTBracket(Bracket):
    """
    ICT-style bracket with PDH/PDL targeting.
    
    Uses pre-computed stop from scanner context (wick-based).
    Targets PDH/PDL if R:R is favorable, otherwise uses min_rr.
    
    Agent config:
        {"type": "ict", "min_rr": 1.5, "use_pdh_pdl": true}
    """
    
    def __init__(self, min_rr: float = 1.5, use_pdh_pdl: bool = True):
        self._min_rr = min_rr
        self._use_pdh_pdl = use_pdh_pdl
    
    @property
    def bracket_type(self) -> BracketType:
        return BracketType.LEVEL
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "min_rr": self._min_rr,
            "use_pdh_pdl": self._use_pdh_pdl,
        }
    
    def compute(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        stop_price: float = None,  # Pre-computed from scanner
        pdh: float = None,
        pdl: float = None,
        **kwargs
    ) -> BracketLevels:
        """
        Compute ICT bracket levels.
        
        Stop is expected from scanner context (at penetrating wick).
        TP targets PDH/PDL if favorable R:R, else min_rr.
        """
        # Use pre-computed stop if provided, otherwise fallback to ATR
        if stop_price is not None:
            computed_stop = stop_price
        else:
            # Fallback if no pre-computed stop
            if direction.upper() == "LONG":
                computed_stop = entry_price - (2.0 * atr)
            else:
                computed_stop = entry_price + (2.0 * atr)
        
        risk_points = abs(entry_price - computed_stop)
        min_reward = self._min_rr * risk_points
        
        if direction.upper() == "LONG":
            # Target PDH if favorable
            if self._use_pdh_pdl and pdh and (pdh - entry_price) >= min_reward:
                tp_price = pdh
            else:
                tp_price = entry_price + min_reward
        else:
            # Target PDL if favorable
            if self._use_pdh_pdl and pdl and (entry_price - pdl) >= min_reward:
                tp_price = pdl
            else:
                tp_price = entry_price - min_reward
        
        reward_points = abs(tp_price - entry_price)
        
        return BracketLevels(
            stop_price=computed_stop,
            tp_price=tp_price,
            risk_points=risk_points,
            reward_points=reward_points,
            r_multiple=reward_points / risk_points if risk_points > 0 else 0
        )


class RangeBracket(Bracket):
    """
    Bracket where SL is based on a context range size and TP is R-multiple.
    
    Perfect for OR strategies: SL = range size, TP = range * tp_multiple
    
    Agent config:
        {"type": "range", "tp_multiple": 2.0}
    
    Requires `range_size` to be passed in kwargs during compute().
    For OR False Break: range_size = OR high - OR low
    """
    
    def __init__(self, tp_multiple: float = 2.0, sl_buffer: float = 0.5):
        self._tp_multiple = tp_multiple
        self._sl_buffer = sl_buffer  # Extra points beyond range for SL
    
    @property
    def bracket_type(self) -> BracketType:
        return BracketType.LEVEL
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "tp_multiple": self._tp_multiple,
            "sl_buffer": self._sl_buffer,
        }
    
    def compute(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        range_size: float = None,  # Provided from trigger context
        target_level: float = None,  # Optional specific target (e.g., OR_low)
        **kwargs
    ) -> BracketLevels:
        """
        Compute range-based bracket levels.
        
        For LONG: SL = entry - range_size - buffer, TP = target_level or entry + range * tp_multiple
        For SHORT: SL = entry + range_size + buffer, TP = target_level or entry - range * tp_multiple
        """
        if range_size is None:
            range_size = atr * 2  # Fallback to ATR-based
        
        risk_points = range_size + self._sl_buffer
        
        if target_level is not None:
            reward_points = abs(target_level - entry_price)
        else:
            reward_points = range_size * self._tp_multiple
        
        if direction.upper() == "LONG":
            stop_price = entry_price - risk_points
            tp_price = target_level if target_level else entry_price + reward_points
        else:
            stop_price = entry_price + risk_points
            tp_price = target_level if target_level else entry_price - reward_points
        
        return BracketLevels(
            stop_price=stop_price,
            tp_price=tp_price,
            risk_points=risk_points,
            reward_points=reward_points,
            r_multiple=reward_points / risk_points if risk_points > 0 else 0
        )


# Registry and factory
BRACKET_REGISTRY = {
    "atr": ATRBracket,
    "percent": PercentBracket,
    "fixed": FixedBracket,
    "ict": ICTBracket,
    "range": RangeBracket,
}


def bracket_from_dict(config: dict) -> Bracket:
    """
    Factory function to create bracket from config dict.
    
    Agent-friendly:
        bracket_from_dict({"type": "atr", "stop_atr": 2.0, "tp_atr": 3.0})
    """
    config = config.copy()
    bracket_type = config.pop("type")
    
    if bracket_type not in BRACKET_REGISTRY:
        raise ValueError(f"Unknown bracket type: {bracket_type}. Available: {list(BRACKET_REGISTRY.keys())}")
    
    return BRACKET_REGISTRY[bracket_type](**config)


def list_brackets() -> list:
    """List available bracket types."""
    return list(BRACKET_REGISTRY.keys())


```

### src/policy/composite_scanner.py

```python
"""
Composite Scanner (The Strategy Engine)

This scanner interprets a JSON Recipe to build a dynamic strategy on the fly.
It replaces the need to write custom Python classes for every new strategy idea.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.features.state import MarketState
from src.features.pipeline import FeatureBundle
from src.policy.scanners import Scanner, ScanResult
from src.policy.triggers.factory import trigger_from_dict
from src.policy.triggers.base import TriggerResult

@dataclass
class CompositeConfig:
    """
    Configuration for a composed strategy.
    
    Example:
    {
        "name": "My Composed Strategy",
        "entry_trigger": {
            "type": "AND",
            "children": [
                {"type": "ema_cross", "fast": 9, "slow": 21},
                {"type": "rsi_threshold", "threshold": 30, "direction": "lt"}
            ]
        },
        "cooldown_bars": 10
    }
    """
    name: str
    entry_trigger: Dict[str, Any]
    cooldown_bars: int = 10


class CompositeScanner(Scanner):
    """
    A Scanner that executes a dynamic configuration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = CompositeConfig(
            name=config.get("name", "composite_strategy"),
            entry_trigger=config["entry_trigger"],
            cooldown_bars=config.get("cooldown_bars", 10)
        )
        
        # Build the Trigger Tree
        self._trigger = trigger_from_dict(self.config.entry_trigger)
        
        # State
        self._last_trigger_idx = -1000
    
    @property
    def scanner_id(self) -> str:
        # Use the name from the config as the ID
        # This ensures it shows up nicely in Trade Viz
        return self.config.name.lower().replace(" ", "_")
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScanResult:
        """
        Evaluate the trigger tree against current market features.
        """
        current_idx = features.bar_idx
        
        # 1. Check Cooldown
        if current_idx - self._last_trigger_idx < self.config.cooldown_bars:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
            
        # 2. Check Trigger
        res: TriggerResult = self._trigger.check(features)
        
        if res.triggered:
            self._last_trigger_idx = current_idx
            
            return ScanResult(
                scanner_id=self.scanner_id,
                triggered=True,
                context={
                    "direction": res.direction.value,
                    "confidence": res.confidence,
                    **res.context
                },
                score=res.confidence
            )
            
        return ScanResult(scanner_id=self.scanner_id, triggered=False)

```

### src/policy/cooldown.py

```python
"""
Cooldown
Manage trade cooldown periods to prevent overtrading.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class CooldownConfig:
    """Cooldown configuration."""
    min_bars_between_trades: int = 15  # Minimum bars between entries
    min_bars_after_loss: int = 30      # Extra cooldown after a loss
    max_trades_per_day: int = 10       # Maximum trades per trading day


class CooldownManager:
    """
    Manage trade cooldown state.
    """
    
    def __init__(self, config: CooldownConfig = None):
        self.config = config or CooldownConfig()
        self._last_trade_bar: int = -999
        self._last_outcome: str = ""
        self._trades_today: int = 0
        self._current_date: Optional[pd.Timestamp] = None
    
    def reset(self):
        """Reset cooldown state."""
        self._last_trade_bar = -999
        self._last_outcome = ""
        self._trades_today = 0
        self._current_date = None
    
    def record_trade(
        self,
        bar_idx: int,
        outcome: str = "",
        timestamp: pd.Timestamp = None
    ):
        """Record that a trade was taken."""
        self._last_trade_bar = bar_idx
        self._last_outcome = outcome
        
        # Track trades per day
        if timestamp:
            trade_date = timestamp.date()
            if self._current_date != trade_date:
                self._current_date = trade_date
                self._trades_today = 0
            self._trades_today += 1
    
    def is_on_cooldown(
        self,
        current_bar: int,
        timestamp: pd.Timestamp = None
    ) -> tuple:
        """
        Check if currently on cooldown.
        
        Returns:
            (on_cooldown: bool, reason: str)
        """
        # Check max trades per day
        if timestamp:
            current_date = timestamp.date()
            if current_date == self._current_date:
                if self._trades_today >= self.config.max_trades_per_day:
                    return (True, f"Max trades per day ({self.config.max_trades_per_day}) reached")
        
        # Check bars since last trade
        bars_since = current_bar - self._last_trade_bar
        
        # Extra cooldown after loss
        if self._last_outcome == 'LOSS':
            min_bars = self.config.min_bars_after_loss
        else:
            min_bars = self.config.min_bars_between_trades
        
        if bars_since < min_bars:
            return (True, f"Cooldown: {bars_since}/{min_bars} bars since last trade")
        
        return (False, "")
    
    def bars_remaining(self, current_bar: int) -> int:
        """Get bars remaining in cooldown."""
        bars_since = current_bar - self._last_trade_bar
        
        if self._last_outcome == 'LOSS':
            min_bars = self.config.min_bars_after_loss
        else:
            min_bars = self.config.min_bars_between_trades
        
        return max(0, min_bars - bars_since)

```

### src/policy/entry_scans.py

```python
"""
Entry Scans - Modular Entry Order Modification

Entry Scans refine HOW a trade enters after a trigger fires.
They don't replace scanners/models, they MODIFY the entry order.

Categories (exclusive within each):
- Entry Type: Market vs Limit
- Stop Method: ATR, Behind Swing, Fixed Bars
- TP Method: ATR, R-Multiple

Usage:
    from src.policy.entry_scans import apply_entry_scans, EntryConfig
    
    config = EntryConfig(
        entry_type='limit',
        stop_method='swing',
        tp_method='r_multiple'
    )
    modified_order = apply_entry_scans(base_order, df_history, config)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class EntryOrder:
    """Represents a trade entry order."""
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    stop_price: float
    tp_price: float
    entry_type: str = 'market'  # 'market' or 'limit'
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntryConfig:
    """Configuration for entry scan processing."""
    entry_type: str = 'market'  # 'market' or 'limit'
    stop_method: str = 'atr'    # 'atr', 'swing', 'fixed_bars'
    tp_method: str = 'atr'      # 'atr', 'r_multiple'
    
    # ATR parameters
    stop_atr_multiple: float = 1.0
    tp_atr_multiple: float = 2.0
    
    # Swing parameters
    swing_lookback: int = 5
    swing_buffer_atr: float = 0.1  # Buffer beyond swing level
    
    # Fixed bars parameters
    fixed_bars_lookback: int = 3
    
    # R-multiple parameters
    tp_r_multiple: float = 2.0


# =============================================================================
# Entry Type Modifiers
# =============================================================================

class EntryTypeModifier(ABC):
    """Base class for entry type modification."""
    
    @abstractmethod
    def modify(self, order: EntryOrder, bar: pd.Series) -> EntryOrder:
        pass


class MarketEntry(EntryTypeModifier):
    """Enter at current market price (close of bar)."""
    
    def modify(self, order: EntryOrder, bar: pd.Series) -> EntryOrder:
        order.entry_type = 'market'
        order.entry_price = float(bar['close'])
        return order


class LimitAtDecision(EntryTypeModifier):
    """Place limit order at the decision point price."""
    
    def modify(self, order: EntryOrder, bar: pd.Series) -> EntryOrder:
        order.entry_type = 'limit'
        # Keep entry_price as-is (from signal)
        return order


# =============================================================================
# Stop Placement Modifiers
# =============================================================================

class StopModifier(ABC):
    """Base class for stop loss modification."""
    
    @abstractmethod
    def modify(self, order: EntryOrder, df_history: pd.DataFrame, config: EntryConfig) -> EntryOrder:
        pass


class StopAtATR(StopModifier):
    """Place stop at N×ATR from entry."""
    
    def modify(self, order: EntryOrder, df_history: pd.DataFrame, config: EntryConfig) -> EntryOrder:
        # Calculate ATR
        high = df_history['high']
        low = df_history['low']
        close = df_history['close'].shift(1)
        tr = pd.concat([high - low, abs(high - close), abs(low - close)], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        
        if order.direction == 'LONG':
            order.stop_price = order.entry_price - (atr * config.stop_atr_multiple)
        else:
            order.stop_price = order.entry_price + (atr * config.stop_atr_multiple)
        
        return order


class StopBehindSwing(StopModifier):
    """Place stop behind the most recent swing high/low."""
    
    def modify(self, order: EntryOrder, df_history: pd.DataFrame, config: EntryConfig) -> EntryOrder:
        lookback = config.swing_lookback
        recent = df_history.tail(lookback * 2)
        
        # Calculate small ATR for buffer
        tr = recent['high'] - recent['low']
        atr = float(tr.mean())
        buffer = atr * config.swing_buffer_atr
        
        if order.direction == 'LONG':
            # Find swing low (lowest low in lookback)
            swing_low = float(recent['low'].min())
            order.stop_price = swing_low - buffer
        else:
            # Find swing high (highest high in lookback)
            swing_high = float(recent['high'].max())
            order.stop_price = swing_high + buffer
        
        return order


class StopAtFixedBars(StopModifier):
    """Place stop at high/low of N bars ago."""
    
    def modify(self, order: EntryOrder, df_history: pd.DataFrame, config: EntryConfig) -> EntryOrder:
        lookback = config.fixed_bars_lookback
        ref_bar = df_history.iloc[-lookback] if len(df_history) >= lookback else df_history.iloc[0]
        
        if order.direction == 'LONG':
            order.stop_price = float(ref_bar['low'])
        else:
            order.stop_price = float(ref_bar['high'])
        
        return order


# =============================================================================
# Take Profit Modifiers
# =============================================================================

class TPModifier(ABC):
    """Base class for take profit modification."""
    
    @abstractmethod
    def modify(self, order: EntryOrder, df_history: pd.DataFrame, config: EntryConfig) -> EntryOrder:
        pass


class TPAtATR(TPModifier):
    """Place TP at N×ATR from entry."""
    
    def modify(self, order: EntryOrder, df_history: pd.DataFrame, config: EntryConfig) -> EntryOrder:
        # Calculate ATR
        high = df_history['high']
        low = df_history['low']
        close = df_history['close'].shift(1)
        tr = pd.concat([high - low, abs(high - close), abs(low - close)], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        
        if order.direction == 'LONG':
            order.tp_price = order.entry_price + (atr * config.tp_atr_multiple)
        else:
            order.tp_price = order.entry_price - (atr * config.tp_atr_multiple)
        
        return order


class TPAtRMultiple(TPModifier):
    """Place TP at R×risk from entry (R-multiple)."""
    
    def modify(self, order: EntryOrder, df_history: pd.DataFrame, config: EntryConfig) -> EntryOrder:
        # Calculate risk (distance to stop)
        risk = abs(order.entry_price - order.stop_price)
        
        if order.direction == 'LONG':
            order.tp_price = order.entry_price + (risk * config.tp_r_multiple)
        else:
            order.tp_price = order.entry_price - (risk * config.tp_r_multiple)
        
        return order


# =============================================================================
# Entry Scan Registry
# =============================================================================

ENTRY_TYPE_MODIFIERS = {
    'market': MarketEntry(),
    'limit': LimitAtDecision(),
}

STOP_MODIFIERS = {
    'atr': StopAtATR(),
    'swing': StopBehindSwing(),
    'fixed_bars': StopAtFixedBars(),
}

TP_MODIFIERS = {
    'atr': TPAtATR(),
    'r_multiple': TPAtRMultiple(),
}


# =============================================================================
# Main Entry Point
# =============================================================================

def apply_entry_scans(
    base_order: EntryOrder,
    df_history: pd.DataFrame,
    config: EntryConfig,
    current_bar: Optional[pd.Series] = None
) -> EntryOrder:
    """
    Apply entry scans to modify the base order.
    
    Args:
        base_order: Initial order from signal
        df_history: Historical OHLCV data
        config: Entry scan configuration
        current_bar: Current bar data (for entry type)
        
    Returns:
        Modified EntryOrder with final levels
    """
    order = base_order
    
    # Apply entry type modifier
    if current_bar is not None and config.entry_type in ENTRY_TYPE_MODIFIERS:
        order = ENTRY_TYPE_MODIFIERS[config.entry_type].modify(order, current_bar)
    
    # Apply stop modifier (must be before TP if using R-multiple)
    if config.stop_method in STOP_MODIFIERS:
        order = STOP_MODIFIERS[config.stop_method].modify(order, df_history, config)
    
    # Apply TP modifier
    if config.tp_method in TP_MODIFIERS:
        order = TP_MODIFIERS[config.tp_method].modify(order, df_history, config)
    
    return order


def create_default_config() -> EntryConfig:
    """Create default entry config (current behavior)."""
    return EntryConfig(
        entry_type='market',
        stop_method='atr',
        tp_method='atr',
        stop_atr_multiple=1.0,
        tp_atr_multiple=2.0
    )

```

### src/policy/filters.py

```python
"""
Filters
Pre-trade filters that block decisions before reaching policy.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from src.features.pipeline import FeatureBundle
from src.features.time_features import Session


@dataclass
class FilterResult:
    """Result from a filter check."""
    passed: bool
    filter_id: str
    reason: str = ""


class Filter(ABC):
    """Base class for pre-trade filters."""
    
    @property
    @abstractmethod
    def filter_id(self) -> str:
        pass
    
    @abstractmethod
    def check(self, features: FeatureBundle) -> FilterResult:
        """Check if filter passes."""
        pass


class SessionFilter(Filter):
    """Only trade during specific sessions."""
    
    def __init__(self, allowed_sessions: List[str] = None):
        self.allowed_sessions = allowed_sessions or ['RTH']
    
    @property
    def filter_id(self) -> str:
        return f"session_{'_'.join(self.allowed_sessions)}"
    
    def check(self, features: FeatureBundle) -> FilterResult:
        if features.time_features is None:
            return FilterResult(passed=False, filter_id=self.filter_id, reason="No time features")
        
        session = features.time_features.session
        passed = session in self.allowed_sessions
        
        return FilterResult(
            passed=passed,
            filter_id=self.filter_id,
            reason="" if passed else f"Session {session} not in {self.allowed_sessions}"
        )


class TimeFilter(Filter):
    """Only trade during specific hours."""
    
    def __init__(
        self,
        allowed_hours: List[int] = None,
        excluded_hours: List[int] = None
    ):
        self.allowed_hours = allowed_hours  # If set, only these hours
        self.excluded_hours = excluded_hours or []  # Always exclude these
    
    @property
    def filter_id(self) -> str:
        return "time_filter"
    
    def check(self, features: FeatureBundle) -> FilterResult:
        if features.time_features is None:
            return FilterResult(passed=False, filter_id=self.filter_id, reason="No time features")
        
        hour = features.time_features.hour_ny
        
        if hour in self.excluded_hours:
            return FilterResult(
                passed=False,
                filter_id=self.filter_id,
                reason=f"Hour {hour} is excluded"
            )
        
        if self.allowed_hours and hour not in self.allowed_hours:
            return FilterResult(
                passed=False,
                filter_id=self.filter_id,
                reason=f"Hour {hour} not in allowed hours"
            )
        
        return FilterResult(passed=True, filter_id=self.filter_id)


class VolatilityFilter(Filter):
    """Filter based on ATR or volatility conditions."""
    
    def __init__(
        self,
        min_atr: float = 0.0,
        max_adr_pct: float = 1.5
    ):
        self.min_atr = min_atr
        self.max_adr_pct = max_adr_pct
    
    @property
    def filter_id(self) -> str:
        return f"volatility_{self.min_atr}_{self.max_adr_pct}"
    
    def check(self, features: FeatureBundle) -> FilterResult:
        # Check minimum ATR
        if self.min_atr > 0 and features.atr < self.min_atr:
            return FilterResult(
                passed=False,
                filter_id=self.filter_id,
                reason=f"ATR {features.atr:.2f} below minimum {self.min_atr}"
            )
        
        # Check ADR consumption
        if features.indicators and features.indicators.adr_pct_used > self.max_adr_pct:
            return FilterResult(
                passed=False,
                filter_id=self.filter_id,
                reason=f"ADR {features.indicators.adr_pct_used:.1%} exceeds max {self.max_adr_pct:.1%}"
            )
        
        return FilterResult(passed=True, filter_id=self.filter_id)


class FilterChain:
    """Run multiple filters in sequence."""
    
    def __init__(self, filters: List[Filter] = None):
        self.filters = filters or []
    
    def add(self, f: Filter) -> 'FilterChain':
        self.filters.append(f)
        return self
    
    def check(self, features: FeatureBundle) -> FilterResult:
        """
        Run all filters. Returns first failure or final pass.
        """
        for f in self.filters:
            result = f.check(features)
            if not result.passed:
                return result
        
        return FilterResult(passed=True, filter_id="all", reason="All filters passed")


# Default filter chain for RTH trading
DEFAULT_FILTERS = FilterChain([
    SessionFilter(['RTH']),
    TimeFilter(excluded_hours=[12]),  # Exclude lunch
])

```

### src/policy/library/__init__.py

```python
"""
Strategy Library
Modular implementations of setup scanners.
"""

```

### src/policy/library/delayed_breakout.py

```python
"""
Delayed Breakout Scanner (1.4 RR)
Triggers trades on 15m swing breakouts only after 11:30 AM (2 hours after open).
Uses a fixed 1.4 Risk:Reward ratio.
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass
from datetime import time

from src.policy.scanners import Scanner, ScanResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle


from src.config import POINT_VALUE, TICK_SIZE

@dataclass
class DelayedBreakoutState:
    """Tracks swing levels and cooldown."""
    last_trigger_bar: int = -1000
    last_swing_high: float = 0.0
    last_swing_low: float = 0.0


class DelayedBreakoutScanner(Scanner):
    """
    Scanner that triggers on breakout of 15m swing structure.
    
    Specific Rules:
    1. TIME FILTER: No trades before 11:30 AM.
    2. ENTRY: Breakout of N-bar 15m high/low.
    3. STOP: Opposite swing level.
    4. TARGET: Fixed 1.4 Risk:Reward Ratio.
    5. SIZING: Dynamic contracts based on max_risk_dollars.
    
    Config:
        lookback_bars: Number of 15m bars to look back for swing detection.
        min_atr_distance: Minimum breakout distance in ATR to avoid noise.
        cooldown_bars: Minimum 1m bars between triggers.
        max_risk_dollars: Maximum dollar risk per trade (default 300).
    """
    
    def __init__(
        self,
        lookback_bars: int = 20, 
        min_atr_distance: float = 0.3,
        cooldown_bars: int = 30,
        max_risk_dollars: float = 300.0
    ):
        self.lookback_bars = lookback_bars
        self.min_atr_distance = min_atr_distance
        self.cooldown_bars = cooldown_bars
        self.max_risk_dollars = max_risk_dollars
        self._state = DelayedBreakoutState()
        
        # 11:30 AM cutoff
        self.start_time = time(11, 30)
        self.end_time = time(15, 30)
    
    @property
    def scanner_id(self) -> str:
        return f"delayed_breakout_1.4rr"
    
    def _compute_swing_levels(self, df_15m: pd.DataFrame, current_time: pd.Timestamp) -> tuple:
        """Compute swing high and swing low from recent 15m bars."""
        if df_15m is None or df_15m.empty:
            return (0.0, 0.0)
        
        # Get bars up to current time
        mask = df_15m['time'] <= current_time
        recent = df_15m.loc[mask].tail(self.lookback_bars + 1)
        
        if len(recent) < 5:
            return (0.0, 0.0)
        
        # Exclude current bar
        lookback = recent.iloc[:-1] if len(recent) > 1 else recent
        
        swing_high = lookback['high'].max()
        swing_low = lookback['low'].min()
        
        return (swing_high, swing_low)
    
    def _find_recent_engulfing(self, df_5m: pd.DataFrame, current_time: pd.Timestamp, direction: str) -> tuple:
        """
        Find the most recent 5m engulfing candle in the specified direction.
        Returns (low, high, found_bool).
        
        Engulfing Definition:
        - Bullish (LONG): Close > Open AND Body engulfs previous Red candle body. 
          (Simpler: Close > Prev Open and Open < Prev Close).
        - Bearish (SHORT): Close < Open AND Body engulfs previous Green candle body.
        
        Lookback: 4 hours (~48 bars).
        """
        if df_5m is None or df_5m.empty:
            return (0.0, 0.0, False)
            
        mask = df_5m['time'] < current_time # Strictly before entry trigger? Or including current? Trigger is current.
        # usually 5m bar is forming. If strategy runs on 1m bars, 5m bar might not be closed.
        # We should look at COMPLETED 5m bars.
        # Assuming df_5m contains closed bars or we check indices.
        # Safest to check bars < current_time
        
        recent = df_5m.loc[mask].tail(48) # 4 hours
        if len(recent) < 2:
            return (0.0, 0.0, False)
        
        # Iterate backwards
        bars = recent.to_dict('records')
        for i in range(len(bars) - 1, 0, -1):
            curr = bars[i]
            prev = bars[i-1]
            
            curr_body_top = max(curr['open'], curr['close'])
            curr_body_bottom = min(curr['open'], curr['close'])
            prev_body_top = max(prev['open'], prev['close'])
            prev_body_bottom = min(prev['open'], prev['close'])
            
            is_bullish = curr['close'] > curr['open']
            is_bearish = curr['close'] < curr['open']
            
            if direction == "LONG":
                # Bullish Engulfing
                if is_bullish and curr_body_top > prev_body_top and curr_body_bottom < prev_body_bottom:
                    return (curr['low'], curr['high'], True)
                    
            elif direction == "SHORT":
                # Bearish Engulfing
                if is_bearish and curr_body_top > prev_body_top and curr_body_bottom < prev_body_bottom:
                    return (curr['low'], curr['high'], True)
                    
        return (0.0, 0.0, False)

    def _calculate_position_size(self, entry: float, stop: float) -> tuple:
        """Calculate max contracts for given risk."""
        dist_points = abs(entry - stop)
        if dist_points < TICK_SIZE: return 0, 0.0
        
        risk_per_contract = dist_points * POINT_VALUE
        if risk_per_contract <= 0: return 0, 0.0
        
        # Max contracts
        contracts = int(self.max_risk_dollars // risk_per_contract)
        contracts = max(1, contracts) 
        
        if contracts * risk_per_contract > self.max_risk_dollars:
             if contracts == 0: return 0, 0.0
        
        contract_risk = contracts * risk_per_contract
        return contracts, contract_risk

    def scan(
        self,
        state: MarketState,
        features: FeatureBundle,
        df_15m: pd.DataFrame = None,
        df_5m: pd.DataFrame = None
    ) -> ScanResult:
        """Check for delayed breakout."""
        t = features.timestamp
        if t is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # 1. TIME FILTER
        current_t = t.time()
        if current_t < self.start_time or current_t > self.end_time:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
            
        # Cooldown check
        if features.bar_idx - self._state.last_trigger_bar < self.cooldown_bars:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Need data
        if df_15m is None or df_15m.empty:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Compute swings (for Entry)
        swing_high, swing_low = self._compute_swing_levels(df_15m, t)
        
        if swing_high == 0 or swing_low == 0:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        current_price = features.current_price
        atr = features.atr if features.atr > 0 else 1.0
        min_breakout = self.min_atr_distance * atr
        
        # Check Breakout
        long_breakout = current_price > swing_high and (current_price - swing_high) >= min_breakout
        short_breakout = current_price < swing_low and (swing_low - current_price) >= min_breakout
        
        if long_breakout or short_breakout:
            direction = "LONG" if long_breakout else "SHORT"
            
            # 2. Risk Management (Stop on Engulfing + ATR Padding)
            # Find Engulfing
            eng_low, eng_high, found_engulfing = self._find_recent_engulfing(df_5m, t, direction)
            
            # Stop Calculation
            padding = 0.5 * atr # "ATR padding" - let's assume 0.5 or 1.0? User said "atr padding". 1.0 is safe.
            # User said "sl the low with atr padding".
            
            if direction == "LONG":
                if found_engulfing:
                    stop_price = eng_low - padding
                else:
                    # Fallback to swing low if no engulfing found
                    stop_price = swing_low - padding
                
                risk = current_price - stop_price
                if risk <= 0: return ScanResult(scanner_id=self.scanner_id, triggered=False)
                tp_price = current_price + (1.4 * risk)
                
            else: # SHORT
                if found_engulfing:
                    stop_price = eng_high + padding
                else:
                    stop_price = swing_high + padding
                
                risk = stop_price - current_price
                if risk <= 0: return ScanResult(scanner_id=self.scanner_id, triggered=False)
                tp_price = current_price - (1.4 * risk)
            
            # 3. Position Sizing
            contracts, actual_risk = self._calculate_position_size(current_price, stop_price)
            if contracts == 0:
                 return ScanResult(scanner_id=self.scanner_id, triggered=False)

            self._state.last_trigger_bar = features.bar_idx
            
            return ScanResult(
                scanner_id=self.scanner_id,
                triggered=True,
                context={
                    'direction': direction,
                    'swing_high': swing_high,
                    'swing_low': swing_low,
                    'engulfing_found': found_engulfing,
                    'entry_price': current_price,
                    'stop_price': stop_price,
                    'tp_price': tp_price,
                    'risk_points': abs(risk),
                    'reward_points': abs(tp_price - current_price),
                    'contracts': contracts,
                    'risk_dollars': actual_risk,
                    'reward_dollars': actual_risk * 1.4,
                    'rr_ratio': 1.4
                },
                score=1.0
            )
            
        return ScanResult(scanner_id=self.scanner_id, triggered=False)

```

### src/policy/library/first_pullback.py

```python
"""
First Pullback Scanner
After opening drive, triggers on first pullback to EMA.
"""

import pandas as pd
from typing import Optional
from dataclasses import dataclass

from src.policy.scanners import Scanner, ScanResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle
from src.config import NY_TZ


@dataclass
class FirstPullbackState:
    """Tracks opening bias and pullback state."""
    date: Optional[pd.Timestamp] = None
    opening_price: float = 0.0
    opening_bias: Optional[str] = None  # 'BULLISH' or 'BEARISH'
    bias_established: bool = False
    pullback_triggered: bool = False


class FirstPullbackScanner(Scanner):
    """
    Scanner that triggers on first pullback to EMA after opening drive.
    
    Logic:
    1. At 10:00 AM NY, establish opening bias based on price vs 9:30 open.
    2. LONG: Bullish bias (price > open) and price pulls back to EMA20.
    3. SHORT: Bearish bias (price < open) and price pulls back to EMA20.
    4. Only one trigger per day.
    
    Config:
        bias_threshold_atr: Min move to establish bias (default 0.5 ATR).
        ema_threshold_atr: How close to EMA for pullback (default 0.3 ATR).
    """
    
    def __init__(
        self,
        bias_threshold_atr: float = 0.5,
        ema_threshold_atr: float = 0.3,
    ):
        self.bias_threshold_atr = bias_threshold_atr
        self.ema_threshold_atr = ema_threshold_atr
        self._state = FirstPullbackState()
    
    @property
    def scanner_id(self) -> str:
        return "first_pullback"
    
    def _is_new_day(self, t: pd.Timestamp) -> bool:
        if self._state.date is None:
            return True
        ny_time = t.astimezone(NY_TZ)
        return ny_time.date() != self._state.date.date()
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScanResult:
        t = features.timestamp
        if t is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        ny_time = t.astimezone(NY_TZ)
        
        # Reset on new day
        if self._is_new_day(t):
            self._state = FirstPullbackState(date=t.astimezone(NY_TZ))
        
        # Capture opening price at 9:30-9:31
        if ny_time.hour == 9 and 30 <= ny_time.minute <= 31 and self._state.opening_price == 0:
            self._state.opening_price = features.current_price
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Establish bias at 10:00
        if ny_time.hour == 10 and ny_time.minute == 0 and not self._state.bias_established:
            if self._state.opening_price > 0:
                price = features.current_price
                atr = features.atr if features.atr > 0 else 1.0
                move = (price - self._state.opening_price) / atr
                
                if move > self.bias_threshold_atr:
                    self._state.opening_bias = "BULLISH"
                    self._state.bias_established = True
                elif move < -self.bias_threshold_atr:
                    self._state.opening_bias = "BEARISH"
                    self._state.bias_established = True
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Only look for pullback between 10:00 and 11:30
        if not (10 <= ny_time.hour < 12 or (ny_time.hour == 11 and ny_time.minute <= 30)):
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Already triggered today
        if self._state.pullback_triggered:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Need bias
        if not self._state.bias_established or self._state.opening_bias is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Need indicators
        if features.indicators is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        price = features.current_price
        ema = features.indicators.ema_5m_20
        atr = features.atr if features.atr > 0 else 1.0
        
        if ema <= 0:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        dist_to_ema = abs(price - ema) / atr
        
        # Check pullback condition
        if dist_to_ema <= self.ema_threshold_atr:
            direction = "LONG" if self._state.opening_bias == "BULLISH" else "SHORT"
            
            # Confirm pullback direction matches bias
            # Bullish: price should be pulling back down toward EMA (price near or below EMA OK)
            # Bearish: price should be pulling back up toward EMA
            valid_pullback = (
                (self._state.opening_bias == "BULLISH" and price <= ema + (self.ema_threshold_atr * atr)) or
                (self._state.opening_bias == "BEARISH" and price >= ema - (self.ema_threshold_atr * atr))
            )
            
            if valid_pullback:
                self._state.pullback_triggered = True
                return ScanResult(
                    scanner_id=self.scanner_id,
                    triggered=True,
                    context={
                        'direction': direction,
                        'opening_bias': self._state.opening_bias,
                        'opening_price': self._state.opening_price,
                        'entry_price': price,
                        'ema_level': ema,
                        'pullback_depth_atr': dist_to_ema,
                    },
                    score=1.0
                )
        
        return ScanResult(scanner_id=self.scanner_id, triggered=False)

```

### src/policy/library/ict_fvg.py

```python
"""
ICT Fair Value Gap Scanner

Strategy Logic:
1. Wait for price to break overnight level (Asian/London high or low)
2. Look for structure change in opposite direction with impulse candle + FVG
3. Wait for price to retrace into FVG at least 50%
4. Enter in direction of structure change
5. Stop at the wick that penetrated the overnight level
6. TP at PDH/PDL or 1:3 R:R minimum

Trade Window: 9:30 AM - 11:30 AM NY
London Cutoff: 8:30 AM NY (levels must be set before this)
Risk: $300 per trade, max 1 trade at a time
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import time

from src.policy.scanners import Scanner, ScanResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle
from src.features.session_levels import (
    SessionLevels, 
    compute_session_levels,
    is_in_trade_window,
    is_london_complete,
    TRADE_WINDOW_START,
    TRADE_WINDOW_END
)
from src.features.fvg import (
    FairValueGap,
    find_fvg,
    find_most_recent_fvg,
    find_impulse_with_fvg
)
from src.config import POINT_VALUE, TICK_SIZE, NY_TZ


@dataclass
class ICTFVGState:
    """Tracks state for ICT FVG strategy."""
    # Level break tracking
    asian_high_broken: bool = False
    asian_low_broken: bool = False
    london_high_broken: bool = False
    london_low_broken: bool = False
    
    # Active setup tracking
    active_fvg: Optional[FairValueGap] = None
    pending_direction: Optional[str] = None
    penetrating_wick: float = 0.0
    broken_level: float = 0.0
    break_type: str = ""  # "asian_high", "asian_low", "london_high", "london_low"
    
    # Trade management
    last_trigger_bar: int = -1000
    last_trigger_date: Optional[Any] = None
    in_position: bool = False
    
    # Session levels for current day
    session_levels: Optional[SessionLevels] = None


class ICTFVGScanner(Scanner):
    """
    ICT Fair Value Gap Strategy Scanner.
    
    Entry Logic:
    1. During trade window (9:30-11:30 NY), after London cutoff (8:30)
    2. Price breaks overnight level (Asian or London H/L)
    3. Market structure changes with impulse candle creating FVG
    4. Wait for 50% retracement into FVG
    5. Enter in direction of structure change
    
    Exit Logic:
    - Stop: Wick that penetrated the overnight level + small buffer
    - TP: PDH/PDL if favorable R:R, else 1:3 R:R minimum
    
    Config:
        cooldown_bars: Min bars between triggers (default 30 = 2.5 hours on 5m)
        max_risk_dollars: Max risk per trade (default 300)
        min_rr: Minimum Risk:Reward ratio (default 1.5)
        fvg_min_pct: Min percentage into FVG for entry (default 0.5)
    """
    
    def __init__(
        self,
        cooldown_bars: int = 30,
        max_risk_dollars: float = 300.0,
        min_rr: float = 1.5,
        fvg_min_pct: float = 0.5,
        atr_buffer: float = 0.25  # Buffer for stop beyond wick
    ):
        self.cooldown_bars = cooldown_bars
        self.max_risk_dollars = max_risk_dollars
        self.min_rr = min_rr
        self.fvg_min_pct = fvg_min_pct
        self.atr_buffer = atr_buffer
        self._state = ICTFVGState()
    
    @property
    def scanner_id(self) -> str:
        return "ict_fvg_5m"
    
    def reset(self):
        """Reset state for new day or simulation run."""
        self._state = ICTFVGState()
    
    def _calculate_position_size(self, entry: float, stop: float) -> tuple:
        """Calculate max contracts for given risk."""
        dist_points = abs(entry - stop)
        if dist_points < TICK_SIZE:
            return 0, 0.0
        
        risk_per_contract = dist_points * POINT_VALUE
        if risk_per_contract <= 0:
            return 0, 0.0
        
        contracts = int(self.max_risk_dollars // risk_per_contract)
        contracts = max(1, contracts)
        
        contract_risk = contracts * risk_per_contract
        if contract_risk > self.max_risk_dollars * 1.1:
            contracts = max(1, contracts - 1)
            contract_risk = contracts * risk_per_contract
        
        return contracts, contract_risk
    
    def _check_level_break(
        self,
        current_price: float,
        current_high: float,
        current_low: float,
        session_levels: SessionLevels
    ) -> Optional[Dict[str, Any]]:
        """
        Check if price has broken an overnight level.
        
        Returns dict with break info or None.
        """
        # Check Asian high break (SHORT setup potential)
        if session_levels.asian_high > 0 and not self._state.asian_high_broken:
            if current_high > session_levels.asian_high:
                self._state.asian_high_broken = True
                return {
                    'level_type': 'asian_high',
                    'level_price': session_levels.asian_high,
                    'break_direction': 'UP',
                    'setup_direction': 'SHORT',  # Trade opposite after structure change
                    'penetrating_wick': current_high
                }
        
        # Check Asian low break (LONG setup potential)
        if session_levels.asian_low > 0 and not self._state.asian_low_broken:
            if current_low < session_levels.asian_low:
                self._state.asian_low_broken = True
                return {
                    'level_type': 'asian_low',
                    'level_price': session_levels.asian_low,
                    'break_direction': 'DOWN',
                    'setup_direction': 'LONG',
                    'penetrating_wick': current_low
                }
        
        # Check London high break (SHORT setup potential)
        if session_levels.london_high > 0 and not self._state.london_high_broken:
            if current_high > session_levels.london_high:
                self._state.london_high_broken = True
                return {
                    'level_type': 'london_high',
                    'level_price': session_levels.london_high,
                    'break_direction': 'UP',
                    'setup_direction': 'SHORT',
                    'penetrating_wick': current_high
                }
        
        # Check London low break (LONG setup potential)
        if session_levels.london_low > 0 and not self._state.london_low_broken:
            if current_low < session_levels.london_low:
                self._state.london_low_broken = True
                return {
                    'level_type': 'london_low',
                    'level_price': session_levels.london_low,
                    'break_direction': 'DOWN',
                    'setup_direction': 'LONG',
                    'penetrating_wick': current_low
                }
        
        return None
    
    def _find_structure_change_fvg(
        self,
        df_5m: pd.DataFrame,
        expected_direction: str,
        atr: float
    ) -> Optional[FairValueGap]:
        """
        Look for a structure change with impulse candle creating FVG.
        
        After a level break UP, we expect a structure change DOWN (bearish FVG).
        After a level break DOWN, we expect a structure change UP (bullish FVG).
        """
        result = find_impulse_with_fvg(df_5m, expected_direction, lookback=10, atr=atr)
        if result:
            _, fvg = result
            return fvg
        return None
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle,
        df_5m: pd.DataFrame = None,
        df_1m: pd.DataFrame = None
    ) -> ScanResult:
        """Check for ICT FVG setup."""
        t = features.timestamp
        if t is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # 1. Check if we're in the trade window (9:30 - 11:30 NY)
        if not is_in_trade_window(t, NY_TZ):
            # Reset state at end of day
            current_date = t.astimezone(NY_TZ).date()
            if self._state.last_trigger_date != current_date:
                self.reset()
                self._state.last_trigger_date = current_date
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # 2. Check if London session is complete (past 8:30)
        if not is_london_complete(t, NY_TZ):
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # 3. Cooldown check
        if features.bar_idx - self._state.last_trigger_bar < self.cooldown_bars:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # 4. Check if in position (only one trade at a time)
        if self._state.in_position:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # 5. Need data
        if df_5m is None or df_5m.empty or df_1m is None or df_1m.empty:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        current_price = features.current_price
        atr = features.atr if features.atr > 0 else 5.0
        
        # 6. Compute session levels if not already done today
        self._state.session_levels = compute_session_levels(df_1m, t, NY_TZ)
        session_levels = self._state.session_levels
        
        if session_levels is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Get current bar info
        current_bar = df_5m.iloc[-1] if len(df_5m) > 0 else None
        if current_bar is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        current_high = current_bar['high']
        current_low = current_bar['low']
        
        # 7. Check for level break if we don't have an active setup
        if self._state.active_fvg is None:
            level_break = self._check_level_break(
                current_price, current_high, current_low, session_levels
            )
            
            if level_break:
                # Level just broke - store info and look for structure change
                self._state.pending_direction = level_break['setup_direction']
                self._state.penetrating_wick = level_break['penetrating_wick']
                self._state.broken_level = level_break['level_price']
                self._state.break_type = level_break['level_type']
                
                # Look for FVG indicating structure change
                fvg = self._find_structure_change_fvg(
                    df_5m, level_break['setup_direction'], atr
                )
                
                if fvg:
                    self._state.active_fvg = fvg
        
        # 8. If no FVG yet but we have a pending direction, keep looking
        if self._state.active_fvg is None and self._state.pending_direction:
            fvg = self._find_structure_change_fvg(
                df_5m, self._state.pending_direction, atr
            )
            if fvg:
                self._state.active_fvg = fvg
        
        # 9. Check for FVG retracement entry
        if self._state.active_fvg is not None:
            fvg = self._state.active_fvg
            
            # Check if price has retraced at least 50% into FVG
            if fvg.contains_price(current_price, self.fvg_min_pct):
                direction = self._state.pending_direction
                
                if direction is None:
                    # Fallback based on FVG direction
                    direction = "LONG" if fvg.direction == "BULLISH" else "SHORT"
                
                # Calculate stop and TP
                if direction == "LONG":
                    # Stop below penetrating wick
                    stop_price = self._state.penetrating_wick - (self.atr_buffer * atr)
                    risk = current_price - stop_price
                    
                    if risk <= 0:
                        return ScanResult(scanner_id=self.scanner_id, triggered=False)
                    
                    # TP: Use PDH if favorable, otherwise 1:3 RR minimum
                    pdh = features.levels.pdh if features.levels else 0
                    tp_at_rr = current_price + (self.min_rr * risk)
                    
                    if pdh > 0 and (pdh - current_price) >= (self.min_rr * risk):
                        tp_price = pdh
                    else:
                        tp_price = tp_at_rr
                        
                else:  # SHORT
                    # Stop above penetrating wick
                    stop_price = self._state.penetrating_wick + (self.atr_buffer * atr)
                    risk = stop_price - current_price
                    
                    if risk <= 0:
                        return ScanResult(scanner_id=self.scanner_id, triggered=False)
                    
                    # TP: Use PDL if favorable, otherwise 1:3 RR minimum
                    pdl = features.levels.pdl if features.levels else 0
                    tp_at_rr = current_price - (self.min_rr * risk)
                    
                    if pdl > 0 and (current_price - pdl) >= (self.min_rr * risk):
                        tp_price = pdl
                    else:
                        tp_price = tp_at_rr
                
                # Calculate position size
                contracts, actual_risk = self._calculate_position_size(current_price, stop_price)
                if contracts == 0:
                    return ScanResult(scanner_id=self.scanner_id, triggered=False)
                
                reward = abs(tp_price - current_price)
                rr_ratio = reward / risk if risk > 0 else 0
                
                # Update state
                self._state.last_trigger_bar = features.bar_idx
                self._state.in_position = True
                
                # Clear active setup
                active_fvg = self._state.active_fvg
                self._state.active_fvg = None
                self._state.pending_direction = None
                
                return ScanResult(
                    scanner_id=self.scanner_id,
                    triggered=True,
                    context={
                        'direction': direction,
                        'entry_price': current_price,
                        'stop_price': stop_price,
                        'tp_price': tp_price,
                        'risk_points': risk,
                        'reward_points': reward,
                        'rr_ratio': rr_ratio,
                        'contracts': contracts,
                        'risk_dollars': actual_risk,
                        'level_broken': self._state.break_type,
                        'broken_level_price': self._state.broken_level,
                        'penetrating_wick': self._state.penetrating_wick,
                        'fvg_high': active_fvg.high if active_fvg else 0,
                        'fvg_low': active_fvg.low if active_fvg else 0,
                        'fvg_midpoint': active_fvg.midpoint if active_fvg else 0,
                        'fvg_direction': active_fvg.direction if active_fvg else "",
                        'asian_high': session_levels.asian_high,
                        'asian_low': session_levels.asian_low,
                        'london_high': session_levels.london_high,
                        'london_low': session_levels.london_low,
                    },
                    score=1.0
                )
        
        return ScanResult(scanner_id=self.scanner_id, triggered=False)

```

### src/policy/library/ict_ifvg.py

```python
"""
ICT Inverted Fair Value Gap (IFVG) Scanner

Strategy Logic:
1. Detect FVGs on 5m timeframe
2. When a new FVG forms opposite to a recent FVG (within 30m), it's an "Inverted FVG"
3. Score the IFVG by counting swept swing levels
4. If score >= 3, place limit order at 50% of new FVG
5. Stop at FVG invalidation, TP at 2R
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import timedelta

from src.features.fvg import find_fvg, FairValueGap
from src.features.swings import find_swings, count_levels_swept, SwingPoint
from src.config import POINT_VALUE, TICK_SIZE


@dataclass
class IFVGSetup:
    """Represents a valid IFVG trade setup."""
    direction: str              # "LONG" or "SHORT"
    entry_price: float          # Limit order at FVG midpoint
    stop_price: float           # FVG invalidation level
    tp_price: float             # 2R target
    new_fvg: FairValueGap       # The triggering FVG
    old_fvg: FairValueGap       # The inverted (opposite) FVG
    liquidity_score: int        # Number of swept swing levels
    bar_idx: int
    bar_time: pd.Timestamp


@dataclass 
class IFVGState:
    """Tracks state for IFVG detection."""
    recent_fvgs: List[FairValueGap] = field(default_factory=list)
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)
    last_trigger_bar: int = -1000
    max_fvg_history: int = 20


class ICTIFVGScanner:
    """
    Scanner for Inverted Fair Value Gap setups.
    
    An IFVG occurs when:
    1. A FVG forms in one direction
    2. Within 30 minutes (6 bars on 5m), a new FVG forms in the opposite direction
    3. The move creating the new FVG swept at least 3 swing levels
    
    Trade execution:
    - Entry: Limit order at 50% of new FVG
    - Stop: At FVG high (for SHORT) or low (for LONG)
    - TP: 2.0 R:R
    """
    
    def __init__(
        self,
        min_liquidity_score: int = 3,
        inversion_window_bars: int = 6,  # 30 min on 5m
        swing_lookback: int = 5,
        min_gap_atr: float = 0.2,
        risk_reward: float = 2.0,
        cooldown_bars: int = 6,
        max_risk_dollars: float = 300.0
    ):
        self.min_liquidity_score = min_liquidity_score
        self.inversion_window_bars = inversion_window_bars
        self.swing_lookback = swing_lookback
        self.min_gap_atr = min_gap_atr
        self.risk_reward = risk_reward
        self.cooldown_bars = cooldown_bars
        self.max_risk_dollars = max_risk_dollars
        self._state = IFVGState()
    
    @property
    def scanner_id(self) -> str:
        return "ict_ifvg"
    
    def reset(self):
        """Reset scanner state for new day."""
        self._state = IFVGState()
    
    def _update_swings(self, df_5m: pd.DataFrame):
        """Update swing points from 5m data."""
        highs, lows = find_swings(df_5m, lookback=self.swing_lookback)
        self._state.swing_highs = highs
        self._state.swing_lows = lows
    
    def _find_opposite_fvg(
        self, 
        new_fvg: FairValueGap,
        current_bar_idx: int
    ) -> Optional[FairValueGap]:
        """
        Find an opposite FVG within the inversion window.
        
        For a BEARISH new_fvg, we look for a recent BULLISH FVG.
        For a BULLISH new_fvg, we look for a recent BEARISH FVG.
        """
        opposite_dir = "BULLISH" if new_fvg.direction == "BEARISH" else "BEARISH"
        
        for old_fvg in reversed(self._state.recent_fvgs):
            if old_fvg.direction != opposite_dir:
                continue
            
            # Check if within window (bar index difference)
            bar_diff = current_bar_idx - old_fvg.bar_idx
            if 0 < bar_diff <= self.inversion_window_bars:
                return old_fvg
        
        return None
    
    def _calculate_liquidity_score(
        self,
        new_fvg: FairValueGap,
        df_5m: pd.DataFrame
    ) -> int:
        """
        Calculate liquidity score based on swept swing levels.
        
        For BEARISH FVG (SHORT setup): Count swing lows swept by the down move
        For BULLISH FVG (LONG setup): Count swing highs swept by the up move
        """
        if len(df_5m) < 3:
            return 0
        
        # Get the impulse bar and its preceding bar to define the move
        fvg_bar = df_5m[df_5m.index == new_fvg.bar_idx]
        if fvg_bar.empty:
            # Try to find it by time
            fvg_bar = df_5m[df_5m['time'] == new_fvg.bar_time]
        
        if fvg_bar.empty:
            return 0
        
        fvg_bar = fvg_bar.iloc[0]
        
        if new_fvg.direction == "BEARISH":
            # Down move - check swing lows that were swept
            move_low = fvg_bar['low']
            # Look back to find the recent swing high as move start
            recent_high = df_5m.tail(10)['high'].max()
            return count_levels_swept(self._state.swing_lows, recent_high, move_low)
        else:
            # Up move - check swing highs that were swept
            move_high = fvg_bar['high']
            recent_low = df_5m.tail(10)['low'].min()
            return count_levels_swept(self._state.swing_highs, recent_low, move_high)
    
    def check(
        self,
        df_5m: pd.DataFrame,
        current_bar_idx: int,
        atr: float = 5.0
    ) -> Optional[IFVGSetup]:
        """
        Check for IFVG setup at current bar.
        
        Args:
            df_5m: 5-minute OHLCV data up to current bar
            current_bar_idx: Current bar index
            atr: Current ATR for gap filtering
            
        Returns:
            IFVGSetup if valid setup found, None otherwise
        """
        # Cooldown check
        if current_bar_idx - self._state.last_trigger_bar < self.cooldown_bars:
            return None
        
        # Update swings
        self._update_swings(df_5m)
        
        # Find new FVGs at current bar
        new_fvgs = find_fvg(df_5m, lookback=3, min_gap_atr=self.min_gap_atr, atr=atr)
        
        if not new_fvgs:
            return None
        
        # Check the most recent FVG
        new_fvg = new_fvgs[0]
        
        # Already in history? Skip
        for existing in self._state.recent_fvgs:
            if existing.bar_idx == new_fvg.bar_idx and existing.direction == new_fvg.direction:
                return None
        
        # Add to history
        self._state.recent_fvgs.append(new_fvg)
        if len(self._state.recent_fvgs) > self._state.max_fvg_history:
            self._state.recent_fvgs.pop(0)
        
        # Check for inversion (opposite FVG nearby)
        old_fvg = self._find_opposite_fvg(new_fvg, current_bar_idx)
        if not old_fvg:
            return None
        
        # Calculate liquidity score
        score = self._calculate_liquidity_score(new_fvg, df_5m)
        if score < self.min_liquidity_score:
            return None
        
        # Valid setup! Calculate levels
        entry = new_fvg.midpoint
        
        if new_fvg.direction == "BEARISH":
            # SHORT setup
            direction = "SHORT"
            stop = new_fvg.high  # Invalidation at top of bearish FVG
            risk = stop - entry
            tp = entry - (risk * self.risk_reward)
        else:
            # LONG setup
            direction = "LONG"
            stop = new_fvg.low  # Invalidation at bottom of bullish FVG
            risk = entry - stop
            tp = entry + (risk * self.risk_reward)
        
        self._state.last_trigger_bar = current_bar_idx
        
        return IFVGSetup(
            direction=direction,
            entry_price=entry,
            stop_price=stop,
            tp_price=tp,
            new_fvg=new_fvg,
            old_fvg=old_fvg,
            liquidity_score=score,
            bar_idx=current_bar_idx,
            bar_time=new_fvg.bar_time
        )
    
    def get_context(self, setup: IFVGSetup) -> Dict[str, Any]:
        """Get scanner context for record output."""
        return {
            "scanner_id": self.scanner_id,
            "direction": setup.direction,
            "entry_price": setup.entry_price,
            "stop_price": setup.stop_price,
            "tp_price": setup.tp_price,
            "liquidity_score": setup.liquidity_score,
            "new_fvg": {
                "direction": setup.new_fvg.direction,
                "high": setup.new_fvg.high,
                "low": setup.new_fvg.low,
                "midpoint": setup.new_fvg.midpoint,
                "gap_size": setup.new_fvg.gap_size,
                "bar_time": setup.new_fvg.bar_time.isoformat() if setup.new_fvg.bar_time else None
            },
            "old_fvg": {
                "direction": setup.old_fvg.direction,
                "high": setup.old_fvg.high,
                "low": setup.old_fvg.low,
                "bar_time": setup.old_fvg.bar_time.isoformat() if setup.old_fvg.bar_time else None
            },
            "risk_reward": self.risk_reward,
            "min_liquidity": self.min_liquidity_score
        }

```

### src/policy/library/mean_reversion.py

```python
"""
Mean Reversion Scanner
Triggers when price extends beyond Keltner Channels (EMA +/- ATR bands).
"""

from typing import Dict, Any
from src.policy.scanners import Scanner, ScanResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle

class MeanReversionScanner(Scanner):
    """
    Triggers when price is outside EMA +/- N * ATR bands.
    Suggests reversion to the mean (EMA).
    """
    
    def __init__(
        self,
        ema_period: int = 20,
        atr_multiple: float = 3.0,
        rsi_min: float = 30.0,
        rsi_max: float = 70.0,
        timeframe: str = '5m'
    ):
        self.ema_period = ema_period
        self.atr_multiple = atr_multiple
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.timeframe = timeframe
        
        # State
        self._was_triggered = False
    
    @property
    def scanner_id(self) -> str:
        return f"mean_reversion_{self.ema_period}_{self.atr_multiple}_{self.timeframe}_{int(self.rsi_min)}_{int(self.rsi_max)}"
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScanResult:
        if features.indicators is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        ind = features.indicators
        
        if self.timeframe == '5m':
            ema = ind.ema_5m_20
            atr = ind.atr_5m_14
            rsi = ind.rsi_5m_14
        elif self.timeframe == '15m':
            ema = ind.ema_15m_20
            atr = ind.atr_15m_14
            rsi = ind.rsi_15m_14
        else:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
            
        if ema == 0 or atr == 0:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
            
        current_price = features.current_price
        
        upper_band = ema + (atr * self.atr_multiple)
        lower_band = ema - (atr * self.atr_multiple)
        
        # Conditions
        short_signal = (current_price > upper_band) and (rsi > self.rsi_max)
        long_signal = (current_price < lower_band) and (rsi < self.rsi_min)
        
        is_signal_active = short_signal or long_signal
        
        triggered = False
        signal = "neutral"
        distance = 0.0
        
        if is_signal_active:
            # Check debounce state
            if not self._was_triggered:
                triggered = True
                self._was_triggered = True
                
                if short_signal:
                    signal = "short"
                    distance = current_price - upper_band
                else:
                    signal = "long"
                    distance = lower_band - current_price
            else:
                # Already triggered, waiting for reset
                triggered = False
        else:
            # Reset state when condition is lost
            self._was_triggered = False
            
        return ScanResult(
            scanner_id=self.scanner_id,
            triggered=triggered,
            context={
                'signal': signal,
                'ema': ema,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'distance': distance,
                'atr': atr,
                'rsi': rsi
            },
            score=1.0 if triggered else 0.0
        )

```

### src/policy/library/mid_day_reversal.py

```python
"""
Mid-Day Reversal Strategy
Modular scanner that looks for reversals during lunch/mid-day.
"""

from src.policy.scanners import Scanner, ScanResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle

class MidDayReversalScanner(Scanner):
    """
    Scanner that triggers for mid-day reversal setups.
    
    Logic:
    1. Must be in RTH (Regular Trading Hours).
    2. Must be Mid-day (11:00 AM - 1:30 PM NY).
    3. Price must show an extreme or RSI must be at an extreme.
    """
    
    def __init__(
        self, 
        start_hour: int = 11, 
        end_hour: int = 13, 
        rsi_extreme: float = 30.0
    ):
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.rsi_extreme = rsi_extreme

    @property
    def scanner_id(self) -> str:
        return f"midday_reversal_{self.start_hour}_{self.end_hour}"

    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScanResult:
        t = features.time_features
        if not t or not t.is_rth:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # 1. Check time window
        is_midday = self.start_hour <= t.hour_ny <= self.end_hour
        if not is_midday:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # 2. Check for reversal signal (Simple RSI extreme for now)
        if features.indicators is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        rsi = features.indicators.rsi_5m_14
        oversold = rsi <= self.rsi_extreme
        overbought = rsi >= (100 - self.rsi_extreme)
        
        triggered = oversold or overbought
        
        return ScanResult(
            scanner_id=self.scanner_id,
            triggered=triggered,
            context={
                'hour': t.hour_ny,
                'rsi': rsi,
                'condition': 'oversold' if oversold else 'overbought' if overbought else 'neutral'
            },
            score=1.0 if triggered else 0.0
        )

```

### src/policy/library/momentum_divergence.py

```python
"""
Momentum Divergence Scanner
Triggers on RSI-price divergence patterns.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

from src.policy.scanners import Scanner, ScanResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle
from src.config import NY_TZ


@dataclass
class DivergenceState:
    """Tracks price and RSI history for divergence detection."""
    price_history: list = field(default_factory=list)
    rsi_history: list = field(default_factory=list)
    last_trigger_bar: int = -1000


class MomentumDivergenceScanner(Scanner):
    """
    Scanner that triggers on RSI-price divergence.
    
    Logic:
    1. Track last N bars of price lows/highs and RSI.
    2. Bullish Divergence: Price makes lower low, RSI makes higher low -> LONG.
    3. Bearish Divergence: Price makes higher high, RSI makes lower high -> SHORT.
    
    Config:
        lookback: Bars for swing detection (default 20).
        rsi_threshold: Min RSI difference for divergence (default 5.0).
        cooldown_bars: Min bars between triggers (default 20).
    """
    
    def __init__(
        self,
        lookback: int = 20,
        rsi_threshold: float = 5.0,
        cooldown_bars: int = 20,
    ):
        self.lookback = lookback
        self.rsi_threshold = rsi_threshold
        self.cooldown_bars = cooldown_bars
        self._state = DivergenceState()
    
    @property
    def scanner_id(self) -> str:
        return f"momentum_divergence_{self.lookback}"
    
    def _find_swing_lows(self, prices: List[float], window: int = 5) -> List[Tuple[int, float]]:
        """Find swing lows in price series."""
        swings = []
        for i in range(window, len(prices) - window):
            is_low = all(prices[i] <= prices[i-j] for j in range(1, window+1))
            is_low = is_low and all(prices[i] <= prices[i+j] for j in range(1, window+1))
            if is_low:
                swings.append((i, prices[i]))
        return swings
    
    def _find_swing_highs(self, prices: List[float], window: int = 5) -> List[Tuple[int, float]]:
        """Find swing highs in price series."""
        swings = []
        for i in range(window, len(prices) - window):
            is_high = all(prices[i] >= prices[i-j] for j in range(1, window+1))
            is_high = is_high and all(prices[i] >= prices[i+j] for j in range(1, window+1))
            if is_high:
                swings.append((i, prices[i]))
        return swings
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScanResult:
        t = features.timestamp
        if t is None:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # RTH check
        ny_time = t.astimezone(NY_TZ)
        is_rth = (ny_time.hour == 9 and ny_time.minute >= 30) or (10 <= ny_time.hour < 16)
        if not is_rth:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
        # Cooldown
        if features.bar_idx - self._state.last_trigger_bar < self.cooldown_bars:
            return ScanResult(scanner_id=self.scanner_id, triggered=False)
        
