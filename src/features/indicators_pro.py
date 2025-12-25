"""
Professional Trading Indicators Library

Comprehensive set of trading primitives for the MLang2 platform:
- Bar Measurement: Heikin-Ashi, range metrics
- Time Series: MACD, Stochastic, ADX, Ichimoku
- Volume: OBV, VWMACD, Chaikin Money Flow
- Levels: Pivot Points, Fibonacci
- Breakouts: Donchian channels, patterns
- Filters: Time-of-day, risk sizing

All functions are registry-compatible and return pandas Series or DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass


# =============================================================================
# 1. BAR MEASUREMENT PRIMITIVES
# =============================================================================

def calculate_heikin_ashi(df: pd.DataFrame, smoothing: float = 1.0) -> pd.DataFrame:
    """
    Calculate Heikin-Ashi candles for trend clarity.
    
    Args:
        df: DataFrame with OHLC columns
        smoothing: Smoothing factor (1.0 = standard, >1.0 = more smoothing)
    
    Returns:
        DataFrame with ha_open, ha_high, ha_low, ha_close
    """
    ha = pd.DataFrame(index=df.index)
    
    # HA Close = (O + H + L + C) / 4
    ha['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    
    # HA Open = (prev HA Open + prev HA Close) / 2
    ha['ha_open'] = 0.0
    ha.iloc[0, ha.columns.get_loc('ha_open')] = (df.iloc[0]['open'] + df.iloc[0]['close']) / 2.0
    
    for i in range(1, len(df)):
        ha.iloc[i, ha.columns.get_loc('ha_open')] = (
            (ha.iloc[i-1]['ha_open'] + ha.iloc[i-1]['ha_close']) / 2.0
        )
    
    # Apply smoothing if needed
    if smoothing != 1.0:
        ha['ha_open'] = ha['ha_open'].ewm(alpha=1.0/smoothing).mean()
        ha['ha_close'] = ha['ha_close'].ewm(alpha=1.0/smoothing).mean()
    
    # HA High = max(H, HA Open, HA Close)
    ha['ha_high'] = df[['high']].join(ha[['ha_open', 'ha_close']]).max(axis=1)
    
    # HA Low = min(L, HA Open, HA Close)
    ha['ha_low'] = df[['low']].join(ha[['ha_open', 'ha_close']]).min(axis=1)
    
    return ha


def calculate_bar_expansion(df: pd.DataFrame, atr_period: int = 14, threshold: float = 1.5) -> pd.Series:
    """
    Detect bars with expansion above threshold × ATR.
    
    Args:
        df: DataFrame with high, low columns
        atr_period: Period for ATR calculation
        threshold: Multiplier for expansion detection (e.g., 1.5 = 150% of ATR)
    
    Returns:
        Boolean Series indicating expansion bars
    """
    from src.features.indicators import calculate_atr
    
    atr = calculate_atr(df, period=atr_period)
    bar_range = df['high'] - df['low']
    
    return bar_range > (threshold * atr)


def calculate_average_bar_size(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate average bar size over N periods.
    
    Args:
        df: DataFrame with high, low columns
        period: Lookback period
    
    Returns:
        Series of average bar sizes
    """
    bar_range = df['high'] - df['low']
    return bar_range.rolling(window=period).mean()


# =============================================================================
# 2. TIME SERIES PRIMITIVES
# =============================================================================

def calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        close: Close price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period
    
    Returns:
        (macd_line, signal_line, histogram)
    """
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smoothing: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        df: DataFrame with high, low, close
        k_period: %K period (lookback for high/low)
        d_period: %D period (smoothing of %K)
        smoothing: Additional smoothing for %K
    
    Returns:
        (%K line, %D line)
    """
    # Calculate %K
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    
    # Smooth %K
    k_smooth = k.rolling(window=smoothing).mean()
    
    # Calculate %D (moving average of %K)
    d = k_smooth.rolling(window=d_period).mean()
    
    return k_smooth, d


def calculate_adx(
    df: pd.DataFrame,
    period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate ADX (Average Directional Index) for trend strength.
    
    Args:
        df: DataFrame with high, low, close
        period: Period for ADX calculation
    
    Returns:
        (adx, plus_di, minus_di)
    """
    # True Range
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = df['high'] - df['high'].shift()
    down_move = df['low'].shift() - df['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's method
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx, plus_di, minus_di


def calculate_ichimoku(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26
) -> Dict[str, pd.Series]:
    """
    Calculate Ichimoku Cloud components.
    
    Args:
        df: DataFrame with high, low, close
        tenkan_period: Conversion line period
        kijun_period: Base line period
        senkou_b_period: Leading span B period
        displacement: Cloud displacement forward
    
    Returns:
        Dictionary with tenkan, kijun, senkou_a, senkou_b, chikou
    """
    # Tenkan-sen (Conversion Line)
    high_tenkan = df['high'].rolling(window=tenkan_period).max()
    low_tenkan = df['low'].rolling(window=tenkan_period).min()
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line)
    high_kijun = df['high'].rolling(window=kijun_period).max()
    low_kijun = df['low'].rolling(window=kijun_period).min()
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_a = ((tenkan + kijun) / 2).shift(displacement)
    
    # Senkou Span B (Leading Span B)
    high_senkou = df['high'].rolling(window=senkou_b_period).max()
    low_senkou = df['low'].rolling(window=senkou_b_period).min()
    senkou_b = ((high_senkou + low_senkou) / 2).shift(displacement)
    
    # Chikou Span (Lagging Span)
    chikou = df['close'].shift(-displacement)
    
    return {
        'tenkan': tenkan,
        'kijun': kijun,
        'senkou_a': senkou_a,
        'senkou_b': senkou_b,
        'chikou': chikou
    }


# =============================================================================
# 3. VOLUME PRIMITIVES
# =============================================================================

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    Args:
        df: DataFrame with close, volume
    
    Returns:
        OBV series
    """
    obv = pd.Series(0, index=df.index)
    obv.iloc[0] = df.iloc[0]['volume']
    
    for i in range(1, len(df)):
        if df.iloc[i]['close'] > df.iloc[i-1]['close']:
            obv.iloc[i] = obv.iloc[i-1] + df.iloc[i]['volume']
        elif df.iloc[i]['close'] < df.iloc[i-1]['close']:
            obv.iloc[i] = obv.iloc[i-1] - df.iloc[i]['volume']
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def calculate_relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate relative volume (current vs average).
    
    Args:
        df: DataFrame with volume
        period: Lookback period for average
    
    Returns:
        Relative volume ratio
    """
    avg_volume = df['volume'].rolling(window=period).mean()
    return df['volume'] / avg_volume


def calculate_chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Chaikin Money Flow.
    
    Args:
        df: DataFrame with high, low, close, volume
        period: Period for CMF calculation
    
    Returns:
        CMF series
    """
    # Money Flow Multiplier
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mf_multiplier = mf_multiplier.fillna(0)  # Handle division by zero
    
    # Money Flow Volume
    mf_volume = mf_multiplier * df['volume']
    
    # CMF
    cmf = mf_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    
    return cmf


def calculate_vwmacd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Volume-Weighted MACD.
    
    Args:
        df: DataFrame with close, volume
        fast_period: Fast period
        slow_period: Slow period
        signal_period: Signal period
    
    Returns:
        (vwmacd_line, signal_line, histogram)
    """
    # Volume-weighted price
    vwp = (df['close'] * df['volume']).ewm(span=1).mean() / df['volume'].ewm(span=1).mean()
    
    # MACD on volume-weighted price
    fast_vwma = vwp.ewm(span=fast_period, adjust=False).mean()
    slow_vwma = vwp.ewm(span=slow_period, adjust=False).mean()
    
    vwmacd_line = fast_vwma - slow_vwma
    signal_line = vwmacd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = vwmacd_line - signal_line
    
    return vwmacd_line, signal_line, histogram


# =============================================================================
# 4. LEVELS PRIMITIVES
# =============================================================================

@dataclass
class PivotLevels:
    """Pivot point levels."""
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float


def calculate_pivot_points(
    high: float,
    low: float,
    close: float,
    method: str = 'standard'
) -> PivotLevels:
    """
    Calculate pivot points.
    
    Args:
        high: Previous period high
        low: Previous period low
        close: Previous period close
        method: 'standard', 'woodie', or 'camarilla'
    
    Returns:
        PivotLevels dataclass
    """
    if method == 'standard':
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
    
    elif method == 'woodie':
        pivot = (high + low + 2 * close) / 4
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
    
    elif method == 'camarilla':
        pivot = (high + low + close) / 3
        range_hl = high - low
        r1 = close + range_hl * 1.1 / 12
        r2 = close + range_hl * 1.1 / 6
        r3 = close + range_hl * 1.1 / 4
        s1 = close - range_hl * 1.1 / 12
        s2 = close - range_hl * 1.1 / 6
        s3 = close - range_hl * 1.1 / 4
    
    else:
        raise ValueError(f"Unknown pivot method: {method}")
    
    return PivotLevels(pivot, r1, r2, r3, s1, s2, s3)


def calculate_fibonacci_levels(
    swing_high: float,
    swing_low: float,
    direction: str = 'retracement'
) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement or extension levels.
    
    Args:
        swing_high: Swing high price
        swing_low: Swing low price
        direction: 'retracement' or 'extension'
    
    Returns:
        Dictionary of Fibonacci levels
    """
    diff = swing_high - swing_low
    
    if direction == 'retracement':
        return {
            '0.0': swing_high,
            '23.6': swing_high - 0.236 * diff,
            '38.2': swing_high - 0.382 * diff,
            '50.0': swing_high - 0.500 * diff,
            '61.8': swing_high - 0.618 * diff,
            '78.6': swing_high - 0.786 * diff,
            '100.0': swing_low
        }
    else:  # extension
        return {
            '0.0': swing_low,
            '61.8': swing_low + 0.618 * diff,
            '100.0': swing_low + diff,
            '161.8': swing_low + 1.618 * diff,
            '261.8': swing_low + 2.618 * diff,
            '423.6': swing_low + 4.236 * diff
        }


def calculate_round_levels(price: float, increment: float = 50.0) -> List[float]:
    """
    Calculate nearby round/psychological levels.
    
    Args:
        price: Current price
        increment: Round number increment (50, 100, etc.)
    
    Returns:
        List of nearby round levels
    """
    base = round(price / increment) * increment
    return [
        base - 2 * increment,
        base - increment,
        base,
        base + increment,
        base + 2 * increment
    ]


# =============================================================================
# 5. BREAKOUTS AND CONTINUATIONS
# =============================================================================

def calculate_donchian_channels(
    df: pd.DataFrame,
    period: int = 20
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Donchian Channels.
    
    Args:
        df: DataFrame with high, low
        period: Lookback period
    
    Returns:
        (upper_band, lower_band, middle_band)
    """
    upper = df['high'].rolling(window=period).max()
    lower = df['low'].rolling(window=period).min()
    middle = (upper + lower) / 2
    
    return upper, lower, middle


def detect_channel_breakout(
    df: pd.DataFrame,
    period: int = 20,
    confirmation_bars: int = 1
) -> pd.Series:
    """
    Detect breakouts above/below Donchian channels.
    
    Args:
        df: DataFrame with high, low, close
        period: Channel period
        confirmation_bars: Bars to confirm breakout
    
    Returns:
        Series with 1 (bullish breakout), -1 (bearish), 0 (no breakout)
    """
    upper, lower, _ = calculate_donchian_channels(df, period)
    
    breakouts = pd.Series(0, index=df.index)
    
    # Bullish breakout: close above upper band for N bars
    bullish = df['close'] > upper
    breakouts[bullish.rolling(window=confirmation_bars).sum() >= confirmation_bars] = 1
    
    # Bearish breakout: close below lower band for N bars
    bearish = df['close'] < lower
    breakouts[bearish.rolling(window=confirmation_bars).sum() >= confirmation_bars] = -1
    
    return breakouts


def detect_momentum_burst(
    df: pd.DataFrame,
    rsi_threshold: float = 70,
    volume_threshold: float = 2.0,
    volume_period: int = 20
) -> pd.Series:
    """
    Detect momentum bursts (RSI spike with volume).
    
    Args:
        df: DataFrame with close, volume
        rsi_threshold: RSI level to trigger
        volume_threshold: Volume multiplier vs average
        volume_period: Period for average volume
    
    Returns:
        Boolean series of momentum bursts
    """
    from src.features.indicators import calculate_rsi
    
    rsi = calculate_rsi(df['close'], period=14)
    rel_vol = calculate_relative_volume(df, period=volume_period)
    
    return (rsi > rsi_threshold) & (rel_vol > volume_threshold)


# =============================================================================
# 6. FILTERS AND RISK PRIMITIVES
# =============================================================================

def filter_time_of_day(
    timestamp: pd.Timestamp,
    allowed_hours: List[Tuple[int, int]],
    timezone: str = 'America/New_York'
) -> bool:
    """
    Check if timestamp is within allowed trading hours.
    
    Args:
        timestamp: Timestamp to check
        allowed_hours: List of (start_hour, end_hour) tuples in 24h format
        timezone: Timezone for hour check
    
    Returns:
        True if timestamp is in allowed hours
    """
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize('UTC')
    
    local_time = timestamp.tz_convert(timezone)
    hour = local_time.hour
    
    for start_hour, end_hour in allowed_hours:
        if start_hour <= hour < end_hour:
            return True
    
    return False


def calculate_kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate Kelly Criterion for position sizing.
    
    Args:
        win_rate: Probability of winning (0-1)
        avg_win: Average win amount
        avg_loss: Average loss amount (positive)
    
    Returns:
        Kelly percentage (fraction of capital to risk)
    """
    if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    # Kelly = (bp - q) / b
    # where b = avg_win/avg_loss, p = win_rate, q = 1-p
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Cap at 25% for safety
    return max(0, min(kelly, 0.25))


def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    entry_price: float,
    stop_price: float,
    contract_value: float = 5.0
) -> int:
    """
    Calculate position size in contracts based on risk.
    
    Args:
        account_balance: Total account balance
        risk_percent: Percentage of account to risk (0-100)
        entry_price: Entry price
        stop_price: Stop loss price
        contract_value: Dollar value per point
    
    Returns:
        Number of contracts to trade
    """
    risk_dollars = account_balance * (risk_percent / 100.0)
    risk_per_contract = abs(entry_price - stop_price) * contract_value
    
    if risk_per_contract == 0:
        return 0
    
    contracts = int(risk_dollars / risk_per_contract)
    
    return max(1, contracts)  # At least 1 contract


def check_risk_reward_ratio(
    entry_price: float,
    stop_price: float,
    target_price: float,
    min_rr: float = 2.0
) -> bool:
    """
    Check if trade meets minimum risk/reward ratio.
    
    Args:
        entry_price: Entry price
        stop_price: Stop loss price
        target_price: Take profit price
        min_rr: Minimum risk/reward ratio
    
    Returns:
        True if RR meets minimum
    """
    risk = abs(entry_price - stop_price)
    reward = abs(target_price - entry_price)
    
    if risk == 0:
        return False
    
    rr_ratio = reward / risk
    
    return rr_ratio >= min_rr
