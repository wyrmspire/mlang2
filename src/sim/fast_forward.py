"""
Fast Forward Engine for Fast Viz Mode

This module provides vectorized trade detection for instant visualization
without full simulation overhead. Used by the Fast Viz feature.

Key differences from full simulation:
- No bar-by-bar stepping (vectorized pandas operations)
- No OCO bracket lifecycle (simple TP/SL comparison)
- No training data export
- Trades are ephemeral until "Saved"
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from src.config import NY_TZ
from src.data.loader import load_continuous_contract
from src.policy.composite_scanner import CompositeScanner
from src.policy.triggers.factory import trigger_from_dict


@dataclass
class FastVizTrade:
    """A lightweight trade representation for Fast Viz."""
    entry_time: str
    entry_price: float
    direction: str  # "LONG" or "SHORT"
    stop_price: float
    target_price: float
    outcome: str  # "WIN", "LOSS", "PENDING"
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    pnl_points: float = 0.0
    trigger_name: str = ""
    confidence: float = 0.0


@dataclass
class FastVizResult:
    """Result of a Fast Viz scan."""
    run_id: str
    strategy_name: str
    start_date: str
    end_date: str
    trades: List[FastVizTrade]
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)


def detect_entries_vectorized(df: pd.DataFrame, trigger_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Vectorized entry detection based on trigger type.
    
    Returns DataFrame with 'signal', 'direction', 'confidence' columns.
    """
    trigger_type = trigger_config.get("type", "ema_cross")
    
    signals = pd.DataFrame(index=df.index)
    signals["signal"] = False
    signals["direction"] = "LONG"
    signals["confidence"] = 0.5
    
    if trigger_type == "ema_cross":
        fast = trigger_config.get("fast", 9)
        slow = trigger_config.get("slow", 21)
        
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        
        # Long signal: fast crosses above slow
        signals["signal"] = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        signals["direction"] = "LONG"
        
        # Short signal: fast crosses below slow
        short_signals = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
        signals.loc[short_signals, "signal"] = True
        signals.loc[short_signals, "direction"] = "SHORT"
        
    elif trigger_type == "rsi_threshold":
        period = trigger_config.get("period", 14)
        long_threshold = trigger_config.get("long_threshold", 30)
        short_threshold = trigger_config.get("short_threshold", 70)
        
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta).where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        # Long: RSI crosses above from below threshold
        signals["signal"] = (rsi > long_threshold) & (rsi.shift(1) <= long_threshold)
        signals["direction"] = "LONG"
        
        # Short: RSI crosses below from above threshold
        short_signals = (rsi < short_threshold) & (rsi.shift(1) >= short_threshold)
        signals.loc[short_signals, "signal"] = True
        signals.loc[short_signals, "direction"] = "SHORT"
        
    elif trigger_type == "time":
        hour = trigger_config.get("hour", 9)
        minute = trigger_config.get("minute", 30)
        direction = trigger_config.get("direction", "LONG")
        
        if "time" in df.columns:
            times = pd.to_datetime(df["time"])
            signals["signal"] = (times.dt.hour == hour) & (times.dt.minute == minute)
            signals["direction"] = direction
            
    return signals


def estimate_outcome(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    direction: str,
    stop_points: float,
    target_points: float,
    max_bars: int = 200
) -> Dict[str, Any]:
    """
    Estimate trade outcome by scanning future bars.
    
    Simple logic: check if TP hit before SL.
    """
    if direction == "LONG":
        tp_price = entry_price + target_points
        sl_price = entry_price - stop_points
    else:
        tp_price = entry_price - target_points
        sl_price = entry_price + stop_points
    
    # Scan forward bars
    future_bars = df.iloc[entry_idx + 1 : entry_idx + 1 + max_bars]
    
    for idx, bar in future_bars.iterrows():
        bar_high = bar["high"]
        bar_low = bar["low"]
        bar_time = bar.get("time", str(idx))
        
        # Check TP and SL
        if direction == "LONG":
            tp_hit = bar_high >= tp_price
            sl_hit = bar_low <= sl_price
        else:
            tp_hit = bar_low <= tp_price
            sl_hit = bar_high >= sl_price
        
        # Determine outcome (TP first wins)
        if tp_hit and sl_hit:
            # Both hit - assume TP first (optimistic)
            return {
                "outcome": "WIN",
                "exit_time": str(bar_time),
                "exit_price": tp_price,
                "pnl_points": target_points
            }
        elif tp_hit:
            return {
                "outcome": "WIN",
                "exit_time": str(bar_time),
                "exit_price": tp_price,
                "pnl_points": target_points
            }
        elif sl_hit:
            return {
                "outcome": "LOSS",
                "exit_time": str(bar_time),
                "exit_price": sl_price,
                "pnl_points": -stop_points
            }
    
    # Timeout - mark as pending/loss
    return {
        "outcome": "PENDING",
        "exit_time": None,
        "exit_price": None,
        "pnl_points": 0.0
    }


def fast_viz_strategy(
    config: Dict[str, Any],
    start_date: str,
    end_date: str,
    run_id: Optional[str] = None
) -> FastVizResult:
    """
    Execute a Fast Viz scan over the given date range.
    
    Args:
        config: Strategy configuration with trigger and bracket settings
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        run_id: Optional custom run ID
        
    Returns:
        FastVizResult with trades list
    """
    # Load data
    df = load_continuous_contract(start_date=start_date, end_date=end_date)
    
    if df is None or len(df) == 0:
        return FastVizResult(
            run_id=run_id or f"fast_viz_{datetime.now().strftime('%H%M%S')}",
            strategy_name="unknown",
            start_date=start_date,
            end_date=end_date,
            trades=[],
            config=config
        )
    
    # Extract trigger and bracket config
    trigger_config = config.get("trigger", {"type": "ema_cross"})
    bracket_config = config.get("bracket", {"type": "atr", "stop_atr": 2.0, "tp_atr": 3.0})
    cooldown = config.get("cooldown_bars", 20)
    
    # Calculate ATR for stops/targets if ATR bracket
    if bracket_config.get("type") == "atr":
        atr_period = bracket_config.get("atr_period", 14)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=atr_period).mean()
    else:
        df["atr"] = 5.0  # Default fixed
    
    # Detect entries
    signals = detect_entries_vectorized(df, trigger_config)
    
    # Reset index for integer indexing
    df = df.reset_index(drop=True)
    signals = signals.reset_index(drop=True)
    
    # Collect trades
    trades: List[FastVizTrade] = []
    last_entry_idx = -cooldown
    
    trigger_name = trigger_config.get("type", "unknown")
    stop_mult = bracket_config.get("stop_atr", 2.0)
    tp_mult = bracket_config.get("tp_atr", 3.0)
    
    for idx in range(len(df)):
        if not signals.loc[idx, "signal"]:
            continue
        
        # Cooldown check
        if idx - last_entry_idx < cooldown:
            continue
        
        last_entry_idx = idx
        
        row = df.iloc[idx]
        entry_price = row["close"]
        direction = signals.loc[idx, "direction"]
        confidence = signals.loc[idx, "confidence"]
        atr = row.get("atr", 5.0)
        
        stop_points = atr * stop_mult
        target_points = atr * tp_mult
        
        if direction == "LONG":
            stop_price = entry_price - stop_points
            target_price = entry_price + target_points
        else:
            stop_price = entry_price + stop_points
            target_price = entry_price - target_points
        
        # Estimate outcome
        outcome_info = estimate_outcome(
            df, idx, entry_price, direction, stop_points, target_points
        )
        
        trade = FastVizTrade(
            entry_time=str(row.get("time", f"bar_{idx}")),
            entry_price=entry_price,
            direction=direction,
            stop_price=stop_price,
            target_price=target_price,
            outcome=outcome_info["outcome"],
            exit_time=outcome_info["exit_time"],
            exit_price=outcome_info["exit_price"],
            pnl_points=outcome_info["pnl_points"],
            trigger_name=trigger_name,
            confidence=confidence
        )
        trades.append(trade)
    
    # Calculate stats
    wins = sum(1 for t in trades if t.outcome == "WIN")
    losses = sum(1 for t in trades if t.outcome == "LOSS")
    total = len(trades)
    win_rate = (wins / total * 100) if total > 0 else 0.0
    
    return FastVizResult(
        run_id=run_id or f"fast_viz_{trigger_name}_{datetime.now().strftime('%H%M%S')}",
        strategy_name=trigger_name,
        start_date=start_date,
        end_date=end_date,
        trades=trades,
        total_trades=total,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        config=config
    )
