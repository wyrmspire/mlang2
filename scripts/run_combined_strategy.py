#!/usr/bin/env python3
"""
Combined Strategy: ORB + Mean Reversion

Run multiple strategies in different time windows:
- 9:30 - 10:30 AM: Opening Range Breakout
- 2:00 - 4:00 PM: Mean Reversion

Single account balance, combined equity curve.

Usage:
    python scripts/run_combined_strategy.py --days 7
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, time
from typing import Dict, Any, List, Tuple
from zoneinfo import ZoneInfo
import json

from src.features.indicators import calculate_atr, calculate_ema, calculate_rsi
from src.storage import ExperimentDB


# =============================================================================
# Time Windows (EST)
# =============================================================================

EST = ZoneInfo("America/New_York")

ORB_START = time(9, 30)
ORB_END = time(10, 30)
ORB_TRADE_END = time(12, 0)  # Stop trading ORB breakouts by noon

MR_START = time(14, 0)
MR_END = time(16, 0)


# =============================================================================
# Strategy 1: Opening Range Breakout
# =============================================================================

def check_orb_signal(
    df: pd.DataFrame,
    idx: int,
    or_high: float,
    or_low: float,
    atr: float,
) -> Tuple[str, float, float]:
    """
    Check for ORB breakout signal.
    
    Returns (direction, stop, tp) or (None, None, None)
    """
    if or_high is None or or_low is None:
        return None, None, None
    
    bar = df.iloc[idx]
    close = bar['close']
    high = bar['high']
    low = bar['low']
    
    # Breakout above OR high
    if high > or_high:
        entry = close
        stop = entry - atr * 0.75
        tp = entry + atr * 1.5
        return 'LONG', stop, tp
    
    # Breakdown below OR low
    if low < or_low:
        entry = close
        stop = entry + atr * 0.75
        tp = entry - atr * 1.5
        return 'SHORT', stop, tp
    
    return None, None, None


# =============================================================================
# Strategy 2: Mean Reversion
# =============================================================================

def check_mr_signal(
    df: pd.DataFrame,
    idx: int,
    ema_20: float,
    atr: float,
    rsi: float,
) -> Tuple[str, float, float]:
    """
    Check for Mean Reversion signal.
    
    LONG: Price > 1.5 ATR below EMA, RSI < 30
    SHORT: Price > 1.5 ATR above EMA, RSI > 70
    
    Returns (direction, stop, tp) or (None, None, None)
    """
    bar = df.iloc[idx]
    close = bar['close']
    
    distance_from_ema = close - ema_20
    
    # Oversold: price below EMA, RSI low
    if distance_from_ema < -atr * 1.5 and rsi < 35:
        entry = close
        stop = entry - atr * 1.0
        tp = ema_20  # Revert to mean
        return 'LONG', stop, tp
    
    # Overbought: price above EMA, RSI high
    if distance_from_ema > atr * 1.5 and rsi > 65:
        entry = close
        stop = entry + atr * 1.0
        tp = ema_20  # Revert to mean
        return 'SHORT', stop, tp
    
    return None, None, None


# =============================================================================
# Combined Simulation
# =============================================================================

def run_combined_strategy(days: int = 7, starting_balance: float = 50000) -> Dict[str, Any]:
    """
    Run combined ORB + MR strategy simulation.
    """
    print("=" * 60)
    print("COMBINED STRATEGY: ORB + MEAN REVERSION")
    print("=" * 60)
    print(f"ORB Window: {ORB_START} - {ORB_END} (breakout until {ORB_TRADE_END})")
    print(f"MR Window:  {MR_START} - {MR_END}")
    print(f"Starting Balance: ${starting_balance:,.0f}")
    print("=" * 60)
    
    # Load data
    actual_days = min(days, 7)
    end = datetime.now()
    start = end - timedelta(days=actual_days)
    
    print(f"\n[1] Loading {actual_days} days of ES data...")
    ticker = yf.Ticker("ES=F")
    df = ticker.history(start=start, end=end, interval="1m")
    
    if df is None or len(df) == 0:
        print("ERROR: No data!")
        return {}
    
    df.columns = [c.lower() for c in df.columns]
    df = df.reset_index()
    df['time'] = pd.to_datetime(
        df['Datetime'] if 'Datetime' in df.columns else df['datetime']
    )
    # Fix timezone safety: localize to UTC if naive, then convert to EST
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('UTC')
    df['time'] = df['time'].dt.tz_convert(EST)
    df['date'] = df['time'].dt.date
    
    print(f"    Loaded {len(df)} bars")
    
    # Compute indicators
    print(f"\n[2] Computing indicators...")
    df['atr'] = calculate_atr(df, period=14).ffill().bfill()
    df['ema_20'] = calculate_ema(df['close'], 20)
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    # Run simulation
    print(f"\n[3] Running combined simulation...")
    
    balance = starting_balance
    equity_curve = [balance]
    trades = []
    active_trade = None
    
    # Daily OR tracking
    daily_or = {}
    
    unique_dates = df['date'].unique()
    
    for i in range(30, len(df)):
        bar = df.iloc[i]
        current_time = bar['time']
        current_date = current_time.date()
        current_time_only = current_time.time()
        
        close = bar['close']
        high = bar['high']
        low = bar['low']
        atr = bar['atr'] if not pd.isna(bar['atr']) else 2.0
        ema_20 = bar['ema_20']
        rsi = bar['rsi'] if not pd.isna(bar['rsi']) else 50
        
        # =====================================================================
        # Compute OR for this day
        # =====================================================================
        if current_date not in daily_or:
            # Find OR data for this day
            or_mask = (df['date'] == current_date) & \
                      (df['time'].dt.time >= ORB_START) & \
                      (df['time'].dt.time <= ORB_END)
            or_data = df[or_mask]
            
            if len(or_data) > 0:
                daily_or[current_date] = {
                    'high': or_data['high'].max(),
                    'low': or_data['low'].min(),
                }
        
        or_info = daily_or.get(current_date, {})
        or_high = or_info.get('high')
        or_low = or_info.get('low')
        
        # =====================================================================
        # Check active trade
        # =====================================================================
        if active_trade is not None:
            if active_trade['direction'] == 'LONG':
                if low <= active_trade['stop']:
                    pnl = (active_trade['stop'] - active_trade['entry']) * 50
                    balance += pnl
                    trades.append({
                        'time': str(current_time),
                        'strategy': active_trade['strategy'],
                        'direction': 'LONG',
                        'result': 'LOSS',
                        'pnl': pnl,
                    })
                    active_trade = None
                elif high >= active_trade['tp']:
                    pnl = (active_trade['tp'] - active_trade['entry']) * 50
                    balance += pnl
                    trades.append({
                        'time': str(current_time),
                        'strategy': active_trade['strategy'],
                        'direction': 'LONG',
                        'result': 'WIN',
                        'pnl': pnl,
                    })
                    active_trade = None
            else:  # SHORT
                if high >= active_trade['stop']:
                    pnl = (active_trade['entry'] - active_trade['stop']) * 50
                    balance += pnl
                    trades.append({
                        'time': str(current_time),
                        'strategy': active_trade['strategy'],
                        'direction': 'SHORT',
                        'result': 'LOSS',
                        'pnl': pnl,
                    })
                    active_trade = None
                elif low <= active_trade['tp']:
                    pnl = (active_trade['entry'] - active_trade['tp']) * 50
                    balance += pnl
                    trades.append({
                        'time': str(current_time),
                        'strategy': active_trade['strategy'],
                        'direction': 'SHORT',
                        'result': 'WIN',
                        'pnl': pnl,
                    })
                    active_trade = None
            
            equity_curve.append(balance)
            continue
        
        # =====================================================================
        # Check for new entries based on time window
        # =====================================================================
        
        # ORB Window (after OR forms, before noon)
        if ORB_END < current_time_only <= ORB_TRADE_END:
            direction, stop, tp = check_orb_signal(df, i, or_high, or_low, atr)
            if direction:
                active_trade = {
                    'entry': close,
                    'stop': stop,
                    'tp': tp,
                    'direction': direction,
                    'strategy': 'ORB',
                    'entry_time': current_time,
                }
        
        # Mean Reversion Window (afternoon)
        elif MR_START <= current_time_only <= MR_END:
            direction, stop, tp = check_mr_signal(df, i, ema_20, atr, rsi)
            if direction:
                active_trade = {
                    'entry': close,
                    'stop': stop,
                    'tp': tp,
                    'direction': direction,
                    'strategy': 'MR',
                    'entry_time': current_time,
                }
        
        equity_curve.append(balance)
    
    # =========================================================================
    # Results
    # =========================================================================
    orb_trades = [t for t in trades if t['strategy'] == 'ORB']
    mr_trades = [t for t in trades if t['strategy'] == 'MR']
    
    orb_wins = sum(1 for t in orb_trades if t['result'] == 'WIN')
    mr_wins = sum(1 for t in mr_trades if t['result'] == 'WIN')
    
    total_pnl = balance - starting_balance
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Starting Balance: ${starting_balance:,.0f}")
    print(f"  Ending Balance:   ${balance:,.0f}")
    print(f"  Total P&L:        ${total_pnl:,.2f}")
    
    print(f"\n  ORB Trades: {len(orb_trades)}")
    if orb_trades:
        print(f"    Win Rate: {orb_wins/len(orb_trades):.1%}")
        print(f"    P&L: ${sum(t['pnl'] for t in orb_trades):,.2f}")
    
    print(f"\n  Mean Reversion Trades: {len(mr_trades)}")
    if mr_trades:
        print(f"    Win Rate: {mr_wins/len(mr_trades):.1%}")
        print(f"    P&L: ${sum(t['pnl'] for t in mr_trades):,.2f}")
    
    # Save equity curve
    output_dir = Path("results/combined")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    eq_df = pd.DataFrame({'equity': equity_curve})
    eq_path = output_dir / "equity_curve.csv"
    eq_df.to_csv(eq_path, index=False)
    print(f"\n[4] Saved equity curve to {eq_path}")
    
    # Print mini curve
    print(f"\n  Equity Curve (sampled):")
    sample_points = np.linspace(0, len(equity_curve)-1, 10, dtype=int)
    for idx in sample_points:
        bar_pct = int((equity_curve[idx] - starting_balance) / starting_balance * 50) + 25
        bar = "█" * max(0, min(50, bar_pct))
        print(f"    {idx:5d}: ${equity_curve[idx]:,.0f} {bar}")
    
    # Store
    db = ExperimentDB()
    run_id = f"combined_orb_mr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.store_run(
        run_id=run_id,
        strategy="combined_orb_mr",
        config={
            'orb_window': f"{ORB_START}-{ORB_END}",
            'mr_window': f"{MR_START}-{MR_END}",
        },
        metrics={
            'total_trades': len(trades),
            'wins': orb_wins + mr_wins,
            'losses': len(trades) - (orb_wins + mr_wins),
            'win_rate': (orb_wins + mr_wins) / len(trades) if trades else 0,
            'total_pnl': total_pnl,
            'orb_trades': len(orb_trades),
            'mr_trades': len(mr_trades),
        }
    )
    print(f"    Stored: {run_id}")
    
    return {
        'total_pnl': total_pnl,
        'ending_balance': balance,
        'trades': len(trades),
        'orb_trades': len(orb_trades),
        'mr_trades': len(mr_trades),
        'equity_curve': equity_curve,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combined ORB + MR Strategy")
    parser.add_argument("--days", type=int, default=7, help="Days to simulate")
    parser.add_argument("--balance", type=float, default=50000, help="Starting balance")
    
    args = parser.parse_args()
    
    results = run_combined_strategy(args.days, args.balance)
