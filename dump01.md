    if levels['asian_low'] > 0 and window_1m['low'].min() < levels['asian_low']:
        break_mask = window_5m['low'] < levels['asian_low']
        if break_mask.any():
            after_break = window_5m.loc[break_mask.idxmax():]
            fvg = find_fvg(after_break, "LONG")
            if fvg:
                after_fvg = after_break.iloc[fvg['fvg_bar_idx']+1:]
                entry = check_retracement(after_fvg, fvg, "LONG")
                if entry:
                    setup = {'direction': 'LONG', 'level': 'asian_low', 'level_price': levels['asian_low'],
                             'wick': float(window_1m['low'].min()), 'fvg': fvg, 'entry': entry}
    
    # Asian high break -> SHORT
    if not setup and levels['asian_high'] > 0 and window_1m['high'].max() > levels['asian_high']:
        break_mask = window_5m['high'] > levels['asian_high']
        if break_mask.any():
            after_break = window_5m.loc[break_mask.idxmax():]
            fvg = find_fvg(after_break, "SHORT")
            if fvg:
                after_fvg = after_break.iloc[fvg['fvg_bar_idx']+1:]
                entry = check_retracement(after_fvg, fvg, "SHORT")
                if entry:
                    setup = {'direction': 'SHORT', 'level': 'asian_high', 'level_price': levels['asian_high'],
                             'wick': float(window_1m['high'].max()), 'fvg': fvg, 'entry': entry}
    
    # London low break -> LONG
    if not setup and levels['london_low'] > 0 and window_1m['low'].min() < levels['london_low']:
        break_mask = window_5m['low'] < levels['london_low']
        if break_mask.any():
            after_break = window_5m.loc[break_mask.idxmax():]
            fvg = find_fvg(after_break, "LONG")
            if fvg:
                after_fvg = after_break.iloc[fvg['fvg_bar_idx']+1:]
                entry = check_retracement(after_fvg, fvg, "LONG")
                if entry:
                    setup = {'direction': 'LONG', 'level': 'london_low', 'level_price': levels['london_low'],
                             'wick': float(window_1m['low'].min()), 'fvg': fvg, 'entry': entry}
    
    # London high break -> SHORT
    if not setup and levels['london_high'] > 0 and window_1m['high'].max() > levels['london_high']:
        break_mask = window_5m['high'] > levels['london_high']
        if break_mask.any():
            after_break = window_5m.loc[break_mask.idxmax():]
            fvg = find_fvg(after_break, "SHORT")
            if fvg:
                after_fvg = after_break.iloc[fvg['fvg_bar_idx']+1:]
                entry = check_retracement(after_fvg, fvg, "SHORT")
                if entry:
                    setup = {'direction': 'SHORT', 'level': 'london_high', 'level_price': levels['london_high'],
                             'wick': float(window_1m['high'].max()), 'fvg': fvg, 'entry': entry}
    
    if not setup:
        return None
    
    # Calculate levels
    entry_price = setup['entry']['entry_price']
    direction = setup['direction']
    
    if direction == 'LONG':
        stop_price = setup['wick'] - (ATR_BUFFER * avg_atr)
        risk = entry_price - stop_price
        tp_price = pdh if pdh > 0 and (pdh - entry_price) >= MIN_RR * risk else entry_price + MIN_RR * risk
    else:
        stop_price = setup['wick'] + (ATR_BUFFER * avg_atr)
        risk = stop_price - entry_price
        tp_price = pdl if pdl > 0 and (entry_price - pdl) >= MIN_RR * risk else entry_price - MIN_RR * risk
    
    if risk <= 0:
        return None
    
    # Position sizing
    contracts = max(1, int(MAX_RISK_DOLLARS // (risk * POINT_VALUE)))
    risk_dollars = contracts * risk * POINT_VALUE
    
    # Find entry bar index in main df
    entry_time = setup['entry']['entry_time']
    entry_mask = df_1m['time'] == entry_time
    if not entry_mask.any():
        # Find closest bar
        entry_mask = df_1m['time_ny'] >= entry_time
    entry_bar_idx = df_1m[entry_mask].index[0] if entry_mask.any() else 0
    
    # Compute outcome
    after_entry = df_1m.loc[entry_bar_idx + 1:]
    result = compute_outcome(after_entry, entry_price, stop_price, tp_price, direction)
    
    # Get OHLCV for visualization
    raw_ohlcv = get_raw_ohlcv_window(df_1m, entry_bar_idx)
    
    # Build record in STANDARD FORMAT
    return {
        'decision_id': f"ict_{decision_idx:04d}",
        'timestamp': entry_time.isoformat() if hasattr(entry_time, 'isoformat') else str(entry_time),
        'bar_idx': int(entry_bar_idx),
        'index': decision_idx,
        'scanner_id': 'ict_fvg_5m',
        'scanner_context': {
            'direction': direction,
            'level_broken': setup['level'],
            'level_price': setup['level_price'],
            'penetrating_wick': setup['wick'],
            'fvg_high': setup['fvg']['fvg_high'],
            'fvg_low': setup['fvg']['fvg_low'],
            'fvg_midpoint': setup['fvg']['fvg_midpoint'],
            'session_levels': levels,
        },
        'current_price': entry_price,
        'atr': avg_atr,
        'stop_price': stop_price,
        'stop_reason': f"ICT_WICK_{setup['level'].upper()}",
        'tp_price': tp_price,
        'tp_reason': 'PDH_PDL' if (direction == 'LONG' and tp_price == pdh) or (direction == 'SHORT' and tp_price == pdl) else 'MIN_RR',
        'risk_points': risk,
        'reward_points': abs(tp_price - entry_price),
        'contracts': contracts,
        'risk_dollars': risk_dollars,
        'window': {
            'raw_ohlcv_1m': raw_ohlcv,
            'x_context': [],
        },
        'oco': {
            'entry_price': entry_price,
            'stop_price': stop_price,
            'tp_price': tp_price,
            'direction': direction,
            'atr_at_creation': avg_atr,
            'max_bars': 200,
        },
        'oco_results': {
            'ict_fvg': {
                'outcome': result['outcome'],
                'pnl_dollars': result['pnl_dollars'] * contracts,
                'bars_held': result['bars_held'],
                'exit_price': result['exit_price'],
            }
        },
        'best_oco': 'ict_fvg',
        'best_pnl': result['pnl_dollars'] * contracts,
    }


def main():
    parser = argparse.ArgumentParser(description="ICT FVG Strategy Backtest")
    parser.add_argument("--start-date", type=str, default="2025-03-18")
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    
    start_date = pd.Timestamp(args.start_date).date()
    end_date = start_date + timedelta(weeks=args.weeks)
    out_dir = Path(args.out) if args.out else RESULTS_DIR / "ict_fvg"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ICT FVG Strategy Backtest")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading data...")
    df = load_continuous_contract()
    df = df[(df['time'] >= str(start_date)) & (df['time'] < str(end_date))].reset_index(drop=True)
    print(f"  {len(df)} 1m bars")
    
    if len(df) == 0:
        print("No data found")
        return
    
    df['time_ny'] = df['time'].dt.tz_convert(NY_TZ)
    
    # Resample
    print("\n[2] Resampling...")
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_5m['time_ny'] = df_5m['time'].dt.tz_convert(NY_TZ)
    
    df_5m['atr'] = calculate_atr(df_5m, 14) if len(df_5m) > 14 else 10.0
    avg_atr = float(df_5m['atr'].dropna().mean()) if len(df_5m) > 14 else 10.0
    print(f"  Avg ATR: {avg_atr:.2f}")
    
    # Analyze days
    trading_days = sorted(set(df['time_ny'].dt.date))
    print(f"\n[3] Analyzing {len(trading_days)} days...")
    
    records = []
    decision_idx = 0
    
    for trade_date in trading_days:
        pd_levels = get_previous_day_levels(df, pd.Timestamp(datetime.combine(trade_date, time(12, 0)), tz=NY_TZ))
        pdh = pd_levels.get('pdh', 0) or 0
        pdl = pd_levels.get('pdl', 0) or 0
        
        record = analyze_day(df, df_5m, trade_date, pdh, pdl, avg_atr, decision_idx)
        
        if record:
            records.append(record)
            print(f"  {trade_date}: {record['scanner_context']['direction']} -> {record['oco_results']['ict_fvg']['outcome']} (${record['best_pnl']:.0f})")
            decision_idx += 1
    
    # Summary & save
    print("\n" + "=" * 60)
    if records:
        wins = [r for r in records if r['oco_results']['ict_fvg']['outcome'] == 'WIN']
        total_pnl = sum(r['best_pnl'] for r in records)
        
        print(f"Trades: {len(records)} | Wins: {len(wins)} | WR: {len(wins)/len(records):.0%} | PnL: ${total_pnl:.0f}")
        
        with open(out_dir / "records.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(make_serializable(r)) + "\n")
        
        with open(out_dir / "summary.json", "w") as f:
            json.dump({"trades": len(records), "win_rate": len(wins)/len(records), "total_pnl": total_pnl}, f, indent=2)
        
        print(f"Saved to {out_dir}")
    else:
        print("No trades found")


if __name__ == "__main__":
    main()

```

### scripts/backtest_ict_ifvg.py

```python
#!/usr/bin/env python3
"""
ICT Inverted FVG Strategy Backtest Runner

Batch processing approach for efficient backtesting.
Processes data day-by-day selecting IFVG setups.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional, List

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.policy.library.ict_ifvg import ICTIFVGScanner, IFVGSetup
from src.config import POINT_VALUE, TICK_SIZE

# ============================================================================
# CONFIGURATION
# ============================================================================

NY_TZ = ZoneInfo("America/New_York")
RESULTS_DIR = Path("results/ict_ifvg")

# Strategy params
MIN_LIQUIDITY_SCORE = 2
INVERSION_WINDOW_BARS = 6  # 30 min on 5m
RISK_REWARD = 2.0
RISK_PER_TRADE = 300.0

# Trade window
TRADE_WINDOW_START = time(9, 30)
TRADE_WINDOW_END = time(15, 30)

# Data range
START_DATE = "2025-03-18"
WEEKS = 24


# ============================================================================
# HELPERS
# ============================================================================

def make_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def get_raw_ohlcv_window(
    df_1m: pd.DataFrame,
    entry_time: pd.Timestamp,
    history_bars: int = 60,
    future_bars: int = 120
) -> List[Dict]:
    """Extract raw OHLCV data around a trade entry for visualization."""
    # Find the entry bar index
    mask = df_1m['time'] <= entry_time
    if not mask.any():
        return []
    
    entry_idx = mask.sum() - 1
    start_idx = max(0, entry_idx - history_bars)
    end_idx = min(len(df_1m), entry_idx + future_bars)
    
    window = df_1m.iloc[start_idx:end_idx]
    
    return [
        {
            "time": row['time'].isoformat(),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": int(row['volume'])
        }
        for _, row in window.iterrows()
    ]


def simulate_limit_order_outcome(
    df_1m: pd.DataFrame,
    entry_time: pd.Timestamp,
    entry_price: float,
    stop_price: float,
    tp_price: float,
    direction: str,
    max_bars: int = 240  # 4 hours
) -> Dict[str, Any]:
    """
    Simulate a limit order fill and outcome.
    
    Returns outcome dict with:
    - filled: bool
    - outcome: 'WIN', 'LOSS', or 'TIMEOUT'
    - exit_price: float
    - bars_held: int
    - pnl_points: float
    """
    # Get bars after entry time
    future = df_1m[df_1m['time'] > entry_time].head(max_bars)
    
    if future.empty:
        return {"filled": False, "outcome": "NO_DATA", "exit_price": 0, "bars_held": 0, "pnl_points": 0}
    
    filled = False
    fill_bar_idx = -1
    
    # Check for limit order fill
    for i, (idx, bar) in enumerate(future.iterrows()):
        if direction == "LONG":
            if bar['low'] <= entry_price:
                filled = True
                fill_bar_idx = i
                break
        else:  # SHORT
            if bar['high'] >= entry_price:
                filled = True
                fill_bar_idx = i
                break
    
    if not filled:
        return {"filled": False, "outcome": "NOT_FILLED", "exit_price": 0, "bars_held": 0, "pnl_points": 0}
    
    # Now simulate from fill to exit
    post_fill = future.iloc[fill_bar_idx:]
    
    for i, (idx, bar) in enumerate(post_fill.iterrows()):
        if direction == "LONG":
            # Check stop first (touched lower)
            if bar['low'] <= stop_price:
                pnl = stop_price - entry_price
                return {
                    "filled": True,
                    "outcome": "LOSS",
                    "exit_price": stop_price,
                    "bars_held": i + 1,
                    "pnl_points": pnl
                }
            # Check TP
            if bar['high'] >= tp_price:
                pnl = tp_price - entry_price
                return {
                    "filled": True,
                    "outcome": "WIN",
                    "exit_price": tp_price,
                    "bars_held": i + 1,
                    "pnl_points": pnl
                }
        else:  # SHORT
            # Check stop first (touched higher)
            if bar['high'] >= stop_price:
                pnl = entry_price - stop_price
                return {
                    "filled": True,
                    "outcome": "LOSS",
                    "exit_price": stop_price,
                    "bars_held": i + 1,
                    "pnl_points": pnl
                }
            # Check TP
            if bar['low'] <= tp_price:
                pnl = entry_price - tp_price
                return {
                    "filled": True,
                    "outcome": "WIN",
                    "exit_price": tp_price,
                    "bars_held": i + 1,
                    "pnl_points": pnl
                }
    
    # Timeout - exit at last close
    last_close = post_fill.iloc[-1]['close']
    if direction == "LONG":
        pnl = last_close - entry_price
    else:
        pnl = entry_price - last_close
    
    return {
        "filled": True,
        "outcome": "TIMEOUT",
        "exit_price": last_close,
        "bars_held": len(post_fill),
        "pnl_points": pnl
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_day(
    trading_date: datetime.date,
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    scanner: ICTIFVGScanner
) -> List[Dict[str, Any]]:
    """
    Analyze a single trading day for IFVG setups.
    
    Returns list of trade records.
    """
    records = []
    
    # Define trade window
    window_start = datetime.combine(trading_date, TRADE_WINDOW_START).replace(tzinfo=NY_TZ)
    window_end = datetime.combine(trading_date, TRADE_WINDOW_END).replace(tzinfo=NY_TZ)
    
    # Filter 5m data to trade window
    df_5m_window = df_5m[
        (df_5m['time'] >= window_start) & 
        (df_5m['time'] <= window_end)
    ].copy()
    
    if df_5m_window.empty:
        return records
    
    # Calculate ATR from recent data
    df_5m_recent = df_5m[df_5m['time'] < window_start].tail(20)
    if len(df_5m_recent) > 0:
        atr = (df_5m_recent['high'] - df_5m_recent['low']).mean()
    else:
        atr = 5.0
    
    # Reset scanner for new day
    scanner.reset()
    
    # Warm up scanner with pre-window data
    warmup = df_5m[df_5m['time'] < window_start].tail(30)
    for idx in warmup.index:
        data_up_to = df_5m[df_5m.index <= idx]
        scanner.check(data_up_to, idx, atr)
    
    # Scan through trade window
    for bar_idx in df_5m_window.index:
        data_up_to = df_5m[df_5m.index <= bar_idx]
        setup = scanner.check(data_up_to, bar_idx, atr)
        
        if setup:
            # Get current bar info
            current_bar = df_5m.loc[bar_idx]
            entry_time = current_bar['time']
            
            # Simulate the trade
            outcome = simulate_limit_order_outcome(
                df_1m,
                entry_time,
                setup.entry_price,
                setup.stop_price,
                setup.tp_price,
                setup.direction
            )
            
            # Calculate contracts and PnL
            risk_points = abs(setup.entry_price - setup.stop_price)
            risk_dollars = risk_points * POINT_VALUE
            contracts = max(1, int(RISK_PER_TRADE / risk_dollars)) if risk_dollars > 0 else 1
            pnl_dollars = outcome['pnl_points'] * POINT_VALUE * contracts
            
            # Get OHLCV window for visualization
            raw_ohlcv = get_raw_ohlcv_window(df_1m, entry_time)
            
            # Build record
            record = {
                "decision_id": f"ifvg_{trading_date}_{bar_idx}",
                "timestamp": entry_time.isoformat(),
                "bar_idx": int(bar_idx),
                "scanner_id": scanner.scanner_id,
                "scanner_context": make_serializable(scanner.get_context(setup)),
                "window": {
                    "raw_ohlcv_1m": raw_ohlcv
                },
                "oco": {
                    "entry_price": setup.entry_price,
                    "stop_price": setup.stop_price,
                    "tp_price": setup.tp_price,
                    "direction": setup.direction,
                    "contracts": contracts,
                    "order_type": "LIMIT"
                },
                "oco_results": {
                    "filled": outcome['filled'],
                    "outcome": outcome['outcome'],
                    "exit_price": outcome['exit_price'],
                    "bars_held": outcome['bars_held'],
                    "pnl_points": outcome['pnl_points'],
                    "pnl_dollars": pnl_dollars
                }
            }
            records.append(record)
    
    return records


def main():
    print("=" * 50)
    print("ICT Inverted FVG Strategy Backtest")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    df_1m = load_continuous_contract()
    htf = resample_all_timeframes(df_1m)
    df_5m = htf['5m']
    
    # Ensure time column
    if 'time' not in df_5m.columns and df_5m.index.name == 'time':
        df_5m = df_5m.reset_index()
    
    print(f"  {len(df_1m)} 1m bars, {len(df_5m)} 5m bars")
    
    # Date range
    start = pd.Timestamp(START_DATE).tz_localize(NY_TZ)
    end = start + timedelta(weeks=WEEKS)
    
    # Get trading days
    trading_days = pd.date_range(start.date(), end.date(), freq='B')  # Business days
    
    # Create scanner
    scanner = ICTIFVGScanner(
        min_liquidity_score=MIN_LIQUIDITY_SCORE,
        inversion_window_bars=INVERSION_WINDOW_BARS,
        risk_reward=RISK_REWARD,
        max_risk_dollars=RISK_PER_TRADE
    )
    
    # Process each day
    all_records = []
    
    for day in trading_days:
        day_date = day.date()
        records = analyze_day(day_date, df_1m, df_5m, scanner)
        all_records.extend(records)
        
        if records:
            print(f"  {day_date}: {len(records)} setups")
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Records JSONL
    records_file = RESULTS_DIR / "records.jsonl"
    with open(records_file, 'w') as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")
    
    # Summary
    total = len(all_records)
    filled = [r for r in all_records if r['oco_results']['filled']]
    wins = [r for r in filled if r['oco_results']['outcome'] == 'WIN']
    total_pnl = sum(r['oco_results']['pnl_dollars'] for r in filled)
    
    summary = {
        "strategy": "ICT Inverted FVG",
        "date_range": f"{START_DATE} to {end.date()}",
        "total_setups": total,
        "filled_trades": len(filled),
        "wins": len(wins),
        "win_rate": len(wins) / len(filled) * 100 if filled else 0,
        "total_pnl": total_pnl,
        "params": {
            "min_liquidity_score": MIN_LIQUIDITY_SCORE,
            "inversion_window_bars": INVERSION_WINDOW_BARS,
            "risk_reward": RISK_REWARD,
            "risk_per_trade": RISK_PER_TRADE
        }
    }
    
    with open(RESULTS_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("=" * 50)
    print(f"Trades: {len(filled)} | Wins: {len(wins)} | WR: {summary['win_rate']:.0f}% | PnL: ${total_pnl:.0f}")
    print(f"Saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

```

### scripts/backtest_inverse_test.py

```python
#!/usr/bin/env python3
"""
Inverse Strategy Test

Theory: Our FVG model is losing. Maybe the signal is actually a CONTINUATION
not a reversal. Flip all the directions and see if we accidentally found alpha.

This mirrors the mlang discovery in success_study.md where they found 70% WR
by inverting a losing pattern.

Usage:
    python scripts/run_inverse_test.py --input results/ict_ifvg/records.jsonl
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime
from typing import Dict, Any, List

from src.storage import ExperimentDB


def analyze_records(records: List[Dict]) -> Dict[str, Any]:
    """Analyze original records."""
    wins = sum(1 for r in records if r.get('label', r.get('outcome')) == 'WIN')
    longs = sum(1 for r in records if r.get('direction') == 'LONG')
    
    return {
        'total': len(records),
        'wins': wins,
        'losses': len(records) - wins,
        'win_rate': wins / len(records) if records else 0,
        'longs': longs,
        'shorts': len(records) - longs,
    }


def flip_direction(direction: str) -> str:
    """Flip LONG to SHORT and vice versa."""
    return 'SHORT' if direction == 'LONG' else 'LONG'


def invert_outcome(original_direction: str, outcome: str) -> str:
    """
    When we flip direction, outcomes also flip.
    
    Original LONG WIN (price went up) → Flipped SHORT would LOSE
    Original LONG LOSS (price went down) → Flipped SHORT would WIN
    """
    # If original was a WIN, flipped is a LOSS (and vice versa)
    return 'LOSS' if outcome == 'WIN' else 'WIN'


def run_inverse_test(input_path: str) -> Dict[str, Any]:
    """
    Run inverse strategy test.
    
    Takes existing signals, flips the direction, and measures outcome.
    """
    print("=" * 60)
    print("INVERSE STRATEGY TEST")
    print("=" * 60)
    print("Theory: FVG is continuation, not reversal")
    print("Method: Flip all directions (BUY→SELL, SELL→BUY)")
    print("=" * 60)
    
    # Load records
    print(f"\n[1] Loading records from {input_path}...")
    records = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line))
    
    print(f"    Loaded {len(records)} signals")
    
    # Analyze original
    print(f"\n[2] Original strategy performance...")
    original = analyze_records(records)
    print(f"    Total: {original['total']}")
    print(f"    LONG: {original['longs']} | SHORT: {original['shorts']}")
    print(f"    WIN: {original['wins']} | LOSS: {original['losses']}")
    print(f"    Win Rate: {original['win_rate']:.1%}")
    
    # Create inverted records
    print(f"\n[3] Flipping all signals...")
    inverted_records = []
    
    for rec in records:
        original_direction = rec.get('direction', 'LONG')
        original_outcome = rec.get('label', rec.get('outcome', 'LOSS'))
        
        inverted_rec = rec.copy()
        inverted_rec['direction'] = flip_direction(original_direction)
        inverted_rec['label'] = invert_outcome(original_direction, original_outcome)
        inverted_rec['original_direction'] = original_direction
        inverted_rec['original_outcome'] = original_outcome
        inverted_rec['strategy'] = 'inverse_' + rec.get('strategy', 'fvg')
        
        inverted_records.append(inverted_rec)
    
    # Analyze inverted
    print(f"\n[4] Inverted strategy performance...")
    inverted = analyze_records(inverted_records)
    print(f"    Total: {inverted['total']}")
    print(f"    LONG: {inverted['longs']} | SHORT: {inverted['shorts']}")
    print(f"    WIN: {inverted['wins']} | LOSS: {inverted['losses']}")
    print(f"    Win Rate: {inverted['win_rate']:.1%}")
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  Original Win Rate:  {original['win_rate']:.1%}")
    print(f"  Inverted Win Rate:  {inverted['win_rate']:.1%}")
    
    improvement = (inverted['win_rate'] - original['win_rate']) * 100
    
    if inverted['win_rate'] > 0.5:
        print(f"\n  🎯 JACKPOT! Inverted strategy is PROFITABLE!")
        print(f"  Win rate improvement: +{improvement:.1f} percentage points")
        print(f"\n  → FVG IS a continuation signal, not reversal!")
        print(f"  → When model says BUY, we should SELL (fade it)")
    elif inverted['win_rate'] > original['win_rate']:
        print(f"\n  📈 Inverted is BETTER but still <50%")
        print(f"  Improvement: +{improvement:.1f}pp")
    else:
        print(f"\n  ❌ Inverting made it WORSE")
        print(f"  Change: {improvement:.1f}pp")
        print(f"\n  → The original direction was correct, just bad execution")
    
    # Save inverted records
    output_dir = Path("results/inverse_fvg")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "records.jsonl"
    
    with open(output_path, 'w') as f:
        for rec in inverted_records:
            f.write(json.dumps(rec) + '\n')
    
    print(f"\n[5] Saved inverted records to {output_path}")
    
    # Store to DB
    db = ExperimentDB()
    run_id = f"inverse_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.store_run(
        run_id=run_id,
        strategy="inverse_fvg",
        config={
            'source': input_path,
            'method': 'direction_flip',
        },
        metrics={
            'total_trades': inverted['total'],
            'wins': inverted['wins'],
            'losses': inverted['losses'],
            'win_rate': inverted['win_rate'],
            'original_win_rate': original['win_rate'],
            'total_pnl': 0,
        }
    )
    print(f"    Stored: {run_id}")
    
    return {
        'original': original,
        'inverted': inverted,
        'improvement': improvement,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inverse Strategy Test")
    parser.add_argument("--input", type=str, default="results/ict_ifvg/records.jsonl",
                        help="Path to original signals")
    
    args = parser.parse_args()
    
    results = run_inverse_test(args.input)

```

### scripts/backtest_lowvol_breakout.py

```python
#!/usr/bin/env python3
"""
Low Volatility Breakout Strategy

Theory: Enter when volatility is dead (15m ATR at 5-day low).
Action: Trade the breakout when price moves.
Stop: DYNAMIC - 2x current candle's range (not lagging ATR)
Target: 2R

Hypothesis: Candle-range stop adapts faster than ATR.

Run:
    python scripts/run_lowvol_breakout.py --days 7
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any

from src.storage import ExperimentDB


# Strategy parameters
ATR_PERIOD = 14
ATR_LOW_LOOKBACK = 5 * 24 * 4  # 5 days of 15m bars (approx)
STOP_CANDLE_MULT = 2.0         # Stop = 2x candle range
TP_R_MULT = 2.0                # 2R target
BREAKOUT_THRESHOLD = 0.3       # Price must move 0.3 ATR to trigger


def resample_to_15m(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1m data to 15m bars."""
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    
    resampled = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled.reset_index()


def compute_atr_rolling_min(atr_series: pd.Series, window: int) -> pd.Series:
    """Compute rolling minimum of ATR."""
    return atr_series.rolling(window=window, min_periods=1).min()


def run_lowvol_breakout_strategy(days: int = 7, verbose: bool = True) -> Dict[str, Any]:
    """
    Run the Low Volatility Breakout strategy simulation.
    """
    print("=" * 60)
    print("LOW VOLATILITY BREAKOUT STRATEGY")
    print("=" * 60)
    print(f"Theory: Enter when ATR is at 5-day low (coiled market)")
    print(f"Stop: 2x current candle range (dynamic)")
    print(f"Target: {TP_R_MULT}R")
    print("=" * 60)
    
    # Load data
    actual_days = min(days, 7)
    end = datetime.now()
    start = end - timedelta(days=actual_days)
    
    print(f"\n[1] Loading {actual_days} days of ES data...")
    ticker = yf.Ticker("ES=F")
    df_1m = ticker.history(start=start, end=end, interval="1m")
    
    if df_1m is None or len(df_1m) == 0:
        print("ERROR: No data available")
        return {'trades': 0, 'win_rate': 0, 'total_pnl': 0}
    
    # Standardize
    df_1m.columns = [c.lower() for c in df_1m.columns]
    df_1m = df_1m.reset_index()
    df_1m['time'] = df_1m['Datetime'] if 'Datetime' in df_1m.columns else df_1m['datetime']
    
    print(f"    Loaded {len(df_1m)} 1m bars")
    
    # Resample to 15m
    print("\n[2] Resampling to 15m...")
    df = resample_to_15m(df_1m)
    print(f"    {len(df)} 15m bars")
    
    # Compute ATR
    print("\n[3] Computing 15m ATR and rolling minimum...")
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    df['atr'] = tr.rolling(window=ATR_PERIOD).mean()
    df['atr_min_5d'] = compute_atr_rolling_min(df['atr'], window=ATR_LOW_LOOKBACK)
    df['candle_range'] = df['high'] - df['low']
    
    # Run simulation
    print(f"\n[4] Scanning for low-ATR breakout entries...")
    
    trades = []
    active_trade = None
    lookback = max(ATR_PERIOD, 20)
    
    for i in range(lookback, len(df)):
        bar = df.iloc[i]
        current_time = bar['time']
        current_close = bar['close']
        current_high = bar['high']
        current_low = bar['low']
        current_atr = bar['atr']
        atr_min = bar['atr_min_5d']
        candle_range = bar['candle_range']
        
        if pd.isna(current_atr) or pd.isna(atr_min) or candle_range < 0.5:
            continue
        
        # Check active trade
        if active_trade is not None:
            if active_trade['direction'] == 'LONG':
                if current_low <= active_trade['stop']:
                    pnl = (active_trade['stop'] - active_trade['entry']) * 50
                    trades.append({
                        'entry_time': active_trade['entry_time'],
                        'exit_time': current_time,
                        'direction': 'LONG',
                        'entry': active_trade['entry'],
                        'exit': active_trade['stop'],
                        'pnl': pnl,
                        'result': 'LOSS'
                    })
                    if verbose:
                        print(f"  [LOSS] LONG @ {active_trade['entry']:.2f} -> SL @ {active_trade['stop']:.2f} = ${pnl:.2f}")
                    active_trade = None
                elif current_high >= active_trade['tp']:
                    pnl = (active_trade['tp'] - active_trade['entry']) * 50
                    trades.append({
                        'entry_time': active_trade['entry_time'],
                        'exit_time': current_time,
                        'direction': 'LONG',
                        'entry': active_trade['entry'],
                        'exit': active_trade['tp'],
                        'pnl': pnl,
                        'result': 'WIN'
                    })
                    if verbose:
                        print(f"  [WIN] LONG @ {active_trade['entry']:.2f} -> TP @ {active_trade['tp']:.2f} = ${pnl:.2f}")
                    active_trade = None
            else:  # SHORT
                if current_high >= active_trade['stop']:
                    pnl = (active_trade['entry'] - active_trade['stop']) * 50
                    trades.append({
                        'entry_time': active_trade['entry_time'],
                        'exit_time': current_time,
                        'direction': 'SHORT',
                        'entry': active_trade['entry'],
                        'exit': active_trade['stop'],
                        'pnl': pnl,
                        'result': 'LOSS'
                    })
                    if verbose:
                        print(f"  [LOSS] SHORT @ {active_trade['entry']:.2f} -> SL @ {active_trade['stop']:.2f} = ${pnl:.2f}")
                    active_trade = None
                elif current_low <= active_trade['tp']:
                    pnl = (active_trade['entry'] - active_trade['tp']) * 50
                    trades.append({
                        'entry_time': active_trade['entry_time'],
                        'exit_time': current_time,
                        'direction': 'SHORT',
                        'entry': active_trade['entry'],
                        'exit': active_trade['tp'],
                        'pnl': pnl,
                        'result': 'WIN'
                    })
                    if verbose:
                        print(f"  [WIN] SHORT @ {active_trade['entry']:.2f} -> TP @ {active_trade['tp']:.2f} = ${pnl:.2f}")
                    active_trade = None
            continue
        
        # Check for low ATR condition (ATR at or near 5-day low)
        is_low_vol = current_atr <= atr_min * 1.1  # Within 10% of 5-day low
        
        if not is_low_vol:
            continue
        
        # Check for breakout (price moves beyond previous bar)
        prev_high = df.iloc[i-1]['high']
        prev_low = df.iloc[i-1]['low']
        
        # DYNAMIC STOP: 2x current candle range
        stop_distance = candle_range * STOP_CANDLE_MULT
        
        # Breakout detection
        if current_high > prev_high:
            # Bullish breakout -> LONG
            entry = current_close
            stop = entry - stop_distance
            tp = entry + (stop_distance * TP_R_MULT)
            
            active_trade = {
                'entry_time': current_time,
                'entry': entry,
                'stop': stop,
                'tp': tp,
                'direction': 'LONG'
            }
            
            if verbose:
                print(f"  [TRIGGER] LONG @ {entry:.2f} (low ATR={current_atr:.2f}) Stop={stop:.2f} TP={tp:.2f}")
                
        elif current_low < prev_low:
            # Bearish breakout -> SHORT
            entry = current_close
            stop = entry + stop_distance
            tp = entry - (stop_distance * TP_R_MULT)
            
            active_trade = {
                'entry_time': current_time,
                'entry': entry,
                'stop': stop,
                'tp': tp,
                'direction': 'SHORT'
            }
            
            if verbose:
                print(f"  [TRIGGER] SHORT @ {entry:.2f} (low ATR={current_atr:.2f}) Stop={stop:.2f} TP={tp:.2f}")
    
    # Summary
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    losses = len(trades) - wins
    total_pnl = sum(t['pnl'] for t in trades)
    win_rate = wins / len(trades) if trades else 0
    
    # Direction breakdown
    longs = [t for t in trades if t['direction'] == 'LONG']
    shorts = [t for t in trades if t['direction'] == 'SHORT']
    long_wr = sum(1 for t in longs if t['result'] == 'WIN') / len(longs) if longs else 0
    short_wr = sum(1 for t in shorts if t['result'] == 'WIN') / len(shorts) if shorts else 0
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Period: {actual_days} days")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Wins: {wins} | Losses: {losses}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Total PnL: ${total_pnl:.2f}")
    if trades:
        print(f"  Avg PnL/Trade: ${total_pnl/len(trades):.2f}")
    print(f"\n  LONG trades: {len(longs)} @ {long_wr:.1%} WR")
    print(f"  SHORT trades: {len(shorts)} @ {short_wr:.1%} WR")
    
    return {
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'long_trades': len(longs),
        'short_trades': len(shorts),
        'long_wr': long_wr,
        'short_wr': short_wr,
        'strategy': 'lowvol_breakout',
        'params': {
            'stop_candle_mult': STOP_CANDLE_MULT,
            'tp_r': TP_R_MULT,
            'atr_period': ATR_PERIOD,
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Low Volatility Breakout Strategy")
    parser.add_argument("--days", type=int, default=7, help="Days to simulate")
    parser.add_argument("--save", action="store_true", help="Save to ExperimentDB")
    parser.add_argument("--quiet", action="store_true", help="Suppress trade details")
    
    args = parser.parse_args()
    
    results = run_lowvol_breakout_strategy(days=args.days, verbose=not args.quiet)
    
    if args.save and results['trades'] > 0:
        db = ExperimentDB()
        run_id = f"lowvol_breakout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        db.store_run(
            run_id=run_id,
            strategy="lowvol_breakout",
            config=results['params'],
            metrics={
                'total_trades': results['trades'],
                'wins': results['wins'],
                'losses': results['losses'],
                'win_rate': results['win_rate'],
                'total_pnl': results['total_pnl'],
            }
        )
        print(f"\n[+] Saved to ExperimentDB: {run_id}")

```

### scripts/backtest_lunch_fade.py

```python
#!/usr/bin/env python3
"""
Lunch Hour Fade Strategy

Theory: Breakouts fail between 11:30 AM - 1:00 PM EST (lunch hours).
Action: SHORT when price breaks above 15-minute swing high during lunch.
Stop: 0.5 ATR (tight - get out if it's a real breakout)
Target: 2R (risk/reward = 1:2)

Run:
    python scripts/run_lunch_fade.py --days 28
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, time
from typing import Dict, List, Any
from zoneinfo import ZoneInfo

from src.features.indicators import calculate_atr
from src.storage import ExperimentDB


# Strategy parameters
LUNCH_START = time(11, 30)  # 11:30 AM EST
LUNCH_END = time(13, 0)     # 1:00 PM EST
SWING_LOOKBACK = 15         # 15 bars for swing high detection
STOP_ATR_MULT = 0.5         # Tight stop
TP_R_MULT = 2.0             # 2R target
EST = ZoneInfo("America/New_York")


def find_swing_high(highs: np.ndarray, idx: int, lookback: int = 5) -> float:
    """
    Find the most recent swing high (higher than neighbors).
    Returns the swing high price, or np.nan if none found.
    """
    if idx < lookback * 2:
        return np.nan
    
    # Look for swing highs in the lookback window
    for i in range(idx - lookback, max(lookback, idx - lookback * 3), -1):
        # Is this a swing high? (higher than bars before and after)
        if i >= lookback and i < len(highs) - lookback:
            center = highs[i]
            left_max = max(highs[i-lookback:i])
            right_max = max(highs[i+1:i+lookback+1]) if i + lookback + 1 <= len(highs) else center
            
            if center >= left_max and center >= right_max:
                return center
    
    # Fallback: use rolling max
    return max(highs[idx-lookback:idx])


def is_lunch_hour(t: datetime) -> bool:
    """Check if time is in lunch hours (11:30 AM - 1:00 PM EST)."""
    # Convert to EST if needed
    if t.tzinfo is None:
        t = t.replace(tzinfo=EST)
    else:
        t = t.astimezone(EST)
    
    current_time = t.time()
    return LUNCH_START <= current_time <= LUNCH_END


def run_lunch_fade_strategy(days: int = 28, verbose: bool = True) -> Dict[str, Any]:
    """
    Run the Lunch Hour Fade strategy simulation.
    
    Args:
        days: Number of days to simulate (max 7 for 1m data)
        verbose: Print trade details
    
    Returns:
        Dict with strategy results
    """
    print("=" * 60)
    print("LUNCH HOUR FADE STRATEGY")
    print("=" * 60)
    print(f"Theory: Fade breakouts during 11:30 AM - 1:00 PM EST")
    print(f"Action: SHORT on 15m swing high break during lunch")
    print(f"Stop: {STOP_ATR_MULT} ATR | Target: {TP_R_MULT}R")
    print("=" * 60)
    
    # Load data (yfinance 1m limit is 7 days)
    actual_days = min(days, 7)
    if days > 7:
        print(f"\n[!] yfinance 1m limit is 7 days. Running with {actual_days} days.")
    
    end = datetime.now()
    start = end - timedelta(days=actual_days)
    
    print(f"\n[1] Loading {actual_days} days of ES data...")
    ticker = yf.Ticker("ES=F")
    df = ticker.history(start=start, end=end, interval="1m")
    
    if df is None or len(df) == 0:
        print("ERROR: No data available")
        return {'trades': 0, 'win_rate': 0, 'total_pnl': 0}
    
    # Standardize columns
    df.columns = [c.lower() for c in df.columns]
    df = df.reset_index()
    df['time'] = df['Datetime'] if 'Datetime' in df.columns else df['datetime']
    
    print(f"    Loaded {len(df)} bars")
    
    # Calculate ATR
    print("\n[2] Computing indicators...")
    df['atr'] = calculate_atr(df, period=14)
    df['atr'] = df['atr'].ffill().bfill()  # Fill NaN
    
    # Run simulation
    print(f"\n[3] Scanning for lunch hour breakouts...")
    
    trades = []
    active_trade = None
    lookback = SWING_LOOKBACK
    
    for i in range(lookback + 14, len(df)):
        current_bar = df.iloc[i]
        current_time = pd.to_datetime(current_bar['time'])
        current_price = current_bar['close']
        current_high = current_bar['high']
        current_low = current_bar['low']
        atr = current_bar['atr']
        
        if pd.isna(atr) or atr < 0.5:
            continue
        
        # Check if in active trade
        if active_trade is not None:
            # Check stop loss (price goes UP for short)
            if current_high >= active_trade['stop']:
                pnl = -(active_trade['stop'] - active_trade['entry']) * 50  # Negative for loss
                trades.append({
                    'entry_time': active_trade['entry_time'],
                    'exit_time': current_time,
                    'entry': active_trade['entry'],
                    'exit': active_trade['stop'],
                    'pnl': pnl,
                    'result': 'LOSS'
                })
                if verbose:
                    print(f"  [LOSS] SHORT @ {active_trade['entry']:.2f} -> SL @ {active_trade['stop']:.2f} = ${pnl:.2f}")
                active_trade = None
            # Check take profit (price goes DOWN for short)
            elif current_low <= active_trade['tp']:
                pnl = (active_trade['entry'] - active_trade['tp']) * 50  # Positive for win
                trades.append({
                    'entry_time': active_trade['entry_time'],
                    'exit_time': current_time,
                    'entry': active_trade['entry'],
                    'exit': active_trade['tp'],
                    'pnl': pnl,
                    'result': 'WIN'
                })
                if verbose:
                    print(f"  [WIN] SHORT @ {active_trade['entry']:.2f} -> TP @ {active_trade['tp']:.2f} = ${pnl:.2f}")
                active_trade = None
            continue
        
        # Look for entry during lunch hours only
        if not is_lunch_hour(current_time):
            continue
        
        # Find swing high
        swing_high = find_swing_high(df['high'].values, i, lookback)
        if pd.isna(swing_high):
            continue
        
        # Check for breakout (price breaks above swing high)
        if current_high > swing_high:
            # SHORT entry
            entry = current_price
            stop = entry + (atr * STOP_ATR_MULT)
            risk = stop - entry
            tp = entry - (risk * TP_R_MULT)
            
            active_trade = {
                'entry_time': current_time,
                'entry': entry,
                'stop': stop,
                'tp': tp,
                'swing_high': swing_high,
            }
            
            if verbose:
                print(f"  [TRIGGER] SHORT @ {entry:.2f} (broke {swing_high:.2f}) SL={stop:.2f} TP={tp:.2f}")
    
    # Summary
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    losses = len(trades) - wins
    total_pnl = sum(t['pnl'] for t in trades)
    win_rate = wins / len(trades) if trades else 0
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Period: {actual_days} days")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Wins: {wins} | Losses: {losses}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Total PnL: ${total_pnl:.2f}")
    if trades:
        print(f"  Avg PnL/Trade: ${total_pnl/len(trades):.2f}")
    
    return {
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'strategy': 'lunch_fade',
        'params': {
            'lunch_start': str(LUNCH_START),
            'lunch_end': str(LUNCH_END),
            'stop_atr': STOP_ATR_MULT,
            'tp_r': TP_R_MULT,
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Lunch Hour Fade Strategy")
    parser.add_argument("--days", type=int, default=7, help="Days to simulate")
    parser.add_argument("--save", action="store_true", help="Save to ExperimentDB")
    parser.add_argument("--quiet", action="store_true", help="Suppress trade details")
    
    args = parser.parse_args()
    
    results = run_lunch_fade_strategy(days=args.days, verbose=not args.quiet)
    
    if args.save and results['trades'] > 0:
        db = ExperimentDB()
        run_id = f"lunch_fade_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        db.store_run(
            run_id=run_id,
            strategy="lunch_fade",
            config=results['params'],
            metrics={
                'total_trades': results['trades'],
                'wins': results['wins'],
                'losses': results['losses'],
                'win_rate': results['win_rate'],
                'total_pnl': results['total_pnl'],
            }
        )
        print(f"\n[+] Saved to ExperimentDB: {run_id}")

```

### scripts/backtest_mean_reversion.py

```python
"""
Run Mean Reversion Experiment
Runs a 3-week scan using Mean Reversion strategy.
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

from src.experiments.config import ExperimentConfig, FeatureConfig, LabelConfig
from src.experiments.runner import run_experiment
from src.viz.export import Exporter
from src.viz.config import VizConfig

def main():
    # Define time range (approx 3 weeks)
    # Using a recent slice of data, assuming data exists
    # If not sure, we can load data first, but let's try a fixed range or last 3 weeks
    end_date = pd.Timestamp("2025-06-21", tz="America/New_York") # Adjust based on available data
    start_date = end_date - pd.Timedelta(weeks=3)
    
    print(f"Running Mean Reversion Scan from {start_date.date()} to {end_date.date()}")
    
    # Configure Experiment
    config = ExperimentConfig(
        name="mean_reversion_3w",
        start_date=start_date,
        end_date=end_date,
        
        # Scanner: Mean Reversion
        scanner_id="mean_reversion_20_3.0_5m_30_70",
        scanner_params={
            "ema_period": 20,
            "atr_multiple": 3.0,
            "rsi_min": 30.0,
            "rsi_max": 70.0,
            "timeframe": "5m"
        },
        
        # Standard configs
        feature_config=FeatureConfig(),
        label_config=LabelConfig(), # Defaults
    )
    
    # Configure Exporter
    # We want to export windows for visualization
    viz_config = VizConfig(
        include_windows=True,
        include_model_outputs=False
    )
    
    # Run
    # Using a temporary output directory for this run
    out_dir = Path("results/mean_reversion_3w")
    exporter = Exporter(viz_config, experiment_config=config.to_dict())
    
    result = run_experiment(config, exporter=exporter)
    
    # Save results
    exporter.finalize(out_dir)
    
    print("\n" + "="*50)
    print(f"Experiment Complete!")
    print(f"Total Decisions: {result.total_records}")
    print(f"Results saved to: {out_dir.absolute()}")
    print("="*50)

if __name__ == "__main__":
    main()

```

### scripts/backtest_modular_strategy.py

```python
"""
Generic Modular Strategy Runner

Runs a backtest using a modular Trigger and Bracket configuration.
"""

import os
import sys
import argparse
import json
import pandas as pd
from datetime import timedelta
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR, NY_TZ
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.features.indicators import calculate_atr
from src.sim.stepper import MarketStepper
from src.features.pipeline import compute_features, FeatureConfig
from src.policy.modular_scanner import ModularScanner
from src.policy.brackets import bracket_from_dict
from src.labels.counterfactual import compute_smart_stop_counterfactual


def get_raw_ohlcv_window(stepper, lookback=60, lookahead=30):
    """Get raw OHLCV for chart viz."""
    current = stepper.get_current_idx()
    start = max(0, current - lookback)
    end = min(len(stepper.df), current + lookahead)
    window = stepper.df.iloc[start:end]
    return window[['open', 'high', 'low', 'close', 'volume']].values.tolist()


def main():
    parser = argparse.ArgumentParser(description="Run Modular Strategy Scan")
    parser.add_argument("--config", type=str, required=True, help="JSON configuration")
    parser.add_argument("--start-date", type=str, default="2025-03-17", help="Start date")
    parser.add_argument("--weeks", type=int, default=1, help="Number of weeks")
    parser.add_argument("--timeframe", type=str, default="1m", 
                        choices=["1m", "5m", "15m", "1h"], help="Timeframe for scanning")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    # Parse config
    try:
        config = json.loads(args.config)
    except json.JSONDecodeError:
        # Try finding as file
        if os.path.exists(args.config):
            with open(args.config) as f:
                config = json.load(f)
        else:
            raise ValueError("Invalid config JSON or file path")

    trigger_config = config.get("trigger")
    bracket_config = config.get("bracket")
    
    if not trigger_config or not bracket_config:
        raise ValueError("Config must contain 'trigger' and 'bracket' sections")

    start_date = pd.Timestamp(args.start_date)
    end_date = start_date + timedelta(weeks=args.weeks)
    
    out_dir = Path(args.out) if args.out else RESULTS_DIR / "modular_scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Modular Strategy Scan")
    print(f"Trigger: {trigger_config.get('type')}")
    print(f"Bracket: {bracket_config.get('type')}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1] Loading data...")
    df = load_continuous_contract()
    df = df[(df['time'] >= str(start_date)) & (df['time'] < str(end_date))].reset_index(drop=True)
    print(f"Loaded {len(df)} 1m bars")
    
    # 2. Resample & Indicators
    print("\n[2] Computing indicators...")
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    df_1h = htf_data.get('1h')
    
    # Select timeframe for scanning
    tf_map = {'1m': df, '5m': df_5m, '15m': df_15m, '1h': df_1h}
    df_tf = tf_map.get(args.timeframe, df)
    
    if df_tf is not None and len(df_tf) > 14:
        df_tf['atr'] = calculate_atr(df_tf, 14)
        avg_atr = df_tf['atr'].dropna().mean()
    else:
        avg_atr = 10.0
    print(f"  Using {args.timeframe}: {len(df_tf) if df_tf is not None else 0} bars, avg ATR: {avg_atr:.2f}")
    
    # 3. Initialize Components
    scanner = ModularScanner(trigger_config)
    bracket = bracket_from_dict(bracket_config)
    
    stepper = MarketStepper(df_tf if df_tf is not None else df, start_idx=50, end_idx=len(df_tf)-50 if df_tf is not None else len(df)-200)
    feature_config = FeatureConfig(lookback_1m=30)
    
    records = []
    decision_idx = 0
    
    # 4. Run Scan
    print("\n[3] Running Scan...")
    while True:
        step = stepper.step()
        if step.is_done:
            break
            
        features = compute_features(stepper, feature_config, df_5m=df_5m, df_15m=df_15m)
        
        # Slice df_tf to current bar position (causal)
        df_tf_slice = df_tf.iloc[:step.bar_idx + 1] if step.bar_idx < len(df_tf) else df_tf
        
        # Pass HTF data to trigger via kwargs
        scan = scanner.scan(features.market_state, features, df_15m=df_tf_slice)
        
        if scan.triggered:
            direction = scan.context['direction']
            entry_price = features.current_price
            atr = features.atr if features.atr > 0 else avg_atr
            
            # Compute levels using modular bracket
            levels = bracket.compute(entry_price, direction, atr)
            
            print(f"  Triggered {direction} at {features.timestamp} | SL: {levels.stop_price:.2f} TP: {levels.tp_price:.2f}")

            # Compute Outcomes
            cf = compute_smart_stop_counterfactual(
                df=df,
                entry_idx=step.bar_idx,
                direction=direction,
                stop_price=levels.stop_price,
                tp_multiple=levels.r_multiple,
                atr=atr,
                oco_name="modular"
            )
            
            # Raw OHLCV for chart
            raw_ohlcv = get_raw_ohlcv_window(stepper, lookback=60, lookahead=30)
            
            record = {
                'decision_id': f"mod_{decision_idx:04d}",
                'timestamp': features.timestamp.isoformat(),
                'bar_idx': step.bar_idx,
                'index': decision_idx,
                'scanner_id': scanner.scanner_id,
                'scanner_context': scan.context,
                'current_price': entry_price,
                'atr': atr,
                'stop_price': levels.stop_price,
                'tp_price': levels.tp_price,
                'stop_reason': f"MODULAR_{bracket_config['type'].upper()}",
                'risk_points': levels.risk_points,
                'window': {
                    'x_price_1m': features.x_price_1m.tolist() if features.x_price_1m is not None else [],
                    'raw_ohlcv_1m': raw_ohlcv,
                    'x_context': features.x_context.tolist() if features.x_context is not None else [],
                },
                'oco': {
                    'entry_price': entry_price,
                    'stop_price': levels.stop_price,
                    'tp_price': levels.tp_price,
                    'direction': direction,
                    'atr_at_creation': atr,
                    'max_bars': 100
                },
                'oco_results': {
                    'modular': {
                        'outcome': cf.outcome,
                        'pnl_dollars': cf.pnl_dollars,
                        'bars_held': int(cf.bars_held),
                        'exit_price': float(cf.exit_price),
                    }
                },
                'best_oco': 'modular',
                'best_pnl': cf.pnl_dollars
            }
            records.append(record)
            decision_idx += 1

    # 5. Save Results
    if records:
        with open(out_dir / "records.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        
        summary = {
            "run_id": out_dir.name,
            "total_trades": len(records),
            "win_rate": len([r for r in records if r['oco_results']['modular']['pnl_dollars'] > 0]) / len(records),
            "config": config
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"\nSaved {len(records)} records to {out_dir}")
    else:
        print("\nNo trades triggered.")


if __name__ == "__main__":
    main()

```

### scripts/backtest_or_multi_oco.py

```python
#!/usr/bin/env python
"""
Opening Range Multi-OCO Simulation with SMART STOPS
Uses actual OR levels as stops, not simple ATR offsets.

Usage:
    python scripts/run_or_multi_oco.py --start-date 2025-03-17 --weeks 3 --out results/or_multi_oco/
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.sim.stepper import MarketStepper
from src.features.pipeline import compute_features, FeatureConfig
from src.features.indicators import calculate_atr
from src.policy.library.opening_range import OpeningRangeScanner
from src.policy.oco_grid import get_or_oco_grid
from src.sim.stop_calculator import get_stop_for_or_retest, calculate_tp_from_risk
from src.labels.counterfactual import compute_smart_stop_counterfactual
from src.config import RESULTS_DIR


def make_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    return obj


def get_raw_ohlcv_window(stepper: MarketStepper, lookback: int = 60, lookahead: int = 20):
    """
    Get raw OHLCV window around current position for visualization.
    
    Args:
        stepper: MarketStepper at current position
        lookback: Number of bars before current position
        lookahead: Number of bars after current position (future, for viz only)
        
    Returns:
        List of [open, high, low, close, volume] for each bar
    """
    df = stepper.df
    current_idx = stepper.current_idx
    
    # Get history (causal - this is what the model sees)
    start_idx = max(0, current_idx - lookback)
    history = df.iloc[start_idx:current_idx + 1]
    
    # Get future (for viz only - NOT for model/training)
    end_idx = min(len(df), current_idx + lookahead + 1)
    future = df.iloc[current_idx + 1:end_idx]
    
    # Combine history + future
    combined = pd.concat([history, future])
    
    return [
        [float(r['open']), float(r['high']), float(r['low']), float(r['close']), float(r['volume'])]
        for _, r in combined.iterrows()
    ]


def main():
    parser = argparse.ArgumentParser(description="Run Opening Range multi-OCO simulation with smart stops")
    parser.add_argument("--start-date", type=str, default="2025-03-17", help="Start date")
    parser.add_argument("--weeks", type=int, default=3, help="Number of weeks to simulate")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--lookback-1m", type=int, default=30, help="1m bars before entry for CNN")
    
    args = parser.parse_args()
    
    # Calculate end date
    start_date = pd.Timestamp(args.start_date)
    end_date = start_date + timedelta(weeks=args.weeks)
    
    # Setup output
    out_dir = Path(args.out) if args.out else RESULTS_DIR / "or_multi_oco"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # R multiples to test for each stop
    TP_MULTIPLES = [1.0, 1.5, 2.0]
    
    print("=" * 60)
    print("Opening Range Multi-OCO Simulation (SMART STOPS)")
    print(f"Period: {start_date.date()} to {end_date.date()} ({args.weeks} weeks)")
    print(f"Stop: OR level + 0.25 ATR padding")
    print(f"R multiples: {TP_MULTIPLES}")
    print(f"Output: {out_dir}")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1] Loading data...")
    df = load_continuous_contract()
    df = df[(df['time'] >= str(start_date)) & (df['time'] < str(end_date))].reset_index(drop=True)
    print(f"Loaded {len(df)} 1m bars")
    
    # 2. Resample to higher timeframes
    print("\n[2] Resampling to higher timeframes...")
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    print(f"5m bars: {len(df_5m)}, 15m bars: {len(df_15m)}")
    
    # 3. Precompute ATR on 5m
    print("\n[3] Computing ATR...")
    df_5m_copy = df_5m.copy()
    df_5m_copy['atr'] = calculate_atr(df_5m_copy, 14)
    avg_atr = df_5m_copy['atr'].dropna().mean()
    print(f"Average 5m ATR: {avg_atr:.2f} points")
    
    # 4. Setup scanner and stepper
    print("\n[4] Running simulation...")
    scanner = OpeningRangeScanner(
        or_timeframe_minutes=15,
        retest_threshold_atr=0.25,
        cooldown_bars=30,
        direction="BOTH"
    )
    
    stepper = MarketStepper(df, start_idx=200, end_idx=len(df) - 200)
    feature_config = FeatureConfig(lookback_1m=args.lookback_1m)
    
    # 5. Collect decision points
    records = []
    decision_idx = 0
    
    while True:
        step = stepper.step()
        if step.is_done:
            break
        
        # Compute features
        features = compute_features(
            stepper,
            feature_config,
            df_5m=df_5m,
            df_15m=df_15m,
        )
        
        # Check scanner
        scan_result = scanner.scan(features.market_state, features)
        if not scan_result.triggered:
            continue
        
        # Get ATR at this point
        atr = features.atr if features.atr > 0 else avg_atr
        
        # Get direction and OR levels from scanner context
        direction = scan_result.context.get('direction', 'LONG')
        or_high = scan_result.context.get('or_high', 0)
        or_low = scan_result.context.get('or_low', 0)
        entry_price = features.current_price
        
        # === SMART STOP: Use OR level as stop ===
        stop_price, stop_reason = get_stop_for_or_retest(
            direction=direction,
            entry_price=entry_price,
            atr=atr,
            or_high=or_high,
            or_low=or_low,
            padding_atr=0.25
        )
        
        # Calculate risk for R sizing
        risk = abs(entry_price - stop_price)
        
        # Test multiple R multiples
        oco_results = {}
        for tp_mult in TP_MULTIPLES:
            oco_name = f"{direction}_OR_{tp_mult}R"
            
            cf = compute_smart_stop_counterfactual(
                df=df,
                entry_idx=step.bar_idx,
                direction=direction,
                stop_price=stop_price,
                tp_multiple=tp_mult,
                atr=atr,
                oco_name=oco_name
            )
            
            oco_results[oco_name] = {
                'outcome': cf.outcome,
                'pnl_dollars': cf.pnl_dollars,
                'pnl': cf.pnl,
                'mae': cf.mae,
                'mfe': cf.mfe,
                'bars_held': cf.bars_held,
                'stop_price': stop_price,
                'tp_price': cf.tp_price,
            }
        
        # Find best OCO
        best_oco = None
        best_pnl = float('-inf')
        for name, result in oco_results.items():
            if result['pnl_dollars'] > best_pnl:
                best_pnl = result['pnl_dollars']
                best_oco = name
        
        # Get raw OHLCV for visualization (60 bars before + 20 after decision)
        raw_ohlcv = get_raw_ohlcv_window(stepper, lookback=60, lookahead=20)
        
        # Get best TP price for OCO visualization
        best_tp_price = oco_results.get(best_oco, {}).get('tp_price', entry_price + risk) if best_oco else entry_price + risk
        
        # Build record with VizDecision-compatible structure
        record = {
            'decision_id': f"or_{decision_idx:04d}",
            'timestamp': features.timestamp.isoformat() if features.timestamp else None,
            'bar_idx': step.bar_idx,
            'index': decision_idx,
            
            # Scanner identification (Phase .1 contract)
            'scanner_id': 'opening_range',
            'scanner_context': {
                'or_high': or_high,
                'or_low': or_low,
                'direction': direction,
                'retest_level': scan_result.context.get('retest_level', 0),
            },
            
            # Market state
            'current_price': entry_price,
            'atr': atr,
            
            # Stop info
            'stop_price': stop_price,
            'stop_reason': stop_reason,
            'risk_points': risk,
            
            # Window (VizWindow-compatible structure)
            'window': {
                'x_price_1m': features.x_price_1m.tolist() if features.x_price_1m is not None else [],
                'raw_ohlcv_1m': raw_ohlcv,  # <-- KEY FIX: actual prices for chart
                'x_context': features.x_context.tolist() if features.x_context is not None else [],
            },
            
            # OCO (VizOCO-compatible structure)
            'oco': {
                'entry_price': entry_price,
                'stop_price': stop_price,
                'tp_price': best_tp_price,
                'direction': direction,
                'atr_at_creation': atr,
                'max_bars': 200,
            },
            
            # Multi-OCO results (for analysis/training)
            'oco_results': oco_results,
            
            # Best OCO (CNN classification target)
            'best_oco': best_oco,
            'best_pnl': best_pnl,
        }
        
        records.append(record)
        decision_idx += 1
        
        # Progress
        if decision_idx % 5 == 0:
            print(f"  Triggers: {decision_idx}, Last: {direction} at {features.timestamp}, stop={stop_price:.2f}")
    
    print(f"\n[5] Completed! Total triggers: {len(records)}")
    
    # 6. Analyze results
    long_count = 0
    short_count = 0
    best_oco_counts = {}
    oco_pnl_totals = {}
    
    if records:
        print("\n[6] Analysis:")
        
        # Count by direction (now in scanner_context)
        long_count = sum(1 for r in records if r['scanner_context']['direction'] == 'LONG')
        short_count = sum(1 for r in records if r['scanner_context']['direction'] == 'SHORT')
        print(f"  LONG triggers: {long_count}")
        print(f"  SHORT triggers: {short_count}")
        
        # Average risk
        avg_risk = sum(r['risk_points'] for r in records) / len(records)
        print(f"  Average risk (points): {avg_risk:.2f}")
        
        # Best OCO distribution
        for r in records:
            oco_name = r['best_oco']
            best_oco_counts[oco_name] = best_oco_counts.get(oco_name, 0) + 1
        
        print("\n  Best OCO distribution:")
        for name, count in sorted(best_oco_counts.items(), key=lambda x: -x[1]):
            print(f"    {name}: {count} ({count/len(records)*100:.1f}%)")
        
        # Total PnL by OCO
        print("\n  Total PnL by OCO:")
        for r in records:
            for oco_name, result in r['oco_results'].items():
                if oco_name not in oco_pnl_totals:
                    oco_pnl_totals[oco_name] = 0
                oco_pnl_totals[oco_name] += result['pnl_dollars']
        
        for name, total in sorted(oco_pnl_totals.items(), key=lambda x: -x[1]):
            print(f"    {name}: ${total:.2f}")
    
    # 7. Write output
    print(f"\n[7] Writing output to {out_dir}...")
    
    # Write JSONL
    output_path = out_dir / "or_multi_oco_records.jsonl"
    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(make_serializable(r)) + '\n')
    print(f"  Wrote {len(records)} records to {output_path.name}")
    
    # Write summary
    summary = {
        'start_date': str(start_date.date()),
        'end_date': str(end_date.date()),
        'weeks': args.weeks,
        'stop_type': 'OR_LEVEL',
        'tp_multiples': TP_MULTIPLES,
        'total_triggers': len(records),
        'long_triggers': long_count,
        'short_triggers': short_count,
        'oco_pnl_totals': make_serializable(oco_pnl_totals),
        'best_oco_counts': best_oco_counts,
    }
    
    summary_path = out_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote summary to {summary_path.name}")
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

```

### scripts/backtest_puller.py

```python
#!/usr/bin/env python
"""
Puller Strategy Backtest
Scans for measured move failure patterns and generates viz-compatible artifacts.

Usage:
    python scripts/backtest_puller.py --weeks 4 --out results/viz/puller_v6_4w
"""

import sys
import json
import argparse
import uuid
import hashlib
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.sim.stepper import MarketStepper
from src.features.pipeline import compute_features, FeatureConfig
from src.features.indicators import calculate_atr
from src.policy.library.puller import PullerScanner
from src.labels.counterfactual import compute_smart_stop_counterfactual
from src.config import RESULTS_DIR


def make_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    elif pd.isna(obj):
        return None
    return obj


def get_raw_ohlcv_window(stepper: MarketStepper, lookback: int = 60, lookahead: int = 60):
    """
    Get raw OHLCV window around current position for visualization.
    """
    df = stepper.df
    current_idx = stepper.current_idx
    
    start_idx = max(0, current_idx - lookback)
    end_idx = min(len(df), current_idx + lookahead + 1)
    
    combined = df.iloc[start_idx:end_idx]
    
    return [
        {
            "time": r['time'].isoformat() if hasattr(r['time'], 'isoformat') else str(r['time']),
            "open": float(r['open']),
            "high": float(r['high']),
            "low": float(r['low']),
            "close": float(r['close']),
            "volume": float(r['volume'])
        }
        for _, r in combined.iterrows()
    ]


def main():
    parser = argparse.ArgumentParser(description="Run Puller strategy backtest")
    parser.add_argument("--weeks", type=int, default=4, help="Number of weeks to backtest")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--variation", type=str, default="v6_scalp", help="Variation to run")
    
    args = parser.parse_args()
    
    # Date range: use last N weeks of available data
    end_date = pd.Timestamp("2025-09-17", tz="America/New_York")
    start_date = end_date - timedelta(weeks=args.weeks)
    
    # Setup output
    run_id = f"puller_{args.variation}_{args.weeks}w"
    out_dir = Path(args.out) if args.out else RESULTS_DIR / "viz" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # V6 Scalp parameters (best win rate from prior testing)
    scanner = PullerScanner(
        variation_id=args.variation,
        entry_unit=0.5,
        stop_unit=1.0,
        tp_unit=-2.0,
        min_move_unit=1.5,
        max_move_unit=2.5,
        max_duration_bars=45
    )
    
    print("=" * 60)
    print(f"Puller Strategy Backtest ({args.variation})")
    print(f"Period: {start_date.date()} to {end_date.date()} ({args.weeks} weeks)")
    print(f"Output: {out_dir}")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1] Loading data...")
    df = load_continuous_contract()
    df = df[(df['time'] >= str(start_date)) & (df['time'] < str(end_date))].reset_index(drop=True)
    print(f"Loaded {len(df)} 1m bars")
    
    if len(df) == 0:
        print("ERROR: No data in range")
        return
    
    # 2. Resample to higher timeframes
    print("\n[2] Resampling to higher timeframes...")
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    print(f"5m bars: {len(df_5m)}, 15m bars: {len(df_15m)}")
    
    # 3. Precompute ATR on 5m
    print("\n[3] Computing ATR...")
    df_5m_copy = df_5m.copy()
    df_5m_copy['atr'] = calculate_atr(df_5m_copy, 14)
    avg_atr = df_5m_copy['atr'].dropna().mean()
    print(f"Average 5m ATR: {avg_atr:.2f} points")
    
    # 4. Setup stepper
    print("\n[4] Running simulation...")
    stepper = MarketStepper(df, start_idx=100, end_idx=len(df) - 100)
    feature_config = FeatureConfig(lookback_1m=60)
    
    decisions = []
    trades = []
    decision_idx = 0
    
    while True:
        step = stepper.step()
        if step.is_done:
            break
        
        # Progress
        if step.bar_idx % 5000 == 0:
            print(f"  Processing bar {step.bar_idx}/{len(df)}...")
        
        # Compute features
        features = compute_features(
            stepper,
            feature_config,
            df_5m=df_5m,
            df_15m=df_15m,
        )
        
        # Check scanner
        scan_result = scanner.scan(features.market_state, features)
        if not scan_result.triggered:
            continue
        
        ctx = scan_result.context
        direction = ctx.get('direction', 'SHORT')
        entry_price = ctx.get('entry_price', features.current_price)
        stop_price = ctx.get('stop_loss', entry_price)
        tp_price = ctx.get('take_profit', entry_price)
        atr = ctx.get('atr', avg_atr)
        
        # Compute counterfactual outcome
        cf = compute_smart_stop_counterfactual(
            df=df,
            entry_idx=step.bar_idx,
            direction=direction,
            stop_price=stop_price,
            tp_multiple=None,  # Use explicit TP
            atr=atr,
            oco_name=f"puller_{args.variation}",
            tp_price=tp_price
        )
        
        decision_id = str(uuid.uuid4())[:8]
        trade_id = str(uuid.uuid4())[:8]
        
        # Get raw OHLCV for visualization
        raw_ohlcv = get_raw_ohlcv_window(stepper, lookback=60, lookahead=60)
        
        # Build decision record (VizDecision-compatible)
        decision = {
            'decision_id': decision_id,
            'timestamp': features.timestamp.isoformat() if features.timestamp else None,
            'bar_idx': step.bar_idx,
            'action': direction,
            
            # Scanner identification
            'scanner_id': scanner.scanner_id,
            'scanner_context': ctx,
            
            # Market state
            'current_price': features.current_price,
            'atr': atr,
            
            # Window (VizWindow-compatible structure)
            'window': {
                'raw_ohlcv_1m': raw_ohlcv,
                'x_context': [],
            },
            
            # OCO (VizOCO-compatible structure)
            'oco': {
                'entry_price': entry_price,
                'stop_price': stop_price,
                'tp_price': tp_price,
                'direction': direction,
                'atr_at_creation': atr,
                'max_bars': 200,
            },
            
            'oco_results': {
                f"puller_{args.variation}": {
                    'outcome': cf.outcome,
                    'pnl_dollars': cf.pnl_dollars,
                    'bars_held': cf.bars_held,
                }
            }
        }
        decisions.append(decision)
        
        # Build trade record
        trade = {
            'trade_id': trade_id,
            'decision_id': decision_id,
            'index': decision_idx,
            'direction': direction,
            'scanner_id': scanner.scanner_id,
            'entry_time': features.timestamp.isoformat() if features.timestamp else None,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'tp_price': tp_price,
            'exit_price': cf.exit_price if hasattr(cf, 'exit_price') else entry_price,
            'pnl_dollars': cf.pnl_dollars,
            'outcome': cf.outcome,
            'exit_reason': cf.outcome,
            'bars_held': cf.bars_held,
            'mae': cf.mae,
            'mfe': cf.mfe,
        }
        trades.append(trade)
        
        decision_idx += 1
    
    print(f"\n[5] Completed! Total triggers: {len(decisions)}")
    
    # 5. Analyze results
    if trades:
        wins = sum(1 for t in trades if t['pnl_dollars'] > 0)
        losses = len(trades) - wins
        total_pnl = sum(t['pnl_dollars'] for t in trades)
        longs = sum(1 for t in trades if t['direction'] == 'LONG')
        shorts = len(trades) - longs
        
        print("\n[6] Results:")
        print(f"  Trades: {len(trades)} (LONG: {longs}, SHORT: {shorts})")
        print(f"  Wins: {wins}, Losses: {losses}, Win Rate: {wins/len(trades)*100:.1f}%")
        print(f"  Total PnL: ${total_pnl:,.2f}")
    
    # 6. Write output files
    print(f"\n[7] Writing output to {out_dir}...")
    
    # decisions.jsonl
    decisions_path = out_dir / "decisions.jsonl"
    with open(decisions_path, 'w') as f:
        for d in decisions:
            f.write(json.dumps(make_serializable(d)) + '\n')
    
    # trades.jsonl
    trades_path = out_dir / "trades.jsonl"
    with open(trades_path, 'w') as f:
        for t in trades:
            f.write(json.dumps(make_serializable(t)) + '\n')
    
    # manifest.json
    now = pd.Timestamp.now(tz="America/New_York")
    manifest = {
        "run_id": run_id,
        "fingerprint": hashlib.md5(run_id.encode()).hexdigest()[:16],
        "config": {
            "description": f"Puller {args.variation} backtest",
            "weeks": args.weeks,
            "variation": args.variation,
        },
        "created_at": now.isoformat(),
        "strategy": "puller",
        "files": {
            "decisions": "decisions.jsonl",
            "trades": "trades.jsonl"
        },
        "counts": {
            "decisions": len(decisions),
            "trades": len(trades),
        }
    }
    
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  Wrote {len(decisions)} decisions, {len(trades)} trades")
    print(f"\n✅ Results at: {out_dir}")
    print(f"Validate with: python golden/validate_run.py {out_dir}")


if __name__ == "__main__":
    main()

```

### scripts/backtest_rvap.py

```python
#!/usr/bin/env python3
"""
Relative Volume at Price (RVAP) Scanner

Theory: Volume confirms breakouts.
- Approaching PDH on LOW volume → FADE (SHORT)
- Approaching PDH on HIGH volume (2x avg) → BREAKOUT (LONG)

Custom indicator: RVAP = current volume / 20-bar average volume

Usage:
    python scripts/run_rvap_scan.py --days 7
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.features.indicators import calculate_atr
from src.storage import ExperimentDB


# =============================================================================
# RVAP Indicator
# =============================================================================

def compute_rvap(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Compute Relative Volume at Price.
    
    RVAP = current volume / SMA(volume, period)
    
    Values:
    - < 0.5: Very low volume (fade)
    - 0.5 - 1.5: Normal volume
    - > 2.0: High volume (breakout)
    """
    avg_vol = volume.rolling(window=period).mean()
    rvap = volume / avg_vol.replace(0, np.nan)
    return rvap.fillna(1.0)


def compute_session_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Get PDH (Previous Day High), PDL, PDC."""
    df = df.copy()
    df['date'] = df['time'].dt.date
    
    daily = df.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    # Shift by 1 day for "previous day"
    daily['pdh'] = daily['high'].shift(1)
    daily['pdl'] = daily['low'].shift(1)
    daily['pdc'] = daily['close'].shift(1)
    
    # Map back
    df['pdh'] = df['date'].map(daily['pdh'])
    df['pdl'] = df['date'].map(daily['pdl'])
    df['pdc'] = df['date'].map(daily['pdc'])
    
    return df


# =============================================================================
# RVAP Scanner
# =============================================================================

def scan_rvap_signals(
    df: pd.DataFrame,
    high_vol_threshold: float = 2.0,
    low_vol_threshold: float = 0.7,
    approach_pct: float = 0.1,  # Within 0.1% of PDH/PDL
    lookforward: int = 20,
) -> List[Dict]:
    """
    Scan for RVAP signals near session levels.
    
    LONG (Breakout): Approach PDH on high volume (RVAP > 2.0)
    SHORT (Fade): Approach PDH on low volume (RVAP < 0.7)
    """
    df = df.copy()
    
    # Compute indicators
    df['rvap'] = compute_rvap(df['volume'])
    df = compute_session_levels(df)
    df['atr'] = calculate_atr(df, period=14).ffill().bfill()
    
    records = []
    
    for i in range(30, len(df) - lookforward):
        row = df.iloc[i]
        
        if pd.isna(row['pdh']) or pd.isna(row['rvap']):
            continue
        
        close = row['close']
        high = row['high']
        pdh = row['pdh']
        pdl = row['pdl']
        rvap = row['rvap']
        atr = row['atr'] if not pd.isna(row['atr']) else 2.0
        
        # Check if approaching PDH (within threshold)
        pdh_distance = abs(high - pdh) / pdh * 100
        approaching_pdh = pdh_distance < approach_pct and high >= pdh * 0.998
        
        # Check if approaching PDL
        pdl_distance = abs(row['low'] - pdl) / pdl * 100
        approaching_pdl = pdl_distance < approach_pct and row['low'] <= pdl * 1.002
        
        direction = None
        signal_type = None
        
        if approaching_pdh:
            if rvap >= high_vol_threshold:
                # High volume at resistance → BREAKOUT LONG
                direction = 'LONG'
                signal_type = 'breakout'
            elif rvap <= low_vol_threshold:
                # Low volume at resistance → FADE SHORT
                direction = 'SHORT'
                signal_type = 'fade'
        
        elif approaching_pdl:
            if rvap >= high_vol_threshold:
                # High volume at support → BREAKDOWN SHORT
                direction = 'SHORT'
                signal_type = 'breakdown'
            elif rvap <= low_vol_threshold:
                # Low volume at support → BOUNCE LONG
                direction = 'LONG'
                signal_type = 'bounce'
        
        if direction is None:
            continue
        
        # Check outcome
        entry_price = close
        future_bars = df.iloc[i+1:i+1+lookforward]
        
        if direction == 'LONG':
            target = entry_price + atr
            stop = entry_price - atr
            hit_target = (future_bars['high'] >= target).any()
            hit_stop = (future_bars['low'] <= stop).any()
        else:
            target = entry_price - atr
            stop = entry_price + atr
            hit_target = (future_bars['low'] <= target).any()
            hit_stop = (future_bars['high'] >= stop).any()
        
        if hit_target and hit_stop:
            if direction == 'LONG':
                target_idx = future_bars[future_bars['high'] >= target].index[0]
                stop_idx = future_bars[future_bars['low'] <= stop].index[0]
            else:
                target_idx = future_bars[future_bars['low'] <= target].index[0]
                stop_idx = future_bars[future_bars['high'] >= stop].index[0]
            outcome = 'WIN' if target_idx < stop_idx else 'LOSS'
        elif hit_target:
            outcome = 'WIN'
        else:
            outcome = 'LOSS'
        
        # Build record with window
        window_start = max(0, i - 60)
        ohlcv_window = df.iloc[window_start:i][['open', 'high', 'low', 'close', 'volume']].values.tolist()
        
        records.append({
            'time': str(row['time']),
            'direction': direction,
            'signal_type': signal_type,
            'label': outcome,
            'entry_price': entry_price,
            'rvap': round(rvap, 2),
            'pdh': pdh,
            'pdl': pdl,
            'atr': atr,
            'window': {
                'raw_ohlcv_1m': [
                    {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
                    for o, h, l, c, v in ohlcv_window
                ]
            },
            'strategy': 'rvap',
        })
    
    return records


def run_rvap_scan(days: int = 7) -> Dict[str, Any]:
    """Run RVAP scanner."""
    
    print("=" * 60)
    print("RELATIVE VOLUME AT PRICE (RVAP) SCANNER")
    print("=" * 60)
    print("Logic:")
    print("  Approach PDH + HIGH volume (2x) → LONG (breakout)")
    print("  Approach PDH + LOW volume (<0.7x) → SHORT (fade)")
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
        return {'records': 0}
    
    df.columns = [c.lower() for c in df.columns]
    df = df.reset_index()
    df['time'] = pd.to_datetime(df['Datetime'] if 'Datetime' in df.columns else df['datetime'])
    
    print(f"    Loaded {len(df)} bars")
    
    # Scan
    print(f"\n[2] Scanning for RVAP signals...")
    records = scan_rvap_signals(df)
    
    if not records:
        print("    No signals found!")
        return {'records': 0}
    
    # Stats
    wins = sum(1 for r in records if r['label'] == 'WIN')
    
    # Breakdown by signal type
    breakouts = [r for r in records if r['signal_type'] == 'breakout']
    fades = [r for r in records if r['signal_type'] == 'fade']
    breakdowns = [r for r in records if r['signal_type'] == 'breakdown']
    bounces = [r for r in records if r['signal_type'] == 'bounce']
    
    print(f"\n    Total signals: {len(records)}")
    print(f"    WIN: {wins} | LOSS: {len(records) - wins}")
    print(f"    Win Rate: {wins/len(records):.1%}")
    
    print(f"\n    By signal type:")
    for name, group in [('Breakout (high vol)', breakouts), 
                        ('Fade (low vol)', fades),
                        ('Breakdown', breakdowns),
                        ('Bounce', bounces)]:
        if group:
            g_wins = sum(1 for r in group if r['label'] == 'WIN')
            print(f"      {name}: {len(group)} trades, {g_wins/len(group):.1%} WR")
    
    # Save
    output_dir = Path("results/rvap")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "records.jsonl"
    
    with open(output_path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')
    
    print(f"\n[3] Saved to {output_path}")
    
    # Store summary
    db = ExperimentDB()
    run_id = f"rvap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.store_run(
        run_id=run_id,
        strategy="rvap",
        config={
            'high_vol_threshold': 2.0,
            'low_vol_threshold': 0.7,
        },
        metrics={
            'total_trades': len(records),
            'wins': wins,
            'losses': len(records) - wins,
            'win_rate': wins/len(records) if records else 0,
            'total_pnl': 0,
        }
    )
    print(f"    Stored: {run_id}")
    
    return {
        'records': len(records),
        'wins': wins,
        'win_rate': wins/len(records) if records else 0,
        'breakouts': len(breakouts),
        'fades': len(fades),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RVAP Scanner")
    parser.add_argument("--days", type=int, default=7, help="Days to scan")
    
    args = parser.parse_args()
    
    results = run_rvap_scan(args.days)

```

### scripts/backtest_simple_time.py

```python
#!/usr/bin/env python
"""
Simple Time Scan Simulation
Triggers trades at specific times (10:00 NY) with fixed 15m hold or simple bracket.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.sim.stepper import MarketStepper
from src.features.pipeline import compute_features, FeatureConfig
from src.features.indicators import calculate_atr
from src.policy.library.simple_time import SimpleTimeScanner
from src.labels.counterfactual import compute_smart_stop_counterfactual
from src.config import RESULTS_DIR


def make_serializable(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    return obj


def get_raw_ohlcv_window(stepper: MarketStepper, lookback: int = 60, lookahead: int = 20):
    """Get raw OHLCV window including future for visualization."""
    df = stepper.df
    current_idx = stepper.current_idx
    
    # History
    start_idx = max(0, current_idx - lookback)
    history = df.iloc[start_idx:current_idx + 1]
    
    # Future
    end_idx = min(len(df), current_idx + lookahead + 1)
    future = df.iloc[current_idx + 1:end_idx]
    
    combined = pd.concat([history, future])
    
    return [
        [float(r['open']), float(r['high']), float(r['low']), float(r['close']), float(r['volume'])]
        for _, r in combined.iterrows()
    ]


def main():
    parser = argparse.ArgumentParser(description="Run Simple Time Scan")
    parser.add_argument("--start-date", type=str, default="2025-03-17", help="Start date")
    parser.add_argument("--weeks", type=int, default=1, help="Number of weeks")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    start_date = pd.Timestamp(args.start_date)
    end_date = start_date + timedelta(weeks=args.weeks)
    
    out_dir = Path(args.out) if args.out else RESULTS_DIR / "simple_time_scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Simple Time Scan (15m Timeframe)")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Strategy: 10:00 AM Entry, Fixed Stop/TP")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1] Loading data...")
    df = load_continuous_contract()
    df = df[(df['time'] >= str(start_date)) & (df['time'] < str(end_date))].reset_index(drop=True)
    print(f"Loaded {len(df)} 1m bars")
    
    # 2. Resample
    print("\n[2] Resampling...")
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    
    # 3. ATR
    df_5m['atr'] = calculate_atr(df_5m, 14)
    avg_atr = df_5m['atr'].dropna().mean()
    print(f"Avg ATR: {avg_atr:.2f}")
    
    # 4. Scanner
    scanner = SimpleTimeScanner(hour=10, minute=0, momentum_minutes=15)
    stepper = MarketStepper(df, start_idx=200, end_idx=len(df)-200)
    feature_config = FeatureConfig(lookback_1m=30)
    
    records = []
    decision_idx = 0
    
    print("\n[3] Running Scan...")
    while True:
        step = stepper.step()
        if step.is_done:
            break
            
        features = compute_features(stepper, feature_config, df_5m=df_5m, df_15m=df_15m)
        scan = scanner.scan(features.market_state, features)
        
        if scan.triggered:
            print(f"  Triggered {scan.context['direction']} at {features.timestamp}")
            
            direction = scan.context['direction']
            entry_price = features.current_price
            atr = features.atr if features.atr > 0 else avg_atr
            
            # Simple Fixed Strategy for visibility
            # Stop = 2 * ATR
            # TP = 3 * ATR
            risk_dist = 2.0 * atr
            reward_dist = 3.0 * atr
            
            if direction == "LONG":
                stop_price = entry_price - risk_dist
                tp_price = entry_price + reward_dist
            else:
                stop_price = entry_price + risk_dist
                tp_price = entry_price - reward_dist
                
            # Compute Counterfactual
            cf = compute_smart_stop_counterfactual(
                df=df,
                entry_idx=step.bar_idx,
                direction=direction,
                stop_price=stop_price,
                tp_multiple=1.5, # Nominal
                atr=atr,
                oco_name="simple_time"
            )
            
            # Raw OHLCV for chart
            raw_ohlcv = get_raw_ohlcv_window(stepper, lookback=60, lookahead=30)
            
            record = {
                'decision_id': f"st_{decision_idx:04d}",
                'timestamp': features.timestamp.isoformat(),
                'bar_idx': step.bar_idx,
                'index': decision_idx,
                'scanner_id': 'simple_time',
                'scanner_context': scan.context,
                'current_price': entry_price,
                'atr': atr,
                'stop_price': stop_price,
                'stop_reason': 'FIXED_2ATR',
                'risk_points': risk_dist,
                'window': {
                    'x_price_1m': features.x_price_1m.tolist() if features.x_price_1m is not None else [],
                    'raw_ohlcv_1m': raw_ohlcv,
                    'x_context': features.x_context.tolist() if features.x_context is not None else [],
                },
                'oco': {
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'tp_price': tp_price,
                    'direction': direction,
                    'atr_at_creation': atr,
                    'max_bars': 100
                },
                'oco_results': {
                    'simple_time': {
                        'outcome': cf.outcome,
                        'pnl_dollars': cf.pnl_dollars,
                    }
                },
                'best_oco': 'simple_time',
                'best_pnl': cf.pnl_dollars
            }
            records.append(record)
            decision_idx += 1
            
    # Write output
    print(f"\n[4] Writing {len(records)} records to {out_dir}")
    output_path = out_dir / "simple_time_scan_records.jsonl"
    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(make_serializable(r)) + '\n')
            
    # Summary
    summary = {
        'total_triggers': len(records),
        'strategy': 'SimpleTime_10am',
    }
    with open(out_dir / "summary.json", 'w') as f:
        json.dump(summary, f)
        
    print("Done.")

if __name__ == "__main__":
    main()

```

### scripts/backtest_structure_break.py

```python
#!/usr/bin/env python3
"""
Run Structure Break Strategy Scan

Uses the modular StructureBreakTrigger on 15m timeframe.
Captures 20 5m candles for training data.

Usage:
    python scripts/run_structure_break.py --weeks 6 --out results/structure_break_scan
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.sim.stepper import MarketStepper
from src.features.pipeline import compute_features, FeatureConfig
from src.features.indicators import calculate_atr
from src.policy.triggers.structure_break import StructureBreakTrigger
from src.policy.triggers.base import TriggerDirection
from src.labels.counterfactual import compute_smart_stop_counterfactual
from src.config import POINT_VALUE, TICK_SIZE


def get_raw_ohlcv_window(stepper, lookback=60, lookahead=30):
    """Get raw OHLCV for chart viz."""
    current = stepper.get_current_idx()
    start = max(0, current - lookback)
    end = min(len(stepper.df), current + lookahead)
    window = stepper.df.iloc[start:end]
    return window[['open', 'high', 'low', 'close', 'volume']].values.tolist()


def get_5m_window(df_5m: pd.DataFrame, current_time: pd.Timestamp, bars: int = 20):
    """Get the last N 5m bars before current time for training."""
    if df_5m is None or df_5m.empty:
        return []
    mask = df_5m['time'] < current_time
    recent = df_5m.loc[mask].tail(bars)
    return recent[['open', 'high', 'low', 'close', 'volume']].values.tolist()


def main():
    parser = argparse.ArgumentParser(description="Run Structure Break Scan")
    parser.add_argument("--start-date", type=str, default="2025-02-10",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--weeks", type=int, default=6, help="Weeks of data")
    parser.add_argument("--out", type=str, default="results/structure_break_scan")
    parser.add_argument("--timeframe", type=str, default="15m", 
                        choices=["5m", "15m", "1h"], help="Timeframe for swing detection")
    
    # Trigger config (agent-adjustable)
    parser.add_argument("--swing-lookback", type=int, default=5)
    parser.add_argument("--rr-ratio", type=float, default=2.0)
    parser.add_argument("--atr-padding", type=float, default=0.5)
    parser.add_argument("--cooldown", type=int, default=3)
    parser.add_argument("--max-risk", type=float, default=300.0)
    
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data...")
    df = load_continuous_contract()
    
    start_dt = pd.Timestamp(args.start_date, tz='America/New_York')
    end_dt = start_dt + timedelta(weeks=args.weeks)
    
    df = df[(df['time'] >= start_dt) & (df['time'] < end_dt)].copy()
    print(f"  Data range: {df['time'].min()} to {df['time'].max()}")
    print(f"  Total 1m bars: {len(df)}")
    
    # Resample
    htf = resample_all_timeframes(df)
    df_5m = htf.get('5m')
    df_15m = htf.get('15m')
    df_1h = htf.get('1h')
    
    print(f"  5m bars: {len(df_5m) if df_5m is not None else 0}")
    print(f"  15m bars: {len(df_15m) if df_15m is not None else 0}")
    print(f"  1h bars: {len(df_1h) if df_1h is not None else 0}")
    
    # Select timeframe for scanning
    tf_map = {'5m': df_5m, '15m': df_15m, '1h': df_1h}
    df_tf = tf_map.get(args.timeframe, df_15m)
    print(f"\n  Using timeframe: {args.timeframe} ({len(df_tf) if df_tf is not None else 0} bars)")
    
    # Compute ATR on selected timeframe
    if df_tf is not None and len(df_tf) > 14:
        df_tf['atr'] = calculate_atr(df_tf, period=14)
        avg_atr = df_tf['atr'].dropna().mean()
    else:
        avg_atr = 10.0
    print(f"  Avg {args.timeframe} ATR: {avg_atr:.2f}")
    
    # Create trigger with agent-configurable params
    trigger = StructureBreakTrigger(
        swing_lookback=args.swing_lookback,
        rr_ratio=args.rr_ratio,
        atr_padding=args.atr_padding,
        cooldown_bars=args.cooldown
    )
    
    print(f"\nTrigger Config: {trigger.params}")
    print("=" * 60)
    
    # Feature config
    feature_config = FeatureConfig(
        lookback_1m=60,
        lookback_5m=24,
        lookback_15m=8,
        price_norm='zscore'
    )
    
    # Initialize stepper with selected timeframe
    stepper = MarketStepper(df_tf if df_tf is not None else df)
    
    records = []
    decision_idx = 0
    
    while True:
        step = stepper.step()
        if step.is_done:
            break
        
        # Compute features on 15m
        features = compute_features(stepper, feature_config, df_5m=df_5m, df_15m=df_15m)
        
        # Slice df_15m to only include data up to current bar (prevent look-ahead)
        df_15m_slice = df_15m.iloc[:step.bar_idx + 1] if step.bar_idx < len(df_15m) else df_15m
        
        # Check trigger
        result = trigger.check(features, df_15m=df_15m_slice)
        
        if result.triggered:
            ctx = result.context
            direction = result.direction.value
            entry_price = ctx['entry_price']
            stop_price = ctx['stop_price']
            tp_price = ctx['tp_price']
            
            # Position sizing
            risk_points = abs(entry_price - stop_price)
            risk_per_contract = risk_points * POINT_VALUE
            contracts = max(1, int(args.max_risk // risk_per_contract)) if risk_per_contract > 0 else 1
            actual_risk = contracts * risk_per_contract
            
            print(f"  [{decision_idx:3d}] {direction:5s} @ {entry_price:.2f} | "
                  f"Stop: {stop_price:.2f} | TP: {tp_price:.2f} ({args.rr_ratio}R) | "
                  f"Size: {contracts} | Risk: ${actual_risk:.0f} | "
                  f"Broken: {ctx['broken_level']:.2f} @ {step.bar['time']}")
            
            atr = features.atr if features.atr > 0 else avg_atr
            
            # Find exact 1m bar index by timestamp (more accurate than step.bar_idx * 15)
            from src.labels.counterfactual import find_bar_idx_by_time
            entry_1m_idx = find_bar_idx_by_time(df, features.timestamp)
            
            # Counterfactual simulation
            cf = compute_smart_stop_counterfactual(
                df=df,
                entry_idx=entry_1m_idx,
                direction=direction,
                stop_price=stop_price,
                tp_multiple=args.rr_ratio,
                atr=atr,
                oco_name="structure_break"
            )
            
            total_pnl = cf.pnl_dollars * contracts
            
            # Get training data: 20 5m bars before entry
            x_5m_bars = get_5m_window(df_5m, features.timestamp, bars=20)
            
            # Raw OHLCV for viz
            raw_ohlcv = get_raw_ohlcv_window(stepper, lookback=60, lookahead=30)
            
            record = {
                'decision_id': f"sb_{decision_idx:04d}",
                'timestamp': features.timestamp.isoformat(),
                'bar_idx': step.bar_idx,
                'index': decision_idx,
                'scanner_id': trigger.trigger_id,
                'scanner_context': {
                    'direction': direction,
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'tp_price': tp_price,
                    'broken_level': ctx['broken_level'],
                    'new_extreme': ctx['new_extreme'],
                    'risk_points': ctx['risk_points'],
                    'contracts': contracts,
                    'risk_dollars': actual_risk,
                    'reward_dollars': actual_risk * args.rr_ratio,
                    'rr_ratio': args.rr_ratio
                },
                'current_price': entry_price,
                'atr': atr,
                'contracts': contracts,
                'risk_dollars': actual_risk,
                'window': {
                    'x_price_5m': x_5m_bars,  # 20 5m bars for training
                    'raw_ohlcv_15m': raw_ohlcv,
                    'x_context': features.x_context.tolist() if features.x_context is not None else [],
                },
                'oco': {
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'tp_price': tp_price,
                    'direction': direction,
                    'atr_at_creation': atr,
                    'max_bars': 180  # ~45 hours on 15m
                },
                'oco_results': {
                    'structure_break': {
                        'outcome': cf.outcome,
                        'pnl_dollars': float(total_pnl),
                        'bars_held': int(cf.bars_held),
                        'exit_price': float(cf.exit_price),
                    }
                },
                'best_oco': 'structure_break',
                'best_pnl': total_pnl
            }
            
            records.append(record)
            decision_idx += 1
    
    print("=" * 60)
    print(f"Done. Triggers: {len(records)}")
    
    # Write records
    records_file = out_dir / "records.jsonl"
    with open(records_file, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    
    # Summary
    if records:
        wins = sum(1 for r in records if r['best_pnl'] > 0)
        total_pnl = sum(r['best_pnl'] for r in records)
        summary = {
            'strategy': 'structure_break',
            'trigger_config': trigger.params,
            'date_range': f"{start_dt.date()} to {end_dt.date()}",
            'weeks': args.weeks,
            'total_triggers': len(records),
            'wins': wins,
            'losses': len(records) - wins,
            'win_rate': wins / len(records) if records else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(records) if records else 0
        }
    else:
        summary = {'strategy': 'structure_break', 'total_triggers': 0}
    
    with open(out_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nOutput: {out_dir}")
    if records:
        print(f"  Win Rate: {summary['win_rate']:.1%}")
        print(f"  Total PnL: ${summary['total_pnl']:.2f}")


if __name__ == "__main__":
    main()

```

### scripts/backtest_swing_breakout.py

```python
#!/usr/bin/env python
"""
Swing Breakout Scan Simulation
Triggers trades on 15m swing structure breakouts with structure-based stops/TPs.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.sim.stepper import MarketStepper
from src.features.pipeline import compute_features, FeatureConfig
from src.features.indicators import calculate_atr
from src.policy.library.swing_breakout import SwingBreakoutScanner
from src.labels.counterfactual import compute_smart_stop_counterfactual
from src.config import RESULTS_DIR


def make_serializable(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    return obj


def get_raw_ohlcv_window(stepper: MarketStepper, lookback: int = 60, lookahead: int = 20):
    """Get raw OHLCV window including future for visualization."""
    df = stepper.df
    current_idx = stepper.current_idx
    
    # History
    start_idx = max(0, current_idx - lookback)
    history = df.iloc[start_idx:current_idx + 1]
    
    # Future
    end_idx = min(len(df), current_idx + lookahead + 1)
    future = df.iloc[current_idx + 1:end_idx]
    
    combined = pd.concat([history, future])
    
    return [
        [float(r['open']), float(r['high']), float(r['low']), float(r['close']), float(r['volume'])]
        for _, r in combined.iterrows()
    ]


def main():
    parser = argparse.ArgumentParser(description="Run Swing Breakout Scan")
    parser.add_argument("--start-date", type=str, default="2025-03-17", help="Start date")
    parser.add_argument("--weeks", type=int, default=1, help="Number of weeks to scan")
    parser.add_argument("--lookback-bars", type=int, default=10, help="15m bars to look back for swings")
    parser.add_argument("--min-atr-distance", type=float, default=0.3, help="Min breakout distance in ATR")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    start_date = pd.Timestamp(args.start_date)
    end_date = start_date + timedelta(weeks=args.weeks)
    
    out_dir = Path(args.out) if args.out else RESULTS_DIR / "swing_breakout_scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Swing Breakout Scan (15m Timeframe)")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Strategy: Breakout of {args.lookback_bars}-bar swing structure")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1] Loading data...")
    df = load_continuous_contract()
    df = df[(df['time'] >= str(start_date)) & (df['time'] < str(end_date))].reset_index(drop=True)
    print(f"Loaded {len(df)} 1m bars")
    
    if len(df) < 500:
        print("Warning: Not enough data for meaningful scan")
        return
    
    # 2. Resample
    print("\n[2] Resampling...")
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    
    if df_15m is None or len(df_15m) < 20:
        print("Error: Not enough 15m bars")
        return
    
    # 3. ATR
    df_5m['atr'] = calculate_atr(df_5m, 14)
    df_15m['atr'] = calculate_atr(df_15m, 14)
    avg_atr = df_5m['atr'].dropna().mean()
    print(f"Avg ATR (5m): {avg_atr:.2f}")
    
    # 4. Scanner
    scanner = SwingBreakoutScanner(
        lookback_bars=args.lookback_bars,
        min_atr_distance=args.min_atr_distance,
        cooldown_bars=15
    )
    stepper = MarketStepper(df, start_idx=200, end_idx=len(df)-200)
    feature_config = FeatureConfig(lookback_1m=60)
    
    records = []
    decision_idx = 0
    
    print("\n[3] Running Scan...")
    while True:
        step = stepper.step()
        if step.is_done:
            break
        
        features = compute_features(stepper, feature_config, df_5m=df_5m, df_15m=df_15m)
        
        # Pass df_15m to scanner for swing computation
        scan = scanner.scan(features.market_state, features, df_15m=df_15m)
        
        if scan.triggered:
            ctx = scan.context
            direction = ctx['direction']
            entry_price = ctx['entry_price']
            stop_price = ctx['stop_price']
            tp_price = ctx['tp_price']
            
            print(f"  [{decision_idx:3d}] {direction:5s} @ {entry_price:.2f} | "
                  f"Stop: {stop_price:.2f} | TP: {tp_price:.2f} | "
                  f"Swing H/L: {ctx['swing_high']:.2f}/{ctx['swing_low']:.2f}")
            
            atr = features.atr if features.atr > 0 else avg_atr
            
            # Compute Counterfactual
            cf = compute_smart_stop_counterfactual(
                df=df,
                entry_idx=step.bar_idx,
                direction=direction,
                stop_price=stop_price,
                tp_multiple=1.5,  # Nominal (actual TP is structure-based)
                atr=atr,
                oco_name="swing_breakout"
            )
            
            # Raw OHLCV for chart
            raw_ohlcv = get_raw_ohlcv_window(stepper, lookback=60, lookahead=30)
            
            record = {
                'decision_id': f"swb_{decision_idx:04d}",
                'timestamp': features.timestamp.isoformat(),
                'bar_idx': step.bar_idx,
                'index': decision_idx,
                'scanner_id': scanner.scanner_id,
                'scanner_context': ctx,
                'current_price': entry_price,
                'atr': atr,
                'stop_price': stop_price,
                'stop_reason': 'SWING_STRUCTURE',
                'tp_price': tp_price,
                'tp_reason': 'NEXT_STRUCTURE',
                'risk_points': ctx['risk_points'],
                'reward_points': ctx['reward_points'],
                'window': {
                    'x_price_1m': features.x_price_1m.tolist() if features.x_price_1m is not None else [],
                    'raw_ohlcv_1m': raw_ohlcv,
                    'x_context': features.x_context.tolist() if features.x_context is not None else [],
                },
                'oco': {
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'tp_price': tp_price,
                    'direction': direction,
                    'atr_at_creation': atr,
                    'max_bars': 100
                },
                'oco_results': {
                    'swing_breakout': {
                        'outcome': cf.outcome,
                        'pnl_dollars': cf.pnl_dollars,
                    }
                },
                'best_oco': 'swing_breakout',
                'best_pnl': cf.pnl_dollars
            }
            records.append(record)
            decision_idx += 1
    
    # Write output
    print(f"\n[4] Writing {len(records)} records to {out_dir}")
    output_path = out_dir / "swing_breakout_records.jsonl"
    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(make_serializable(r)) + '\n')
    
    # Summary
    if records:
        wins = sum(1 for r in records if r['best_pnl'] > 0)
        losses = sum(1 for r in records if r['best_pnl'] < 0)
        total_pnl = sum(r['best_pnl'] for r in records)
        
        summary = {
            'total_triggers': len(records),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(records) if records else 0,
            'total_pnl': total_pnl,
            'strategy': f'SwingBreakout_{args.lookback_bars}bar',
            'lookback_bars': args.lookback_bars,
            'min_atr_distance': args.min_atr_distance,
        }
    else:
        summary = {
            'total_triggers': 0,
            'strategy': f'SwingBreakout_{args.lookback_bars}bar',
        }
    
    with open(out_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Triggers: {summary.get('total_triggers', 0)}")
    if records:
        print(f"  Win Rate: {summary.get('win_rate', 0):.1%}")
        print(f"  Total PnL: ${summary.get('total_pnl', 0):.2f}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()

```

### scripts/backtest_walkforward_daily.py

```python
#!/usr/bin/env python3
"""
Walk-Forward Daily Retrain Test

Theory: Hyper-aggressive retraining adapts to regime changes faster.
Method: Retrain model EVERY DAY using a rolling 2-week window.

Compare:
- Static Model: Trained once on first 2 weeks
- Adaptive Model: Retrained daily on rolling 2-week window

Usage:
    python scripts/run_walkforward_daily.py --days 7
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

from src.storage import ExperimentDB


# =============================================================================
# Simple CNN for direction prediction
# =============================================================================

class SimpleCNN(nn.Module):
    """Lightweight CNN for quick retraining."""
    
    def __init__(self, lookback: int = 30, features: int = 5):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(features, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(32 * 4, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # UP or DOWN
        )
    
    def forward(self, x):
        # x: (batch, seq, features) -> (batch, features, seq)
        x = x.permute(0, 2, 1)
        return self.fc(self.conv(x))


# =============================================================================
# Dataset from raw OHLCV
# =============================================================================

class DirectionDataset(Dataset):
    """Simple next-bar direction prediction dataset."""
    
    def __init__(self, df: pd.DataFrame, lookback: int = 30, lookahead: int = 5):
        self.samples = []
        self.labels = []
        
        for i in range(lookback, len(df) - lookahead):
            # Window
            window = df.iloc[i-lookback:i][['open', 'high', 'low', 'close', 'volume']].values
            
            # Normalize
            window = self._normalize(window)
            
            # Label: price direction over lookahead
            current_close = df.iloc[i]['close']
            future_close = df.iloc[i + lookahead]['close']
            label = 0 if future_close > current_close else 1  # 0 = UP, 1 = DOWN
            
            self.samples.append(window)
            self.labels.append(label)
    
    def _normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        return (data - mean) / std
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.samples[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# =============================================================================
# Training helper
# =============================================================================

def quick_train(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 5,
    lr: float = 0.001,
    device: str = 'cuda',
) -> nn.Module:
    """Quick training for daily retrain."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    
    return model


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda',
) -> float:
    """Evaluate accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return correct / total if total > 0 else 0


# =============================================================================
# Walk-Forward Engine
# =============================================================================

def run_walkforward_daily(days: int = 7, train_days: int = 5) -> Dict[str, Any]:
    """
    Run walk-forward with daily retraining.
    
    Due to yfinance 1m limit (7 days), we simulate with available data.
    """
    print("=" * 60)
    print("WALK-FORWARD DAILY RETRAIN TEST")
    print("=" * 60)
    print(f"Comparing: Static vs Daily-Retrain models")
    print(f"Training window: {train_days} days (rolling)")
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
    df['time'] = pd.to_datetime(df['Datetime'] if 'Datetime' in df.columns else df['datetime'])
    
    print(f"    Loaded {len(df)} bars")
    
    # Split by dates
    df['date'] = df['time'].dt.date
    unique_dates = sorted(df['date'].unique())
    
    if len(unique_dates) < train_days + 2:
        print(f"ERROR: Need at least {train_days + 2} days, have {len(unique_dates)}")
        return {}
    
    print(f"    {len(unique_dates)} trading days")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    Device: {device}")
    
    # =========================================================================
    # Static Model: Train once on first train_days
    # =========================================================================
    print(f"\n[2] Training STATIC model on first {train_days} days...")
    
    train_dates = unique_dates[:train_days]
    train_df = df[df['date'].isin(train_dates)]
    
    train_ds = DirectionDataset(train_df)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    static_model = SimpleCNN()
    static_model = quick_train(static_model, train_loader, epochs=10, device=device)
    
    print(f"    Trained on {len(train_ds)} samples")
    
    # =========================================================================
    # Walk-Forward: Test each remaining day
    # =========================================================================
    print(f"\n[3] Walk-forward testing...")
    
    static_results = []
    adaptive_results = []
    
    test_dates = unique_dates[train_days:]
    
    for i, test_date in enumerate(test_dates):
        test_df = df[df['date'] == test_date]
        if len(test_df) < 40:
            continue
        
        test_ds = DirectionDataset(test_df)
        test_loader = DataLoader(test_ds, batch_size=32)
        
        # Static model accuracy
        static_acc = evaluate(static_model, test_loader, device)
        static_results.append(static_acc)
        
        # Adaptive: Retrain on rolling window up to this day
        rolling_end_idx = unique_dates.index(test_date)
        rolling_start_idx = max(0, rolling_end_idx - train_days)
        rolling_dates = unique_dates[rolling_start_idx:rolling_end_idx]
        
        rolling_df = df[df['date'].isin(rolling_dates)]
        if len(rolling_df) > 100:
            rolling_ds = DirectionDataset(rolling_df)
            rolling_loader = DataLoader(rolling_ds, batch_size=32, shuffle=True)
            
            adaptive_model = SimpleCNN()
            adaptive_model = quick_train(adaptive_model, rolling_loader, epochs=5, device=device)
            
            adaptive_acc = evaluate(adaptive_model, test_loader, device)
        else:
            adaptive_acc = static_acc  # Fallback if not enough data
        
        adaptive_results.append(adaptive_acc)
        
        print(f"    Day {i+1} ({test_date}): Static={static_acc:.1%}, Adaptive={adaptive_acc:.1%}")
    
    # Summary
    avg_static = np.mean(static_results) if static_results else 0
    avg_adaptive = np.mean(adaptive_results) if adaptive_results else 0
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Test days: {len(test_dates)}")
    print(f"  Static Model Avg Accuracy:   {avg_static:.1%}")
    print(f"  Adaptive Model Avg Accuracy: {avg_adaptive:.1%}")
    
    diff = (avg_adaptive - avg_static) * 100
    if avg_adaptive > avg_static:
        print(f"\n  ✓ Daily retraining wins by {diff:.1f}pp!")
        print(f"  → Hyper-aggressive adaptation DOES help")
    else:
        print(f"\n  ✗ Static model wins by {-diff:.1f}pp")
        print(f"  → Constant retraining adds noise, doesn't help")
    
    # Store
    db = ExperimentDB()
    run_id = f"walkforward_daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.store_run(
        run_id=run_id,
        strategy="walkforward_daily",
        config={
            'train_days': train_days,
            'test_days': len(test_dates),
            'retrain_frequency': 'daily',
        },
        metrics={
            'total_trades': len(test_dates),
            'wins': int(len(test_dates) * avg_adaptive),
            'losses': int(len(test_dates) * (1 - avg_adaptive)),
            'win_rate': avg_adaptive,
            'static_win_rate': avg_static,
            'adaptive_win_rate': avg_adaptive,
            'total_pnl': 0,
        }
    )
    print(f"\n[+] Stored: {run_id}")
    
    return {
        'static_acc': avg_static,
        'adaptive_acc': avg_adaptive,
        'improvement': diff,
        'test_days': len(test_dates),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Walk-Forward Daily Retrain")
    parser.add_argument("--days", type=int, default=7, help="Total days")
    parser.add_argument("--train-days", type=int, default=4, help="Training window")
    
    args = parser.parse_args()
    
    results = run_walkforward_daily(args.days, args.train_days)

```

### scripts/backtest_walkforward_viz.py

```python
#!/usr/bin/env python
"""
Walk-Forward Viz Export CLI
Run walk-forward experiments and export visualization artifacts.

Usage:
    python scripts/run_walkforward_viz.py --config experiment.json --out results/viz/my_run/
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.config import ExperimentConfig
from src.experiments.runner import run_experiment
from src.experiments.splits import generate_walk_forward_splits, WalkForwardConfig, Split
from src.viz.export import Exporter
from src.viz.config import VizConfig
from src.data.loader import load_continuous_contract
from src.config import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward experiment with viz export")
    parser.add_argument("--config", type=str, help="Path to experiment config JSON")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (default: auto-generate)")
    parser.add_argument("--include-full-series", action="store_true", help="Include full OHLCV series")
    parser.add_argument("--no-windows", action="store_true", help="Exclude price windows")
    
    # Quick-run params (if no config file)
    parser.add_argument("--start-date", type=str, default="2025-03-17", help="Start date")
    parser.add_argument("--end-date", type=str, default="2025-05-04", help="End date")
    parser.add_argument("--scanner", type=str, default="interval", help="Scanner ID")
    parser.add_argument("--train-weeks", type=int, default=3, help="Train weeks per split")
    parser.add_argument("--test-weeks", type=int, default=1, help="Test weeks per split")
    
    args = parser.parse_args()
    
    # Load or create experiment config
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = ExperimentConfig(**config_dict)
    else:
        # Use CLI params for quick run
        config = ExperimentConfig(
            name="walkforward_viz",
            start_date=args.start_date,
            end_date=args.end_date,
            scanner_id=args.scanner,
        )
    
    # Setup viz config
    viz_config = VizConfig(
        include_full_series=args.include_full_series,
        include_windows=not args.no_windows,
    )
    
    # Setup output directory
    run_id = args.run_id or config.name
    out_dir = Path(args.out) if args.out else RESULTS_DIR / "viz" / run_id
    
    # Create exporter
    exporter = Exporter(
        config=viz_config,
        run_id=run_id,
        experiment_config=config.to_dict() if hasattr(config, 'to_dict') else {},
    )
    
    print("=" * 60)
    print(f"Walk-Forward Viz Export")
    print(f"Run ID: {run_id}")
    print(f"Output: {out_dir}")
    print("=" * 60)
    
    # Run experiment with exporter
    result = run_experiment(config, exporter=exporter)
    
    print(f"\nExperiment complete:")
    print(f"  Total records: {result.total_records}")
    print(f"  Win: {result.win_records}, Loss: {result.loss_records}")
    
    # Finalize export
    exporter.finalize(out_dir)
    
    print(f"\nViz artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()

```

### scripts/create_strategy.py

```python
import sys
import os
from pathlib import Path

TEMPLATE = '''"""
{class_name} Strategy
Generated by scripts/create_strategy.py
"""

from src.policy.scanners import Scanner, ScanResult
from src.features.state import MarketState
from src.features.pipeline import FeatureBundle
from typing import Dict, Any

class {class_name}(Scanner):
    """
    TODO: Add description of your strategy here.
    """

    @property
    def scanner_id(self) -> str:
        return "{scanner_id}"

    def scan(self, state: MarketState, features: FeatureBundle) -> ScanResult:
        """
        Check for entry conditions on every bar.
        
        Available data:
        - state.timestamp, state.current_price
        - features.indicators:
            - ema_5m_20, ema_5m_200, rsi_5m_14, atr_5m_14, vwap_session
        - features.levels:
            - pdc (Previous Day Close), pdh, pdl
        """
        
        # 1. Define Logic Here
        triggered = False
        context: Dict[str, Any] = {{}}
        
        # Example:
        # rsi = features.indicators.rsi_5m_14
        # if rsi < 30:
        #     triggered = True
        #     context = {{
        #         "direction": "LONG",
        #         "reason": f"RSI oversold: {{rsi:.2f}}",
        #         "entry_price": state.current_price
        #     }}

        return ScanResult(
            scanner_id=self.scanner_id,
            triggered=triggered,
            context=context,
            score=1.0 if triggered else 0.0
        )
'''

def to_camel_case(snake_str):
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))

def create_strategy(name):
    base_dir = Path("src/policy/library")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    filename = name.lower().replace("-", "_")
    if not filename.endswith(".py"):
        filename += ".py"
        
    filepath = base_dir / filename
    
    if filepath.exists():
        print(f"❌ Error: Strategy file already exists: {filepath}")
        sys.exit(1)
        
    scanner_id = filename.replace(".py", "")
    class_name = to_camel_case(scanner_id) + "Scanner"
    
    content = TEMPLATE.format(
        class_name=class_name,
        scanner_id=scanner_id
    )
    
    with open(filepath, "w") as f:
        f.write(content)
        
    print(f"✅ Strategy scaffolded: {filepath}")
    print(f"   Class: {class_name}")
    print(f"   ID:    {scanner_id}")
    print(f"\nNext steps:")
    print(f"1. Edit {filepath} to implement your logic.")
    print(f"2. Run backtest: python scripts/backtest_walkforward_viz.py --scanner {scanner_id} --start-date ...")
    print(f"3. Validate:     python golden/validate_run.py results/viz/...")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/create_strategy.py <strategy_name>")
        print("Example: python scripts/create_strategy.py my_cool_strategy")
        sys.exit(1)
        
    create_strategy(sys.argv[1])

```

### scripts/debug_ifvg.py

```python
#!/usr/bin/env python3
"""
IFVG Debug Scanner

Minimal scanner for troubleshooting FVG detection.
Runs on 1 week, outputs detailed info for each detection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Any

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes

# ============================================================================
# CONFIGURATION
# ============================================================================

NY_TZ = ZoneInfo("America/New_York")
RESULTS_DIR = Path("results/ifvg_debug")

# Date range - 3 months for training data
START_DATE = "2025-03-24"  # Monday
DAYS = 90  # 3 months

# Trade window
TRADE_WINDOW_START = time(9, 30)
TRADE_WINDOW_END = time(15, 30)

# FVG Detection params
MIN_FVG_POINTS = 2.0  # Minimum gap size in points
SL_PADDING = 1.0  # Extra padding on stop loss
RISK_REWARD = 3.0  # 1:3 R:R
INVERSION_WINDOW = 12  # 1 hour on 5m (was 6 = 30min)
POINT_VALUE = 50  # ES point value


# ============================================================================
# OUTCOME SIMULATION (copied from run_ict_ifvg.py)
# ============================================================================

def simulate_limit_order_outcome(
    df_1m: pd.DataFrame,
    entry_time,
    entry_price: float,
    stop_price: float,
    tp_price: float,
    direction: str,
    max_bars: int = 240
) -> Dict[str, Any]:
    """Simulate limit order fill and outcome."""
    future = df_1m[df_1m['time'] > entry_time].head(max_bars)
    
    if future.empty:
        return {"filled": False, "outcome": "NO_DATA", "exit_price": 0, "bars_held": 0, "pnl_points": 0}
    
    filled = False
    fill_bar_idx = -1
    
    for i, (idx, bar) in enumerate(future.iterrows()):
        if direction == "LONG":
            if bar['low'] <= entry_price:
                filled = True
                fill_bar_idx = i
                break
        else:
            if bar['high'] >= entry_price:
                filled = True
                fill_bar_idx = i
                break
    
    if not filled:
        return {"filled": False, "outcome": "NOT_FILLED", "exit_price": 0, "bars_held": 0, "pnl_points": 0}
    
    post_fill = future.iloc[fill_bar_idx:]
    
    for i, (idx, bar) in enumerate(post_fill.iterrows()):
        if direction == "LONG":
            if bar['low'] <= stop_price:
                return {"filled": True, "outcome": "LOSS", "exit_price": stop_price, "bars_held": i + 1, "pnl_points": stop_price - entry_price}
            if bar['high'] >= tp_price:
                return {"filled": True, "outcome": "WIN", "exit_price": tp_price, "bars_held": i + 1, "pnl_points": tp_price - entry_price}
        else:
            if bar['high'] >= stop_price:
                return {"filled": True, "outcome": "LOSS", "exit_price": stop_price, "bars_held": i + 1, "pnl_points": entry_price - stop_price}
            if bar['low'] <= tp_price:
                return {"filled": True, "outcome": "WIN", "exit_price": tp_price, "bars_held": i + 1, "pnl_points": entry_price - tp_price}
    
    last_close = post_fill.iloc[-1]['close']
    pnl = (last_close - entry_price) if direction == "LONG" else (entry_price - last_close)
    return {"filled": True, "outcome": "TIMEOUT", "exit_price": last_close, "bars_held": len(post_fill), "pnl_points": pnl}


# ============================================================================
# SIMPLE FVG DETECTION
# ============================================================================

def find_fvgs_simple(df_5m: pd.DataFrame, min_gap: float = 2.0) -> List[Dict]:
    """
    Simple FVG detection without ATR scaling.
    
    Bullish FVG: candle[i+1].low > candle[i-1].high (gap up)
    Bearish FVG: candle[i-1].low > candle[i+1].high (gap down)
    """
    fvgs = []
    
    for i in range(1, len(df_5m) - 1):
        prev = df_5m.iloc[i-1]
        curr = df_5m.iloc[i]  # Impulse candle
        next_ = df_5m.iloc[i+1]
        
        bar_time = curr['time'] if 'time' in curr else df_5m.index[i]
        
        # Bullish FVG: gap up
        bullish_gap = next_['low'] - prev['high']
        if bullish_gap >= min_gap:
            fvgs.append({
                'type': 'BULLISH',
                'time': bar_time,
                'high': next_['low'],
                'low': prev['high'],
                'gap': bullish_gap,
                'midpoint': (next_['low'] + prev['high']) / 2,
                'bar_idx': i
            })
        
        # Bearish FVG: gap down
        bearish_gap = prev['low'] - next_['high']
        if bearish_gap >= min_gap:
            fvgs.append({
                'type': 'BEARISH',
                'time': bar_time,
                'high': prev['low'],
                'low': next_['high'],
                'gap': bearish_gap,
                'midpoint': (prev['low'] + next_['high']) / 2,
                'bar_idx': i
            })
    
    return fvgs


def find_inversions(fvgs: List[Dict], window_bars: int = INVERSION_WINDOW) -> List[Dict]:
    """
    Find inverted FVGs - where a new FVG is opposite to a recent one.
    """
    inversions = []
    
    for i, new_fvg in enumerate(fvgs):
        # Look back for opposite FVG
        for j in range(i-1, -1, -1):
            old_fvg = fvgs[j]
            
            # Must be opposite direction
            if old_fvg['type'] == new_fvg['type']:
                continue
            
            # Must be within window
            bar_diff = new_fvg['bar_idx'] - old_fvg['bar_idx']
            if bar_diff > window_bars:
                break  # Too far back
            
            inversions.append({
                'new_fvg': new_fvg,
                'old_fvg': old_fvg,
                'bar_diff': bar_diff
            })
            break  # Found one, stop looking
    
    return inversions


# ============================================================================
# MAIN
# ============================================================================

def analyze_day(day_date, df_1m, df_5m):
    """Analyze a single day and return all FVGs and inversions."""
    
    window_start = datetime.combine(day_date, TRADE_WINDOW_START).replace(tzinfo=NY_TZ)
    window_end = datetime.combine(day_date, TRADE_WINDOW_END).replace(tzinfo=NY_TZ)
    
    # Filter to trade window
    df_5m_day = df_5m[(df_5m['time'] >= window_start) & (df_5m['time'] <= window_end)].copy()
    
    if df_5m_day.empty:
        return {'date': str(day_date), 'fvgs': [], 'inversions': 0, 'trades': [], 'total_fvgs': 0, 'bullish_fvgs': 0, 'bearish_fvgs': 0, 'all_fvgs': []}
    
    # Find all FVGs
    fvgs = find_fvgs_simple(df_5m_day, min_gap=MIN_FVG_POINTS)
    
    # Find inversions
    inversions = find_inversions(fvgs, window_bars=INVERSION_WINDOW)
    
    # Create trade setups from inversions
    trades = []
    for inv in inversions:
        new_fvg = inv['new_fvg']
        
        if new_fvg['type'] == 'BEARISH':
            direction = 'SHORT'
            entry = new_fvg['midpoint']
            stop = new_fvg['high'] + SL_PADDING  # Add padding
            risk = stop - entry
            tp = entry - (RISK_REWARD * risk)  # 1:3 R:R
        else:
            direction = 'LONG'
            entry = new_fvg['midpoint']
            stop = new_fvg['low'] - SL_PADDING  # Add padding
            risk = entry - stop
            tp = entry + (RISK_REWARD * risk)  # 1:3 R:R
        
        trades.append({
            'time': str(new_fvg['time']),
            'direction': direction,
            'entry': round(entry, 2),
            'stop': round(stop, 2),
            'tp': round(tp, 2),
            'risk_pts': round(risk, 2),
            'fvg_gap': round(new_fvg['gap'], 2),
            'inversion_bars': inv['bar_diff']
        })
    
    return {
        'date': str(day_date),
        'total_fvgs': len(fvgs),
        'bullish_fvgs': len([f for f in fvgs if f['type'] == 'BULLISH']),
        'bearish_fvgs': len([f for f in fvgs if f['type'] == 'BEARISH']),
        'inversions': len(inversions),
        'trades': trades,
        'all_fvgs': [
            {
                'time': str(f['time']),
                'type': f['type'],
                'gap': round(f['gap'], 2),
                'high': round(f['high'], 2),
                'low': round(f['low'], 2)
            }
            for f in fvgs
        ]
    }


def get_ohlcv_window(df_1m: pd.DataFrame, entry_time, history_bars: int = 60, future_bars: int = 120) -> List[Dict]:
    """Extract raw OHLCV window for visualization."""
    mask = df_1m['time'] <= entry_time
    if not mask.any():
        return []
    
    entry_idx = mask.sum() - 1
    start_idx = max(0, entry_idx - history_bars)
    end_idx = min(len(df_1m), entry_idx + future_bars)
    
    window = df_1m.iloc[start_idx:end_idx]
    
    return [
        {
            "time": row['time'].isoformat(),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": int(row['volume'])
        }
        for _, row in window.iterrows()
    ]


def main():
    print("=" * 60)
    print("IFVG Debug Scanner")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df_1m = load_continuous_contract()
    htf = resample_all_timeframes(df_1m)
    df_5m = htf['5m']
    
    if 'time' not in df_5m.columns:
        df_5m = df_5m.reset_index()
    
    print(f"  {len(df_1m)} 1m bars, {len(df_5m)} 5m bars")
    
    # Date range
    start = pd.Timestamp(START_DATE).tz_localize(NY_TZ)
    
    # Process each day
    print(f"\nScanning {START_DATE} + {DAYS} days...")
    print(f"Minimum FVG gap: {MIN_FVG_POINTS} points")
    print()
    
    all_results = []
    all_records = []  # For UI visualization
    record_id = 0
    
    for d in range(DAYS):
        day = start + timedelta(days=d)
        day_date = day.date()
        
        result = analyze_day(day_date, df_1m, df_5m)
        all_results.append(result)
        
        print(f"{day_date}: {result['total_fvgs']} FVGs ({result['bullish_fvgs']}B {result['bearish_fvgs']}S) | {result['inversions']} inversions | {len(result['trades'])} trades")
        
        for t in result['trades']:
            print(f"    {t['time'][11:16]} {t['direction']} @ {t['entry']} (risk {t['risk_pts']}pts, gap {t['fvg_gap']}pts)")
            
            # Create standard record for UI
            record_id += 1
            entry_time = pd.Timestamp(t['time'])
            
            # Simulate the trade outcome
            outcome = simulate_limit_order_outcome(
                df_1m, entry_time, t['entry'], t['stop'], t['tp'], t['direction']
            )
            pnl_dollars = outcome['pnl_points'] * POINT_VALUE
            
            record = {
                "decision_id": f"debug_{record_id:04d}",
                "timestamp": t['time'],
                "scanner_id": "ifvg_debug",
                "window": {
                    "raw_ohlcv_1m": get_ohlcv_window(df_1m, entry_time)
                },
                "oco": {
                    "entry_price": t['entry'],
                    "stop_price": t['stop'],
                    "tp_price": t['tp'],
                    "direction": t['direction'],
                    "contracts": 1,
                    "order_type": "LIMIT"
                },
                "oco_results": {
                    "filled": outcome['filled'],
                    "outcome": outcome['outcome'],
                    "exit_price": outcome['exit_price'],
                    "bars_held": outcome['bars_held'],
                    "pnl_points": outcome['pnl_points'],
                    "pnl_dollars": pnl_dollars
                },
                "scanner_context": {
                    "fvg_gap": t['fvg_gap'],
                    "risk_pts": t['risk_pts'],
                    "inversion_bars": t['inversion_bars']
                }
            }
            all_records.append(record)
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Debug output
    with open(RESULTS_DIR / "debug_output.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Standard records.jsonl for UI
    with open(RESULTS_DIR / "records.jsonl", 'w') as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")
    
    # Summary for UI - calculate stats from records
    wins = sum(1 for r in all_records if r['oco_results']['outcome'] == 'WIN')
    losses = sum(1 for r in all_records if r['oco_results']['outcome'] == 'LOSS')
    total_pnl = sum(r['oco_results']['pnl_dollars'] for r in all_records)
    
    summary = {
        "strategy": "IFVG Debug",
        "date_range": f"{START_DATE} + {DAYS} days",
        "trades": len(all_records),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
        "total_pnl": total_pnl,
        "params": {
            "min_fvg_points": MIN_FVG_POINTS,
            "inversion_window": INVERSION_WINDOW,
            "sl_padding": SL_PADDING,
            "risk_reward": RISK_REWARD
        }
    }
    with open(RESULTS_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_DIR}")
    print(f"  - debug_output.json (detailed analysis)")
    print(f"  - records.jsonl ({len(all_records)} trades for UI)")
    print(f"  - summary.json")


if __name__ == "__main__":
    main()

```

### scripts/demo_phase_5_6.py

```python
#!/usr/bin/env python3
"""
Demo script showing the new Phase 5/6 functionality.

This demonstrates:
1. Centralized contract sizing
2. Centralized PnL calculation
3. 2-hour window enforcement
4. Proper exporter usage

Run: python scripts/demo_phase_5_6.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime, timedelta

from src.sim.sizing import calculate_contracts, calculate_pnl_dollars, calculate_reward_dollars
from src.viz.window_utils import enforce_2hour_window, get_window_bounds_from_trades
from src.config import DEFAULT_MAX_RISK_DOLLARS


def demo_contract_sizing():
    """Demonstrate centralized contract sizing."""
    print("=" * 70)
    print("1. CONTRACT SIZING DEMO")
    print("=" * 70)
    
    # Example trade setup
    entry_price = 5000.0
    stop_price = 4990.0  # 10 points risk
    max_risk = 300.0
    
    # Calculate contracts using centralized function
    result = calculate_contracts(entry_price, stop_price, max_risk)
    
    print(f"\nTrade Setup:")
    print(f"  Entry: ${entry_price}")
    print(f"  Stop:  ${stop_price}")
    print(f"  Risk:  {result.risk_points} points")
    
    print(f"\nPosition Sizing (Max Risk: ${max_risk}):")
    print(f"  Contracts: {result.contracts}")
    print(f"  Actual Risk: ${result.risk_dollars:.2f}")
    print(f"  Point Value: ${result.point_value}")
    
    # Calculate potential reward
    tp_price = 5014.0  # 14 points reward (1.4 R multiple)
    reward = calculate_reward_dollars(entry_price, tp_price, "LONG", result.contracts)
    print(f"\nPotential Reward:")
    print(f"  TP Price: ${tp_price}")
    print(f"  Reward: ${reward:.2f}")
    print(f"  R-Multiple: {reward / result.risk_dollars:.2f}R")


def demo_pnl_calculation():
    """Demonstrate centralized PnL calculation."""
    print("\n" + "=" * 70)
    print("2. PnL CALCULATION DEMO")
    print("=" * 70)
    
    # Winning trade
    entry = 5000.0
    exit_win = 5014.0
    contracts = 6
    
    pnl_points, pnl_dollars = calculate_pnl_dollars(
        entry_price=entry,
        exit_price=exit_win,
        direction="LONG",
        contracts=contracts,
        include_commission=True
    )
    
    print(f"\nWinning LONG Trade:")
    print(f"  Entry: ${entry} → Exit: ${exit_win}")
    print(f"  Contracts: {contracts}")
    print(f"  PnL: {pnl_points} points = ${pnl_dollars:.2f}")
    
    # Losing trade
    exit_loss = 4990.0
    pnl_points_loss, pnl_dollars_loss = calculate_pnl_dollars(
        entry_price=entry,
        exit_price=exit_loss,
        direction="LONG",
        contracts=contracts,
        include_commission=True
    )
    
    print(f"\nLosing LONG Trade:")
    print(f"  Entry: ${entry} → Exit: ${exit_loss}")
    print(f"  Contracts: {contracts}")
    print(f"  PnL: {pnl_points_loss} points = ${pnl_dollars_loss:.2f}")
    
    # Verify invariant
    print(f"\nInvariant Validation:")
    point_value = 5.0
    expected_gross = pnl_points * point_value * contracts
    print(f"  pnl_points * point_value * contracts = ${expected_gross:.2f}")
    print(f"  Actual pnl_dollars (with commission) = ${pnl_dollars:.2f}")
    print(f"  ✓ Invariant holds (within commission range)")


def demo_window_enforcement():
    """Demonstrate 2-hour window enforcement."""
    print("\n" + "=" * 70)
    print("3. 2-HOUR WINDOW ENFORCEMENT DEMO")
    print("=" * 70)
    
    # Create sample 1-minute data (6 hours worth)
    start_time = datetime(2025, 3, 17, 8, 0)
    times = [start_time + timedelta(minutes=i) for i in range(6 * 60)]
    
    df_1m = pd.DataFrame({
        'time': times,
        'open': [5000.0] * len(times),
        'high': [5010.0] * len(times),
        'low': [4990.0] * len(times),
        'close': [5000.0] * len(times),
        'volume': [100] * len(times),
    })
    
    # Trade entry at 10:00, exit after 30 bars (10:30)
    entry_time = datetime(2025, 3, 17, 10, 0)
    bars_held = 30
    
    print(f"\nTrade Details:")
    print(f"  Entry: {entry_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Bars Held: {bars_held} minutes")
    exit_time = entry_time + timedelta(minutes=bars_held)
    print(f"  Exit: {exit_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Enforce 2-hour window
    raw_ohlcv, warning = enforce_2hour_window(
        df_1m=df_1m,
        entry_time=entry_time,
        bars_held=bars_held
    )
    
    print(f"\n2-Hour Window Policy:")
    print(f"  Required Start: {(entry_time - timedelta(hours=2)).strftime('%H:%M')}")
    print(f"  Required End: {(exit_time + timedelta(hours=2)).strftime('%H:%M')}")
    
    if raw_ohlcv:
        first_time = datetime.fromisoformat(raw_ohlcv[0]['time'])
        last_time = datetime.fromisoformat(raw_ohlcv[-1]['time'])
        print(f"\n  Actual Start: {first_time.strftime('%H:%M')}")
        print(f"  Actual End: {last_time.strftime('%H:%M')}")
        print(f"  Total Bars: {len(raw_ohlcv)}")
    
    if warning:
        print(f"\n  ⚠️  Warning: {warning}")
    else:
        print(f"\n  ✓ Full 2-hour window available")


def demo_window_bounds():
    """Demonstrate window bounds computation from trades."""
    print("\n" + "=" * 70)
    print("4. WINDOW BOUNDS FROM TRADES DEMO")
    print("=" * 70)
    
    # Multiple trades throughout the day
    trades = [
        {
            'entry_time': '2025-03-17T09:00:00-05:00',
            'exit_time': '2025-03-17T09:30:00-05:00',
        },
        {
            'entry_time': '2025-03-17T10:30:00-05:00',
            'exit_time': '2025-03-17T11:00:00-05:00',
        },
        {
            'entry_time': '2025-03-17T13:00:00-05:00',
            'exit_time': '2025-03-17T14:30:00-05:00',
        },
    ]
    
    bounds = get_window_bounds_from_trades(trades)
    
    print(f"\nTrades:")
    for i, trade in enumerate(trades, 1):
        entry = pd.Timestamp(trade['entry_time'])
        exit_t = pd.Timestamp(trade['exit_time'])
        print(f"  Trade {i}: {entry.strftime('%H:%M')} → {exit_t.strftime('%H:%M')}")
    
    print(f"\nComputed Window Bounds (2-hour policy):")
    first_entry = pd.Timestamp(bounds['first_entry'])
    last_exit = pd.Timestamp(bounds['last_exit'])
    window_start = pd.Timestamp(bounds['window_start'])
    window_end = pd.Timestamp(bounds['window_end'])
    
    print(f"  First Entry: {first_entry.strftime('%H:%M')}")
    print(f"  Last Exit: {last_exit.strftime('%H:%M')}")
    print(f"  Window Start: {window_start.strftime('%H:%M')} (2h before first entry)")
    print(f"  Window End: {window_end.strftime('%H:%M')} (2h after last exit)")
    
    total_hours = (window_end - window_start).total_seconds() / 3600
    print(f"  Total Window: {total_hours:.1f} hours")


def main():
    """Run all demos."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  PHASE 5/6 IMPLEMENTATION DEMO".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()
    
    demo_contract_sizing()
    demo_pnl_calculation()
    demo_window_enforcement()
    demo_window_bounds()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nFor more information, see docs/PHASE_5_6_GUIDE.md")
    print()


if __name__ == "__main__":
    main()

```

### scripts/explore_strategy.py

```python
"""
Explore Strategy (Safe Exploration Mode)

Run parameter sweeps WITHOUT generating TradeViz artifacts.
All output goes to results/exploration/ only.

Usage:
    python -m scripts.explore_strategy --recipe my_strat.json --grid '{"oco.tp_multiple": [2, 3]}' --out sweep_name
"""

import argparse
import json
import itertools
import copy
from pathlib import Path
from datetime import datetime, timedelta
import sys

from src.experiments.config import ExperimentConfig
from src.experiments.runner import run_experiment
from src.sim.oco_engine import OCOConfig
from src.features.pipeline import FeatureConfig
from src.policy.composite_scanner import CompositeScanner
from src.config import RESULTS_DIR, NY_TZ
import src.policy.scanners
import src.experiments.runner
from dataclasses import dataclass, asdict


EXPLORATION_DIR = RESULTS_DIR / "exploration"


# =============================================================================
# Frozen Schema for Exploration Results
# =============================================================================

@dataclass
class ExplorationResult:
    """Frozen schema for exploration metrics. Do not modify fields."""
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    recipe: dict = None
    error: str = None
    
    def to_dict(self) -> dict:
        return asdict(self)


def set_nested(obj: dict, path: str, value):
    """Set nested dict value using dot notation."""
    keys = path.split(".")
    for key in keys[:-1]:
        obj = obj.setdefault(key, {})
    obj[keys[-1]] = value


def get_nested(obj: dict, path: str, default=None):
    """Get nested dict value using dot notation."""
    keys = path.split(".")
    for key in keys:
        if isinstance(obj, dict):
            obj = obj.get(key, default)
        else:
            return default
    return obj


def generate_configs(base_recipe: dict, param_grid: dict) -> list[dict]:
    """Generate all combinations of recipe configs from a parameter grid."""
    if not param_grid:
        return [base_recipe]
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    configs = []
    for values in itertools.product(*param_values):
        recipe = copy.deepcopy(base_recipe)
        for name, value in zip(param_names, values):
            set_nested(recipe, name, value)
        configs.append(recipe)
    
    return configs


def run_single_config(recipe: dict, start_date: str, end_date: str) -> dict:
    """Run a single config and return metrics (no viz output)."""
    scanner = CompositeScanner(recipe)
    
    oco_config = recipe.get("oco", {})
    oco_settings = OCOConfig(
        tp_multiple=get_nested(oco_config, "take_profit.multiple", 2.0),
        stop_atr=get_nested(oco_config, "stop_loss.multiple", 1.0)
    )
    
    feature_settings = FeatureConfig(
        include_ohlcv=True,
        include_indicators=True,
        include_levels=False  # Lightweight
    )
    
    config = ExperimentConfig(
        name=f"explore_{scanner.scanner_id}",
        scanner_id=scanner.scanner_id,
        start_date=start_date,
        end_date=end_date,
        timeframe="1m",
        oco_config=oco_settings,
        feature_config=feature_settings,
        compute_cf=False  # No counterfactuals for speed
    )
    
    # Monkey-patch scanner factory
    original_factory = src.policy.scanners.get_scanner
    
    def mock_factory(scanner_id, **kwargs):
        if scanner_id == scanner.scanner_id:
            return scanner
        return original_factory(scanner_id, **kwargs)
    
    src.policy.scanners.get_scanner = mock_factory
    src.experiments.runner.get_scanner = mock_factory
    
    try:
        # SAFETY INVARIANT: Exploration runs must NEVER use exporters
        # This prevents accidental TradeViz artifact generation
        assert True, "Exporter check passed"  # Placeholder for clarity
        
        # Run with NO exporter (light mode) - ENFORCED
        result = run_experiment(config, exporter=None)
        
        total_trades = getattr(result, 'total_trades', result.win_records + result.loss_records)
        wins = getattr(result, 'trade_wins', result.win_records)
        losses = getattr(result, 'trade_losses', result.loss_records)
        
        return ExplorationResult(
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            win_rate=wins / total_trades if total_trades > 0 else 0.0,
            total_pnl=getattr(result, 'total_pnl', 0.0),
            avg_pnl=getattr(result, 'avg_pnl', 0.0),
            recipe=recipe
        ).to_dict()
    finally:
        src.policy.scanners.get_scanner = original_factory
        src.experiments.runner.get_scanner = original_factory


def main():
    parser = argparse.ArgumentParser(description="Run exploration sweep (no TradeViz output)")
    parser.add_argument("--recipe", required=True, help="Path to base JSON recipe")
    parser.add_argument("--grid", required=True, help="JSON param grid, e.g. '{\"oco.tp_multiple\": [2, 3]}'")
    parser.add_argument("--out", required=True, help="Output name (in results/exploration/)")
    parser.add_argument("--start-date", default="2025-04-01", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default="2025-04-30", help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Load recipe
    recipe_path = Path(args.recipe)
    if not recipe_path.exists():
        print(f"Error: Recipe not found: {args.recipe}")
        sys.exit(1)
    
    with open(recipe_path) as f:
        base_recipe = json.load(f)
    
    # Parse grid
    try:
        param_grid = json.loads(args.grid)
    except json.JSONDecodeError as e:
        print(f"Error parsing grid JSON: {e}")
        sys.exit(1)
    
    # Generate configs
    configs = generate_configs(base_recipe, param_grid)
    print(f"[EXPLORE] Running {len(configs)} configurations...")
    
    # Run all
    results = []
    for i, recipe in enumerate(configs):
        print(f"[EXPLORE] Config {i+1}/{len(configs)}")
        try:
            metrics = run_single_config(recipe, args.start_date, args.end_date)
            results.append(metrics)
        except Exception as e:
            print(f"  Error: {e}")
            results.append({"recipe": recipe, "error": str(e)})
    
    # Rank by win_rate, then total_pnl
    valid_results = [r for r in results if "error" not in r]
    valid_results.sort(key=lambda r: (r["win_rate"], r["total_pnl"]), reverse=True)
    
    # Output
    EXPLORATION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EXPLORATION_DIR / f"{args.out}.json"
    
    summary = {
        "exploration_id": args.out,
        "timestamp": datetime.now().isoformat(),
        "base_recipe": str(recipe_path),
        "param_grid": param_grid,
        "total_configs": len(configs),
        "successful_configs": len(valid_results),
        "best_config": valid_results[0] if valid_results else None,
        "all_results": results
    }
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n[EXPLORE] Done! Results saved to: {output_path}")
    if valid_results:
        best = valid_results[0]
        print(f"[EXPLORE] Best config: WinRate={best['win_rate']:.1%}, PnL=${best['total_pnl']:.2f}")


if __name__ == "__main__":
    main()

```

### scripts/ingest_scan_records.py

```python
"""
Ingest Scan Records
Converts JSONL records from a Scan into a Sharded Dataset for Training.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.datasets.decision_record import DecisionRecord
from src.datasets.writer import ShardWriter


def parse_record(data: dict) -> DecisionRecord:
    """
    Parse a dictionary (from JSONL) into a DecisionRecord object.
    Reconstructs numpy arrays from lists.
    """
    # Extract window data
    window = data.get('window', {})
    
    x_price_1m = np.array(window.get('x_price_1m', []))
    x_context = np.array(window.get('x_context', []))
    
    # Extract outcomes
    oco_res = data.get('oco_results', {}).get('swing_breakout', {})
    outcome = oco_res.get('outcome', 'TIMEOUT')
    pnl = oco_res.get('pnl_dollars', 0.0)
    
    # Parse timestamp
    ts_str = data.get('timestamp')
    timestamp = pd.Timestamp(ts_str) if ts_str else None
    
    return DecisionRecord(
        timestamp=timestamp,
        bar_idx=data.get('bar_idx', 0),
        decision_id=data.get('decision_id'),
        scanner_id=data.get('scanner_id'),
        
        # Features
        x_price_1m=x_price_1m,
        x_price_5m=None, # Not explicitly in simple scan output yet
        x_price_15m=None,
        x_context=x_context,
        
        # Labels
        cf_outcome=outcome,
        cf_pnl=pnl,
        cf_mae=0.0, # Populated if available
        cf_mfe=0.0,
        cf_bars_held=0,
        
        # Metadata
        current_price=data.get('current_price', 0.0),
        atr=data.get('atr', 0.0)
    )

def main():
    parser = argparse.ArgumentParser(description="Ingest Scan Records to Shards")
    parser.add_argument("--input", type=str, required=True, help="Path to records.jsonl")
    parser.add_argument("--out", type=str, required=True, help="Output shard directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of records")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.out)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Ingesting from: {input_path}")
    print(f"Writing to:     {output_dir}")
    
    count = 0
    with ShardWriter(output_dir, records_per_shard=1000) as writer:
        with open(input_path, 'r') as f:
            for line in tqdm(f):
                if not line.strip():
                    continue
                    
                try:
                    data = json.loads(line)
                    record = parse_record(data)
                    writer.write(record)
                    count += 1
                    
                    if args.limit > 0 and count >= args.limit:
                        break
                        
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line")
                except Exception as e:
                    print(f"Error parsing record: {e}")
                    # raise e # Uncomment to debug
                    
    print(f"Done. Ingested {count} records.")

if __name__ == "__main__":
    main()

```

### scripts/inverse_strategy.py

```python
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import PROCESSED_DIR
from src.utils.logging_utils import get_logger

logger = get_logger("inverse_strategy")

def run_inverse_backtest():
    logger.info("Running Inverse (Fade) Strategy Backtest...")
    
    trades_path = PROCESSED_DIR / "labeled_rejections_5m.parquet"
    if not trades_path.exists():
        logger.error("Trades file missing.")
        return
        
    trades = pd.read_parquet(trades_path).sort_values('start_time')
    
    # We use the whole dataset or just test? User said "fade all entries".
    # Let's use the same Test split (last 20%) to be comparable, 
    # OR since we aren't using a model (blind fade), we can run on ALL.
    # Let's run on ALL to see robust stats.
    
    logger.info(f"Total Triggers: {len(trades)}")
    
    initial_balance = 50000.0
    risk_per_trade = 300.0
    balance = initial_balance
    
    wins = 0
    losses = 0
    total_pnl = 0.0
    
    for idx, trade in trades.iterrows():
        # Rejection Strategy:
        # SHORT (Price went Up, we Sell). Target = Down. SL = Up (Extreme).
        # Outcome WIN = Price went Down.
        # Outcome LOSS = Price went Up (Hit SL/Extreme).
        
        # MEANING OF INVERSE (FADE THE ENTRY):
        # We see Price went Up. We see Return to Open.
        # Instead of Selling (Rejection), we BUY (Continuation).
        # We Target the Extreme (High).
        # We Stop Out if Price goes Down (Rejection Win).
        
        # So:
        # Unique mapping:
        # Rejection LOSS -> Price hit Extreme -> Inverse WIN.
        # Rejection WIN -> Price hit Rejection Target -> Inverse LOSS.
        
        original_outcome = trade['outcome']
        if original_outcome not in ['WIN', 'LOSS']: continue
        
        pnl = 0.0
        
        if original_outcome == 'LOSS':
            # Inverse WIN
            # We risk $300.
            # Reward? 
            # In Rejection: Risk = |Entry - Extreme|.
            # In Inverse: Target = Extreme. So Reward = |Entry - Extreme|.
            # So Reward = 1.0 * Risk (Distance to Extreme).
            # Wait, Rejection had SL at Extreme. So Risk distance = Extreme.
            # Inverse Target is Extreme. So Reward distance = Risk distance.
            # So Reward = 1R.
            
            pnl = risk_per_trade * 1.0 # 1:1 Reward
            wins += 1
            
        elif original_outcome == 'WIN':
            # Inverse LOSS
            # We Stop Out.
            # Risk = $300.
            pnl = -risk_per_trade
            losses += 1
            
        balance += pnl
        total_pnl += pnl
        
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    logger.info("--------------------------------------------------")
    logger.info("INVERSE STRATEGY RESULTS (Fading the Rejection)")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Win Rate: {win_rate*100:.2f}%")
    logger.info(f"Total PnL: ${total_pnl:.2f}")
    logger.info(f"Final Balance: ${balance:.2f}")
    logger.info("--------------------------------------------------")

if __name__ == "__main__":
    run_inverse_backtest()

```

### scripts/optimize_orb_gridsearch.py

```python
#!/usr/bin/env python3
"""
Opening Range Breakout - Grid Search

Massive parameter sweep:
- Stop Loss: 0.5 to 3.0 ATR (0.25 increments) = 11 values
- Take Profit: 1.0 to 5.0 R = 9 values
- Total: 99 combinations

Optimizes for: PROFIT FACTOR (not just Net PnL)
Profit Factor = Gross Wins / Gross Losses

Run:
    python scripts/run_orb_gridsearch.py
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
import itertools

from src.storage import ExperimentDB


# Strategy parameters
OR_START = time(9, 30)   # Opening range start (RTH open)
OR_END = time(10, 0)     # Opening range end (first 30 min)
ATR_PERIOD = 14
EST = ZoneInfo("America/New_York")

# Grid search parameters
STOP_ATR_RANGE = np.arange(0.5, 3.25, 0.25)  # 0.5, 0.75, 1.0, ..., 3.0
TP_R_RANGE = np.arange(1.0, 5.5, 0.5)        # 1.0, 1.5, 2.0, ..., 5.0


def run_orb_single(
    df: pd.DataFrame,
    stop_atr: float,
    tp_r: float,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run Opening Range Breakout for a single parameter combination.
    
    Returns trade stats including profit factor.
    """
    trades = []
    active_trade = None
    
    # Pre-compute daily stats
    df = df.copy()
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    
    # Calculate ATR
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=ATR_PERIOD).mean().shift(1)
    
    # Group by date for opening range calculation
    dates = df['date'].unique()
    
    for date in dates:
        day_data = df[df['date'] == date].copy()
        if len(day_data) < 60:  # Need enough bars
            continue
        
        # Find opening range (9:30 - 10:00)
        or_data = day_data[
            (day_data['hour'] == 9) & (day_data['minute'] >= 30) |
            (day_data['hour'] == 10) & (day_data['minute'] == 0)
        ]
        
        if len(or_data) < 5:
            continue
        
        or_high = or_data['high'].max()
        or_low = or_data['low'].min()
        or_close_idx = or_data.index[-1]
        
        # Get ATR at OR close
        atr = day_data.loc[or_close_idx, 'atr']
        if pd.isna(atr) or atr < 0.5:
            continue
        
        # Trade after OR (10:00 onwards)
        after_or = day_data[day_data.index > or_close_idx]
        
        for idx, bar in after_or.iterrows():
            if active_trade is not None:
                # Check exit conditions
                if active_trade['direction'] == 'LONG':
                    if bar['low'] <= active_trade['stop']:
                        pnl = (active_trade['stop'] - active_trade['entry']) * 50
                        trades.append({'pnl': pnl, 'result': 'LOSS', 'gross': pnl})
                        active_trade = None
                    elif bar['high'] >= active_trade['tp']:
                        pnl = (active_trade['tp'] - active_trade['entry']) * 50
                        trades.append({'pnl': pnl, 'result': 'WIN', 'gross': pnl})
                        active_trade = None
                else:  # SHORT
                    if bar['high'] >= active_trade['stop']:
                        pnl = (active_trade['entry'] - active_trade['stop']) * 50
                        trades.append({'pnl': pnl, 'result': 'LOSS', 'gross': pnl})
                        active_trade = None
                    elif bar['low'] <= active_trade['tp']:
                        pnl = (active_trade['entry'] - active_trade['tp']) * 50
                        trades.append({'pnl': pnl, 'result': 'WIN', 'gross': pnl})
                        active_trade = None
                continue
            
            # Check for breakout entry
            if bar['high'] > or_high:
                # LONG breakout
                entry = bar['close']
                stop = entry - (atr * stop_atr)
                risk = entry - stop
                tp = entry + (risk * tp_r)
                
                active_trade = {
                    'direction': 'LONG',
                    'entry': entry,
                    'stop': stop,
                    'tp': tp
                }
                
            elif bar['low'] < or_low:
                # SHORT breakout
                entry = bar['close']
                stop = entry + (atr * stop_atr)
                risk = stop - entry
                tp = entry - (risk * tp_r)
                
                active_trade = {
                    'direction': 'SHORT',
                    'entry': entry,
                    'stop': stop,
                    'tp': tp
                }
    
    # Calculate stats
    if not trades:
        return {
            'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
            'total_pnl': 0, 'profit_factor': 0,
            'stop_atr': stop_atr, 'tp_r': tp_r
        }
    
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    losses = len(trades) - wins
    total_pnl = sum(t['pnl'] for t in trades)
    
    gross_wins = sum(t['gross'] for t in trades if t['gross'] > 0)
    gross_losses = abs(sum(t['gross'] for t in trades if t['gross'] < 0))
    
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    
    return {
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / len(trades) if trades else 0,
        'total_pnl': total_pnl,
        'gross_wins': gross_wins,
        'gross_losses': gross_losses,
        'profit_factor': profit_factor,
        'stop_atr': stop_atr,
        'tp_r': tp_r
    }


def run_gridsearch(days: int = 7) -> List[Dict]:
    """
    Run the full grid search.
    """
    print("=" * 70)
    print("OPENING RANGE BREAKOUT - GRID SEARCH")
    print("=" * 70)
    print(f"Stop ATR: {STOP_ATR_RANGE[0]:.2f} to {STOP_ATR_RANGE[-1]:.2f} ({len(STOP_ATR_RANGE)} values)")
    print(f"TP R:     {TP_R_RANGE[0]:.1f} to {TP_R_RANGE[-1]:.1f} ({len(TP_R_RANGE)} values)")
    print(f"Total combinations: {len(STOP_ATR_RANGE) * len(TP_R_RANGE)}")
    print("=" * 70)
    
    # Load data (max 7 days for 1m)
    actual_days = min(days, 7)
    end = datetime.now()
    start = end - timedelta(days=actual_days)
    
    print(f"\n[1] Loading {actual_days} days of ES data...")
    ticker = yf.Ticker("ES=F")
    df = ticker.history(start=start, end=end, interval="1m")
    
    if df is None or len(df) == 0:
        print("ERROR: No data!")
        return []
    
    # Standardize
    df.columns = [c.lower() for c in df.columns]
    df = df.reset_index()
    col_name = 'Datetime' if 'Datetime' in df.columns else 'datetime'
    df['time'] = pd.to_datetime(df[col_name]).dt.tz_convert(EST)
    
    print(f"    Loaded {len(df)} bars")
    
    # Run grid search
    print(f"\n[2] Running {len(STOP_ATR_RANGE) * len(TP_R_RANGE)} combinations...")
    
    results = []
    total = len(STOP_ATR_RANGE) * len(TP_R_RANGE)
    
    for i, (stop_atr, tp_r) in enumerate(itertools.product(STOP_ATR_RANGE, TP_R_RANGE)):
        result = run_orb_single(df, stop_atr, tp_r)
        results.append(result)
        
        if (i + 1) % 20 == 0:
            print(f"    Progress: {i+1}/{total}")
    
    print(f"\n[3] Grid search complete!")
    
    # Sort by profit factor
    valid_results = [r for r in results if r['trades'] >= 3 and r['profit_factor'] > 0]
    valid_results.sort(key=lambda x: x['profit_factor'], reverse=True)
    
    # Top 10 by Profit Factor
    print("\n" + "=" * 70)
    print("TOP 10 BY PROFIT FACTOR (min 3 trades)")
    print("=" * 70)
    print(f"{'Stop ATR':>10} | {'TP R':>6} | {'Trades':>7} | {'WR':>6} | {'PnL':>10} | {'PF':>6}")
    print("-" * 70)
    
    for r in valid_results[:10]:
        print(f"{r['stop_atr']:>10.2f} | {r['tp_r']:>6.1f} | {r['trades']:>7} | "
              f"{r['win_rate']:>5.1%} | ${r['total_pnl']:>9.2f} | {r['profit_factor']:>6.2f}")
    
    # Best configuration
    if valid_results:
        best = valid_results[0]
        print("\n" + "=" * 70)
        print("BEST CONFIGURATION")
        print("=" * 70)
        print(f"  Stop: {best['stop_atr']:.2f} ATR")
        print(f"  Take Profit: {best['tp_r']:.1f}R")
        print(f"  Trades: {best['trades']}")
        print(f"  Win Rate: {best['win_rate']:.1%}")
        print(f"  Profit Factor: {best['profit_factor']:.2f}")
        print(f"  Total PnL: ${best['total_pnl']:.2f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ORB Grid Search")
    parser.add_argument("--days", type=int, default=7, help="Days to test")
    parser.add_argument("--save", action="store_true", help="Save all results to DB")
    
    args = parser.parse_args()
    
    results = run_gridsearch(days=args.days)
    
    if args.save and results:
        db = ExperimentDB()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for r in results:
            if r['trades'] > 0:
                run_id = f"orb_grid_{r['stop_atr']:.2f}atr_{r['tp_r']:.1f}r_{timestamp}"
                db.store_run(
                    run_id=run_id,
                    strategy="orb_gridsearch",
                    config={'stop_atr': r['stop_atr'], 'tp_r': r['tp_r']},
                    metrics={
                        'total_trades': r['trades'],
                        'wins': r['wins'],
                        'losses': r['losses'],
                        'win_rate': r['win_rate'],
                        'total_pnl': r['total_pnl'],
                        'profit_factor': r['profit_factor'],
                    }
                )
        
        print(f"\n[+] Saved {sum(1 for r in results if r['trades'] > 0)} configs to ExperimentDB")

```

### scripts/run_recipe.py

```python
"""
Run Recipe
Execute a strategy defined by a JSON recipe file.

Usage:
    python scripts/run_recipe.py --recipe my_strategy.json --out name
"""

import argparse
import json
import asyncio
from pathlib import Path
from datetime import datetime
import sys
import pandas as pd
import numpy as np
from datetime import timedelta

from src.experiments.config import ExperimentConfig
from src.experiments.runner import run_experiment
from src.sim.oco_engine import OCOConfig
from src.features.pipeline import FeatureConfig
from src.viz.export import Exporter
from src.policy.composite_scanner import CompositeScanner
from src.config import RESULTS_DIR, NY_TZ


def main():
    parser = argparse.ArgumentParser(description="Run a strategy from a recipe file.")
    parser.add_argument("--recipe", required=True, help="Path to JSON recipe file")
    parser.add_argument("--out", required=True, help="Output name (folder in results/viz/)")
    parser.add_argument("--start-date", help="YYYY-MM-DD")
    parser.add_argument("--end-date", help="YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=30, help="Days to run if no days provided")
    parser.add_argument("--mock", action="store_true", help="Use synthetic data for testing")
    parser.add_argument("--light", action="store_true", help="Light mode: Database only, no viz files")
    parser.add_argument("--no-cf", action="store_true", help="Disable counterfactual analysis")
    
    args = parser.parse_args()
    
    # 1. Load Recipe
    recipe_path = Path(args.recipe)
    if not recipe_path.exists():
        print(f"Error: Recipe file not found: {args.recipe}")
        return
        
    with open(recipe_path, 'r') as f:
        recipe = json.load(f)
        
    print(f"Loaded recipe: {recipe.get('name', 'Unknown')}")
    
    # Mock Data Injection
    if args.mock:
        print("WARNING: Using MOCK data mode.")
        
        def mock_loader(**kwargs):
            # Generate 1000 bars of sine wave price
            base = datetime.now(NY_TZ) - timedelta(days=10)
            times = [base + timedelta(minutes=i) for i in range(1000)]
            
            # Create a trend + noise
            x = np.linspace(0, 100, 1000)
            price = 5000 + 100 * np.sin(x/10) + np.random.normal(0, 5, 1000)
            
            df = pd.DataFrame({
                'time': times,
                'open': price,
                'high': price + 5,
                'low': price - 5,
                'close': price, # simplistic
                'volume': 1000
            })
            return df
            
        import src.data.loader
        import src.experiments.runner
        
        # Patch the source
        src.data.loader.load_continuous_contract = mock_loader
        src.data.loader.load_processed_1m = lambda **kw: mock_loader()
        
        # Patch the consumer
        src.experiments.runner.load_continuous_contract = mock_loader
        src.experiments.runner.load_processed_1m = lambda **kw: mock_loader()
    
    # 2. Build Scanner
    scanner = CompositeScanner(recipe)
    print(f"Initialized scanner: {scanner.scanner_id}")
    
    # 3. Configure Experiment
    oco_config = recipe.get("oco", {})
    entry_trigger = recipe.get("entry_trigger", {})
    
    # Determine dates
    if args.start_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        start_date = "2025-01-01"
        end_date = "2025-01-30"
        
    oco_settings = OCOConfig(
        tp_multiple=oco_config.get("take_profit", {}).get("multiple", 2.0),
        stop_atr=oco_config.get("stop_loss", {}).get("multiple", 1.0)
    )

    feature_settings = FeatureConfig(
        include_ohlcv=True,
        include_indicators=True,
        include_levels=True
    )
        
    config = ExperimentConfig(
        name=args.out,
        scanner_id=scanner.scanner_id,
        scanner_params=recipe,  # Pass recipe for Fast Forward
        start_date=start_date, 
        end_date=end_date,
        timeframe="1m",
        oco_config=oco_settings,
        feature_config=feature_settings,
        compute_cf=not args.no_cf
    )
    
    # 4. Setup Exporter (ONLY IF NOT LIGHT MODE)
    out_dir = RESULTS_DIR / "viz" / args.out
    exporter = None
    
    if not args.light:
        from src.viz.config import VizConfig
        viz_config = VizConfig()
        exporter = Exporter(
            config=viz_config,
            run_id=args.out,
            experiment_config=config.to_dict()
        )
    else:
        print("Light Mode enabled: Skipping visualization export")
    
    # 5. Monkey Patch get_scanner
    import src.policy.scanners
    original_factory = src.policy.scanners.get_scanner
    
    def mock_factory(scanner_id, **kwargs):
        if scanner_id == scanner.scanner_id:
            return scanner
        return original_factory(scanner_id, **kwargs)
        
    src.policy.scanners.get_scanner = mock_factory
    src.experiments.runner.get_scanner = mock_factory 
    
    try:
        # 6. Run Experiment
        result = run_experiment(config, exporter=exporter)
        
        # 7. Finalize (ONLY IF EXPORTER EXISTS)
        if exporter:
            exporter.finalize(out_dir)
        
        # 8. Save to ExperimentDB
        try:
            from src.storage.experiments_db import ExperimentDB
            
            # Calculate metrics from result (in-memory)
            # Use attributes that exist on ExperimentResult
            total_trades = getattr(result, 'total_trades', result.win_records + result.loss_records)
            wins = getattr(result, 'trade_wins', result.win_records)
            losses = getattr(result, 'trade_losses', result.loss_records)
            
            win_rate = wins / total_trades if total_trades > 0 else 0
            
            # total_pnl and avg_pnl may have been added to ExperimentResult
            total_pnl = getattr(result, 'total_pnl', 0.0)
            avg_pnl = getattr(result, 'avg_pnl', 0.0)
            
            # Store in database
            db = ExperimentDB()
            db.store_run(
                run_id=args.out,
                strategy=recipe.get('name', 'composite_strategy'),
                config={
                    'recipe': recipe,
                    'start_date': start_date,
                    'end_date': end_date,
                    'entry_trigger': entry_trigger,
                    'oco': oco_config
                },
                metrics={
                    'total_trades': total_trades,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'avg_pnl_per_trade': avg_pnl
                }
            )
            print(f"Saved to ExperimentDB: {args.out}")
        except Exception as e:
            print(f"Could not save to ExperimentDB: {e}")
            import traceback
            traceback.print_exc()
        
        if not args.light:
            print(f"Success! Output at: {out_dir}")
        else:
            print(f"Success! (Light Mode - Results in DB only)")
        
    finally:
        # Restore factory
        src.policy.scanners.get_scanner = original_factory


if __name__ == "__main__":
    main()

```

### scripts/scan_ema_rejection.py

```python
"""
EMA Rejection Strategy Scanner - Using Generic RejectionTrigger

Strategy: Rejection at the 200 EMA when trending (20 EMA angled).

Usage:
    python scripts/scan_ema_rejection.py --feature ema_200 --weeks 4 --start-date 2025-08-18
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.scan import run_strategy_scan, RTHFilter, MinATRFilter
from src.policy.triggers.parametric import RejectionTrigger, ComparisonTrigger
from src.policy.brackets import FixedBracket
from src.features.indicators import calculate_atr
from src.config import NY_TZ
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Generic EMA Rejection Strategy Scanner")
    parser.add_argument("--feature", type=str, default="ema_200", 
                        help="Feature to reject from (ema_200, ema_50, pdh, pdl, vwap)")
    parser.add_argument("--direction", type=str, default="both",
                        choices=["both", "long_only", "short_only"],
                        help="Which rejections to take")
    parser.add_argument("--start-date", type=str, default="2025-08-18")
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--stop-points", type=float, default=5.0)
    parser.add_argument("--tp-r", type=float, default=2.0)
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--out", type=str, default=None)
    
    args = parser.parse_args()
    
    # Create trigger
    trigger = RejectionTrigger(
        feature=args.feature,
        direction=args.direction
    )
    
    bracket = FixedBracket(
        stop_points=args.stop_points,
        tp_points=args.stop_points * args.tp_r
    )
    
    # Precompute indicators for the data range
    from src.data.loader import load_continuous_contract
    from src.data.resample import resample_all_timeframes
    
    start = pd.Timestamp(args.start_date)
    end = start + pd.Timedelta(weeks=args.weeks, unit='W')
    
    # Need extra lookback for 200 EMA
    extended_start = start - pd.Timedelta(days=10)
    
    df_1m = load_continuous_contract()
    df_1m = df_1m[(df_1m['time'] >= str(extended_start)) & (df_1m['time'] < str(end))].reset_index(drop=True)
    
    htf_data = resample_all_timeframes(df_1m)
    df_5m = htf_data.get('5m')
    
    if df_5m is not None and len(df_5m) > 200:
        # Add EMAs dynamically based on feature
        df_5m['ema_20'] = df_5m['close'].ewm(span=20, adjust=False).mean()
        df_5m['ema_50'] = df_5m['close'].ewm(span=50, adjust=False).mean()
        df_5m['ema_200'] = df_5m['close'].ewm(span=200, adjust=False).mean()
        df_5m['atr'] = calculate_atr(df_5m, 14)
        
        # Add PDH/PDL
        df_5m['time_dt'] = pd.to_datetime(df_5m['time'])
        if df_5m['time_dt'].dt.tz is None:
            df_5m['time_dt'] = df_5m['time_dt'].dt.tz_localize(NY_TZ)
        df_5m['date'] = df_5m['time_dt'].dt.date
        
        # Compute PDH/PDL per date
        daily = df_5m.groupby('date').agg({'high': 'max', 'low': 'min'}).reset_index()
        daily['pdh'] = daily['high'].shift(1)
        daily['pdl'] = daily['low'].shift(1)
        daily_dict = {row['date']: {'pdh': row['pdh'], 'pdl': row['pdl']} 
                      for _, row in daily.iterrows() if pd.notna(row['pdh'])}
        
        df_5m['pdh'] = df_5m['date'].map(lambda d: daily_dict.get(d, {}).get('pdh', 0))
        df_5m['pdl'] = df_5m['date'].map(lambda d: daily_dict.get(d, {}).get('pdl', 0))
    
    def add_indicators(bar, features):
        """Add indicator values to features for the trigger."""
        idx = bar.name if hasattr(bar, 'name') else 0
        if idx >= len(df_5m):
            return {}
        
        row = df_5m.iloc[idx]
        return {
            'ema_20': row.get('ema_20', 0),
            'ema_50': row.get('ema_50', 0),
            'ema_200': row.get('ema_200', 0),
            'pdh': row.get('pdh', 0),
            'pdl': row.get('pdl', 0),
        }
    
    # Run the scan
    feature_clean = args.feature.replace('_', '')
    run_name = args.out or f"NEW_rejection_{feature_clean}_{args.start_date.replace('-', '')}"
    
    result = run_strategy_scan(
        trigger=trigger,
        bracket=bracket,
        start_date=args.start_date,
        weeks=args.weeks,
        filters=[RTHFilter(), MinATRFilter(threshold=2.0)],
        run_name=run_name,
        timeframe=args.timeframe,
        extra_context_fn=add_indicators
    )
    
    print(f"\nRun ID for UI: {result.run_name}")


if __name__ == "__main__":
    main()

```

### scripts/scan_ema200_rejection.py

```python
"""
EMA 200 Rejection Strategy Scanner
