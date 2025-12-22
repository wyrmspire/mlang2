# Git Diff Report

**Generated**: Mon, Dec 22, 2025 12:18:14 AM

**Local Branch**: move

**Comparing Against**: origin/move

---

## Uncommitted Changes (working directory)

### Modified/Staged Files

```
?? gitrdiff.md
```

### New Untracked Files

#### `gitrdiff.md`

```
```

---

## Commits Ahead (local changes not on remote)

```
803737f feat: Add Lab research page for agent strategy testing
f25fbe6 feat: Add simple EMA scanner for model training
5e039f7 feat: Add trade validation rails for simulation integrity
12e84a6 feat: Add settlement and session levels to indicators
```

## Commits Behind (remote changes not pulled)

```
```

---

## File Changes (YOUR UNPUSHED CHANGES)

```
 scripts/run_combined_strategy.py | 389 ++++++++++++++++++++++++++++++++++
 scripts/run_ema_scan.py          | 437 +++++++++++++++++++++++++++++++++++++++
 scripts/run_inverse_test.py      | 182 ++++++++++++++++
 scripts/run_lowvol_breakout.py   | 312 ++++++++++++++++++++++++++++
 scripts/run_lunch_fade.py        | 258 +++++++++++++++++++++++
 scripts/run_orb_gridsearch.py    | 297 ++++++++++++++++++++++++++
 scripts/run_rvap_scan.py         | 309 +++++++++++++++++++++++++++
 scripts/run_walkforward_daily.py | 322 +++++++++++++++++++++++++++++
 scripts/train_fusion_mtf.py      | 414 +++++++++++++++++++++++++++++++++++++
 scripts/train_lstm_compare.py    | 336 ++++++++++++++++++++++++++++++
 src/App.tsx                      |  42 +++-
 src/api/client.ts                |  18 ++
 src/components/LabPage.tsx       | 307 +++++++++++++++++++++++++++
 src/features/indicators.py       |  95 +++++++++
 src/server/main.py               | 142 +++++++++++++
 src/sim/validation.py            | 228 ++++++++++++++++++++
 16 files changed, 4086 insertions(+), 2 deletions(-)
```

---

## Full Diff of Your Unpushed Changes

Green (+) = lines you ADDED locally
Red (-) = lines you REMOVED locally

```diff
diff --git a/scripts/run_combined_strategy.py b/scripts/run_combined_strategy.py
new file mode 100644
index 0000000..e0a50da
--- /dev/null
+++ b/scripts/run_combined_strategy.py
@@ -0,0 +1,389 @@
+#!/usr/bin/env python3
+"""
+Combined Strategy: ORB + Mean Reversion
+
+Run multiple strategies in different time windows:
+- 9:30 - 10:30 AM: Opening Range Breakout
+- 2:00 - 4:00 PM: Mean Reversion
+
+Single account balance, combined equity curve.
+
+Usage:
+    python scripts/run_combined_strategy.py --days 7
+"""
+
+import sys
+from pathlib import Path
+sys.path.insert(0, str(Path(__file__).parent.parent))
+
+import numpy as np
+import pandas as pd
+import yfinance as yf
+from datetime import datetime, timedelta, time
+from typing import Dict, Any, List, Tuple
+from zoneinfo import ZoneInfo
+import json
+
+from src.features.indicators import calculate_atr, calculate_ema, calculate_rsi
+from src.storage import ExperimentDB
+
+
+# =============================================================================
+# Time Windows (EST)
+# =============================================================================
+
+EST = ZoneInfo("America/New_York")
+
+ORB_START = time(9, 30)
+ORB_END = time(10, 30)
+ORB_TRADE_END = time(12, 0)  # Stop trading ORB breakouts by noon
+
+MR_START = time(14, 0)
+MR_END = time(16, 0)
+
+
+# =============================================================================
+# Strategy 1: Opening Range Breakout
+# =============================================================================
+
+def check_orb_signal(
+    df: pd.DataFrame,
+    idx: int,
+    or_high: float,
+    or_low: float,
+    atr: float,
+) -> Tuple[str, float, float]:
+    """
+    Check for ORB breakout signal.
+    
+    Returns (direction, stop, tp) or (None, None, None)
+    """
+    if or_high is None or or_low is None:
+        return None, None, None
+    
+    bar = df.iloc[idx]
+    close = bar['close']
+    high = bar['high']
+    low = bar['low']
+    
+    # Breakout above OR high
+    if high > or_high:
+        entry = close
+        stop = entry - atr * 0.75
+        tp = entry + atr * 1.5
+        return 'LONG', stop, tp
+    
+    # Breakdown below OR low
+    if low < or_low:
+        entry = close
+        stop = entry + atr * 0.75
+        tp = entry - atr * 1.5
+        return 'SHORT', stop, tp
+    
+    return None, None, None
+
+
+# =============================================================================
+# Strategy 2: Mean Reversion
+# =============================================================================
+
+def check_mr_signal(
+    df: pd.DataFrame,
+    idx: int,
+    ema_20: float,
+    atr: float,
+    rsi: float,
+) -> Tuple[str, float, float]:
+    """
+    Check for Mean Reversion signal.
+    
+    LONG: Price > 1.5 ATR below EMA, RSI < 30
+    SHORT: Price > 1.5 ATR above EMA, RSI > 70
+    
+    Returns (direction, stop, tp) or (None, None, None)
+    """
+    bar = df.iloc[idx]
+    close = bar['close']
+    
+    distance_from_ema = close - ema_20
+    
+    # Oversold: price below EMA, RSI low
+    if distance_from_ema < -atr * 1.5 and rsi < 35:
+        entry = close
+        stop = entry - atr * 1.0
+        tp = ema_20  # Revert to mean
+        return 'LONG', stop, tp
+    
+    # Overbought: price above EMA, RSI high
+    if distance_from_ema > atr * 1.5 and rsi > 65:
+        entry = close
+        stop = entry + atr * 1.0
+        tp = ema_20  # Revert to mean
+        return 'SHORT', stop, tp
+    
+    return None, None, None
+
+
+# =============================================================================
+# Combined Simulation
+# =============================================================================
+
+def run_combined_strategy(days: int = 7, starting_balance: float = 50000) -> Dict[str, Any]:
+    """
+    Run combined ORB + MR strategy simulation.
+    """
+    print("=" * 60)
+    print("COMBINED STRATEGY: ORB + MEAN REVERSION")
+    print("=" * 60)
+    print(f"ORB Window: {ORB_START} - {ORB_END} (breakout until {ORB_TRADE_END})")
+    print(f"MR Window:  {MR_START} - {MR_END}")
+    print(f"Starting Balance: ${starting_balance:,.0f}")
+    print("=" * 60)
+    
+    # Load data
+    actual_days = min(days, 7)
+    end = datetime.now()
+    start = end - timedelta(days=actual_days)
+    
+    print(f"\n[1] Loading {actual_days} days of ES data...")
+    ticker = yf.Ticker("ES=F")
+    df = ticker.history(start=start, end=end, interval="1m")
+    
+    if df is None or len(df) == 0:
+        print("ERROR: No data!")
+        return {}
+    
+    df.columns = [c.lower() for c in df.columns]
+    df = df.reset_index()
+    df['time'] = pd.to_datetime(
+        df['Datetime'] if 'Datetime' in df.columns else df['datetime']
+    ).dt.tz_convert(EST)
+    df['date'] = df['time'].dt.date
+    
+    print(f"    Loaded {len(df)} bars")
+    
+    # Compute indicators
+    print(f"\n[2] Computing indicators...")
+    df['atr'] = calculate_atr(df, period=14).ffill().bfill()
+    df['ema_20'] = calculate_ema(df['close'], 20)
+    df['rsi'] = calculate_rsi(df['close'], 14)
+    
+    # Run simulation
+    print(f"\n[3] Running combined simulation...")
+    
+    balance = starting_balance
+    equity_curve = [balance]
+    trades = []
+    active_trade = None
+    
+    # Daily OR tracking
+    daily_or = {}
+    
+    unique_dates = df['date'].unique()
+    
+    for i in range(30, len(df)):
+        bar = df.iloc[i]
+        current_time = bar['time']
+        current_date = current_time.date()
+        current_time_only = current_time.time()
+        
+        close = bar['close']
+        high = bar['high']
+        low = bar['low']
+        atr = bar['atr'] if not pd.isna(bar['atr']) else 2.0
+        ema_20 = bar['ema_20']
+        rsi = bar['rsi'] if not pd.isna(bar['rsi']) else 50
+        
+        # =====================================================================
+        # Compute OR for this day
+        # =====================================================================
+        if current_date not in daily_or:
+            # Find OR data for this day
+            or_mask = (df['date'] == current_date) & \
+                      (df['time'].dt.time >= ORB_START) & \
+                      (df['time'].dt.time <= ORB_END)
+            or_data = df[or_mask]
+            
+            if len(or_data) > 0:
+                daily_or[current_date] = {
+                    'high': or_data['high'].max(),
+                    'low': or_data['low'].min(),
+                }
+        
+        or_info = daily_or.get(current_date, {})
+        or_high = or_info.get('high')
+        or_low = or_info.get('low')
+        
+        # =====================================================================
+        # Check active trade
+        # =====================================================================
+        if active_trade is not None:
+            if active_trade['direction'] == 'LONG':
+                if low <= active_trade['stop']:
+                    pnl = (active_trade['stop'] - active_trade['entry']) * 50
+                    balance += pnl
+                    trades.append({
+                        'time': str(current_time),
+                        'strategy': active_trade['strategy'],
+                        'direction': 'LONG',
+                        'result': 'LOSS',
+                        'pnl': pnl,
+                    })
+                    active_trade = None
+                elif high >= active_trade['tp']:
+                    pnl = (active_trade['tp'] - active_trade['entry']) * 50
+                    balance += pnl
+                    trades.append({
+                        'time': str(current_time),
+                        'strategy': active_trade['strategy'],
+                        'direction': 'LONG',
+                        'result': 'WIN',
+                        'pnl': pnl,
+                    })
+                    active_trade = None
+            else:  # SHORT
+                if high >= active_trade['stop']:
+                    pnl = (active_trade['entry'] - active_trade['stop']) * 50
+                    balance += pnl
+                    trades.append({
+                        'time': str(current_time),
+                        'strategy': active_trade['strategy'],
+                        'direction': 'SHORT',
+                        'result': 'LOSS',
+                        'pnl': pnl,
+                    })
+                    active_trade = None
+                elif low <= active_trade['tp']:
+                    pnl = (active_trade['entry'] - active_trade['tp']) * 50
+                    balance += pnl
+                    trades.append({
+                        'time': str(current_time),
+                        'strategy': active_trade['strategy'],
+                        'direction': 'SHORT',
+                        'result': 'WIN',
+                        'pnl': pnl,
+                    })
+                    active_trade = None
+            
+            equity_curve.append(balance)
+            continue
+        
+        # =====================================================================
+        # Check for new entries based on time window
+        # =====================================================================
+        
+        # ORB Window (after OR forms, before noon)
+        if ORB_END < current_time_only <= ORB_TRADE_END:
+            direction, stop, tp = check_orb_signal(df, i, or_high, or_low, atr)
+            if direction:
+                active_trade = {
+                    'entry': close,
+                    'stop': stop,
+                    'tp': tp,
+                    'direction': direction,
+                    'strategy': 'ORB',
+                    'entry_time': current_time,
+                }
+        
+        # Mean Reversion Window (afternoon)
+        elif MR_START <= current_time_only <= MR_END:
+            direction, stop, tp = check_mr_signal(df, i, ema_20, atr, rsi)
+            if direction:
+                active_trade = {
+                    'entry': close,
+                    'stop': stop,
+                    'tp': tp,
+                    'direction': direction,
+                    'strategy': 'MR',
+                    'entry_time': current_time,
+                }
+        
+        equity_curve.append(balance)
+    
+    # =========================================================================
+    # Results
+    # =========================================================================
+    orb_trades = [t for t in trades if t['strategy'] == 'ORB']
+    mr_trades = [t for t in trades if t['strategy'] == 'MR']
+    
+    orb_wins = sum(1 for t in orb_trades if t['result'] == 'WIN')
+    mr_wins = sum(1 for t in mr_trades if t['result'] == 'WIN')
+    
+    total_pnl = balance - starting_balance
+    
+    print("\n" + "=" * 60)
+    print("RESULTS")
+    print("=" * 60)
+    print(f"  Starting Balance: ${starting_balance:,.0f}")
+    print(f"  Ending Balance:   ${balance:,.0f}")
+    print(f"  Total P&L:        ${total_pnl:,.2f}")
+    
+    print(f"\n  ORB Trades: {len(orb_trades)}")
+    if orb_trades:
+        print(f"    Win Rate: {orb_wins/len(orb_trades):.1%}")
+        print(f"    P&L: ${sum(t['pnl'] for t in orb_trades):,.2f}")
+    
+    print(f"\n  Mean Reversion Trades: {len(mr_trades)}")
+    if mr_trades:
+        print(f"    Win Rate: {mr_wins/len(mr_trades):.1%}")
+        print(f"    P&L: ${sum(t['pnl'] for t in mr_trades):,.2f}")
+    
+    # Save equity curve
+    output_dir = Path("results/combined")
+    output_dir.mkdir(parents=True, exist_ok=True)
+    
+    eq_df = pd.DataFrame({'equity': equity_curve})
+    eq_path = output_dir / "equity_curve.csv"
+    eq_df.to_csv(eq_path, index=False)
+    print(f"\n[4] Saved equity curve to {eq_path}")
+    
+    # Print mini curve
+    print(f"\n  Equity Curve (sampled):")
+    sample_points = np.linspace(0, len(equity_curve)-1, 10, dtype=int)
+    for idx in sample_points:
+        bar_pct = int((equity_curve[idx] - starting_balance) / starting_balance * 50) + 25
+        bar = "█" * max(0, min(50, bar_pct))
+        print(f"    {idx:5d}: ${equity_curve[idx]:,.0f} {bar}")
+    
+    # Store
+    db = ExperimentDB()
+    run_id = f"combined_orb_mr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
+    db.store_run(
+        run_id=run_id,
+        strategy="combined_orb_mr",
+        config={
+            'orb_window': f"{ORB_START}-{ORB_END}",
+            'mr_window': f"{MR_START}-{MR_END}",
+        },
+        metrics={
+            'total_trades': len(trades),
+            'wins': orb_wins + mr_wins,
+            'losses': len(trades) - (orb_wins + mr_wins),
+            'win_rate': (orb_wins + mr_wins) / len(trades) if trades else 0,
+            'total_pnl': total_pnl,
+            'orb_trades': len(orb_trades),
+            'mr_trades': len(mr_trades),
+        }
+    )
+    print(f"    Stored: {run_id}")
+    
+    return {
+        'total_pnl': total_pnl,
+        'ending_balance': balance,
+        'trades': len(trades),
+        'orb_trades': len(orb_trades),
+        'mr_trades': len(mr_trades),
+        'equity_curve': equity_curve,
+    }
+
+
+if __name__ == "__main__":
+    import argparse
+    
+    parser = argparse.ArgumentParser(description="Combined ORB + MR Strategy")
+    parser.add_argument("--days", type=int, default=7, help="Days to simulate")
+    parser.add_argument("--balance", type=float, default=50000, help="Starting balance")
+    
+    args = parser.parse_args()
+    
+    results = run_combined_strategy(args.days, args.balance)
diff --git a/scripts/run_ema_scan.py b/scripts/run_ema_scan.py
new file mode 100644
index 0000000..52333b4
--- /dev/null
+++ b/scripts/run_ema_scan.py
@@ -0,0 +1,437 @@
+#!/usr/bin/env python3
+"""
+Simple EMA Scanner
+
+Generates clean, objective labels for model training.
+Much simpler than ICT patterns - clear cause-effect relationships.
+
+Strategies:
+1. EMA Cross: 9 EMA crosses 21 EMA
+2. EMA Bounce: Price touches 20 EMA and reverses
+3. EMA Stack: All EMAs aligned (9 > 21 > 50 > 200)
+
+Usage:
+    python scripts/run_ema_scan.py --days 7 --strategy cross
+"""
+
+import sys
+from pathlib import Path
+sys.path.insert(0, str(Path(__file__).parent.parent))
+
+import json
+import numpy as np
+import pandas as pd
+import yfinance as yf
+from datetime import datetime, timedelta
+from typing import Dict, Any, List
+
+from src.features.indicators import calculate_ema, calculate_atr
+from src.storage import ExperimentDB
+
+
+# =============================================================================
+# EMA Strategies
+# =============================================================================
+
+def detect_ema_cross(
+    df: pd.DataFrame,
+    fast_period: int = 9,
+    slow_period: int = 21,
+    lookforward: int = 20,
+) -> List[Dict]:
+    """
+    Detect EMA crossover signals.
+    
+    LONG: Fast EMA crosses above slow EMA
+    SHORT: Fast EMA crosses below slow EMA
+    Label: WIN if price moves in direction within lookforward bars
+    """
+    df = df.copy()
+    df['ema_fast'] = calculate_ema(df['close'], fast_period)
+    df['ema_slow'] = calculate_ema(df['close'], slow_period)
+    
+    # Calculate ATR for target sizing
+    df['atr'] = calculate_atr(df, period=14).ffill()
+    
+    records = []
+    
+    for i in range(slow_period + 1, len(df) - lookforward):
+        fast_prev = df['ema_fast'].iloc[i-1]
+        fast_curr = df['ema_fast'].iloc[i]
+        slow_prev = df['ema_slow'].iloc[i-1]
+        slow_curr = df['ema_slow'].iloc[i]
+        
+        # Detect cross
+        cross_up = fast_prev <= slow_prev and fast_curr > slow_curr
+        cross_down = fast_prev >= slow_prev and fast_curr < slow_curr
+        
+        if not cross_up and not cross_down:
+            continue
+        
+        direction = 'LONG' if cross_up else 'SHORT'
+        entry_price = df['close'].iloc[i]
+        atr = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else 2.0
+        
+        # Check outcome
+        future_bars = df.iloc[i+1:i+1+lookforward]
+        
+        if direction == 'LONG':
+            # Win if price goes up by 1 ATR before going down 1 ATR
+            target = entry_price + atr
+            stop = entry_price - atr
+            hit_target = (future_bars['high'] >= target).any()
+            hit_stop = (future_bars['low'] <= stop).any()
+            
+            if hit_target and hit_stop:
+                # Both hit - check which first
+                target_idx = future_bars[future_bars['high'] >= target].index[0]
+                stop_idx = future_bars[future_bars['low'] <= stop].index[0]
+                outcome = 'WIN' if target_idx < stop_idx else 'LOSS'
+            elif hit_target:
+                outcome = 'WIN'
+            else:
+                outcome = 'LOSS'
+        else:
+            # SHORT
+            target = entry_price - atr
+            stop = entry_price + atr
+            hit_target = (future_bars['low'] <= target).any()
+            hit_stop = (future_bars['high'] >= stop).any()
+            
+            if hit_target and hit_stop:
+                target_idx = future_bars[future_bars['low'] <= target].index[0]
+                stop_idx = future_bars[future_bars['high'] >= stop].index[0]
+                outcome = 'WIN' if target_idx < stop_idx else 'LOSS'
+            elif hit_target:
+                outcome = 'WIN'
+            else:
+                outcome = 'LOSS'
+        
+        # Build record with window for model training
+        window_start = max(0, i - 60)
+        ohlcv_window = df.iloc[window_start:i][['open', 'high', 'low', 'close', 'volume']].values.tolist()
+        
+        records.append({
+            'time': str(df['time'].iloc[i]),
+            'direction': direction,
+            'label': outcome,
+            'entry_price': entry_price,
+            'atr': atr,
+            'window': {
+                'raw_ohlcv_1m': [
+                    {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
+                    for o, h, l, c, v in ohlcv_window
+                ]
+            },
+            'strategy': 'ema_cross',
+            'params': {'fast': fast_period, 'slow': slow_period},
+        })
+    
+    return records
+
+
+def detect_ema_bounce(
+    df: pd.DataFrame,
+    ema_period: int = 20,
+    touch_threshold: float = 0.1,  # % distance from EMA to count as "touch"
+    lookforward: int = 20,
+) -> List[Dict]:
+    """
+    Detect EMA bounce signals.
+    
+    LONG: Price touches EMA from above and bounces up
+    SHORT: Price touches EMA from below and bounces down
+    """
+    df = df.copy()
+    df['ema'] = calculate_ema(df['close'], ema_period)
+    df['atr'] = calculate_atr(df, period=14).ffill()
+    
+    records = []
+    
+    for i in range(ema_period + 5, len(df) - lookforward):
+        ema = df['ema'].iloc[i]
+        low = df['low'].iloc[i]
+        high = df['high'].iloc[i]
+        close = df['close'].iloc[i]
+        atr = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else 2.0
+        
+        # Touch threshold in points
+        threshold = ema * touch_threshold / 100
+        
+        # Check for touches
+        touch_from_above = (low <= ema + threshold) and (close > ema) and (df['close'].iloc[i-1] > ema)
+        touch_from_below = (high >= ema - threshold) and (close < ema) and (df['close'].iloc[i-1] < ema)
+        
+        if not touch_from_above and not touch_from_below:
+            continue
+        
+        direction = 'LONG' if touch_from_above else 'SHORT'
+        entry_price = close
+        
+        # Check outcome
+        future_bars = df.iloc[i+1:i+1+lookforward]
+        
+        if direction == 'LONG':
+            target = entry_price + atr
+            stop = entry_price - atr
+            hit_target = (future_bars['high'] >= target).any()
+            hit_stop = (future_bars['low'] <= stop).any()
+            
+            if hit_target and hit_stop:
+                target_idx = future_bars[future_bars['high'] >= target].index[0]
+                stop_idx = future_bars[future_bars['low'] <= stop].index[0]
+                outcome = 'WIN' if target_idx < stop_idx else 'LOSS'
+            elif hit_target:
+                outcome = 'WIN'
+            else:
+                outcome = 'LOSS'
+        else:
+            target = entry_price - atr
+            stop = entry_price + atr
+            hit_target = (future_bars['low'] <= target).any()
+            hit_stop = (future_bars['high'] >= stop).any()
+            
+            if hit_target and hit_stop:
+                target_idx = future_bars[future_bars['low'] <= target].index[0]
+                stop_idx = future_bars[future_bars['high'] >= stop].index[0]
+                outcome = 'WIN' if target_idx < stop_idx else 'LOSS'
+            elif hit_target:
+                outcome = 'WIN'
+            else:
+                outcome = 'LOSS'
+        
+        # Build record
+        window_start = max(0, i - 60)
+        ohlcv_window = df.iloc[window_start:i][['open', 'high', 'low', 'close', 'volume']].values.tolist()
+        
+        records.append({
+            'time': str(df['time'].iloc[i]),
+            'direction': direction,
+            'label': outcome,
+            'entry_price': entry_price,
+            'atr': atr,
+            'window': {
+                'raw_ohlcv_1m': [
+                    {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
+                    for o, h, l, c, v in ohlcv_window
+                ]
+            },
+            'strategy': 'ema_bounce',
+            'params': {'ema_period': ema_period},
+        })
+    
+    return records
+
+
+def detect_ema_stack(
+    df: pd.DataFrame,
+    periods: List[int] = [9, 21, 50, 200],
+    lookforward: int = 20,
+) -> List[Dict]:
+    """
+    Detect EMA stack alignment signals.
+    
+    LONG: 9 > 21 > 50 > 200 (bullish stack)
+    SHORT: 9 < 21 < 50 < 200 (bearish stack)
+    
+    Entry when stack first forms.
+    """
+    df = df.copy()
+    
+    for p in periods:
+        df[f'ema_{p}'] = calculate_ema(df['close'], p)
+    
+    df['atr'] = calculate_atr(df, period=14).ffill()
+    
+    records = []
+    prev_bullish = False
+    prev_bearish = False
+    
+    for i in range(max(periods) + 1, len(df) - lookforward):
+        emas = [df[f'ema_{p}'].iloc[i] for p in periods]
+        
+        # Check stack alignment
+        bullish_stack = all(emas[j] > emas[j+1] for j in range(len(emas)-1))
+        bearish_stack = all(emas[j] < emas[j+1] for j in range(len(emas)-1))
+        
+        # Detect new stack formation
+        new_bullish = bullish_stack and not prev_bullish
+        new_bearish = bearish_stack and not prev_bearish
+        
+        prev_bullish = bullish_stack
+        prev_bearish = bearish_stack
+        
+        if not new_bullish and not new_bearish:
+            continue
+        
+        direction = 'LONG' if new_bullish else 'SHORT'
+        entry_price = df['close'].iloc[i]
+        atr = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else 2.0
+        
+        # Check outcome
+        future_bars = df.iloc[i+1:i+1+lookforward]
+        
+        if direction == 'LONG':
+            target = entry_price + atr * 1.5  # Bigger target for trend trades
+            stop = entry_price - atr
+        else:
+            target = entry_price - atr * 1.5
+            stop = entry_price + atr
+        
+        if direction == 'LONG':
+            hit_target = (future_bars['high'] >= target).any()
+            hit_stop = (future_bars['low'] <= stop).any()
+        else:
+            hit_target = (future_bars['low'] <= target).any()
+            hit_stop = (future_bars['high'] >= stop).any()
+        
+        if hit_target and hit_stop:
+            if direction == 'LONG':
+                target_idx = future_bars[future_bars['high'] >= target].index[0]
+                stop_idx = future_bars[future_bars['low'] <= stop].index[0]
+            else:
+                target_idx = future_bars[future_bars['low'] <= target].index[0]
+                stop_idx = future_bars[future_bars['high'] >= stop].index[0]
+            outcome = 'WIN' if target_idx < stop_idx else 'LOSS'
+        elif hit_target:
+            outcome = 'WIN'
+        else:
+            outcome = 'LOSS'
+        
+        # Build record
+        window_start = max(0, i - 60)
+        ohlcv_window = df.iloc[window_start:i][['open', 'high', 'low', 'close', 'volume']].values.tolist()
+        
+        records.append({
+            'time': str(df['time'].iloc[i]),
+            'direction': direction,
+            'label': outcome,
+            'entry_price': entry_price,
+            'atr': atr,
+            'window': {
+                'raw_ohlcv_1m': [
+                    {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
+                    for o, h, l, c, v in ohlcv_window
+                ]
+            },
+            'strategy': 'ema_stack',
+            'params': {'periods': periods},
+        })
+    
+    return records
+
+
+# =============================================================================
+# Main
+# =============================================================================
+
+def run_ema_scan(
+    strategy: str = 'cross',
+    days: int = 7,
+    save: bool = True,
+) -> Dict[str, Any]:
+    """Run EMA scan and save records."""
+    
+    print("=" * 60)
+    print(f"EMA SCANNER - {strategy.upper()}")
+    print("=" * 60)
+    
+    # Load data
+    actual_days = min(days, 7)
+    end = datetime.now()
+    start = end - timedelta(days=actual_days)
+    
+    print(f"\n[1] Loading {actual_days} days of ES data...")
+    ticker = yf.Ticker("ES=F")
+    df = ticker.history(start=start, end=end, interval="1m")
+    
+    if df is None or len(df) == 0:
+        print("ERROR: No data!")
+        return {'records': 0}
+    
+    df.columns = [c.lower() for c in df.columns]
+    df = df.reset_index()
+    df['time'] = pd.to_datetime(df['Datetime'] if 'Datetime' in df.columns else df['datetime'])
+    
+    print(f"    Loaded {len(df)} bars")
+    
+    # Run scan
+    print(f"\n[2] Scanning for {strategy} signals...")
+    
+    if strategy == 'cross':
+        records = detect_ema_cross(df)
+    elif strategy == 'bounce':
+        records = detect_ema_bounce(df)
+    elif strategy == 'stack':
+        records = detect_ema_stack(df)
+    else:
+        print(f"Unknown strategy: {strategy}")
+        return {'records': 0}
+    
+    # Stats
+    wins = sum(1 for r in records if r['label'] == 'WIN')
+    longs = sum(1 for r in records if r['direction'] == 'LONG')
+    shorts = len(records) - longs
+    
+    print(f"\n    Found {len(records)} signals")
+    print(f"    LONG: {longs} | SHORT: {shorts}")
+    print(f"    WIN: {wins} | LOSS: {len(records) - wins}")
+    print(f"    Win Rate: {wins/len(records):.1%}" if records else "    No trades")
+    
+    # Save
+    if save and records:
+        output_dir = Path(f"results/ema_{strategy}")
+        output_dir.mkdir(parents=True, exist_ok=True)
+        output_path = output_dir / "records.jsonl"
+        
+        with open(output_path, 'w') as f:
+            for rec in records:
+                f.write(json.dumps(rec) + '\n')
+        
+        print(f"\n[3] Saved to {output_path}")
+        
+        # Also store summary in DB
+        db = ExperimentDB()
+        run_id = f"ema_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
+        db.store_run(
+            run_id=run_id,
+            strategy=f"ema_{strategy}",
+            config={'strategy': strategy, 'days': actual_days},
+            metrics={
+                'total_trades': len(records),
+                'wins': wins,
+                'losses': len(records) - wins,
+                'win_rate': wins/len(records) if records else 0,
+                'total_pnl': 0,
+            }
+        )
+        print(f"    Stored summary: {run_id}")
+    
+    return {
+        'records': len(records),
+        'wins': wins,
+        'losses': len(records) - wins,
+        'win_rate': wins/len(records) if records else 0,
+        'longs': longs,
+        'shorts': shorts,
+    }
+
+
+if __name__ == "__main__":
+    import argparse
+    
+    parser = argparse.ArgumentParser(description="EMA Scanner")
+    parser.add_argument("--strategy", type=str, default="cross",
+                        choices=['cross', 'bounce', 'stack'],
+                        help="Strategy type")
+    parser.add_argument("--days", type=int, default=7, help="Days to scan")
+    parser.add_argument("--all", action="store_true", help="Run all strategies")
+    
+    args = parser.parse_args()
+    
+    if args.all:
+        for strat in ['cross', 'bounce', 'stack']:
+            print("\n")
+            run_ema_scan(strategy=strat, days=args.days)
+    else:
+        run_ema_scan(strategy=args.strategy, days=args.days)
diff --git a/scripts/run_inverse_test.py b/scripts/run_inverse_test.py
new file mode 100644
index 0000000..13fd33d
--- /dev/null
+++ b/scripts/run_inverse_test.py
@@ -0,0 +1,182 @@
+#!/usr/bin/env python3
+"""
+Inverse Strategy Test
+
+Theory: Our FVG model is losing. Maybe the signal is actually a CONTINUATION
+not a reversal. Flip all the directions and see if we accidentally found alpha.
+
+This mirrors the mlang discovery in success_study.md where they found 70% WR
+by inverting a losing pattern.
+
+Usage:
+    python scripts/run_inverse_test.py --input results/ict_ifvg/records.jsonl
+"""
+
+import sys
+from pathlib import Path
+sys.path.insert(0, str(Path(__file__).parent.parent))
+
+import json
+from datetime import datetime
+from typing import Dict, Any, List
+
+from src.storage import ExperimentDB
+
+
+def analyze_records(records: List[Dict]) -> Dict[str, Any]:
+    """Analyze original records."""
+    wins = sum(1 for r in records if r.get('label', r.get('outcome')) == 'WIN')
+    longs = sum(1 for r in records if r.get('direction') == 'LONG')
+    
+    return {
+        'total': len(records),
+        'wins': wins,
+        'losses': len(records) - wins,
+        'win_rate': wins / len(records) if records else 0,
+        'longs': longs,
+        'shorts': len(records) - longs,
+    }
+
+
+def flip_direction(direction: str) -> str:
+    """Flip LONG to SHORT and vice versa."""
+    return 'SHORT' if direction == 'LONG' else 'LONG'
+
+
+def invert_outcome(original_direction: str, outcome: str) -> str:
+    """
+    When we flip direction, outcomes also flip.
+    
+    Original LONG WIN (price went up) → Flipped SHORT would LOSE
+    Original LONG LOSS (price went down) → Flipped SHORT would WIN
+    """
+    # If original was a WIN, flipped is a LOSS (and vice versa)
+    return 'LOSS' if outcome == 'WIN' else 'WIN'
+
+
+def run_inverse_test(input_path: str) -> Dict[str, Any]:
+    """
+    Run inverse strategy test.
+    
+    Takes existing signals, flips the direction, and measures outcome.
+    """
+    print("=" * 60)
+    print("INVERSE STRATEGY TEST")
+    print("=" * 60)
+    print("Theory: FVG is continuation, not reversal")
+    print("Method: Flip all directions (BUY→SELL, SELL→BUY)")
+    print("=" * 60)
+    
+    # Load records
+    print(f"\n[1] Loading records from {input_path}...")
+    records = []
+    with open(input_path) as f:
+        for line in f:
+            records.append(json.loads(line))
+    
+    print(f"    Loaded {len(records)} signals")
+    
+    # Analyze original
+    print(f"\n[2] Original strategy performance...")
+    original = analyze_records(records)
+    print(f"    Total: {original['total']}")
+    print(f"    LONG: {original['longs']} | SHORT: {original['shorts']}")
+    print(f"    WIN: {original['wins']} | LOSS: {original['losses']}")
+    print(f"    Win Rate: {original['win_rate']:.1%}")
+    
+    # Create inverted records
+    print(f"\n[3] Flipping all signals...")
+    inverted_records = []
+    
+    for rec in records:
+        original_direction = rec.get('direction', 'LONG')
+        original_outcome = rec.get('label', rec.get('outcome', 'LOSS'))
+        
+        inverted_rec = rec.copy()
+        inverted_rec['direction'] = flip_direction(original_direction)
+        inverted_rec['label'] = invert_outcome(original_direction, original_outcome)
+        inverted_rec['original_direction'] = original_direction
+        inverted_rec['original_outcome'] = original_outcome
+        inverted_rec['strategy'] = 'inverse_' + rec.get('strategy', 'fvg')
+        
+        inverted_records.append(inverted_rec)
+    
+    # Analyze inverted
+    print(f"\n[4] Inverted strategy performance...")
+    inverted = analyze_records(inverted_records)
+    print(f"    Total: {inverted['total']}")
+    print(f"    LONG: {inverted['longs']} | SHORT: {inverted['shorts']}")
+    print(f"    WIN: {inverted['wins']} | LOSS: {inverted['losses']}")
+    print(f"    Win Rate: {inverted['win_rate']:.1%}")
+    
+    # Compare
+    print("\n" + "=" * 60)
+    print("COMPARISON")
+    print("=" * 60)
+    print(f"  Original Win Rate:  {original['win_rate']:.1%}")
+    print(f"  Inverted Win Rate:  {inverted['win_rate']:.1%}")
+    
+    improvement = (inverted['win_rate'] - original['win_rate']) * 100
+    
+    if inverted['win_rate'] > 0.5:
+        print(f"\n  🎯 JACKPOT! Inverted strategy is PROFITABLE!")
+        print(f"  Win rate improvement: +{improvement:.1f} percentage points")
+        print(f"\n  → FVG IS a continuation signal, not reversal!")
+        print(f"  → When model says BUY, we should SELL (fade it)")
+    elif inverted['win_rate'] > original['win_rate']:
+        print(f"\n  📈 Inverted is BETTER but still <50%")
+        print(f"  Improvement: +{improvement:.1f}pp")
+    else:
+        print(f"\n  ❌ Inverting made it WORSE")
+        print(f"  Change: {improvement:.1f}pp")
+        print(f"\n  → The original direction was correct, just bad execution")
+    
+    # Save inverted records
+    output_dir = Path("results/inverse_fvg")
+    output_dir.mkdir(parents=True, exist_ok=True)
+    output_path = output_dir / "records.jsonl"
+    
+    with open(output_path, 'w') as f:
+        for rec in inverted_records:
+            f.write(json.dumps(rec) + '\n')
+    
+    print(f"\n[5] Saved inverted records to {output_path}")
+    
+    # Store to DB
+    db = ExperimentDB()
+    run_id = f"inverse_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
+    db.store_run(
+        run_id=run_id,
+        strategy="inverse_fvg",
+        config={
+            'source': input_path,
+            'method': 'direction_flip',
+        },
+        metrics={
+            'total_trades': inverted['total'],
+            'wins': inverted['wins'],
+            'losses': inverted['losses'],
+            'win_rate': inverted['win_rate'],
+            'original_win_rate': original['win_rate'],
+            'total_pnl': 0,
+        }
+    )
+    print(f"    Stored: {run_id}")
+    
+    return {
+        'original': original,
+        'inverted': inverted,
+        'improvement': improvement,
+    }
+
+
+if __name__ == "__main__":
+    import argparse
+    
+    parser = argparse.ArgumentParser(description="Inverse Strategy Test")
+    parser.add_argument("--input", type=str, default="results/ict_ifvg/records.jsonl",
+                        help="Path to original signals")
+    
+    args = parser.parse_args()
+    
+    results = run_inverse_test(args.input)
diff --git a/scripts/run_lowvol_breakout.py b/scripts/run_lowvol_breakout.py
new file mode 100644
index 0000000..a66534e
--- /dev/null
+++ b/scripts/run_lowvol_breakout.py
@@ -0,0 +1,312 @@
+#!/usr/bin/env python3
+"""
+Low Volatility Breakout Strategy
+
+Theory: Enter when volatility is dead (15m ATR at 5-day low).
+Action: Trade the breakout when price moves.
+Stop: DYNAMIC - 2x current candle's range (not lagging ATR)
+Target: 2R
+
+Hypothesis: Candle-range stop adapts faster than ATR.
+
+Run:
+    python scripts/run_lowvol_breakout.py --days 7
+"""
+
+import sys
+from pathlib import Path
+sys.path.insert(0, str(Path(__file__).parent.parent))
+
+import numpy as np
+import pandas as pd
+import yfinance as yf
+from datetime import datetime, timedelta
+from typing import Dict, Any
+
+from src.storage import ExperimentDB
+
+
+# Strategy parameters
+ATR_PERIOD = 14
+ATR_LOW_LOOKBACK = 5 * 24 * 4  # 5 days of 15m bars (approx)
+STOP_CANDLE_MULT = 2.0         # Stop = 2x candle range
+TP_R_MULT = 2.0                # 2R target
+BREAKOUT_THRESHOLD = 0.3       # Price must move 0.3 ATR to trigger
+
+
+def resample_to_15m(df: pd.DataFrame) -> pd.DataFrame:
+    """Resample 1m data to 15m bars."""
+    df = df.copy()
+    df['time'] = pd.to_datetime(df['time'])
+    df = df.set_index('time')
+    
+    resampled = df.resample('15min').agg({
+        'open': 'first',
+        'high': 'max',
+        'low': 'min',
+        'close': 'last',
+        'volume': 'sum'
+    }).dropna()
+    
+    return resampled.reset_index()
+
+
+def compute_atr_rolling_min(atr_series: pd.Series, window: int) -> pd.Series:
+    """Compute rolling minimum of ATR."""
+    return atr_series.rolling(window=window, min_periods=1).min()
+
+
+def run_lowvol_breakout_strategy(days: int = 7, verbose: bool = True) -> Dict[str, Any]:
+    """
+    Run the Low Volatility Breakout strategy simulation.
+    """
+    print("=" * 60)
+    print("LOW VOLATILITY BREAKOUT STRATEGY")
+    print("=" * 60)
+    print(f"Theory: Enter when ATR is at 5-day low (coiled market)")
+    print(f"Stop: 2x current candle range (dynamic)")
+    print(f"Target: {TP_R_MULT}R")
+    print("=" * 60)
+    
+    # Load data
+    actual_days = min(days, 7)
+    end = datetime.now()
+    start = end - timedelta(days=actual_days)
+    
+    print(f"\n[1] Loading {actual_days} days of ES data...")
+    ticker = yf.Ticker("ES=F")
+    df_1m = ticker.history(start=start, end=end, interval="1m")
+    
+    if df_1m is None or len(df_1m) == 0:
+        print("ERROR: No data available")
+        return {'trades': 0, 'win_rate': 0, 'total_pnl': 0}
+    
+    # Standardize
+    df_1m.columns = [c.lower() for c in df_1m.columns]
+    df_1m = df_1m.reset_index()
+    df_1m['time'] = df_1m['Datetime'] if 'Datetime' in df_1m.columns else df_1m['datetime']
+    
+    print(f"    Loaded {len(df_1m)} 1m bars")
+    
+    # Resample to 15m
+    print("\n[2] Resampling to 15m...")
+    df = resample_to_15m(df_1m)
+    print(f"    {len(df)} 15m bars")
+    
+    # Compute ATR
+    print("\n[3] Computing 15m ATR and rolling minimum...")
+    high = df['high']
+    low = df['low']
+    close = df['close']
+    
+    tr1 = high - low
+    tr2 = abs(high - close.shift(1))
+    tr3 = abs(low - close.shift(1))
+    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
+    
+    df['atr'] = tr.rolling(window=ATR_PERIOD).mean()
+    df['atr_min_5d'] = compute_atr_rolling_min(df['atr'], window=ATR_LOW_LOOKBACK)
+    df['candle_range'] = df['high'] - df['low']
+    
+    # Run simulation
+    print(f"\n[4] Scanning for low-ATR breakout entries...")
+    
+    trades = []
+    active_trade = None
+    lookback = max(ATR_PERIOD, 20)
+    
+    for i in range(lookback, len(df)):
+        bar = df.iloc[i]
+        current_time = bar['time']
+        current_close = bar['close']
+        current_high = bar['high']
+        current_low = bar['low']
+        current_atr = bar['atr']
+        atr_min = bar['atr_min_5d']
+        candle_range = bar['candle_range']
+        
+        if pd.isna(current_atr) or pd.isna(atr_min) or candle_range < 0.5:
+            continue
+        
+        # Check active trade
+        if active_trade is not None:
+            if active_trade['direction'] == 'LONG':
+                if current_low <= active_trade['stop']:
+                    pnl = (active_trade['stop'] - active_trade['entry']) * 50
+                    trades.append({
+                        'entry_time': active_trade['entry_time'],
+                        'exit_time': current_time,
+                        'direction': 'LONG',
+                        'entry': active_trade['entry'],
+                        'exit': active_trade['stop'],
+                        'pnl': pnl,
+                        'result': 'LOSS'
+                    })
+                    if verbose:
+                        print(f"  [LOSS] LONG @ {active_trade['entry']:.2f} -> SL @ {active_trade['stop']:.2f} = ${pnl:.2f}")
+                    active_trade = None
+                elif current_high >= active_trade['tp']:
+                    pnl = (active_trade['tp'] - active_trade['entry']) * 50
+                    trades.append({
+                        'entry_time': active_trade['entry_time'],
+                        'exit_time': current_time,
+                        'direction': 'LONG',
+                        'entry': active_trade['entry'],
+                        'exit': active_trade['tp'],
+                        'pnl': pnl,
+                        'result': 'WIN'
+                    })
+                    if verbose:
+                        print(f"  [WIN] LONG @ {active_trade['entry']:.2f} -> TP @ {active_trade['tp']:.2f} = ${pnl:.2f}")
+                    active_trade = None
+            else:  # SHORT
+                if current_high >= active_trade['stop']:
+                    pnl = (active_trade['entry'] - active_trade['stop']) * 50
+                    trades.append({
+                        'entry_time': active_trade['entry_time'],
+                        'exit_time': current_time,
+                        'direction': 'SHORT',
+                        'entry': active_trade['entry'],
+                        'exit': active_trade['stop'],
+                        'pnl': pnl,
+                        'result': 'LOSS'
+                    })
+                    if verbose:
+                        print(f"  [LOSS] SHORT @ {active_trade['entry']:.2f} -> SL @ {active_trade['stop']:.2f} = ${pnl:.2f}")
+                    active_trade = None
+                elif current_low <= active_trade['tp']:
+                    pnl = (active_trade['entry'] - active_trade['tp']) * 50
+                    trades.append({
+                        'entry_time': active_trade['entry_time'],
+                        'exit_time': current_time,
+                        'direction': 'SHORT',
+                        'entry': active_trade['entry'],
+                        'exit': active_trade['tp'],
+                        'pnl': pnl,
+                        'result': 'WIN'
+                    })
+                    if verbose:
+                        print(f"  [WIN] SHORT @ {active_trade['entry']:.2f} -> TP @ {active_trade['tp']:.2f} = ${pnl:.2f}")
+                    active_trade = None
+            continue
+        
+        # Check for low ATR condition (ATR at or near 5-day low)
+        is_low_vol = current_atr <= atr_min * 1.1  # Within 10% of 5-day low
+        
+        if not is_low_vol:
+            continue
+        
+        # Check for breakout (price moves beyond previous bar)
+        prev_high = df.iloc[i-1]['high']
+        prev_low = df.iloc[i-1]['low']
+        
+        # DYNAMIC STOP: 2x current candle range
+        stop_distance = candle_range * STOP_CANDLE_MULT
+        
+        # Breakout detection
+        if current_high > prev_high:
+            # Bullish breakout -> LONG
+            entry = current_close
+            stop = entry - stop_distance
+            tp = entry + (stop_distance * TP_R_MULT)
+            
+            active_trade = {
+                'entry_time': current_time,
+                'entry': entry,
+                'stop': stop,
+                'tp': tp,
+                'direction': 'LONG'
+            }
+            
+            if verbose:
+                print(f"  [TRIGGER] LONG @ {entry:.2f} (low ATR={current_atr:.2f}) Stop={stop:.2f} TP={tp:.2f}")
+                
+        elif current_low < prev_low:
+            # Bearish breakout -> SHORT
+            entry = current_close
+            stop = entry + stop_distance
+            tp = entry - (stop_distance * TP_R_MULT)
+            
+            active_trade = {
+                'entry_time': current_time,
+                'entry': entry,
+                'stop': stop,
+                'tp': tp,
+                'direction': 'SHORT'
+            }
+            
+            if verbose:
+                print(f"  [TRIGGER] SHORT @ {entry:.2f} (low ATR={current_atr:.2f}) Stop={stop:.2f} TP={tp:.2f}")
+    
+    # Summary
+    wins = sum(1 for t in trades if t['result'] == 'WIN')
+    losses = len(trades) - wins
+    total_pnl = sum(t['pnl'] for t in trades)
+    win_rate = wins / len(trades) if trades else 0
+    
+    # Direction breakdown
+    longs = [t for t in trades if t['direction'] == 'LONG']
+    shorts = [t for t in trades if t['direction'] == 'SHORT']
+    long_wr = sum(1 for t in longs if t['result'] == 'WIN') / len(longs) if longs else 0
+    short_wr = sum(1 for t in shorts if t['result'] == 'WIN') / len(shorts) if shorts else 0
+    
+    print("\n" + "=" * 60)
+    print("RESULTS")
+    print("=" * 60)
+    print(f"  Period: {actual_days} days")
+    print(f"  Total Trades: {len(trades)}")
+    print(f"  Wins: {wins} | Losses: {losses}")
+    print(f"  Win Rate: {win_rate:.1%}")
+    print(f"  Total PnL: ${total_pnl:.2f}")
+    if trades:
+        print(f"  Avg PnL/Trade: ${total_pnl/len(trades):.2f}")
+    print(f"\n  LONG trades: {len(longs)} @ {long_wr:.1%} WR")
+    print(f"  SHORT trades: {len(shorts)} @ {short_wr:.1%} WR")
+    
+    return {
+        'trades': len(trades),
+        'wins': wins,
+        'losses': losses,
+        'win_rate': win_rate,
+        'total_pnl': total_pnl,
+        'long_trades': len(longs),
+        'short_trades': len(shorts),
+        'long_wr': long_wr,
+        'short_wr': short_wr,
+        'strategy': 'lowvol_breakout',
+        'params': {
+            'stop_candle_mult': STOP_CANDLE_MULT,
+            'tp_r': TP_R_MULT,
+            'atr_period': ATR_PERIOD,
+        }
+    }
+
+
+if __name__ == "__main__":
+    import argparse
+    
+    parser = argparse.ArgumentParser(description="Low Volatility Breakout Strategy")
+    parser.add_argument("--days", type=int, default=7, help="Days to simulate")
+    parser.add_argument("--save", action="store_true", help="Save to ExperimentDB")
+    parser.add_argument("--quiet", action="store_true", help="Suppress trade details")
+    
+    args = parser.parse_args()
+    
+    results = run_lowvol_breakout_strategy(days=args.days, verbose=not args.quiet)
+    
+    if args.save and results['trades'] > 0:
+        db = ExperimentDB()
+        run_id = f"lowvol_breakout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
+        db.store_run(
+            run_id=run_id,
+            strategy="lowvol_breakout",
+            config=results['params'],
+            metrics={
+                'total_trades': results['trades'],
+                'wins': results['wins'],
+                'losses': results['losses'],
+                'win_rate': results['win_rate'],
+                'total_pnl': results['total_pnl'],
+            }
+        )
+        print(f"\n[+] Saved to ExperimentDB: {run_id}")
diff --git a/scripts/run_lunch_fade.py b/scripts/run_lunch_fade.py
new file mode 100644
index 0000000..5277a81
--- /dev/null
+++ b/scripts/run_lunch_fade.py
@@ -0,0 +1,258 @@
+#!/usr/bin/env python3
+"""
+Lunch Hour Fade Strategy
+
+Theory: Breakouts fail between 11:30 AM - 1:00 PM EST (lunch hours).
+Action: SHORT when price breaks above 15-minute swing high during lunch.
+Stop: 0.5 ATR (tight - get out if it's a real breakout)
+Target: 2R (risk/reward = 1:2)
+
+Run:
+    python scripts/run_lunch_fade.py --days 28
+"""
+
+import sys
+from pathlib import Path
+sys.path.insert(0, str(Path(__file__).parent.parent))
+
+import numpy as np
+import pandas as pd
+import yfinance as yf
+from datetime import datetime, timedelta, time
+from typing import Dict, List, Any
+from zoneinfo import ZoneInfo
+
+from src.features.indicators import calculate_atr
+from src.storage import ExperimentDB
+
+
+# Strategy parameters
+LUNCH_START = time(11, 30)  # 11:30 AM EST
+LUNCH_END = time(13, 0)     # 1:00 PM EST
+SWING_LOOKBACK = 15         # 15 bars for swing high detection
+STOP_ATR_MULT = 0.5         # Tight stop
+TP_R_MULT = 2.0             # 2R target
+EST = ZoneInfo("America/New_York")
+
+
+def find_swing_high(highs: np.ndarray, idx: int, lookback: int = 5) -> float:
+    """
+    Find the most recent swing high (higher than neighbors).
+    Returns the swing high price, or np.nan if none found.
+    """
+    if idx < lookback * 2:
+        return np.nan
+    
+    # Look for swing highs in the lookback window
+    for i in range(idx - lookback, max(lookback, idx - lookback * 3), -1):
+        # Is this a swing high? (higher than bars before and after)
+        if i >= lookback and i < len(highs) - lookback:
+            center = highs[i]
+            left_max = max(highs[i-lookback:i])
+            right_max = max(highs[i+1:i+lookback+1]) if i + lookback + 1 <= len(highs) else center
+            
+            if center >= left_max and center >= right_max:
+                return center
+    
+    # Fallback: use rolling max
+    return max(highs[idx-lookback:idx])
+
+
+def is_lunch_hour(t: datetime) -> bool:
+    """Check if time is in lunch hours (11:30 AM - 1:00 PM EST)."""
+    # Convert to EST if needed
+    if t.tzinfo is None:
+        t = t.replace(tzinfo=EST)
+    else:
+        t = t.astimezone(EST)
+    
+    current_time = t.time()
+    return LUNCH_START <= current_time <= LUNCH_END
+
+
+def run_lunch_fade_strategy(days: int = 28, verbose: bool = True) -> Dict[str, Any]:
+    """
+    Run the Lunch Hour Fade strategy simulation.
+    
+    Args:
+        days: Number of days to simulate (max 7 for 1m data)
+        verbose: Print trade details
+    
+    Returns:
+        Dict with strategy results
+    """
+    print("=" * 60)
+    print("LUNCH HOUR FADE STRATEGY")
+    print("=" * 60)
+    print(f"Theory: Fade breakouts during 11:30 AM - 1:00 PM EST")
+    print(f"Action: SHORT on 15m swing high break during lunch")
+    print(f"Stop: {STOP_ATR_MULT} ATR | Target: {TP_R_MULT}R")
+    print("=" * 60)
+    
+    # Load data (yfinance 1m limit is 7 days)
+    actual_days = min(days, 7)
+    if days > 7:
+        print(f"\n[!] yfinance 1m limit is 7 days. Running with {actual_days} days.")
+    
+    end = datetime.now()
+    start = end - timedelta(days=actual_days)
+    
+    print(f"\n[1] Loading {actual_days} days of ES data...")
+    ticker = yf.Ticker("ES=F")
+    df = ticker.history(start=start, end=end, interval="1m")
+    
+    if df is None or len(df) == 0:
+        print("ERROR: No data available")
+        return {'trades': 0, 'win_rate': 0, 'total_pnl': 0}
+    
+    # Standardize columns
+    df.columns = [c.lower() for c in df.columns]
+    df = df.reset_index()
+    df['time'] = df['Datetime'] if 'Datetime' in df.columns else df['datetime']
+    
+    print(f"    Loaded {len(df)} bars")
+    
+    # Calculate ATR
+    print("\n[2] Computing indicators...")
+    df['atr'] = calculate_atr(df, period=14)
+    df['atr'] = df['atr'].ffill().bfill()  # Fill NaN
+    
+    # Run simulation
+    print(f"\n[3] Scanning for lunch hour breakouts...")
+    
+    trades = []
+    active_trade = None
+    lookback = SWING_LOOKBACK
+    
+    for i in range(lookback + 14, len(df)):
+        current_bar = df.iloc[i]
+        current_time = pd.to_datetime(current_bar['time'])
+        current_price = current_bar['close']
+        current_high = current_bar['high']
+        current_low = current_bar['low']
+        atr = current_bar['atr']
+        
+        if pd.isna(atr) or atr < 0.5:
+            continue
+        
+        # Check if in active trade
+        if active_trade is not None:
+            # Check stop loss (price goes UP for short)
+            if current_high >= active_trade['stop']:
+                pnl = -(active_trade['stop'] - active_trade['entry']) * 50  # Negative for loss
+                trades.append({
+                    'entry_time': active_trade['entry_time'],
+                    'exit_time': current_time,
+                    'entry': active_trade['entry'],
+                    'exit': active_trade['stop'],
+                    'pnl': pnl,
+                    'result': 'LOSS'
+                })
+                if verbose:
+                    print(f"  [LOSS] SHORT @ {active_trade['entry']:.2f} -> SL @ {active_trade['stop']:.2f} = ${pnl:.2f}")
+                active_trade = None
+            # Check take profit (price goes DOWN for short)
+            elif current_low <= active_trade['tp']:
+                pnl = (active_trade['entry'] - active_trade['tp']) * 50  # Positive for win
+                trades.append({
+                    'entry_time': active_trade['entry_time'],
+                    'exit_time': current_time,
+                    'entry': active_trade['entry'],
+                    'exit': active_trade['tp'],
+                    'pnl': pnl,
+                    'result': 'WIN'
+                })
+                if verbose:
+                    print(f"  [WIN] SHORT @ {active_trade['entry']:.2f} -> TP @ {active_trade['tp']:.2f} = ${pnl:.2f}")
+                active_trade = None
+            continue
+        
+        # Look for entry during lunch hours only
+        if not is_lunch_hour(current_time):
+            continue
+        
+        # Find swing high
+        swing_high = find_swing_high(df['high'].values, i, lookback)
+        if pd.isna(swing_high):
+            continue
+        
+        # Check for breakout (price breaks above swing high)
+        if current_high > swing_high:
+            # SHORT entry
+            entry = current_price
+            stop = entry + (atr * STOP_ATR_MULT)
+            risk = stop - entry
+            tp = entry - (risk * TP_R_MULT)
+            
+            active_trade = {
+                'entry_time': current_time,
+                'entry': entry,
+                'stop': stop,
+                'tp': tp,
+                'swing_high': swing_high,
+            }
+            
+            if verbose:
+                print(f"  [TRIGGER] SHORT @ {entry:.2f} (broke {swing_high:.2f}) SL={stop:.2f} TP={tp:.2f}")
+    
+    # Summary
+    wins = sum(1 for t in trades if t['result'] == 'WIN')
+    losses = len(trades) - wins
+    total_pnl = sum(t['pnl'] for t in trades)
+    win_rate = wins / len(trades) if trades else 0
+    
+    print("\n" + "=" * 60)
+    print("RESULTS")
+    print("=" * 60)
+    print(f"  Period: {actual_days} days")
+    print(f"  Total Trades: {len(trades)}")
+    print(f"  Wins: {wins} | Losses: {losses}")
+    print(f"  Win Rate: {win_rate:.1%}")
+    print(f"  Total PnL: ${total_pnl:.2f}")
+    if trades:
+        print(f"  Avg PnL/Trade: ${total_pnl/len(trades):.2f}")
+    
+    return {
+        'trades': len(trades),
+        'wins': wins,
+        'losses': losses,
+        'win_rate': win_rate,
+        'total_pnl': total_pnl,
+        'strategy': 'lunch_fade',
+        'params': {
+            'lunch_start': str(LUNCH_START),
+            'lunch_end': str(LUNCH_END),
+            'stop_atr': STOP_ATR_MULT,
+            'tp_r': TP_R_MULT,
+        }
+    }
+
+
+if __name__ == "__main__":
+    import argparse
+    
+    parser = argparse.ArgumentParser(description="Lunch Hour Fade Strategy")
+    parser.add_argument("--days", type=int, default=7, help="Days to simulate")
+    parser.add_argument("--save", action="store_true", help="Save to ExperimentDB")
+    parser.add_argument("--quiet", action="store_true", help="Suppress trade details")
+    
+    args = parser.parse_args()
+    
+    results = run_lunch_fade_strategy(days=args.days, verbose=not args.quiet)
+    
+    if args.save and results['trades'] > 0:
+        db = ExperimentDB()
+        run_id = f"lunch_fade_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
+        db.store_run(
+            run_id=run_id,
+            strategy="lunch_fade",
+            config=results['params'],
+            metrics={
+                'total_trades': results['trades'],
+                'wins': results['wins'],
+                'losses': results['losses'],
+                'win_rate': results['win_rate'],
+                'total_pnl': results['total_pnl'],
+            }
+        )
+        print(f"\n[+] Saved to ExperimentDB: {run_id}")
diff --git a/scripts/run_orb_gridsearch.py b/scripts/run_orb_gridsearch.py
new file mode 100644
index 0000000..c876e4d
--- /dev/null
+++ b/scripts/run_orb_gridsearch.py
@@ -0,0 +1,297 @@
+#!/usr/bin/env python3
+"""
+Opening Range Breakout - Grid Search
+
+Massive parameter sweep:
+- Stop Loss: 0.5 to 3.0 ATR (0.25 increments) = 11 values
+- Take Profit: 1.0 to 5.0 R = 9 values
+- Total: 99 combinations
+
+Optimizes for: PROFIT FACTOR (not just Net PnL)
+Profit Factor = Gross Wins / Gross Losses
+
+Run:
+    python scripts/run_orb_gridsearch.py
+"""
+
+import sys
+from pathlib import Path
+sys.path.insert(0, str(Path(__file__).parent.parent))
+
+import numpy as np
+import pandas as pd
+import yfinance as yf
+from datetime import datetime, timedelta, time
+from typing import Dict, Any, List, Tuple
+from zoneinfo import ZoneInfo
+import itertools
+
+from src.storage import ExperimentDB
+
+
+# Strategy parameters
+OR_START = time(9, 30)   # Opening range start (RTH open)
+OR_END = time(10, 0)     # Opening range end (first 30 min)
+ATR_PERIOD = 14
+EST = ZoneInfo("America/New_York")
+
+# Grid search parameters
+STOP_ATR_RANGE = np.arange(0.5, 3.25, 0.25)  # 0.5, 0.75, 1.0, ..., 3.0
+TP_R_RANGE = np.arange(1.0, 5.5, 0.5)        # 1.0, 1.5, 2.0, ..., 5.0
+
+
+def run_orb_single(
+    df: pd.DataFrame,
+    stop_atr: float,
+    tp_r: float,
+    verbose: bool = False
+) -> Dict[str, Any]:
+    """
+    Run Opening Range Breakout for a single parameter combination.
+    
+    Returns trade stats including profit factor.
+    """
+    trades = []
+    active_trade = None
+    
+    # Pre-compute daily stats
+    df = df.copy()
+    df['date'] = df['time'].dt.date
+    df['hour'] = df['time'].dt.hour
+    df['minute'] = df['time'].dt.minute
+    
+    # Calculate ATR
+    high = df['high']
+    low = df['low']
+    close = df['close']
+    tr = pd.concat([
+        high - low,
+        abs(high - close.shift(1)),
+        abs(low - close.shift(1))
+    ], axis=1).max(axis=1)
+    df['atr'] = tr.rolling(window=ATR_PERIOD).mean().shift(1)
+    
+    # Group by date for opening range calculation
+    dates = df['date'].unique()
+    
+    for date in dates:
+        day_data = df[df['date'] == date].copy()
+        if len(day_data) < 60:  # Need enough bars
+            continue
+        
+        # Find opening range (9:30 - 10:00)
+        or_data = day_data[
+            (day_data['hour'] == 9) & (day_data['minute'] >= 30) |
+            (day_data['hour'] == 10) & (day_data['minute'] == 0)
+        ]
+        
+        if len(or_data) < 5:
+            continue
+        
+        or_high = or_data['high'].max()
+        or_low = or_data['low'].min()
+        or_close_idx = or_data.index[-1]
+        
+        # Get ATR at OR close
+        atr = day_data.loc[or_close_idx, 'atr']
+        if pd.isna(atr) or atr < 0.5:
+            continue
+        
+        # Trade after OR (10:00 onwards)
+        after_or = day_data[day_data.index > or_close_idx]
+        
+        for idx, bar in after_or.iterrows():
+            if active_trade is not None:
+                # Check exit conditions
+                if active_trade['direction'] == 'LONG':
+                    if bar['low'] <= active_trade['stop']:
+                        pnl = (active_trade['stop'] - active_trade['entry']) * 50
+                        trades.append({'pnl': pnl, 'result': 'LOSS', 'gross': pnl})
+                        active_trade = None
+                    elif bar['high'] >= active_trade['tp']:
+                        pnl = (active_trade['tp'] - active_trade['entry']) * 50
+                        trades.append({'pnl': pnl, 'result': 'WIN', 'gross': pnl})
+                        active_trade = None
+                else:  # SHORT
+                    if bar['high'] >= active_trade['stop']:
+                        pnl = (active_trade['entry'] - active_trade['stop']) * 50
+                        trades.append({'pnl': pnl, 'result': 'LOSS', 'gross': pnl})
+                        active_trade = None
+                    elif bar['low'] <= active_trade['tp']:
+                        pnl = (active_trade['entry'] - active_trade['tp']) * 50
+                        trades.append({'pnl': pnl, 'result': 'WIN', 'gross': pnl})
+                        active_trade = None
+                continue
+            
+            # Check for breakout entry
+            if bar['high'] > or_high:
+                # LONG breakout
+                entry = bar['close']
+                stop = entry - (atr * stop_atr)
+                risk = entry - stop
+                tp = entry + (risk * tp_r)
+                
+                active_trade = {
+                    'direction': 'LONG',
+                    'entry': entry,
+                    'stop': stop,
+                    'tp': tp
+                }
+                
+            elif bar['low'] < or_low:
+                # SHORT breakout
+                entry = bar['close']
+                stop = entry + (atr * stop_atr)
+                risk = stop - entry
+                tp = entry - (risk * tp_r)
+                
+                active_trade = {
+                    'direction': 'SHORT',
+                    'entry': entry,
+                    'stop': stop,
+                    'tp': tp
+                }
+    
+    # Calculate stats
+    if not trades:
+        return {
+            'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
+            'total_pnl': 0, 'profit_factor': 0,
+            'stop_atr': stop_atr, 'tp_r': tp_r
+        }
+    
+    wins = sum(1 for t in trades if t['result'] == 'WIN')
+    losses = len(trades) - wins
+    total_pnl = sum(t['pnl'] for t in trades)
+    
+    gross_wins = sum(t['gross'] for t in trades if t['gross'] > 0)
+    gross_losses = abs(sum(t['gross'] for t in trades if t['gross'] < 0))
+    
+    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
+    
+    return {
+        'trades': len(trades),
+        'wins': wins,
+        'losses': losses,
+        'win_rate': wins / len(trades) if trades else 0,
+        'total_pnl': total_pnl,
+        'gross_wins': gross_wins,
+        'gross_losses': gross_losses,
+        'profit_factor': profit_factor,
+        'stop_atr': stop_atr,
+        'tp_r': tp_r
+    }
+
+
+def run_gridsearch(days: int = 7) -> List[Dict]:
+    """
+    Run the full grid search.
+    """
+    print("=" * 70)
+    print("OPENING RANGE BREAKOUT - GRID SEARCH")
+    print("=" * 70)
+    print(f"Stop ATR: {STOP_ATR_RANGE[0]:.2f} to {STOP_ATR_RANGE[-1]:.2f} ({len(STOP_ATR_RANGE)} values)")
+    print(f"TP R:     {TP_R_RANGE[0]:.1f} to {TP_R_RANGE[-1]:.1f} ({len(TP_R_RANGE)} values)")
+    print(f"Total combinations: {len(STOP_ATR_RANGE) * len(TP_R_RANGE)}")
+    print("=" * 70)
+    
+    # Load data (max 7 days for 1m)
+    actual_days = min(days, 7)
+    end = datetime.now()
+    start = end - timedelta(days=actual_days)
+    
+    print(f"\n[1] Loading {actual_days} days of ES data...")
+    ticker = yf.Ticker("ES=F")
+    df = ticker.history(start=start, end=end, interval="1m")
+    
+    if df is None or len(df) == 0:
+        print("ERROR: No data!")
+        return []
+    
+    # Standardize
+    df.columns = [c.lower() for c in df.columns]
+    df = df.reset_index()
+    col_name = 'Datetime' if 'Datetime' in df.columns else 'datetime'
+    df['time'] = pd.to_datetime(df[col_name]).dt.tz_convert(EST)
+    
+    print(f"    Loaded {len(df)} bars")
+    
+    # Run grid search
+    print(f"\n[2] Running {len(STOP_ATR_RANGE) * len(TP_R_RANGE)} combinations...")
+    
+    results = []
+    total = len(STOP_ATR_RANGE) * len(TP_R_RANGE)
+    
+    for i, (stop_atr, tp_r) in enumerate(itertools.product(STOP_ATR_RANGE, TP_R_RANGE)):
+        result = run_orb_single(df, stop_atr, tp_r)
+        results.append(result)
+        
+        if (i + 1) % 20 == 0:
+            print(f"    Progress: {i+1}/{total}")
+    
+    print(f"\n[3] Grid search complete!")
+    
+    # Sort by profit factor
+    valid_results = [r for r in results if r['trades'] >= 3 and r['profit_factor'] > 0]
+    valid_results.sort(key=lambda x: x['profit_factor'], reverse=True)
+    
+    # Top 10 by Profit Factor
+    print("\n" + "=" * 70)
+    print("TOP 10 BY PROFIT FACTOR (min 3 trades)")
+    print("=" * 70)
+    print(f"{'Stop ATR':>10} | {'TP R':>6} | {'Trades':>7} | {'WR':>6} | {'PnL':>10} | {'PF':>6}")
+    print("-" * 70)
+    
+    for r in valid_results[:10]:
+        print(f"{r['stop_atr']:>10.2f} | {r['tp_r']:>6.1f} | {r['trades']:>7} | "
+              f"{r['win_rate']:>5.1%} | ${r['total_pnl']:>9.2f} | {r['profit_factor']:>6.2f}")
+    
+    # Best configuration
+    if valid_results:
+        best = valid_results[0]
+        print("\n" + "=" * 70)
+        print("BEST CONFIGURATION")
+        print("=" * 70)
+        print(f"  Stop: {best['stop_atr']:.2f} ATR")
+        print(f"  Take Profit: {best['tp_r']:.1f}R")
+        print(f"  Trades: {best['trades']}")
+        print(f"  Win Rate: {best['win_rate']:.1%}")
+        print(f"  Profit Factor: {best['profit_factor']:.2f}")
+        print(f"  Total PnL: ${best['total_pnl']:.2f}")
+    
+    return results
+
+
+if __name__ == "__main__":
+    import argparse
+    
+    parser = argparse.ArgumentParser(description="ORB Grid Search")
+    parser.add_argument("--days", type=int, default=7, help="Days to test")
+    parser.add_argument("--save", action="store_true", help="Save all results to DB")
+    
+    args = parser.parse_args()
+    
+    results = run_gridsearch(days=args.days)
+    
+    if args.save and results:
+        db = ExperimentDB()
+        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
+        
+        for r in results:
+            if r['trades'] > 0:
+                run_id = f"orb_grid_{r['stop_atr']:.2f}atr_{r['tp_r']:.1f}r_{timestamp}"
+                db.store_run(
+                    run_id=run_id,
+                    strategy="orb_gridsearch",
+                    config={'stop_atr': r['stop_atr'], 'tp_r': r['tp_r']},
+                    metrics={
+                        'total_trades': r['trades'],
+                        'wins': r['wins'],
+                        'losses': r['losses'],
+                        'win_rate': r['win_rate'],
+                        'total_pnl': r['total_pnl'],
+                        'profit_factor': r['profit_factor'],
+                    }
+                )
+        
+        print(f"\n[+] Saved {sum(1 for r in results if r['trades'] > 0)} configs to ExperimentDB")
diff --git a/scripts/run_rvap_scan.py b/scripts/run_rvap_scan.py
new file mode 100644
index 0000000..11f267b
--- /dev/null
+++ b/scripts/run_rvap_scan.py
@@ -0,0 +1,309 @@
+#!/usr/bin/env python3
+"""
+Relative Volume at Price (RVAP) Scanner
+
+Theory: Volume confirms breakouts.
+- Approaching PDH on LOW volume → FADE (SHORT)
+- Approaching PDH on HIGH volume (2x avg) → BREAKOUT (LONG)
+
+Custom indicator: RVAP = current volume / 20-bar average volume
+
+Usage:
+    python scripts/run_rvap_scan.py --days 7
+"""
+
+import sys
+from pathlib import Path
+sys.path.insert(0, str(Path(__file__).parent.parent))
+
+import json
+import numpy as np
+import pandas as pd
+import yfinance as yf
+from datetime import datetime, timedelta
+from typing import Dict, Any, List
+
+from src.features.indicators import calculate_atr
+from src.storage import ExperimentDB
+
+
+# =============================================================================
+# RVAP Indicator
+# =============================================================================
+
+def compute_rvap(volume: pd.Series, period: int = 20) -> pd.Series:
+    """
+    Compute Relative Volume at Price.
+    
+    RVAP = current volume / SMA(volume, period)
+    
+    Values:
+    - < 0.5: Very low volume (fade)
+    - 0.5 - 1.5: Normal volume
+    - > 2.0: High volume (breakout)
+    """
+    avg_vol = volume.rolling(window=period).mean()
+    rvap = volume / avg_vol.replace(0, np.nan)
+    return rvap.fillna(1.0)
+
+
+def compute_session_levels(df: pd.DataFrame) -> pd.DataFrame:
+    """Get PDH (Previous Day High), PDL, PDC."""
+    df = df.copy()
+    df['date'] = df['time'].dt.date
+    
+    daily = df.groupby('date').agg({
+        'high': 'max',
+        'low': 'min',
+        'close': 'last'
+    })
+    
+    # Shift by 1 day for "previous day"
+    daily['pdh'] = daily['high'].shift(1)
+    daily['pdl'] = daily['low'].shift(1)
+    daily['pdc'] = daily['close'].shift(1)
+    
+    # Map back
+    df['pdh'] = df['date'].map(daily['pdh'])
+    df['pdl'] = df['date'].map(daily['pdl'])
+    df['pdc'] = df['date'].map(daily['pdc'])
+    
+    return df
+
+
+# =============================================================================
+# RVAP Scanner
+# =============================================================================
+
+def scan_rvap_signals(
+    df: pd.DataFrame,
+    high_vol_threshold: float = 2.0,
+    low_vol_threshold: float = 0.7,
+    approach_pct: float = 0.1,  # Within 0.1% of PDH/PDL
+    lookforward: int = 20,
+) -> List[Dict]:
+    """
+    Scan for RVAP signals near session levels.
+    
+    LONG (Breakout): Approach PDH on high volume (RVAP > 2.0)
+    SHORT (Fade): Approach PDH on low volume (RVAP < 0.7)
+    """
+    df = df.copy()
+    
+    # Compute indicators
+    df['rvap'] = compute_rvap(df['volume'])
+    df = compute_session_levels(df)
+    df['atr'] = calculate_atr(df, period=14).ffill().bfill()
+    
+    records = []
+    
+    for i in range(30, len(df) - lookforward):
+        row = df.iloc[i]
+        
+        if pd.isna(row['pdh']) or pd.isna(row['rvap']):
+            continue
+        
+        close = row['close']
+        high = row['high']
+        pdh = row['pdh']
+        pdl = row['pdl']
+        rvap = row['rvap']
+        atr = row['atr'] if not pd.isna(row['atr']) else 2.0
+        
+        # Check if approaching PDH (within threshold)
+        pdh_distance = abs(high - pdh) / pdh * 100
+        approaching_pdh = pdh_distance < approach_pct and high >= pdh * 0.998
+        
+        # Check if approaching PDL
+        pdl_distance = abs(row['low'] - pdl) / pdl * 100
+        approaching_pdl = pdl_distance < approach_pct and row['low'] <= pdl * 1.002
+        
+        direction = None
+        signal_type = None
+        
+        if approaching_pdh:
+            if rvap >= high_vol_threshold:
+                # High volume at resistance → BREAKOUT LONG
+                direction = 'LONG'
+                signal_type = 'breakout'
+            elif rvap <= low_vol_threshold:
+                # Low volume at resistance → FADE SHORT
+                direction = 'SHORT'
+                signal_type = 'fade'
+        
+        elif approaching_pdl:
+            if rvap >= high_vol_threshold:
+                # High volume at support → BREAKDOWN SHORT
+                direction = 'SHORT'
+                signal_type = 'breakdown'
+            elif rvap <= low_vol_threshold:
+                # Low volume at support → BOUNCE LONG
+                direction = 'LONG'
+                signal_type = 'bounce'
+        
+        if direction is None:
+            continue
+        
+        # Check outcome
+        entry_price = close
+        future_bars = df.iloc[i+1:i+1+lookforward]
+        
+        if direction == 'LONG':
+            target = entry_price + atr
+            stop = entry_price - atr
+            hit_target = (future_bars['high'] >= target).any()
+            hit_stop = (future_bars['low'] <= stop).any()
+        else:
+            target = entry_price - atr
+            stop = entry_price + atr
+            hit_target = (future_bars['low'] <= target).any()
+            hit_stop = (future_bars['high'] >= stop).any()
+        
+        if hit_target and hit_stop:
+            if direction == 'LONG':
+                target_idx = future_bars[future_bars['high'] >= target].index[0]
+                stop_idx = future_bars[future_bars['low'] <= stop].index[0]
+            else:
+                target_idx = future_bars[future_bars['low'] <= target].index[0]
+                stop_idx = future_bars[future_bars['high'] >= stop].index[0]
+            outcome = 'WIN' if target_idx < stop_idx else 'LOSS'
+        elif hit_target:
+            outcome = 'WIN'
+        else:
+            outcome = 'LOSS'
+        
+        # Build record with window
+        window_start = max(0, i - 60)
+        ohlcv_window = df.iloc[window_start:i][['open', 'high', 'low', 'close', 'volume']].values.tolist()
+        
+        records.append({
+            'time': str(row['time']),
+            'direction': direction,
+            'signal_type': signal_type,
+            'label': outcome,
+            'entry_price': entry_price,
+            'rvap': round(rvap, 2),
+            'pdh': pdh,
+            'pdl': pdl,
+            'atr': atr,
+            'window': {
+                'raw_ohlcv_1m': [
+                    {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}
+                    for o, h, l, c, v in ohlcv_window
+                ]
+            },
+            'strategy': 'rvap',
+        })
+    
+    return records
+
+
+def run_rvap_scan(days: int = 7) -> Dict[str, Any]:
+    """Run RVAP scanner."""
+    
+    print("=" * 60)
+    print("RELATIVE VOLUME AT PRICE (RVAP) SCANNER")
+    print("=" * 60)
+    print("Logic:")
+    print("  Approach PDH + HIGH volume (2x) → LONG (breakout)")
+    print("  Approach PDH + LOW volume (<0.7x) → SHORT (fade)")
+    print("=" * 60)
+    
+    # Load data
+    actual_days = min(days, 7)
+    end = datetime.now()
+    start = end - timedelta(days=actual_days)
+    
+    print(f"\n[1] Loading {actual_days} days of ES data...")
+    ticker = yf.Ticker("ES=F")
+    df = ticker.history(start=start, end=end, interval="1m")
+    
+    if df is None or len(df) == 0:
+        print("ERROR: No data!")
+        return {'records': 0}
+    
+    df.columns = [c.lower() for c in df.columns]
+    df = df.reset_index()
+    df['time'] = pd.to_datetime(df['Datetime'] if 'Datetime' in df.columns else df['datetime'])
+    
+    print(f"    Loaded {len(df)} bars")
+    
+    # Scan
+    print(f"\n[2] Scanning for RVAP signals...")
+    records = scan_rvap_signals(df)
+    
+    if not records:
+        print("    No signals found!")
+        return {'records': 0}
+    
+    # Stats
+    wins = sum(1 for r in records if r['label'] == 'WIN')
+    
+    # Breakdown by signal type
+    breakouts = [r for r in records if r['signal_type'] == 'breakout']
+    fades = [r for r in records if r['signal_type'] == 'fade']
+    breakdowns = [r for r in records if r['signal_type'] == 'breakdown']
+    bounces = [r for r in records if r['signal_type'] == 'bounce']
+    
+    print(f"\n    Total signals: {len(records)}")
+    print(f"    WIN: {wins} | LOSS: {len(records) - wins}")
+    print(f"    Win Rate: {wins/len(records):.1%}")
+    
+    print(f"\n    By signal type:")
+    for name, group in [('Breakout (high vol)', breakouts), 
+                        ('Fade (low vol)', fades),
+                        ('Breakdown', breakdowns),
+                        ('Bounce', bounces)]:
+        if group:
+            g_wins = sum(1 for r in group if r['label'] == 'WIN')
+            print(f"      {name}: {len(group)} trades, {g_wins/len(group):.1%} WR")
+    
+    # Save
+    output_dir = Path("results/rvap")
+    output_dir.mkdir(parents=True, exist_ok=True)
+    output_path = output_dir / "records.jsonl"
+    
+    with open(output_path, 'w') as f:
+        for rec in records:
+            f.write(json.dumps(rec) + '\n')
+    
+    print(f"\n[3] Saved to {output_path}")
+    
+    # Store summary
+    db = ExperimentDB()
+    run_id = f"rvap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
+    db.store_run(
+        run_id=run_id,
+        strategy="rvap",
+        config={
+            'high_vol_threshold': 2.0,
+            'low_vol_threshold': 0.7,
+        },
+        metrics={
+            'total_trades': len(records),
+            'wins': wins,
+            'losses': len(records) - wins,
+            'win_rate': wins/len(records) if records else 0,
+            'total_pnl': 0,
+        }
+    )
+    print(f"    Stored: {run_id}")
+    
+    return {
+        'records': len(records),
+        'wins': wins,
+        'win_rate': wins/len(records) if records else 0,
+        'breakouts': len(breakouts),
+        'fades': len(fades),
+    }
+
+
+if __name__ == "__main__":
+    import argparse
+    
+    parser = argparse.ArgumentParser(description="RVAP Scanner")
+    parser.add_argument("--days", type=int, default=7, help="Days to scan")
+    
+    args = parser.parse_args()
+    
+    results = run_rvap_scan(args.days)
diff --git a/scripts/run_walkforward_daily.py b/scripts/run_walkforward_daily.py
new file mode 100644
index 0000000..346e707
--- /dev/null
+++ b/scripts/run_walkforward_daily.py
@@ -0,0 +1,322 @@
+#!/usr/bin/env python3
+"""
+Walk-Forward Daily Retrain Test
+
+Theory: Hyper-aggressive retraining adapts to regime changes faster.
+Method: Retrain model EVERY DAY using a rolling 2-week window.
+
+Compare:
+- Static Model: Trained once on first 2 weeks
+- Adaptive Model: Retrained daily on rolling 2-week window
+
+Usage:
+    python scripts/run_walkforward_daily.py --days 7
+"""
+
+import sys
+from pathlib import Path
+sys.path.insert(0, str(Path(__file__).parent.parent))
+
+import numpy as np
+import pandas as pd
+import torch
+import torch.nn as nn
+from torch.utils.data import Dataset, DataLoader
+import yfinance as yf
+from datetime import datetime, timedelta
+from typing import Dict, Any, List, Tuple
+
+from src.storage import ExperimentDB
+
+
+# =============================================================================
+# Simple CNN for direction prediction
+# =============================================================================
+
+class SimpleCNN(nn.Module):
+    """Lightweight CNN for quick retraining."""
+    
+    def __init__(self, lookback: int = 30, features: int = 5):
+        super().__init__()
+        
+        self.conv = nn.Sequential(
+            nn.Conv1d(features, 16, kernel_size=3, padding=1),
+            nn.ReLU(),
+            nn.MaxPool1d(2),
+            nn.Conv1d(16, 32, kernel_size=3, padding=1),
+            nn.ReLU(),
+            nn.AdaptiveAvgPool1d(4),
+            nn.Flatten(),
+        )
+        
+        self.fc = nn.Sequential(
+            nn.Linear(32 * 4, 16),
+            nn.ReLU(),
+            nn.Linear(16, 2),  # UP or DOWN
+        )
+    
+    def forward(self, x):
+        # x: (batch, seq, features) -> (batch, features, seq)
+        x = x.permute(0, 2, 1)
+        return self.fc(self.conv(x))
+
+
+# =============================================================================
+# Dataset from raw OHLCV
+# =============================================================================
+
+class DirectionDataset(Dataset):
+    """Simple next-bar direction prediction dataset."""
+    
+    def __init__(self, df: pd.DataFrame, lookback: int = 30, lookahead: int = 5):
+        self.samples = []
+        self.labels = []
+        
+        for i in range(lookback, len(df) - lookahead):
+            # Window
+            window = df.iloc[i-lookback:i][['open', 'high', 'low', 'close', 'volume']].values
+            
+            # Normalize
+            window = self._normalize(window)
+            
+            # Label: price direction over lookahead
+            current_close = df.iloc[i]['close']
+            future_close = df.iloc[i + lookahead]['close']
+            label = 0 if future_close > current_close else 1  # 0 = UP, 1 = DOWN
+            
+            self.samples.append(window)
+            self.labels.append(label)
+    
+    def _normalize(self, data):
+        mean = np.mean(data, axis=0)
+        std = np.std(data, axis=0) + 1e-8
+        return (data - mean) / std
+    
+    def __len__(self):
+        return len(self.samples)
+    
+    def __getitem__(self, idx):
+        return (
+            torch.tensor(self.samples[idx], dtype=torch.float32),
+            torch.tensor(self.labels[idx], dtype=torch.long)
+        )
+
+
+# =============================================================================
+# Training helper
+# =============================================================================
+
+def quick_train(
+    model: nn.Module,
+    train_loader: DataLoader,
+    epochs: int = 5,
+    lr: float = 0.001,
+    device: str = 'cuda',
+) -> nn.Module:
+    """Quick training for daily retrain."""
+    model = model.to(device)
+    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
+    criterion = nn.CrossEntropyLoss()
+    
+    model.train()
+    for epoch in range(epochs):
+        for x, y in train_loader:
+            x, y = x.to(device), y.to(device)
+            optimizer.zero_grad()
+            out = model(x)
+            loss = criterion(out, y)
+            loss.backward()
+            optimizer.step()
+    
+    return model
+
+
+def evaluate(
+    model: nn.Module,
+    test_loader: DataLoader,
+    device: str = 'cuda',
+) -> float:
+    """Evaluate accuracy."""
+    model.eval()
+    correct = 0
+    total = 0
+    
+    with torch.no_grad():
+        for x, y in test_loader:
+            x, y = x.to(device), y.to(device)
+            out = model(x)
+            _, pred = out.max(1)
+            correct += (pred == y).sum().item()
+            total += y.size(0)
+    
+    return correct / total if total > 0 else 0
+
+
+# =============================================================================
+# Walk-Forward Engine
+# =============================================================================
+
+def run_walkforward_daily(days: int = 7, train_days: int = 5) -> Dict[str, Any]:
+    """
+    Run walk-forward with daily retraining.
+    
+    Due to yfinance 1m limit (7 days), we simulate with available data.
+    """
+    print("=" * 60)
+    print("WALK-FORWARD DAILY RETRAIN TEST")
+    print("=" * 60)
+    print(f"Comparing: Static vs Daily-Retrain models")
+    print(f"Training window: {train_days} days (rolling)")
+    print("=" * 60)
+    
+    # Load data
+    actual_days = min(days, 7)
+    end = datetime.now()
+    start = end - timedelta(days=actual_days)
+    
+    print(f"\n[1] Loading {actual_days} days of ES data...")
+    ticker = yf.Ticker("ES=F")
+    df = ticker.history(start=start, end=end, interval="1m")
+    
+    if df is None or len(df) == 0:
+        print("ERROR: No data!")
+        return {}
+    
+    df.columns = [c.lower() for c in df.columns]
+    df = df.reset_index()
+    df['time'] = pd.to_datetime(df['Datetime'] if 'Datetime' in df.columns else df['datetime'])
+    
+    print(f"    Loaded {len(df)} bars")
+    
+    # Split by dates
+    df['date'] = df['time'].dt.date
+    unique_dates = sorted(df['date'].unique())
+    
+    if len(unique_dates) < train_days + 2:
+        print(f"ERROR: Need at least {train_days + 2} days, have {len(unique_dates)}")
+        return {}
+    
+    print(f"    {len(unique_dates)} trading days")
+    
+    device = 'cuda' if torch.cuda.is_available() else 'cpu'
+    print(f"    Device: {device}")
+    
+    # =========================================================================
+    # Static Model: Train once on first train_days
+    # =========================================================================
+    print(f"\n[2] Training STATIC model on first {train_days} days...")
+    
+    train_dates = unique_dates[:train_days]
+    train_df = df[df['date'].isin(train_dates)]
+    
+    train_ds = DirectionDataset(train_df)
+    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
+    
+    static_model = SimpleCNN()
+    static_model = quick_train(static_model, train_loader, epochs=10, device=device)
+    
+    print(f"    Trained on {len(train_ds)} samples")
+    
+    # =========================================================================
+    # Walk-Forward: Test each remaining day
+    # =========================================================================
+    print(f"\n[3] Walk-forward testing...")
+    
+    static_results = []
+    adaptive_results = []
+    
+    test_dates = unique_dates[train_days:]
+    
+    for i, test_date in enumerate(test_dates):
+        test_df = df[df['date'] == test_date]
+        if len(test_df) < 40:
+            continue
+        
+        test_ds = DirectionDataset(test_df)
+        test_loader = DataLoader(test_ds, batch_size=32)
+        
+        # Static model accuracy
+        static_acc = evaluate(static_model, test_loader, device)
+        static_results.append(static_acc)
+        
+        # Adaptive: Retrain on rolling window up to this day
+        rolling_end_idx = unique_dates.index(test_date)
+        rolling_start_idx = max(0, rolling_end_idx - train_days)
+        rolling_dates = unique_dates[rolling_start_idx:rolling_end_idx]
+        
+        rolling_df = df[df['date'].isin(rolling_dates)]
+        if len(rolling_df) > 100:
+            rolling_ds = DirectionDataset(rolling_df)
+            rolling_loader = DataLoader(rolling_ds, batch_size=32, shuffle=True)
+            
+            adaptive_model = SimpleCNN()
+            adaptive_model = quick_train(adaptive_model, rolling_loader, epochs=5, device=device)
+            
+            adaptive_acc = evaluate(adaptive_model, test_loader, device)
+        else:
+            adaptive_acc = static_acc  # Fallback if not enough data
+        
+        adaptive_results.append(adaptive_acc)
+        
+        print(f"    Day {i+1} ({test_date}): Static={static_acc:.1%}, Adaptive={adaptive_acc:.1%}")
+    
+    # Summary
+    avg_static = np.mean(static_results) if static_results else 0
+    avg_adaptive = np.mean(adaptive_results) if adaptive_results else 0
+    
+    print("\n" + "=" * 60)
+    print("RESULTS")
+    print("=" * 60)
+    print(f"  Test days: {len(test_dates)}")
+    print(f"  Static Model Avg Accuracy:   {avg_static:.1%}")
+    print(f"  Adaptive Model Avg Accuracy: {avg_adaptive:.1%}")
+    
+    diff = (avg_adaptive - avg_static) * 100
+    if avg_adaptive > avg_static:
+        print(f"\n  ✓ Daily retraining wins by {diff:.1f}pp!")
+        print(f"  → Hyper-aggressive adaptation DOES help")
+    else:
+        print(f"\n  ✗ Static model wins by {-diff:.1f}pp")
+        print(f"  → Constant retraining adds noise, doesn't help")
+    
+    # Store
+    db = ExperimentDB()
+    run_id = f"walkforward_daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
+    db.store_run(
+        run_id=run_id,
+        strategy="walkforward_daily",
+        config={
+            'train_days': train_days,
+            'test_days': len(test_dates),
+            'retrain_frequency': 'daily',
+        },
+        metrics={
+            'total_trades': len(test_dates),
+            'wins': int(len(test_dates) * avg_adaptive),
+            'losses': int(len(test_dates) * (1 - avg_adaptive)),
+            'win_rate': avg_adaptive,
+            'static_win_rate': avg_static,
+            'adaptive_win_rate': avg_adaptive,
+            'total_pnl': 0,
+        }
+    )
+    print(f"\n[+] Stored: {run_id}")
+    
+    return {
+        'static_acc': avg_static,
+        'adaptive_acc': avg_adaptive,
+        'improvement': diff,
+        'test_days': len(test_dates),
+    }
+
+
+if __name__ == "__main__":
+    import argparse
+    
+    parser = argparse.ArgumentParser(description="Walk-Forward Daily Retrain")
+    parser.add_argument("--days", type=int, default=7, help="Total days")
+    parser.add_argument("--train-days", type=int, default=4, help="Training window")
+    
+    args = parser.parse_args()
+    
+    results = run_walkforward_daily(args.days, args.train_days)
diff --git a/scripts/train_fusion_mtf.py b/scripts/train_fusion_mtf.py
new file mode 100644
index 0000000..7620893
--- /dev/null
+++ b/scripts/train_fusion_mtf.py
@@ -0,0 +1,414 @@
+#!/usr/bin/env python3
+"""
+Multi-Timeframe Fusion Model
+
+Theory: Only trade if higher timeframe (1H) agrees with entry direction.
+Input: 30 bars of 1m data + 5 bars of 1H data
+Filter: If 1H is bullish (close > open over window), only take LONGS.
+
+Usage:
+    python scripts/train_fusion_mtf.py
+"""
+
+import sys
+from pathlib import Path
+sys.path.insert(0, str(Path(__file__).parent.parent))
+
+import numpy as np
+import pandas as pd
+import torch
+import torch.nn as nn
+import yfinance as yf
+from torch.utils.data import Dataset, DataLoader, random_split
+from datetime import datetime, timedelta
+from typing import Dict, Any, List, Tuple
+
+from src.storage import ExperimentDB
+
+
+# =============================================================================
+# Multi-Timeframe Fusion Model
+# =============================================================================
+
+class MTFFusionModel(nn.Module):
+    """
+    Fusion model that processes 1m and 1H data separately, then combines.
+    
+    Architecture:
+    - 1m Branch: CNN for short-term patterns (30 bars × 5 features)
+    - 1H Branch: MLP for trend context (5 bars × 5 features)
+    - Fusion: Concatenate + FC layers
+    """
+    
+    def __init__(
+        self,
+        bars_1m: int = 30,
+        bars_1h: int = 5,
+        num_features: int = 5,  # OHLCV
+        num_classes: int = 2,   # LONG or SHORT
+    ):
+        super().__init__()
+        
+        self.bars_1m = bars_1m
+        self.bars_1h = bars_1h
+        
+        # 1-Minute Branch (CNN)
+        self.cnn_1m = nn.Sequential(
+            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
+            nn.ReLU(),
+            nn.MaxPool1d(2),
+            nn.Conv1d(32, 64, kernel_size=3, padding=1),
+            nn.ReLU(),
+            nn.AdaptiveAvgPool1d(4),
+            nn.Flatten(),
+        )
+        cnn_out_size = 64 * 4  # 256
+        
+        # 1-Hour Branch (MLP)
+        self.mlp_1h = nn.Sequential(
+            nn.Flatten(),
+            nn.Linear(bars_1h * num_features, 32),
+            nn.ReLU(),
+            nn.Linear(32, 16),
+            nn.ReLU(),
+        )
+        mlp_out_size = 16
+        
+        # Fusion Head
+        self.fusion = nn.Sequential(
+            nn.Linear(cnn_out_size + mlp_out_size, 64),
+            nn.ReLU(),
+            nn.Dropout(0.3),
+            nn.Linear(64, num_classes),
+        )
+    
+    def forward(self, x_1m, x_1h):
+        """
+        Args:
+            x_1m: (batch, bars_1m, features) - 1-minute data
+            x_1h: (batch, bars_1h, features) - 1-hour data
+        """
+        # CNN expects (batch, channels, seq_len)
+        x_1m = x_1m.permute(0, 2, 1)
+        
+        # Process each branch
+        feat_1m = self.cnn_1m(x_1m)      # (batch, 256)
+        feat_1h = self.mlp_1h(x_1h)       # (batch, 16)
+        
+        # Fuse
+        fused = torch.cat([feat_1m, feat_1h], dim=1)
+        
+        return self.fusion(fused)
+    
+    def get_1h_trend(self, x_1h):
+        """Determine if 1H trend is bullish (returns True/False per sample)."""
+        # Simple: compare first close to last close
+        # x_1h: (batch, bars, features) where features = [O, H, L, C, V]
+        first_close = x_1h[:, 0, 3]  # First bar close
+        last_close = x_1h[:, -1, 3]  # Last bar close
+        return last_close > first_close  # Bullish if rising
+
+
+# =============================================================================
+# Dataset with MTF data
+# =============================================================================
+
+class MTFDataset(Dataset):
+    """Generate multi-timeframe samples from market data."""
+    
+    def __init__(
+        self,
+        df_1m: pd.DataFrame,
+        df_1h: pd.DataFrame,
+        bars_1m: int = 30,
+        bars_1h: int = 5,
+    ):
+        self.samples_1m = []
+        self.samples_1h = []
+        self.labels = []
+        self.bars_1m = bars_1m
+        self.bars_1h = bars_1h
+        
+        # Align 1m and 1h data
+        df_1m = df_1m.copy()
+        df_1h = df_1h.copy()
+        
+        df_1m['hour'] = df_1m['time'].dt.floor('h')
+        
+        # For each valid point, create a sample
+        for i in range(bars_1m + 60, len(df_1m) - 10):  # Leave room for outcome
+            current_time = df_1m.iloc[i]['time']
+            current_hour = current_time.floor('h')
+            
+            # Get 1m window
+            window_1m = df_1m.iloc[i-bars_1m:i][['open', 'high', 'low', 'close', 'volume']].values
+            
+            # Get 1h window (last 5 hours before current)
+            hourly_mask = df_1h['time'] < current_hour
+            hourly_data = df_1h[hourly_mask].tail(bars_1h)
+            
+            if len(hourly_data) < bars_1h:
+                continue
+            
+            window_1h = hourly_data[['open', 'high', 'low', 'close', 'volume']].values
+            
+            # Normalize each window separately (Z-score)
+            window_1m = self._normalize(window_1m)
+            window_1h = self._normalize(window_1h)
+            
+            # Simple label: next 10 bars go up = LONG(0), down = SHORT(1)
+            future_close = df_1m.iloc[i + 10]['close']
+            current_close = df_1m.iloc[i]['close']
+            label = 0 if future_close > current_close else 1
+            
+            self.samples_1m.append(window_1m)
+            self.samples_1h.append(window_1h)
+            self.labels.append(label)
+        
+        print(f"MTF Dataset: {len(self.samples_1m)} samples")
+        print(f"  LONG (up): {sum(1 for l in self.labels if l == 0)}")
+        print(f"  SHORT (down): {sum(1 for l in self.labels if l == 1)}")
+    
+    def _normalize(self, data):
+        """Z-score normalize."""
+        mean = np.mean(data, axis=0)
+        std = np.std(data, axis=0) + 1e-8
+        return (data - mean) / std
+    
+    def __len__(self):
+        return len(self.samples_1m)
+    
+    def __getitem__(self, idx):
+        x_1m = torch.tensor(self.samples_1m[idx], dtype=torch.float32)
+        x_1h = torch.tensor(self.samples_1h[idx], dtype=torch.float32)
+        y = torch.tensor(self.labels[idx], dtype=torch.long)
+        return x_1m, x_1h, y
+
+
+# =============================================================================
+# Training with Trend Filter
+# =============================================================================
+
+def train_fusion_model(
+    model: nn.Module,
+    train_loader: DataLoader,
+    val_loader: DataLoader,
+    epochs: int = 30,
+    lr: float = 0.001,
+    device: str = 'cuda',
+    use_trend_filter: bool = True,
+) -> Tuple[dict, float, Dict]:
+    """
+    Train fusion model with optional 1H trend filter.
+    
+    If use_trend_filter=True:
+    - Only count LONG predictions as correct if 1H is bullish
+    - Only count SHORT predictions as correct if 1H is bearish
+    """
+    model = model.to(device)
+    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
+    criterion = nn.CrossEntropyLoss()
+    
+    best_acc = 0.0
+    best_state = None
+    stats = {'filtered_trades': 0, 'total_trades': 0}
+    
+    for epoch in range(epochs):
+        model.train()
+        total_loss = 0
+        
+        for x_1m, x_1h, y in train_loader:
+            x_1m, x_1h, y = x_1m.to(device), x_1h.to(device), y.to(device)
+            
+            optimizer.zero_grad()
+            out = model(x_1m, x_1h)
+            loss = criterion(out, y)
+            loss.backward()
+            optimizer.step()
+            
+            total_loss += loss.item()
+        
+        # Validate with trend filter
+        model.eval()
+        correct_unfiltered = 0
+        correct_filtered = 0
+        total = 0
+        filtered_trades = 0
+        
+        with torch.no_grad():
+            for x_1m, x_1h, y in val_loader:
+                x_1m, x_1h, y = x_1m.to(device), x_1h.to(device), y.to(device)
+                
+                out = model(x_1m, x_1h)
+                _, pred = out.max(1)
+                
+                # Get 1H trend
+                trend_bullish = model.get_1h_trend(x_1h)
+                
+                for i in range(len(pred)):
+                    total += 1
+                    
+                    # Unfiltered accuracy
+                    if pred[i] == y[i]:
+                        correct_unfiltered += 1
+                    
+                    # Filtered: only count if direction matches trend
+                    if use_trend_filter:
+                        is_long = pred[i] == 0
+                        should_take = (is_long and trend_bullish[i]) or (not is_long and not trend_bullish[i])
+                        
+                        if should_take:
+                            filtered_trades += 1
+                            if pred[i] == y[i]:
+                                correct_filtered += 1
+                    else:
+                        filtered_trades += 1
+                        if pred[i] == y[i]:
+                            correct_filtered += 1
+        
+        acc_unfiltered = correct_unfiltered / total if total > 0 else 0
+        acc_filtered = correct_filtered / filtered_trades if filtered_trades > 0 else 0
+        
+        if acc_filtered > best_acc:
+            best_acc = acc_filtered
+            best_state = model.state_dict().copy()
+            stats = {
+                'filtered_trades': filtered_trades,
+                'total_trades': total,
+                'unfiltered_acc': acc_unfiltered,
+                'filtered_acc': acc_filtered,
+            }
+        
+        if (epoch + 1) % 10 == 0 or epoch == 0:
+            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} "
+                  f"- Val Acc: {acc_unfiltered:.1%} (unfiltered) / {acc_filtered:.1%} (filtered)")
+    
+    return best_state, best_acc, stats
+
+
+# =============================================================================
+# Main
+# =============================================================================
+
+def run_fusion_comparison(days: int = 7) -> Dict[str, Any]:
+    """
+    Train fusion model and compare filtered vs unfiltered expectancy.
+    """
+    print("=" * 60)
+    print("MULTI-TIMEFRAME FUSION MODEL")
+    print("=" * 60)
+    print("Input: 30 bars (1m) + 5 bars (1H)")
+    print("Filter: Only trade if 1H trend agrees with direction")
+    print("=" * 60)
+    
+    # Load data
+    actual_days = min(days, 7)
+    end = datetime.now()
+    start = end - timedelta(days=actual_days)
+    
+    print(f"\n[1] Loading {actual_days} days of ES data...")
+    ticker = yf.Ticker("ES=F")
+    
+    df_1m = ticker.history(start=start, end=end, interval="1m")
+    df_1h = ticker.history(start=start - timedelta(days=30), end=end, interval="1h")
+    
+    if df_1m is None or len(df_1m) == 0:
+        print("ERROR: No data!")
+        return {}
+    
+    # Standardize
+    for df in [df_1m, df_1h]:
+        df.columns = [c.lower() for c in df.columns]
+    
+    df_1m = df_1m.reset_index()
+    df_1h = df_1h.reset_index()
+    df_1m['time'] = pd.to_datetime(df_1m['Datetime'] if 'Datetime' in df_1m.columns else df_1m['datetime'])
+    df_1h['time'] = pd.to_datetime(df_1h['Datetime'] if 'Datetime' in df_1h.columns else df_1h['datetime'])
+    
+    print(f"    1m: {len(df_1m)} bars")
+    print(f"    1h: {len(df_1h)} bars")
+    
+    # Create dataset
+    print(f"\n[2] Creating MTF dataset...")
+    dataset = MTFDataset(df_1m, df_1h, bars_1m=30, bars_1h=5)
+    
+    if len(dataset) < 50:
+        print("ERROR: Not enough samples")
+        return {}
+    
+    train_size = int(0.8 * len(dataset))
+    val_size = len(dataset) - train_size
+    train_ds, val_ds = random_split(dataset, [train_size, val_size])
+    
+    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
+    val_loader = DataLoader(val_ds, batch_size=32)
+    
+    # Train model
+    print(f"\n[3] Training Fusion Model...")
+    device = 'cuda' if torch.cuda.is_available() else 'cpu'
+    print(f"    Device: {device}")
+    
+    model = MTFFusionModel(bars_1m=30, bars_1h=5, num_classes=2)
+    
+    best_state, best_acc, stats = train_fusion_model(
+        model, train_loader, val_loader,
+        epochs=30, lr=0.001, device=device,
+        use_trend_filter=True
+    )
+    
+    # Save
+    model_path = Path("models/mtf_fusion.pth")
+    model_path.parent.mkdir(exist_ok=True)
+    torch.save(best_state, model_path)
+    
+    # Results
+    print("\n" + "=" * 60)
+    print("RESULTS: TREND FILTER IMPACT")
+    print("=" * 60)
+    print(f"  Unfiltered Accuracy: {stats['unfiltered_acc']:.1%}")
+    print(f"  Filtered Accuracy:   {stats['filtered_acc']:.1%}")
+    print(f"  Trades Taken:        {stats['filtered_trades']}/{stats['total_trades']} "
+          f"({100*stats['filtered_trades']/stats['total_trades']:.0f}%)")
+    
+    improvement = (stats['filtered_acc'] - stats['unfiltered_acc']) * 100
+    if improvement > 0:
+        print(f"\n  ✓ Filter IMPROVED expectancy by {improvement:.1f} percentage points!")
+    else:
+        print(f"\n  ✗ Filter did not improve expectancy ({improvement:.1f}pp)")
+    
+    # Store
+    db = ExperimentDB()
+    run_id = f"mtf_fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
+    db.store_run(
+        run_id=run_id,
+        strategy="mtf_fusion",
+        config={
+            'bars_1m': 30,
+            'bars_1h': 5,
+            'use_trend_filter': True,
+        },
+        metrics={
+            'total_trades': stats['total_trades'],
+            'filtered_trades': stats['filtered_trades'],
+            'wins': int(stats['filtered_trades'] * stats['filtered_acc']),
+            'losses': int(stats['filtered_trades'] * (1 - stats['filtered_acc'])),
+            'win_rate': stats['filtered_acc'],
+            'unfiltered_win_rate': stats['unfiltered_acc'],
+            'total_pnl': 0,
+        },
+        model_path=str(model_path)
+    )
+    print(f"\n[+] Saved to ExperimentDB: {run_id}")
+    
+    return stats
+
+
+if __name__ == "__main__":
+    import argparse
+    
+    parser = argparse.ArgumentParser(description="MTF Fusion Model")
+    parser.add_argument("--days", type=int, default=7, help="Days to train on")
+    
+    args = parser.parse_args()
+    
+    results = run_fusion_comparison(args.days)
diff --git a/scripts/train_lstm_compare.py b/scripts/train_lstm_compare.py
new file mode 100644
index 0000000..398bf3d
--- /dev/null
+++ b/scripts/train_lstm_compare.py
@@ -0,0 +1,336 @@
+#!/usr/bin/env python3
+"""
+LSTM vs CNN Comparison
+
+Engineer theory: CNN focuses on shapes, LSTM captures price flow/sequence.
+Test: Train an LSTM on 60-bar close price sequences, compare to baseline CNN.
+
+Usage:
+    python scripts/train_lstm_compare.py --input results/ict_ifvg/records.jsonl
+"""
+
+import sys
+from pathlib import Path
+sys.path.insert(0, str(Path(__file__).parent.parent))
+
+import json
+import numpy as np
+import torch
+import torch.nn as nn
+from torch.utils.data import Dataset, DataLoader, random_split
+from datetime import datetime
+from typing import Dict, Any, List, Tuple
+
+from src.storage import ExperimentDB
+from src.features.engine import normalize_ohlcv_window, FeatureConfig
+
+
+# =============================================================================
+# LSTM Model Architecture
+# =============================================================================
+
+class PriceLSTM(nn.Module):
+    """
+    LSTM for price sequence classification.
+    
+    Input: (batch, seq_len, features)
+    Output: (batch, num_classes)
+    """
+    def __init__(
+        self,
+        input_size: int = 1,      # Just close prices
+        hidden_size: int = 64,
+        num_layers: int = 2,
+        num_classes: int = 4,     # LONG_WIN, LONG_LOSS, SHORT_WIN, SHORT_LOSS
+        dropout: float = 0.3,
+    ):
+        super().__init__()
+        
+        self.lstm = nn.LSTM(
+            input_size=input_size,
+            hidden_size=hidden_size,
+            num_layers=num_layers,
+            batch_first=True,
+            dropout=dropout if num_layers > 1 else 0,
+            bidirectional=False,  # Causal - only look back
+        )
+        
+        self.fc = nn.Sequential(
+            nn.Linear(hidden_size, 32),
+            nn.ReLU(),
+            nn.Dropout(dropout),
+            nn.Linear(32, num_classes),
+        )
+    
+    def forward(self, x):
+        # x: (batch, seq_len, features)
+        lstm_out, (h_n, c_n) = self.lstm(x)
+        
+        # Use last hidden state
+        last_hidden = h_n[-1]  # (batch, hidden_size)
+        
+        return self.fc(last_hidden)
+
+
+# =============================================================================
+# Dataset
+# =============================================================================
+
+class LSTMDataset(Dataset):
+    """Dataset for LSTM training using close price sequences."""
+    
+    LABEL_MAP = {
+        'LONG_WIN': 0, 'LONG_LOSS': 1,
+        'SHORT_WIN': 2, 'SHORT_LOSS': 3,
+    }
+    
+    def __init__(self, records: List[Dict], lookback: int = 60):
+        self.samples = []
+        self.labels = []
+        self.lookback = lookback
+        
+        for rec in records:
+            # Get label
+            direction = rec.get('direction', 'LONG')
+            outcome = rec.get('label', rec.get('outcome', 'LOSS'))
+            label_str = f"{direction}_{outcome}"
+            
+            if label_str not in self.LABEL_MAP:
+                continue
+            
+            # Get close prices from window
+            window = rec.get('window', {})
+            ohlcv = window.get('raw_ohlcv_1m', [])
+            
+            if len(ohlcv) < lookback:
+                continue
+            
+            # Extract close prices - handle both dict and array format
+            closes = []
+            for bar in ohlcv[-lookback:]:
+                if isinstance(bar, dict):
+                    closes.append(bar.get('close', bar.get('Close', 0)))
+                else:
+                    closes.append(bar[3])  # Index 3 = close in array format
+            
+            closes = np.array(closes, dtype=np.float32)
+            
+            # Normalize: Z-score
+            mean = np.mean(closes)
+            std = np.std(closes) + 1e-8
+            closes_norm = (closes - mean) / std
+            
+            self.samples.append(closes_norm)
+            self.labels.append(self.LABEL_MAP[label_str])
+        
+        print(f"Dataset: {len(self.samples)} samples")
+        for label, idx in self.LABEL_MAP.items():
+            count = sum(1 for l in self.labels if l == idx)
+            print(f"  {label} ({idx}): {count}")
+    
+    def __len__(self):
+        return len(self.samples)
+    
+    def __getitem__(self, idx):
+        x = torch.tensor(self.samples[idx], dtype=torch.float32).unsqueeze(-1)  # (seq, 1)
+        y = torch.tensor(self.labels[idx], dtype=torch.long)
+        return x, y
+
+
+# =============================================================================
+# Training
+# =============================================================================
+
+def train_lstm(
+    model: nn.Module,
+    train_loader: DataLoader,
+    val_loader: DataLoader,
+    epochs: int = 50,
+    lr: float = 0.001,
+    device: str = 'cuda',
+) -> Tuple[dict, float]:
+    """Train LSTM and return best state dict and accuracy."""
+    
+    model = model.to(device)
+    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
+    criterion = nn.CrossEntropyLoss()
+    
+    best_acc = 0.0
+    best_state = None
+    
+    for epoch in range(epochs):
+        # Train
+        model.train()
+        total_loss = 0
+        
+        for x, y in train_loader:
+            x, y = x.to(device), y.to(device)
+            
+            optimizer.zero_grad()
+            out = model(x)
+            loss = criterion(out, y)
+            loss.backward()
+            optimizer.step()
+            
+            total_loss += loss.item()
+        
+        # Validate
+        model.eval()
+        correct = 0
+        total = 0
+        
+        with torch.no_grad():
+            for x, y in val_loader:
+                x, y = x.to(device), y.to(device)
+                out = model(x)
+                _, pred = out.max(1)
+                correct += (pred == y).sum().item()
+                total += y.size(0)
+        
+        acc = correct / total if total > 0 else 0
+        
+        if acc > best_acc:
+            best_acc = acc
+            best_state = model.state_dict().copy()
+        
+        if (epoch + 1) % 10 == 0 or epoch == 0:
+            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {acc:.1%}")
+    
+    return best_state, best_acc
+
+
+# =============================================================================
+# Main
+# =============================================================================
+
+def run_comparison(input_path: str, lookback: int = 60) -> Dict[str, Any]:
+    """
+    Train LSTM and compare to CNN baseline.
+    """
+    print("=" * 60)
+    print("LSTM vs CNN COMPARISON")
+    print("=" * 60)
+    print(f"Lookback: {lookback} bars (close prices only)")
+    print("=" * 60)
+    
+    # Load records
+    print(f"\n[1] Loading records from {input_path}...")
+    records = []
+    with open(input_path) as f:
+        for line in f:
+            records.append(json.loads(line))
+    print(f"    Loaded {len(records)} records")
+    
+    # Create dataset
+    print(f"\n[2] Creating LSTM dataset...")
+    dataset = LSTMDataset(records, lookback=lookback)
+    
+    if len(dataset) < 20:
+        print("ERROR: Not enough samples for training")
+        return {'lstm_acc': 0, 'cnn_acc': 0}
+    
+    # Split
+    train_size = int(0.8 * len(dataset))
+    val_size = len(dataset) - train_size
+    train_ds, val_ds = random_split(dataset, [train_size, val_size])
+    
+    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
+    val_loader = DataLoader(val_ds, batch_size=16)
+    
+    print(f"    Train: {len(train_ds)}, Val: {len(val_ds)}")
+    
+    # Train LSTM
+    print(f"\n[3] Training LSTM...")
+    device = 'cuda' if torch.cuda.is_available() else 'cpu'
+    print(f"    Device: {device}")
+    
+    lstm_model = PriceLSTM(
+        input_size=1,
+        hidden_size=64,
+        num_layers=2,
+        num_classes=4,
+    )
+    
+    lstm_state, lstm_acc = train_lstm(
+        lstm_model, train_loader, val_loader,
+        epochs=50, lr=0.001, device=device
+    )
+    
+    # Save LSTM
+    lstm_path = Path("models/lstm_price_seq.pth")
+    lstm_path.parent.mkdir(exist_ok=True)
+    torch.save(lstm_state, lstm_path)
+    print(f"\n    LSTM saved to {lstm_path}")
+    print(f"    LSTM Val Accuracy: {lstm_acc:.1%}")
+    
+    # Load CNN baseline if exists
+    cnn_acc = 0.0
+    cnn_path = Path("models/ifvg_4class_cnn.pth")
+    if cnn_path.exists():
+        print(f"\n[4] Loading CNN baseline from {cnn_path}...")
+        # We'd need to evaluate CNN on same data - for now use stored metrics
+        db = ExperimentDB()
+        cnn_runs = db.query_best("win_rate", strategy="cnn_training", top_k=1)
+        if cnn_runs:
+            cnn_acc = cnn_runs[0].get('win_rate', 0)
+            print(f"    CNN baseline accuracy: {cnn_acc:.1%}")
+    
+    # Compare
+    print("\n" + "=" * 60)
+    print("RESULTS COMPARISON")
+    print("=" * 60)
+    print(f"  LSTM Accuracy: {lstm_acc:.1%}")
+    print(f"  CNN Accuracy:  {cnn_acc:.1%}")
+    
+    if lstm_acc > cnn_acc:
+        diff = (lstm_acc - cnn_acc) * 100
+        print(f"\n  ✓ LSTM wins by {diff:.1f} percentage points!")
+        print("  -> Sequence/flow matters more than shape for this data")
+    elif cnn_acc > lstm_acc:
+        diff = (cnn_acc - lstm_acc) * 100
+        print(f"\n  ✓ CNN wins by {diff:.1f} percentage points!")
+        print("  -> Shape patterns matter more than sequence for this data")
+    else:
+        print(f"\n  = TIE - both models perform similarly")
+    
+    # Store LSTM result
+    db = ExperimentDB()
+    run_id = f"lstm_vs_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
+    db.store_run(
+        run_id=run_id,
+        strategy="lstm_comparison",
+        config={
+            'lookback': lookback,
+            'hidden_size': 64,
+            'num_layers': 2,
+            'architecture': 'PriceLSTM',
+        },
+        metrics={
+            'total_trades': len(dataset),
+            'wins': int(len(dataset) * lstm_acc),
+            'losses': int(len(dataset) * (1 - lstm_acc)),
+            'win_rate': lstm_acc,
+            'total_pnl': 0,
+        },
+        model_path=str(lstm_path)
+    )
+    print(f"\n[+] Saved to ExperimentDB: {run_id}")
+    
+    return {
+        'lstm_acc': lstm_acc,
+        'cnn_acc': cnn_acc,
+        'lstm_path': str(lstm_path),
+        'samples': len(dataset),
+    }
+
+
+if __name__ == "__main__":
+    import argparse
+    
+    parser = argparse.ArgumentParser(description="LSTM vs CNN Comparison")
+    parser.add_argument("--input", type=str, default="results/ict_ifvg/records.jsonl")
+    parser.add_argument("--lookback", type=int, default=60, help="Sequence length")
+    
+    args = parser.parse_args()
+    
+    results = run_comparison(args.input, args.lookback)
diff --git a/src/App.tsx b/src/App.tsx
index 3d1fb00..a4f3078 100644
--- a/src/App.tsx
+++ b/src/App.tsx
@@ -8,8 +8,12 @@ import { DetailsPanel } from './components/DetailsPanel';
 import { ChatAgent } from './components/ChatAgent';
 import { SimulationView } from './components/SimulationView';
 import { StatsPanel } from './components/StatsPanel';
+import { LabPage } from './components/LabPage';
+
+type PageType = 'trade' | 'lab';
 
 const App: React.FC = () => {
+  const [currentPage, setCurrentPage] = useState<PageType>('trade');
   const [currentRun, setCurrentRun] = useState<string>('');
   const [mode, setMode] = useState<'DECISION' | 'TRADE'>('DECISION');
   const [index, setIndex] = useState<number>(0);
@@ -116,14 +120,48 @@ const App: React.FC = () => {
 
   const maxIndex = mode === 'DECISION' ? decisions.length - 1 : trades.length - 1;
 
-  // Always show main UI - no blocking load screen
+  // If Lab page is active, render it instead
+  if (currentPage === 'lab') {
+    return (
+      <div className="flex flex-col h-screen w-full bg-slate-900">
+        {/* Page Navigation */}
+        <div className="h-12 flex items-center gap-4 px-4 bg-slate-800 border-b border-slate-700">
+          <button
+            onClick={() => setCurrentPage('trade')}
+            className="text-slate-400 hover:text-white px-3 py-1"
+          >
+            Trade View
+          </button>
+          <button
+            onClick={() => setCurrentPage('lab')}
+            className="text-white bg-blue-600 px-3 py-1 rounded"
+          >
+            🔬 Lab
+          </button>
+        </div>
+        <div className="flex-1">
+          <LabPage />
+        </div>
+      </div>
+    );
+  }
+
+  // Trade View (default)
   return (
     <div className="flex h-screen w-full bg-slate-900 overflow-hidden">
 
       {/* LEFT SIDEBAR */}
       <div className="w-80 flex flex-col border-r border-slate-700 bg-slate-800">
         <div className="h-16 flex items-center justify-between px-4 border-b border-slate-700">
-          <h1 className="font-bold text-white text-lg">Trade Viz Agent</h1>
+          <div className="flex items-center gap-3">
+            <h1 className="font-bold text-white text-lg">Trade Viz</h1>
+            <button
+              onClick={() => setCurrentPage('lab')}
+              className="bg-green-600 hover:bg-green-500 text-white text-xs px-2 py-1 rounded"
+            >
+              🔬 Lab
+            </button>
+          </div>
           <button
             onClick={() => setShowSimulation(true)}
             className="bg-purple-600 hover:bg-purple-500 text-white text-xs px-3 py-1 rounded"
diff --git a/src/api/client.ts b/src/api/client.ts
index a6d5c37..b986455 100644
--- a/src/api/client.ts
+++ b/src/api/client.ts
@@ -135,6 +135,24 @@ export const api = {
         }
     },
 
+    postLabAgent: async (messages: ChatMessage[]): Promise<any> => {
+        const hasBackend = await checkBackend();
+        if (!hasBackend) {
+            return { reply: 'Backend not connected. Start with: ./start.sh' };
+        }
+        try {
+            const response = await fetch(`${API_BASE}/lab/agent`, {
+                method: 'POST',
+                headers: { 'Content-Type': 'application/json' },
+                body: JSON.stringify({ messages }),
+            });
+            if (!response.ok) return { reply: 'Error contacting lab agent.' };
+            return response.json();
+        } catch {
+            return { reply: 'Error contacting lab agent.' };
+        }
+    },
+
     runStrategy: async (payload: any): Promise<any> => {
         const hasBackend = await checkBackend();
         if (!hasBackend) {
diff --git a/src/components/LabPage.tsx b/src/components/LabPage.tsx
new file mode 100644
index 0000000..6628f3b
--- /dev/null
+++ b/src/components/LabPage.tsx
@@ -0,0 +1,307 @@
+import React, { useState, useRef, useEffect } from 'react';
+import { api } from '../api/client';
+
+interface Message {
+    role: 'user' | 'assistant';
+    content: string;
+    type?: 'text' | 'table' | 'chart' | 'code';
+    data?: any;
+}
+
+interface LabResult {
+    strategy: string;
+    trades: number;
+    wins: number;
+    losses: number;
+    win_rate: number;
+    total_pnl: number;
+    equity_curve?: number[];
+}
+
+export const LabPage: React.FC = () => {
+    const [messages, setMessages] = useState<Message[]>([
+        {
+            role: 'assistant',
+            content: 'Welcome to the Research Lab! I can help you test strategies, run scans, train models, and analyze results. What would you like to explore?',
+            type: 'text'
+        }
+    ]);
+    const [input, setInput] = useState('');
+    const [loading, setLoading] = useState(false);
+    const [currentResult, setCurrentResult] = useState<LabResult | null>(null);
+    const scrollRef = useRef<HTMLDivElement>(null);
+
+    useEffect(() => {
+        if (scrollRef.current) {
+            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
+        }
+    }, [messages]);
+
+    const handleSubmit = async (e: React.FormEvent) => {
+        e.preventDefault();
+        if (!input.trim()) return;
+
+        const userMsg: Message = { role: 'user', content: input, type: 'text' };
+        setMessages(prev => [...prev, userMsg]);
+        setInput('');
+        setLoading(true);
+
+        try {
+            // Call the lab agent endpoint
+            const response = await api.postLabAgent([...messages, userMsg]);
+
+            // Add assistant response
+            const assistantMsg: Message = {
+                role: 'assistant',
+                content: response.reply || 'Processing...',
+                type: response.type || 'text',
+                data: response.data
+            };
+            setMessages(prev => [...prev, assistantMsg]);
+
+            // If there's a result, store it
+            if (response.result) {
+                setCurrentResult(response.result);
+            }
+        } catch (err) {
+            setMessages(prev => [...prev, {
+                role: 'assistant',
+                content: 'Error contacting lab agent. Is the backend running?',
+                type: 'text'
+            }]);
+        } finally {
+            setLoading(false);
+        }
+    };
+
+    // Quick action buttons
+    const quickActions = [
+        { label: 'Run EMA Scan', prompt: 'Run an EMA cross scan on the last 7 days' },
+        { label: 'Test ORB Strategy', prompt: 'Test the Opening Range Breakout strategy' },
+        { label: 'Compare Models', prompt: 'Compare the LSTM vs CNN model accuracy' },
+        { label: 'Show Best Config', prompt: 'What is the best configuration from recent experiments?' },
+        { label: 'Run Grid Search', prompt: 'Run a grid search on ORB stop and target parameters' },
+    ];
+
+    const sendQuickAction = (prompt: string) => {
+        setInput(prompt);
+    };
+
+    // Render a result table
+    const renderResultTable = (result: LabResult) => (
+        <div className="bg-slate-800 rounded-lg p-4 my-3 border border-slate-600">
+            <div className="text-sm font-bold text-blue-400 mb-3">{result.strategy}</div>
+            <div className="grid grid-cols-3 gap-4 text-center">
+                <div>
+                    <div className="text-2xl font-bold text-white">{result.trades}</div>
+                    <div className="text-xs text-slate-400">Trades</div>
+                </div>
+                <div>
+                    <div className={`text-2xl font-bold ${result.win_rate >= 0.5 ? 'text-green-400' : 'text-red-400'}`}>
+                        {(result.win_rate * 100).toFixed(1)}%
+                    </div>
+                    <div className="text-xs text-slate-400">Win Rate</div>
+                </div>
+                <div>
+                    <div className={`text-2xl font-bold ${result.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
+                        ${result.total_pnl.toLocaleString()}
+                    </div>
+                    <div className="text-xs text-slate-400">P&L</div>
+                </div>
+            </div>
+
+            {/* Win/Loss Bar */}
+            <div className="mt-4">
+                <div className="flex h-3 rounded overflow-hidden">
+                    <div
+                        className="bg-green-500"
+                        style={{ width: `${result.win_rate * 100}%` }}
+                    />
+                    <div
+                        className="bg-red-500"
+                        style={{ width: `${(1 - result.win_rate) * 100}%` }}
+                    />
+                </div>
+                <div className="flex justify-between text-xs text-slate-400 mt-1">
+                    <span>{result.wins} Wins</span>
+                    <span>{result.losses} Losses</span>
+                </div>
+            </div>
+
+            {/* Mini Equity Curve */}
+            {result.equity_curve && result.equity_curve.length > 0 && (
+                <div className="mt-4">
+                    <div className="text-xs text-slate-400 mb-2">Equity Curve</div>
+                    <div className="h-16 flex items-end gap-px">
+                        {result.equity_curve.slice(-50).map((val, idx) => {
+                            const min = Math.min(...result.equity_curve!.slice(-50));
+                            const max = Math.max(...result.equity_curve!.slice(-50));
+                            const height = max > min ? ((val - min) / (max - min)) * 100 : 50;
+                            return (
+                                <div
+                                    key={idx}
+                                    className={`flex-1 ${val >= result.equity_curve![0] ? 'bg-green-500' : 'bg-red-500'}`}
+                                    style={{ height: `${Math.max(5, height)}%` }}
+                                />
+                            );
+                        })}
+                    </div>
+                </div>
+            )}
+        </div>
+    );
+
+    // Render message based on type
+    const renderMessage = (msg: Message, idx: number) => {
+        if (msg.role === 'user') {
+            return (
+                <div key={idx} className="flex justify-end">
+                    <div className="max-w-[80%] bg-blue-600 text-white rounded-xl px-4 py-2">
+                        {msg.content}
+                    </div>
+                </div>
+            );
+        }
+
+        return (
+            <div key={idx} className="flex justify-start">
+                <div className="max-w-[90%]">
+                    <div className="bg-slate-700 text-slate-100 rounded-xl px-4 py-3">
+                        {/* Render markdown-like content */}
+                        {msg.content.split('\n').map((line, i) => {
+                            if (line.startsWith('##')) {
+                                return <h3 key={i} className="font-bold text-lg text-blue-400 mt-2">{line.replace('##', '').trim()}</h3>;
+                            }
+                            if (line.startsWith('**') && line.endsWith('**')) {
+                                return <p key={i} className="font-bold text-white">{line.replace(/\*\*/g, '')}</p>;
+                            }
+                            if (line.startsWith('- ')) {
+                                return <p key={i} className="text-slate-300 ml-3">• {line.substring(2)}</p>;
+                            }
+                            if (line.startsWith('|')) {
+                                // Simple table row
+                                return (
+                                    <div key={i} className="font-mono text-xs text-slate-300 bg-slate-800 px-2 py-1">
+                                        {line}
+                                    </div>
+                                );
+                            }
+                            return <p key={i} className="text-slate-200">{line}</p>;
+                        })}
+                    </div>
+
+                    {/* Render result if attached */}
+                    {msg.data?.result && renderResultTable(msg.data.result)}
+                </div>
+            </div>
+        );
+    };
+
+    return (
+        <div className="flex flex-col h-screen bg-slate-900">
+            {/* Header */}
+            <div className="h-14 flex items-center justify-between px-6 border-b border-slate-700 bg-slate-800">
+                <div className="flex items-center gap-3">
+                    <span className="text-2xl">🔬</span>
+                    <h1 className="text-xl font-bold text-white">Research Lab</h1>
+                </div>
+                <div className="text-sm text-slate-400">
+                    AI-Powered Strategy Research
+                </div>
+            </div>
+
+            {/* Main Content */}
+            <div className="flex flex-1 overflow-hidden">
+                {/* Chat Area */}
+                <div className="flex-1 flex flex-col">
+                    {/* Messages */}
+                    <div className="flex-1 overflow-y-auto p-6 space-y-4" ref={scrollRef}>
+                        {messages.map((msg, idx) => renderMessage(msg, idx))}
+                        {loading && (
+                            <div className="flex justify-start">
+                                <div className="bg-slate-700 text-slate-300 rounded-xl px-4 py-3 animate-pulse">
+                                    <span className="text-blue-400">Agent is thinking...</span>
+                                </div>
+                            </div>
+                        )}
+                    </div>
+
+                    {/* Quick Actions */}
+                    <div className="px-6 py-3 border-t border-slate-700 bg-slate-800">
+                        <div className="flex gap-2 flex-wrap">
+                            {quickActions.map((action, idx) => (
+                                <button
+                                    key={idx}
+                                    onClick={() => sendQuickAction(action.prompt)}
+                                    className="bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs px-3 py-1.5 rounded-full transition"
+                                >
+                                    {action.label}
+                                </button>
+                            ))}
+                        </div>
+                    </div>
+
+                    {/* Input */}
+                    <form onSubmit={handleSubmit} className="p-4 border-t border-slate-700 bg-slate-800">
+                        <div className="flex gap-3">
+                            <input
+                                value={input}
+                                onChange={e => setInput(e.target.value)}
+                                placeholder="Ask me to run a strategy, test a theory, or analyze results..."
+                                className="flex-1 bg-slate-900 border border-slate-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
+                            />
+                            <button
+                                type="submit"
+                                disabled={loading}
+                                className="bg-blue-600 hover:bg-blue-500 text-white rounded-lg px-6 py-3 font-bold disabled:opacity-50"
+                            >
+                                Send
+                            </button>
+                        </div>
+                    </form>
+                </div>
+
+                {/* Right Sidebar - Current Result */}
+                <div className="w-80 border-l border-slate-700 bg-slate-800 p-4 overflow-y-auto">
+                    <h2 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">
+                        Latest Result
+                    </h2>
+
+                    {currentResult ? (
+                        renderResultTable(currentResult)
+                    ) : (
+                        <div className="text-slate-500 text-sm text-center py-8">
+                            Run a strategy to see results here
+                        </div>
+                    )}
+
+                    {/* Experiment History */}
+                    <div className="mt-6">
+                        <h2 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-3">
+                            Quick Commands
+                        </h2>
+                        <div className="space-y-2 text-xs">
+                            <div className="bg-slate-700 p-2 rounded text-slate-300">
+                                <code>"Run EMA cross scan"</code>
+                            </div>
+                            <div className="bg-slate-700 p-2 rounded text-slate-300">
+                                <code>"Test lunch hour fade"</code>
+                            </div>
+                            <div className="bg-slate-700 p-2 rounded text-slate-300">
+                                <code>"Train LSTM on bounce data"</code>
+                            </div>
+                            <div className="bg-slate-700 p-2 rounded text-slate-300">
+                                <code>"Compare ORB vs MR strategy"</code>
+                            </div>
+                            <div className="bg-slate-700 p-2 rounded text-slate-300">
+                                <code>"Show experiment history"</code>
+                            </div>
+                        </div>
+                    </div>
+                </div>
+            </div>
+        </div>
+    );
+};
+
+export default LabPage;
diff --git a/src/features/indicators.py b/src/features/indicators.py
index 28e3f4b..864e1da 100644
--- a/src/features/indicators.py
+++ b/src/features/indicators.py
@@ -194,6 +194,101 @@ def calculate_vwap(
     return vwap.fillna(method='ffill')
 
 
+# =============================================================================
+# Settlement Price
+# =============================================================================
+
+def calculate_settlement(
+    df: pd.DataFrame,
+    settlement_time: str = "15:00",  # 3 PM
+    tz: ZoneInfo = NY_TZ
+) -> pd.Series:
+    """
+    Calculate settlement price (typically 3 PM close).
+    
+    Args:
+        df: DataFrame with close and time
+        settlement_time: Time of settlement (HH:MM format)
+    
+    Returns:
+        Series with settlement values (forward-filled)
+    """
+    df = df.copy()
+    
+    if 'time' not in df.columns:
+        raise ValueError("DataFrame must have 'time' column")
+    
+    df['time_tz'] = pd.to_datetime(df['time']).dt.tz_convert(tz)
+    hour, minute = map(int, settlement_time.split(':'))
+    settlement_time_obj = df['time_tz'].iloc[0].replace(hour=hour, minute=minute).time()
+    
+    settlement = pd.Series(np.nan, index=df.index)
+    current_settlement = None
+    prev_hour = None
+    
+    for i in range(len(df)):
+        t = df['time_tz'].iloc[i]
+        
+        # Check if crossed settlement time
+        if prev_hour is not None:
+            if prev_hour < hour <= t.hour or (prev_hour >= hour and t.hour >= hour and t.minute >= minute):
+                current_settlement = df['close'].iloc[i]
+        
+        if current_settlement is not None:
+            settlement.iloc[i] = current_settlement
+        
+        prev_hour = t.hour
+    
+    return settlement.ffill()
+
+
+# =============================================================================
+# Session Levels (PDH, PDL, PDC)
+# =============================================================================
+
+def calculate_session_levels(
+    df: pd.DataFrame,
+    tz: ZoneInfo = NY_TZ
+) -> pd.DataFrame:
+    """
+    Calculate Previous Day High, Low, Close.
+    
+    Args:
+        df: DataFrame with OHLC and time
+    
+    Returns:
+        DataFrame with columns: pdh, pdl, pdc (Previous Day High/Low/Close)
+    """
+    df = df.copy()
+    
+    if 'time' not in df.columns:
+        raise ValueError("DataFrame must have 'time' column")
+    
+    df['date'] = pd.to_datetime(df['time']).dt.date
+    
+    # Calculate daily stats
+    daily = df.groupby('date').agg({
+        'high': 'max',
+        'low': 'min', 
+        'close': 'last'
+    }).rename(columns={
+        'high': 'pdh',
+        'low': 'pdl',
+        'close': 'pdc'
+    })
+    
+    # Shift by 1 day (previous day's values)
+    daily = daily.shift(1)
+    
+    # Map back to each bar
+    levels = pd.DataFrame(index=df.index)
+    levels['pdh'] = df['date'].map(daily['pdh'])
+    levels['pdl'] = df['date'].map(daily['pdl'])
+    levels['pdc'] = df['date'].map(daily['pdc'])
+    
+    return levels
+
+
 # =============================================================================
 # Indicator Bundle
 # =============================================================================
diff --git a/src/server/main.py b/src/server/main.py
index 45bef9a..0cb2740 100644
--- a/src/server/main.py
+++ b/src/server/main.py
@@ -486,6 +486,148 @@ async def agent_chat(request: ChatRequest) -> AgentResponse:
             return AgentResponse(reply=f"Error calling Gemini: {str(e)}")
 
 
+# =============================================================================
+# ENDPOINTS: Lab Research Agent
+# =============================================================================
+
+class LabChatRequest(BaseModel):
+    messages: List[ChatMessage]
+
+
+@app.post("/lab/agent")
+async def lab_agent(request: LabChatRequest):
+    """
+    Lab research agent - can execute strategies and return structured results.
+    """
+    import subprocess
+    
+    if not request.messages:
+        return {"reply": "No message provided."}
+    
+    last_message = request.messages[-1].content.lower()
+    
+    # Parse intent from message
+    result = None
+    reply = ""
+    
+    # Check for strategy execution requests
+    if "ema" in last_message and ("scan" in last_message or "run" in last_message):
+        strategy = "cross"
+        if "bounce" in last_message:
+            strategy = "bounce"
+        elif "stack" in last_message:
+            strategy = "stack"
+        
+        # Run the EMA scan
+        try:
+            proc = subprocess.run(
+                ["python", "scripts/run_ema_scan.py", "--strategy", strategy, "--days", "7"],
+                capture_output=True, text=True, timeout=120, cwd=str(RESULTS_DIR.parent)
+            )
+            
+            # Parse output for results
+            output = proc.stdout
+            if "Win Rate:" in output:
+                lines = output.split("\n")
+                for line in lines:
+                    if "Found" in line:
+                        reply += line + "\n"
+                    if "WIN:" in line or "LONG:" in line or "Win Rate:" in line:
+                        reply += line + "\n"
+                
+                # Try to extract numbers for result card
+                import re
+                trades_match = re.search(r"Found (\d+) signals", output)
+                wins_match = re.search(r"WIN: (\d+)", output)
+                wr_match = re.search(r"Win Rate: ([\d.]+)%", output)
+                
+                if trades_match and wins_match and wr_match:
+                    trades = int(trades_match.group(1))
+                    wins = int(wins_match.group(1))
+                    wr = float(wr_match.group(1)) / 100
+                    result = {
+                        "strategy": f"EMA {strategy.title()}",
+                        "trades": trades,
+                        "wins": wins,
+                        "losses": trades - wins,
+                        "win_rate": wr,
+                        "total_pnl": 0
+                    }
+            else:
+                reply = f"Ran EMA {strategy} scan:\n{output}"
+        except Exception as e:
+            reply = f"Error running strategy: {str(e)}"
+    
+    elif "orb" in last_message or "opening range" in last_message:
+        try:
+            proc = subprocess.run(
+                ["python", "scripts/run_orb_gridsearch.py", "--days", "7"],
+                capture_output=True, text=True, timeout=180, cwd=str(RESULTS_DIR.parent)
+            )
+            output = proc.stdout
+            reply = "ORB Grid Search Complete!\n\n"
+            if "BEST CONFIGURATION" in output:
+                start = output.index("BEST CONFIGURATION")
+                reply += output[start:]
+        except Exception as e:
+            reply = f"Error: {str(e)}"
+    
+    elif "lunch" in last_message and "fade" in last_message:
+        try:
+            proc = subprocess.run(
+                ["python", "scripts/run_lunch_fade.py", "--days", "7"],
+                capture_output=True, text=True, timeout=120, cwd=str(RESULTS_DIR.parent)
+            )
+            output = proc.stdout
+            reply = "Lunch Hour Fade Results:\n" + output.split("RESULTS")[1] if "RESULTS" in output else output
+        except Exception as e:
+            reply = f"Error: {str(e)}"
+    
+    elif "combined" in last_message or ("orb" in last_message and "reversion" in last_message):
+        try:
+            proc = subprocess.run(
+                ["python", "scripts/run_combined_strategy.py", "--days", "7"],
+                capture_output=True, text=True, timeout=120, cwd=str(RESULTS_DIR.parent)
+            )
+            output = proc.stdout
+            reply = "Combined Strategy Results:\n"
+            if "RESULTS" in output:
+                reply += output.split("RESULTS")[1]
+        except Exception as e:
+            reply = f"Error: {str(e)}"
+    
+    elif "experiment" in last_message or "history" in last_message or "best" in last_message:
+        # Query experiment database
+        try:
+            from src.storage import ExperimentDB
+            db = ExperimentDB()
+            best = db.query_best("win_rate", top_k=5)
+            reply = "## Top 5 Experiments by Win Rate\n\n"
+            for exp in best:
+                reply += f"- **{exp.get('strategy', 'unknown')}**: {exp.get('win_rate', 0):.1%} WR, {exp.get('total_trades', 0)} trades\n"
+        except Exception as e:
+            reply = f"Error querying experiments: {str(e)}"
+    
+    else:
+        # General response
+        reply = """I can help you run strategies and analyze results. Try:
+
+- "Run EMA cross scan"
+- "Test the lunch hour fade strategy"
+- "Run ORB grid search"
+- "Show experiment history"
+- "Combined ORB + Mean Reversion"
+
+What would you like to test?"""
+    
+    return {
+        "reply": reply,
+        "type": "text",
+        "data": {"result": result} if result else None,
+        "result": result
+    }
+
+
 # =============================================================================
 # ENDPOINTS: Strategy Runner (Agent Tool)
 # =============================================================================
diff --git a/src/sim/validation.py b/src/sim/validation.py
new file mode 100644
index 0000000..ef268c6
--- /dev/null
+++ b/src/sim/validation.py
@@ -0,0 +1,228 @@
+"""
+Trade Validation Rails
+
+Safety checks for simulation integrity:
+- Minimum distance between entry and stop/TP
+- Prevents trades that can't be simulated on 1m data
+- Flags suspicious fills
+
+Usage:
+    from src.sim.validation import validate_trade_distances, MIN_TRADE_DISTANCE
+    
+    if not validate_trade_distances(entry, stop, tp, candle_range):
+        print("Trade too tight for simulation!")
+"""
+
+from dataclasses import dataclass
+from typing import Optional, Dict, Any
+
+
+# =============================================================================
+# Minimum Trade Distance Rules
+# =============================================================================
+
+# Minimum distance in points - trades smaller than this can't be reliably simulated
+MIN_TRADE_DISTANCE_POINTS = 1.0  # 1 point minimum (4 ticks on MES)
+
+# Alternative: minimum as multiple of average candle range
+MIN_DISTANCE_CANDLE_MULT = 0.5  # Stop/TP must be at least 0.5x average candle range
+
+
+@dataclass
+class ValidationResult:
+    """Result of trade validation."""
+    valid: bool
+    reason: str = ""
+    warnings: list = None
+    
+    def __post_init__(self):
+        if self.warnings is None:
+            self.warnings = []
+
+
+def validate_trade_distances(
+    entry: float,
+    stop: float,
+    tp: float,
+    avg_candle_range: float,
+    min_points: float = MIN_TRADE_DISTANCE_POINTS,
+    min_candle_mult: float = MIN_DISTANCE_CANDLE_MULT,
+) -> ValidationResult:
+    """
+    Validate that trade distances are large enough to simulate.
+    
+    Args:
+        entry: Entry price
+        stop: Stop loss price
+        tp: Take profit price
+        avg_candle_range: Average candle range (high - low) for the timeframe
+        min_points: Minimum distance in points
+        min_candle_mult: Minimum distance as multiple of candle range
+    
+    Returns:
+        ValidationResult with valid flag and reason if invalid
+    """
+    stop_distance = abs(entry - stop)
+    tp_distance = abs(entry - tp)
+    min_candle_distance = avg_candle_range * min_candle_mult
+    
+    warnings = []
+    
+    # Check stop distance
+    if stop_distance < min_points:
+        return ValidationResult(
+            valid=False,
+            reason=f"Stop too tight: {stop_distance:.2f} pts < {min_points:.2f} pts minimum"
+        )
+    
+    if stop_distance < min_candle_distance:
+        return ValidationResult(
+            valid=False,
+            reason=f"Stop smaller than candle: {stop_distance:.2f} < {min_candle_distance:.2f} (0.5x avg candle)"
+        )
+    
+    # Check TP distance
+    if tp_distance < min_points:
+        return ValidationResult(
+            valid=False,
+            reason=f"TP too tight: {tp_distance:.2f} pts < {min_points:.2f} pts minimum"
+        )
+    
+    if tp_distance < min_candle_distance:
+        return ValidationResult(
+            valid=False,
+            reason=f"TP smaller than candle: {tp_distance:.2f} < {min_candle_distance:.2f} (0.5x avg candle)"
+        )
+    
+    # Warnings for marginal cases
+    if stop_distance < avg_candle_range:
+        warnings.append(f"Stop ({stop_distance:.2f}) < avg candle range ({avg_candle_range:.2f})")
+    
+    if tp_distance < avg_candle_range:
+        warnings.append(f"TP ({tp_distance:.2f}) < avg candle range ({avg_candle_range:.2f})")
+    
+    return ValidationResult(valid=True, warnings=warnings)
+
+
+def get_minimum_stop_distance(avg_candle_range: float, atr: float = None) -> float:
+    """
+    Calculate the minimum stop distance for reliable simulation.
+    
+    Returns the larger of:
+    - MIN_TRADE_DISTANCE_POINTS
+    - 0.5x average candle range
+    - 0.5x ATR (if provided)
+    
+    Use this when placing stops to ensure simulation validity.
+    """
+    candidates = [MIN_TRADE_DISTANCE_POINTS]
+    
+    candidates.append(avg_candle_range * MIN_DISTANCE_CANDLE_MULT)
+    
+    if atr is not None:
+        candidates.append(atr * 0.5)
+    
+    return max(candidates)
+
+
+def check_same_bar_fill_risk(
+    entry: float,
+    stop: float,
+    tp: float,
+    bar_high: float,
+    bar_low: float,
+) -> Dict[str, Any]:
+    """
+    Check if both stop and TP could hit on the same bar (ambiguous).
+    
+    This happens when the bar's range contains both the stop and TP.
+    When this occurs, we can't determine which hit first.
+    
+    Returns:
+        Dict with 'at_risk' bool and 'details' string
+    """
+    bar_range = bar_high - bar_low
+    stop_distance = abs(entry - stop)
+    tp_distance = abs(entry - tp)
+    
+    # Check if bar range exceeds both distances
+    both_in_range = (
+        bar_range >= stop_distance and
+        bar_range >= tp_distance
+    )
+    
+    # Check if bar actually touched both
+    stop_in_bar = bar_low <= stop <= bar_high or bar_low <= stop <= bar_high
+    tp_in_bar = bar_low <= tp <= bar_high or bar_low <= tp <= bar_high
+    
+    at_risk = both_in_range or (stop_in_bar and tp_in_bar)
+    
+    return {
+        'at_risk': at_risk,
+        'bar_range': bar_range,
+        'stop_distance': stop_distance,
+        'tp_distance': tp_distance,
+        'details': f"Bar range: {bar_range:.2f}, Stop: {stop_distance:.2f}, TP: {tp_distance:.2f}"
+    }
+
+
+# =============================================================================
+# Helper for grid searches
+# =============================================================================
+
+def filter_valid_grid_params(
+    stop_atr_range: list,
+    tp_r_range: list,
+    avg_candle_range: float,
+    avg_atr: float,
+) -> list:
+    """
+    Filter grid search parameters to only include valid combinations.
+    
+    Returns list of (stop_atr, tp_r) tuples that pass validation.
+    """
+    valid_combos = []
+    min_stop = get_minimum_stop_distance(avg_candle_range, avg_atr)
+    
+    for stop_atr in stop_atr_range:
+        stop_distance = avg_atr * stop_atr
+        
+        if stop_distance < min_stop:
+            continue  # Stop too tight
+        
+        for tp_r in tp_r_range:
+            tp_distance = stop_distance * tp_r
+            
+            if tp_distance < min_stop:
+                continue  # TP too tight
+            
+            valid_combos.append((stop_atr, tp_r))
+    
+    return valid_combos
+
+
+if __name__ == "__main__":
+    # Quick test
+    print("Trade Validation Rails")
+    print("=" * 40)
+    
+    # Simulate typical MES values
+    avg_candle_range = 2.5  # 2.5 points average 1m candle
+    atr = 4.0  # 15m ATR
+    
+    print(f"Avg candle range: {avg_candle_range}")
+    print(f"ATR: {atr}")
+    print(f"Min stop distance: {get_minimum_stop_distance(avg_candle_range, atr):.2f}")
+    print()
+    
+    # Test some trades
+    test_cases = [
+        (6000.0, 5999.5, 6001.0),  # Too tight
+        (6000.0, 5998.0, 6004.0),  # OK
+        (6000.0, 5997.0, 6006.0),  # Good
+    ]
+    
+    for entry, stop, tp in test_cases:
+        result = validate_trade_distances(entry, stop, tp, avg_candle_range)
+        status = "✓ VALID" if result.valid else f"✗ INVALID: {result.reason}"
+        print(f"Entry={entry}, Stop={stop}, TP={tp}: {status}")
```
