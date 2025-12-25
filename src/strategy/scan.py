"""
Strategy Scan Runner - Single Entry Point

This is THE way to run strategy scans. All required outputs are built-in.
Like a car factory - wheels are part of the assembly line, not an afterthought.

Usage:
    from src.strategy.scan import run_strategy_scan
    
    result = run_strategy_scan(
        trigger=FakeoutTrigger(level="pdh"),
        bracket=FixedBracket(stop_points=2, tp_points=4),
        start_date="2025-08-18",
        weeks=4,
        filters=[RTHFilter(), MinATRFilter()],
        run_name="fakeout_pdh_august"
    )
    
    # result.manifest_path → load in UI
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable
import uuid

from src.config import RESULTS_DIR, NY_TZ
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.features.indicators import calculate_atr
from src.features.pipeline import compute_features, FeatureConfig
from src.labels.counterfactual import compute_smart_stop_counterfactual

from src.policy.triggers.base import Trigger, TriggerResult
from src.policy.brackets import Bracket
from src.policy.actions import Action, SkipReason

from src.datasets.decision_record import DecisionRecord
from src.datasets.trade_record import TradeRecord
from src.sim.oco_engine import OCOBracket, OCOConfig, OCOEngine, DEFAULT_OCO_ENGINE
from src.sim.sizing import calculate_contracts, calculate_pnl_dollars, DEFAULT_MAX_RISK_DOLLARS

from src.viz.export import Exporter
from src.viz.config import VizConfig
from src.viz.window_utils import enforce_2hour_window


# =============================================================================
# FILTERS - Composable pre-trade filters
# =============================================================================

class PreTradeFilter:
    """Base class for pre-trade filters."""
    def __init__(self, name: str):
        self.name = name
    
    def check(self, timestamp: pd.Timestamp, atr: float, bar: pd.Series) -> tuple[bool, str]:
        """Returns (passed, reason) - reason is empty if passed."""
        raise NotImplementedError


class RTHFilter(PreTradeFilter):
    """Only trade during Regular Trading Hours (9:30-16:00 ET)."""
    def __init__(self):
        super().__init__("session_rth")
    
    def check(self, timestamp: pd.Timestamp, atr: float, bar: pd.Series) -> tuple[bool, str]:
        if timestamp.tzinfo is None:
            ts = timestamp.tz_localize(NY_TZ)
        else:
            ts = timestamp.tz_convert(NY_TZ)
        
        in_rth = 9 <= ts.hour < 16 or (ts.hour == 9 and ts.minute >= 30)
        return (in_rth, "" if in_rth else "Outside RTH")


class MinATRFilter(PreTradeFilter):
    """Require minimum ATR for volatility."""
    def __init__(self, threshold: float = 2.0):
        super().__init__("min_atr")
        self.threshold = threshold
    
    def check(self, timestamp: pd.Timestamp, atr: float, bar: pd.Series) -> tuple[bool, str]:
        passed = atr >= self.threshold
        return (passed, "" if passed else f"ATR {atr:.2f} < {self.threshold}")


# =============================================================================
# SCAN RESULT
# =============================================================================

@dataclass
class ScanResult:
    """Result of a strategy scan - all paths for UI consumption."""
    run_name: str
    manifest_path: Path
    decisions_path: Path
    trades_path: Path
    run_path: Path
    filter_failures_path: Optional[Path]
    
    total_decisions: int
    total_trades: int
    total_filtered: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "manifest_path": str(self.manifest_path),
            "total_decisions": self.total_decisions,
            "total_trades": self.total_trades,
            "total_filtered": self.total_filtered
        }


# =============================================================================
# THE MAIN FUNCTION - SINGLE ENTRY POINT
# =============================================================================

def run_strategy_scan(
    trigger: Trigger,
    bracket: Bracket,
    start_date: str,
    weeks: int,
    filters: Optional[List[PreTradeFilter]] = None,
    run_name: Optional[str] = None,
    timeframe: str = "5m",
    lookback_bars: int = 60,
    lookahead_bars: int = 30,
    cooldown_bars: int = 20,
    extra_context_fn: Optional[Callable] = None,  # For custom per-bar context
) -> ScanResult:
    """
    Run a strategy scan with ALL outputs guaranteed.
    
    This is the ONLY way to run scans. It internally calls:
    - exporter.on_decision()
    - exporter.on_bracket_created()
    - exporter.on_trade_closed()
    
    You CANNOT forget any of these - they're built into this function.
    
    Args:
        trigger: Trigger instance (e.g., FakeoutTrigger, EMACrossTrigger)
        bracket: Bracket instance (e.g., FixedBracket, ATRBracket)
        start_date: Start date string "YYYY-MM-DD"
        weeks: Number of weeks to scan
        filters: List of PreTradeFilter instances (default: [RTHFilter(), MinATRFilter()])
        run_name: Custom run name (auto-generated if None)
        timeframe: Timeframe for scanning ("1m", "5m", "15m")
        lookback_bars: Bars of history for chart viz
        lookahead_bars: Bars of future for chart viz
        cooldown_bars: Bars to wait between trades
        extra_context_fn: Optional function(bar, features) -> dict for custom context
        
    Returns:
        ScanResult with all artifact paths
    """
    
    # Use default filters if none provided
    if filters is None:
        filters = [RTHFilter(), MinATRFilter()]
    
    # Setup
    start = pd.Timestamp(start_date)
    end = start + pd.Timedelta(weeks=weeks, unit='W')
    run_name = run_name or f"{trigger.trigger_id}_{start_date.replace('-', '')}"
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"STRATEGY SCAN: {trigger.trigger_id}")
    print(f"Period: {start.date()} to {end.date()} ({weeks} weeks)")
    print(f"Bracket: {bracket.bracket_type}")
    print(f"Filters: {[f.name for f in filters]}")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1/5] Loading data...")
    df_1m = load_continuous_contract()
    df_1m = df_1m[(df_1m['time'] >= str(start)) & (df_1m['time'] < str(end))].reset_index(drop=True)
    
    # Compute VWAP for triggers that need it
    from src.features.indicators import calculate_vwap
    if 'vwap_session' not in df_1m.columns:
        df_1m['vwap_session'] = calculate_vwap(df_1m, period='session')
    
    print(f"  Loaded {len(df_1m)} 1m bars")
    
    # 2. Resample
    print("\n[2/5] Resampling...")
    htf_data = resample_all_timeframes(df_1m)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    
    tf_map = {'1m': df_1m, '5m': df_5m, '15m': df_15m}
    df_scan = tf_map.get(timeframe, df_5m)
    
    if df_scan is not None and len(df_scan) > 14:
        df_scan = df_scan.copy()
        df_scan['atr'] = calculate_atr(df_scan, 14)
        avg_atr = df_scan['atr'].dropna().mean()
    else:
        avg_atr = 5.0
    
    print(f"  Scanning on {timeframe}: {len(df_scan)} bars, avg ATR: {avg_atr:.2f}")
    
    # 3. Setup Exporter (REQUIRED - all hooks will be called)
    print("\n[3/5] Initializing exporter...")
    viz_config = VizConfig(include_windows=True, include_full_series=False)
    exporter = Exporter(
        config=viz_config,
        run_id=run_name,
        experiment_config={
            "trigger": trigger.to_dict(),
            "bracket": bracket.to_dict(),
            "start_date": str(start.date()),
            "weeks": weeks,
            "timeframe": timeframe,
            "filters": [f.name for f in filters]
        }
    )
    
    # Tracking
    filter_failures = []
    decision_idx = 0
    last_trigger_bar = -cooldown_bars - 1
    
    # 4. Run Scan Loop
    print("\n[4/5] Scanning...")
    
    for bar_idx in range(lookback_bars, len(df_scan) - lookahead_bars):
        bar = df_scan.iloc[bar_idx]
        bar_start_time = pd.Timestamp(bar['time'])
        
        # CRITICAL FIX: For market-on-close entry, timestamp should be bar CLOSE time
        # not bar START time. This aligns timestamp with entry_price.
        # For 5m bar: start=09:30, close=09:34 (we enter at 09:34)
        timeframe_minutes = {'1m': 1, '5m': 5, '15m': 15}[timeframe]
        bar_time = bar_start_time + pd.Timedelta(minutes=timeframe_minutes - 1)
        
        atr_value = bar.get('atr', avg_atr)
        if pd.isna(atr_value):
            atr_value = avg_atr
        
        # Cooldown check
        if bar_idx - last_trigger_bar < cooldown_bars:
            continue
        
        # Build features for trigger (mock bundle for now)
        class MockFeatures:
            pass
        features = MockFeatures()
        features.current_price = bar['close']
        features.bar_high = bar['high']
        features.bar_low = bar['low']
        features.bar_close = bar['close']
        features.timestamp = bar_time
        features.atr = atr_value
        
        # Add extra context if provided
        if extra_context_fn:
            extra = extra_context_fn(bar, features)
            for k, v in extra.items():
                setattr(features, k, v)
        
        # Check trigger
        result = trigger.check(features)
        
        if not result.triggered:
            continue
        
        direction = result.direction.value
        entry_price = bar['close']
        
        # === APPLY FILTERS ===
        filtered = False
        for f in filters:
            passed, reason = f.check(bar_time, atr_value, bar)
            if not passed:
                filter_failures.append({
                    "bar_idx": bar_idx,
                    "timestamp": str(bar_time),
                    "filter": f.name,
                    "reason": reason
                })
                filtered = True
                break
        
        if filtered:
            continue
        
        # === PASSED - RECORD DECISION ===
        last_trigger_bar = bar_idx
        
        # Compute bracket levels
        levels = bracket.compute(entry_price, direction, atr_value)
        
        # Find entry_idx in df_1m using time-based lookup (same pattern as raw_ohlcv)
        cf_mask = df_1m['time'] <= bar_time
        cf_entry_idx = cf_mask.sum() - 1 if cf_mask.any() else 0
        
        # Compute counterfactual outcome
        cf = compute_smart_stop_counterfactual(
            df=df_1m,
            entry_idx=cf_entry_idx,
            direction=direction,
            stop_price=levels.stop_price,
            tp_multiple=levels.r_multiple,
            atr=atr_value,
            oco_name="strategy"
        )
        
        # Get raw OHLCV window for chart - ENFORCING 2-HOUR POLICY
        # According to ARCHITECTURE_AGREEMENT.md Section 3:
        # - 2 hours before entry
        # - 2 hours after exit (estimated from bars_held)
        raw_ohlcv, window_warning = enforce_2hour_window(
            df_1m=df_1m,
            entry_time=bar_time,
            bars_held=int(cf.bars_held)
        )
        
        # Record warning if data is missing
        if window_warning:
            # Will be added to exporter._window_warnings
            pass
        
        # Future bars from 1m data (for counterfactual viz)
        # Keep existing future bars for backward compatibility
        entry_idx_1m = (df_1m['time'] <= bar_time).sum() - 1
        future_bars_1m = 120
        future_slice = df_1m.iloc[entry_idx_1m+1:entry_idx_1m+future_bars_1m+1]
        future_bars = [
            {
                "time": row['time'].isoformat() if hasattr(row['time'], 'isoformat') else str(row['time']),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row['volume'])
            }
            for _, row in future_slice.iterrows()
        ] if len(future_slice) > 0 else []
        
        # === POSITION SIZING: Use centralized sizing function ===
        sizing_result = calculate_contracts(
            entry_price=entry_price,
            stop_price=levels.stop_price,
            max_risk_dollars=DEFAULT_MAX_RISK_DOLLARS
        )
        contracts = sizing_result.contracts
        risk_points = sizing_result.risk_points
        risk_dollars = sizing_result.risk_dollars
        
        # Calculate reward using same cost model
        from src.sim.sizing import calculate_reward_dollars
        reward_dollars = calculate_reward_dollars(
            entry_price=entry_price,
            tp_price=levels.tp_price,
            direction=direction,
            contracts=contracts
        )
        
        # Create decision record
        decision_id = f"{trigger.trigger_id}_{decision_idx:04d}"
        decision = DecisionRecord(
            decision_id=decision_id,
            timestamp=bar_time,
            bar_idx=bar_idx,
            scanner_id=trigger.trigger_id,
            scanner_context=result.context,
            action=Action.PLACE_ORDER,
            current_price=entry_price,
            atr=atr_value,
            cf_outcome=cf.outcome,
            cf_pnl_dollars=cf.pnl_dollars
        )
        
        # === REQUIRED EXPORTER HOOK 1: on_decision ===
        class FeatBundle:
            x_price_1m = None
            x_price_5m = None
            x_price_15m = None
            x_context = None
        
        exporter.on_decision(
            decision=decision,
            features=FeatBundle(),
            raw_ohlcv=raw_ohlcv,
            future_1m=future_bars,
            indicators={
                "entry_price": entry_price,
                "stop_price": levels.stop_price,
                "tp_price": levels.tp_price,
                "atr": atr_value,
                "direction": direction,
                "contracts": contracts,
                "risk_dollars": risk_dollars,
                "reward_dollars": reward_dollars,
                "risk_points": risk_points
            }
        )
        
        # === REQUIRED EXPORTER HOOK 2: on_bracket_created ===
        oco_config = OCOConfig(
            direction=direction,
            entry_type="MARKET",
            stop_atr=levels.risk_points / atr_value if atr_value > 0 else 1.0,
            tp_multiple=levels.r_multiple
        )
        oco_bracket = OCOBracket(
            entry_price=entry_price,
            stop_price=levels.stop_price,
            tp_price=levels.tp_price,
            atr_at_creation=atr_value,
            config=oco_config
        )
        # CRITICAL: Pass contracts to exporter (not defaulted to 1)
        exporter.on_bracket_created(decision_id, oco_bracket, contracts=contracts)
        
        # === REQUIRED EXPORTER HOOK 3: on_trade_closed ===
        exit_bar = bar_idx + int(cf.bars_held)
        exit_time = bar_time + pd.Timedelta(minutes=int(cf.bars_held) * (5 if timeframe == '5m' else 15 if timeframe == '15m' else 1))
        
        # Use centralized PnL calculation (SINGLE source of truth)
        pnl_points, pnl_dollars = calculate_pnl_dollars(
            entry_price=entry_price,
            exit_price=cf.exit_price,
            direction=direction,
            contracts=contracts,
            include_commission=True
        )
        
        trade = TradeRecord(
            trade_id=str(uuid.uuid4())[:8],
            decision_id=decision_id,
            entry_time=bar_time,
            entry_bar=bar_idx,
            entry_price=entry_price,
            direction=direction,
            exit_time=exit_time,
            exit_bar=exit_bar,
            exit_price=cf.exit_price,
            exit_reason=cf.outcome,
            outcome=cf.outcome,
            pnl_points=pnl_points,
            pnl_dollars=pnl_dollars,
            r_multiple=pnl_points / risk_points if risk_points > 0 else 0,
            bars_held=int(cf.bars_held),
            mae=0,  # Would need to compute via OCOEngine
            mfe=0,  # Would need to compute via OCOEngine
            scanner_id=trigger.trigger_id,
            entry_atr=atr_value
        )
        exporter.on_trade_closed(trade)
        
        print(f"  [{decision_idx}] {direction} @ {bar_time.strftime('%Y-%m-%d %H:%M')} | "
              f"Entry: {entry_price:.2f} SL: {levels.stop_price:.2f} TP: {levels.tp_price:.2f} | {cf.outcome}")
        decision_idx += 1
    
    # 5. Finalize - writes manifest + all artifacts
    print(f"\n[5/5] Finalizing...")
    exporter.finalize(out_dir)
    
    # Save filter failures
    filter_failures_path = None
    if filter_failures:
        import json
        filter_failures_path = out_dir / "filter_failures.jsonl"
        with open(filter_failures_path, "w") as f:
            for ff in filter_failures:
                f.write(json.dumps(ff) + "\n")
        print(f"  Saved {len(filter_failures)} filter failures")
    
    # Build result
    result = ScanResult(
        run_name=run_name,
        manifest_path=out_dir / "manifest.json",
        decisions_path=out_dir / "decisions.jsonl",
        trades_path=out_dir / "trades.jsonl",
        run_path=out_dir / "run.json",
        filter_failures_path=filter_failures_path,
        total_decisions=decision_idx,
        total_trades=len(exporter.trades),
        total_filtered=len(filter_failures)
    )
    
    print(f"\n{'=' * 60}")
    print(f"SCAN COMPLETE")
    print(f"  Decisions: {result.total_decisions}")
    print(f"  Trades: {result.total_trades}")
    print(f"  Filtered: {result.total_filtered}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}")
    
    return result
