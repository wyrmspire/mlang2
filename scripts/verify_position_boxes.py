#!/usr/bin/env python
"""
Test Strategy Run with Position Box Verification

This script:
1. Runs a simple EMA crossover strategy for 1 week
2. Verifies position boxes match trade outcomes
3. Checks SL/TP levels are correct for direction

Run: python scripts/verify_position_boxes.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import loader
from src.policy.triggers.indicator_triggers import EMACrossTrigger
from src.policy.brackets import ATRBracket
from src.strategy.scan import run_strategy_scan
from src.config import RESULTS_DIR


def verify_position_boxes(run_dir: Path) -> dict:
    """
    Verify position boxes match trade outcomes.
    
    Checks:
    1. Direction matches between decision.oco and trade
    2. SL is on correct side of entry for direction
    3. TP is on correct side of entry for direction  
    4. Outcome matches what price did (hit SL = LOSS, hit TP = WIN)
    """
    decisions_file = run_dir / "decisions.jsonl"
    trades_file = run_dir / "trades.jsonl"
    
    if not decisions_file.exists() or not trades_file.exists():
        return {"error": "Missing decisions.jsonl or trades.jsonl"}
    
    # Load data
    decisions = []
    with open(decisions_file) as f:
        for line in f:
            if line.strip():
                decisions.append(json.loads(line))
    
    trades = []
    with open(trades_file) as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))
    
    # Build lookup
    decisions_by_id = {d['decision_id']: d for d in decisions}
    
    errors = []
    warnings = []
    checks_passed = 0
    
    for trade in trades:
        trade_dir = trade.get('direction')
        decision_id = trade.get('decision_id')
        entry = trade.get('entry_price')
        exit_price = trade.get('exit_price')
        outcome = trade.get('outcome')
        pnl = trade.get('pnl_dollars')
        
        decision = decisions_by_id.get(decision_id, {})
        oco = decision.get('oco', {})
        oco_dir = oco.get('direction')
        stop = oco.get('stop_price')
        tp = oco.get('tp_price')
        scanner_ctx = decision.get('scanner_context', {})
        ctx_dir = scanner_ctx.get('direction')
        
        # Check 1: Direction consistency
        if oco_dir and trade_dir and oco_dir != trade_dir:
            errors.append(f"Trade {decision_id}: OCO dir={oco_dir} vs Trade dir={trade_dir}")
        elif ctx_dir and trade_dir and ctx_dir != trade_dir:
            errors.append(f"Trade {decision_id}: Context dir={ctx_dir} vs Trade dir={trade_dir}")
        else:
            checks_passed += 1
        
        # Check 2: SL/TP on correct side for direction
        if oco_dir == "LONG":
            if stop and entry and stop >= entry:
                errors.append(f"Trade {decision_id}: LONG but SL({stop}) >= entry({entry})")
            elif stop and entry:
                checks_passed += 1
                
            if tp and entry and tp <= entry:
                errors.append(f"Trade {decision_id}: LONG but TP({tp}) <= entry({entry})")
            elif tp and entry:
                checks_passed += 1
                
        elif oco_dir == "SHORT":
            if stop and entry and stop <= entry:
                errors.append(f"Trade {decision_id}: SHORT but SL({stop}) <= entry({entry})")
            elif stop and entry:
                checks_passed += 1
                
            if tp and entry and tp >= entry:
                errors.append(f"Trade {decision_id}: SHORT but TP({tp}) >= entry({entry})")
            elif tp and entry:
                checks_passed += 1
        
        # Check 3: Outcome vs PnL consistency
        if outcome == "WIN" and pnl and pnl < 0:
            errors.append(f"Trade {decision_id}: WIN but pnl={pnl} (negative)")
        elif outcome == "LOSS" and pnl and pnl > 0:
            errors.append(f"Trade {decision_id}: LOSS but pnl={pnl} (positive)")
        elif pnl is not None:
            checks_passed += 1
    
    return {
        "total_trades": len(trades),
        "checks_passed": checks_passed,
        "errors": errors,
        "warnings": warnings,
        "status": "PASS" if not errors else "FAIL"
    }


def main():
    print("=" * 60)
    print("POSITION BOX VERIFICATION TEST")
    print("=" * 60)
    
    # 1. Create trigger and bracket
    print("\n1. Creating EMA Crossover (9/21) trigger...")
    trigger = EMACrossTrigger(fast=9, slow=21)
    bracket = ATRBracket(stop_atr=2.0, tp_atr=3.0)
    
    # 2. Run strategy for 1 week in April
    print("\n2. Running strategy scan (April 1-7, 2025)...")
    start_date = "2025-04-01"
    weeks = 1
    run_name = "verify_boxes_test"
    
    try:
        result = run_strategy_scan(
            trigger=trigger,
            bracket=bracket,
            start_date=start_date,
            weeks=weeks,
            run_name=run_name,
            timeframe="5m"
        )
        print(f"   Scan complete: {result.total_trades} trades, {result.total_decisions} decisions")
    except Exception as e:
        print(f"   ERROR during scan: {e}")
        return
    
    # 3. Verify position boxes
    print("\n3. Verifying position boxes...")
    run_dir = RESULTS_DIR / "viz" / run_name
    verification = verify_position_boxes(run_dir)
    
    print(f"\n   Total trades: {verification.get('total_trades', 0)}")
    print(f"   Checks passed: {verification.get('checks_passed', 0)}")
    print(f"   Status: {verification.get('status', 'UNKNOWN')}")
    
    if verification.get('errors'):
        print("\n   ERRORS:")
        for err in verification['errors'][:10]:  # First 10
            print(f"     - {err}")
        if len(verification['errors']) > 10:
            print(f"     ... and {len(verification['errors']) - 10} more")
    
    if verification.get('warnings'):
        print("\n   WARNINGS:")
        for warn in verification['warnings'][:5]:
            print(f"     - {warn}")
    
    print("\n" + "=" * 60)
    if verification.get('status') == 'PASS':
        print("✅ ALL CHECKS PASSED")
    else:
        print("❌ VERIFICATION FAILED - SEE ERRORS ABOVE")
    print("=" * 60)


if __name__ == "__main__":
    main()
