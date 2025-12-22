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
