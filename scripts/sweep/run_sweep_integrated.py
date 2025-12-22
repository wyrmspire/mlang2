"""
Sweep Runner - Integrated with mlang2 Architecture

This wrapper connects the sweep tools from mlang to:
- ExperimentDB for storing results
- FeatureEngine for consistent normalization
- ModelRegistry for model management

Usage:
    python scripts/sweep/run_sweep_integrated.py --help
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from datetime import datetime
from typing import Dict, Any, List

# Sweep configs
from scripts.sweep.config import (
    PatternSweepConfig, 
    CandleComposition, 
    OCOBracketConfig,
    ModelSweepConfig,
    CANDLE_COMPOSITIONS,
    OCO_SWEEP_VALUES,
)

# mlang2 integrations
from src.storage import ExperimentDB
from src.features.engine import normalize_ohlcv_window, FeatureConfig


def run_sweep_variant(
    pattern_config: PatternSweepConfig,
    oco_config: OCOBracketConfig,
    model_config: ModelSweepConfig,
    data_path: str = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a single sweep variant and return results.
    
    Args:
        pattern_config: Pattern mining geometry
        oco_config: OCO bracket configuration
        model_config: Model architecture settings
        data_path: Path to market data (uses default if None)
        verbose: Print progress
    
    Returns:
        Dict with trades, win_rate, pnl, config
    """
    import numpy as np
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # Load data via yfinance (max 7 days for 1m data)
    end = datetime.now()
    start = end - timedelta(days=7)
    
    try:
        ticker = yf.Ticker("ES=F")  # MES proxy
        df = ticker.history(start=start, end=end, interval="1m")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_pnl': 0, 'config': {}}
    
    if df is None or len(df) == 0:
        print("No market data available - market may be closed!")
        return {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'total_pnl': 0, 'config': {}}
    
    # Standardize columns
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 
        'Close': 'close', 'Volume': 'volume'
    })
    df = df.reset_index()
    
    if verbose:
        print(f"Loaded {len(df)} bars via yfinance")
    
    # Create feature config from candle composition
    lookback = model_config.candle_composition.candles_1m
    feature_config = FeatureConfig(lookback=lookback)
    
    # Simulate trades based on pattern config
    trades = []
    wins = 0
    total_pnl = 0.0
    
    # Simplified trading simulation
    atr_period = 14
    
    for i in range(lookback + atr_period, len(df) - 10):
        # Calculate ATR
        highs = df['high'].iloc[i-atr_period:i].values
        lows = df['low'].iloc[i-atr_period:i].values
        closes = df['close'].iloc[i-atr_period:i].values
        
        tr = np.maximum(highs - lows, 
                        np.maximum(np.abs(highs - np.roll(closes, 1)),
                                  np.abs(lows - np.roll(closes, 1))))
        atr = np.mean(tr[1:])
        
        if atr < 0.5:
            continue
        
        # Get normalized window
        window_data = df.iloc[i-lookback:i][['open', 'high', 'low', 'close', 'volume']].values
        x_norm = normalize_ohlcv_window(window_data, feature_config)
        
        # Simple trigger: large move followed by pullback
        recent_range = df['high'].iloc[i-5:i].max() - df['low'].iloc[i-5:i].min()
        if recent_range > atr * pattern_config.rise_ratio_min:
            # Trigger trade based on OCO config
            entry = df['close'].iloc[i]
            
            if oco_config.direction == "LONG":
                stop = entry - atr * oco_config.stop_atr_pct
                tp = entry + atr * oco_config.r_multiple * oco_config.stop_atr_pct
            else:
                stop = entry + atr * oco_config.stop_atr_pct
                tp = entry - atr * oco_config.r_multiple * oco_config.stop_atr_pct
            
            # Check outcome in next bars
            for j in range(i+1, min(i+50, len(df))):
                if oco_config.direction == "LONG":
                    if df['low'].iloc[j] <= stop:
                        pnl = -abs(entry - stop) * 50  # MES $50/pt
                        trades.append({'win': False, 'pnl': pnl})
                        total_pnl += pnl
                        break
                    elif df['high'].iloc[j] >= tp:
                        pnl = abs(tp - entry) * 50
                        wins += 1
                        trades.append({'win': True, 'pnl': pnl})
                        total_pnl += pnl
                        break
                else:
                    if df['high'].iloc[j] >= stop:
                        pnl = -abs(stop - entry) * 50
                        trades.append({'win': False, 'pnl': pnl})
                        total_pnl += pnl
                        break
                    elif df['low'].iloc[j] <= tp:
                        pnl = abs(entry - tp) * 50
                        wins += 1
                        trades.append({'win': True, 'pnl': pnl})
                        total_pnl += pnl
                        break
    
    win_rate = wins / len(trades) if trades else 0
    
    return {
        'total_trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': total_pnl / len(trades) if trades else 0,
        'config': {
            'pattern': pattern_config.to_dict(),
            'oco': oco_config.to_dict(),
            'model': model_config.to_dict(),
        }
    }


def run_mini_sweep(directions: List[str] = None, r_multiples: List[float] = None) -> List[Dict]:
    """
    Run a small sweep over key parameters.
    
    Args:
        directions: List of directions to test ["LONG", "SHORT"]
        r_multiples: List of R multiples to test [1.0, 1.4, 2.0]
    
    Returns:
        List of results sorted by win_rate
    """
    if directions is None:
        directions = ["LONG", "SHORT"]
    if r_multiples is None:
        r_multiples = [1.0, 1.4, 2.0]
    
    db = ExperimentDB()
    results = []
    
    pattern_config = PatternSweepConfig(
        rise_ratio_min=1.5,
        lookback_bars=30,
    )
    
    model_config = ModelSweepConfig(
        candle_composition=CandleComposition(candles_1m=30)
    )
    
    print("=" * 60)
    print("MINI SWEEP - Testing configurations")
    print("=" * 60)
    
    for direction in directions:
        for r_mult in r_multiples:
            oco_config = OCOBracketConfig(
                direction=direction,
                r_multiple=r_mult,
                stop_atr_pct=0.5,
                config_id=f"{direction}_{r_mult}R"
            )
            
            print(f"\nTesting: {oco_config.config_id}...")
            
            result = run_sweep_variant(
                pattern_config=pattern_config,
                oco_config=oco_config,
                model_config=model_config,
                verbose=False
            )
            
            print(f"  Trades: {result['total_trades']} | "
                  f"WR: {result['win_rate']:.1%} | "
                  f"PnL: ${result['total_pnl']:.2f}")
            
            # Store to DB
            run_id = f"sweep_{oco_config.config_id}_{datetime.now().strftime('%H%M%S')}"
            db.store_run(
                run_id=run_id,
                strategy="sweep_test",
                config=result['config'],
                metrics={
                    'total_trades': result['total_trades'],
                    'wins': result['wins'],
                    'losses': result['losses'],
                    'win_rate': result['win_rate'],
                    'total_pnl': result['total_pnl'],
                }
            )
            
            results.append({
                'config_id': oco_config.config_id,
                **result
            })
    
    # Sort by win_rate
    results.sort(key=lambda x: x['win_rate'], reverse=True)
    
    print("\n" + "=" * 60)
    print("RESULTS (sorted by win rate)")
    print("=" * 60)
    for r in results:
        print(f"  {r['config_id']}: {r['win_rate']:.1%} WR, ${r['total_pnl']:.2f}")
    
    print(f"\nStored {len(results)} experiments to DB")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sweep with mlang2 integration")
    parser.add_argument("--directions", nargs="+", default=["LONG", "SHORT"])
    parser.add_argument("--r-multiples", nargs="+", type=float, default=[1.0, 1.4, 2.0])
    parser.add_argument("--quick", action="store_true", help="Run minimal sweep")
    
    args = parser.parse_args()
    
    if args.quick:
        results = run_mini_sweep(["LONG"], [1.0])
    else:
        results = run_mini_sweep(args.directions, args.r_multiples)
    
    print(f"\nBest config: {results[0]['config_id']} with {results[0]['win_rate']:.1%}")
