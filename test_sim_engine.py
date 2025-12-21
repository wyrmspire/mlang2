"""
Test script for simulation engine.

Tests the core simulation engine components.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.sim.engine import (
    SimulationEngine, 
    DataStream, 
    StrategyRunner, 
    OrderManagementSystem
)

def create_test_data(n_bars=100):
    """Create test OHLCV data."""
    start_time = pd.Timestamp('2025-01-01 09:30:00', tz='America/New_York')
    times = [start_time + timedelta(minutes=i) for i in range(n_bars)]
    
    # Generate simple random walk
    np.random.seed(42)
    closes = np.cumsum(np.random.randn(n_bars) * 2) + 5850
    
    data = []
    for i, (t, c) in enumerate(zip(times, closes)):
        high = c + abs(np.random.randn() * 5)
        low = c - abs(np.random.randn() * 5)
        open_price = closes[i-1] if i > 0 else c
        
        data.append({
            'time': t,
            'open': open_price,
            'high': high,
            'low': low,
            'close': c,
            'volume': np.random.randint(1000, 10000)
        })
    
    return pd.DataFrame(data)


def test_data_stream():
    """Test DataStream component."""
    print("Testing DataStream...")
    
    df = create_test_data(10)
    stream = DataStream(df, start_idx=0, end_idx=10)
    
    bars_seen = 0
    for idx, bar in stream:
        bars_seen += 1
        assert bar['close'] is not None
    
    assert bars_seen == 10
    print(f"  ✓ DataStream works - processed {bars_seen} bars")


def test_strategy_runner():
    """Test StrategyRunner component."""
    print("Testing StrategyRunner...")
    
    df = create_test_data(10)
    
    # Test random strategy
    runner = StrategyRunner('random', {})
    signal = runner.evaluate(df.iloc[5], df.iloc[:5])
    
    assert signal.confidence >= 0.0 and signal.confidence <= 1.0
    print(f"  ✓ StrategyRunner works - signal: {signal.direction}, conf: {signal.confidence:.2f}")


def test_oms():
    """Test Order Management System."""
    print("Testing OMS...")
    
    oms = OrderManagementSystem()
    df = create_test_data(10)
    
    # Submit an OCO
    from src.sim.oco import OCOConfig
    config = OCOConfig(
        direction='LONG',
        entry_type='MARKET',
        stop_atr=1.0,
        tp_multiple=1.5
    )
    
    oco_id = oms.submit_oco(config, 5850.0, 10.0, bar_idx=0)
    assert oco_id is not None
    
    # Process a bar
    events = oms.process_bar(df.iloc[1], 1)
    
    print(f"  ✓ OMS works - OCO {oco_id} submitted, {len(events)} events")


def test_simulation_engine():
    """Test full simulation engine."""
    print("Testing SimulationEngine...")
    
    df = create_test_data(20)
    
    engine = SimulationEngine(
        df=df,
        strategy_name='random',
        config={
            'entry_type': 'MARKET',
            'stop_atr': 1.0,
            'tp_multiple': 1.5,
            'auto_submit_ocos': True
        },
        start_idx=0,
        end_idx=20
    )
    
    # Step through a few bars
    steps = 0
    events_seen = []
    
    for _ in range(5):
        result = engine.step()
        if result.get('done'):
            break
        
        steps += 1
        events_seen.extend(result.get('events', []))
    
    state = engine.get_state()
    
    print(f"  ✓ SimulationEngine works - {steps} steps, {len(events_seen)} events")
    print(f"    Progress: {state['progress']:.1%}")
    print(f"    Stats: {state['stats']}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Simulation Engine Tests")
    print("=" * 60)
    
    try:
        test_data_stream()
        test_strategy_runner()
        test_oms()
        test_simulation_engine()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
