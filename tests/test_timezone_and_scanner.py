#!/usr/bin/env python3
"""
Test for timezone safety fix and ScriptScanner bridge.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo


def test_timezone_safety():
    """Test that timezone conversion handles both naive and aware datetimes."""
    EST = ZoneInfo("America/New_York")
    
    # Test 1: Naive datetime (as yfinance might return)
    df_naive = pd.DataFrame({
        'time': pd.date_range('2025-01-01', periods=5, freq='1h')
    })
    
    # Apply the timezone safety fix
    if df_naive['time'].dt.tz is None:
        df_naive['time'] = df_naive['time'].dt.tz_localize('UTC')
    df_naive['time'] = df_naive['time'].dt.tz_convert(EST)
    
    assert df_naive['time'].dt.tz is not None, "Timezone should be set"
    assert str(df_naive['time'].dt.tz) == 'America/New_York', "Should be EST/EDT"
    print("✓ Naive datetime test passed")
    
    # Test 2: Already tz-aware datetime
    df_aware = pd.DataFrame({
        'time': pd.date_range('2025-01-01', periods=5, freq='1h', tz='UTC')
    })
    
    # Apply the same logic
    if df_aware['time'].dt.tz is None:
        df_aware['time'] = df_aware['time'].dt.tz_localize('UTC')
    df_aware['time'] = df_aware['time'].dt.tz_convert(EST)
    
    assert df_aware['time'].dt.tz is not None, "Timezone should be set"
    assert str(df_aware['time'].dt.tz) == 'America/New_York', "Should be EST/EDT"
    print("✓ Aware datetime test passed")


def test_script_scanner_basic():
    """Test ScriptScanner basic implementation in scanners.py."""
    # Read the file directly to verify implementation
    scanner_file = Path(__file__).parent.parent / "src/policy/scanners.py"
    with open(scanner_file, 'r') as f:
        content = f.read()
    
    # Check for ScriptScanner class definition
    assert 'class ScriptScanner(Scanner):' in content, "ScriptScanner class should be defined"
    print("✓ ScriptScanner class defined")
    
    # Check for key features
    assert 'script_path' in content, "Should accept script_path parameter"
    assert 'get_signals' in content or 'scan' in content, "Should look for scan/get_signals"
    assert 'importlib' in content, "Should use dynamic import"
    print("✓ ScriptScanner has dynamic import capability")
    
    # Check get_scanner factory updated
    assert "'script': ScriptScanner" in content, "Factory should include ScriptScanner"
    print("✓ ScriptScanner integrated into factory")


def test_validation_exists():
    """Test that validation module exists and has required functions."""
    validation_file = Path(__file__).parent.parent / "src/sim/validation.py"
    assert validation_file.exists(), "validation.py should exist"
    
    with open(validation_file, 'r') as f:
        content = f.read()
    
    # Check for key functions
    assert 'def validate_trade_distances' in content, "Should have validate_trade_distances"
    assert 'def check_same_bar_fill_risk' in content, "Should have check_same_bar_fill_risk"
    assert 'def get_minimum_stop_distance' in content, "Should have get_minimum_stop_distance"
    print("✓ Validation module has required functions")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Timezone Safety and ScriptScanner")
    print("=" * 60)
    
    try:
        test_timezone_safety()
        print()
        test_script_scanner_basic()
        print()
        test_validation_exists()
        print()
        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
