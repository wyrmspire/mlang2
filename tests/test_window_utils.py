"""
Tests for Window Utilities

Validates that 2-hour window policy is enforced correctly.
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.viz.window_utils import enforce_2hour_window, get_window_bounds_from_trades


class TestEnforce2HourWindow(unittest.TestCase):
    """Test enforce_2hour_window function."""
    
    def setUp(self):
        """Create sample 1-minute OHLCV data."""
        # Create 6 hours of 1-minute bars
        start_time = datetime(2025, 3, 17, 8, 0)
        times = [start_time + timedelta(minutes=i) for i in range(6 * 60)]
        
        self.df_1m = pd.DataFrame({
            'time': times,
            'open': [5000.0] * len(times),
            'high': [5010.0] * len(times),
            'low': [4990.0] * len(times),
            'close': [5000.0] * len(times),
            'volume': [100] * len(times),
        })
    
    def test_2hour_before_entry(self):
        """Test that window includes 2 hours before entry."""
        entry_time = datetime(2025, 3, 17, 10, 0)  # 10:00 AM
        
        raw_ohlcv, warning = enforce_2hour_window(
            df_1m=self.df_1m,
            entry_time=entry_time,
            bars_held=30  # 30 minutes
        )
        
        # Should have no warning (we have data from 8:00 AM)
        self.assertIsNone(warning)
        
        # Parse first timestamp
        first_time = datetime.fromisoformat(raw_ohlcv[0]['time'])
        
        # Should be approximately 2 hours before entry (8:00 AM)
        expected_start = entry_time - timedelta(hours=2)
        self.assertEqual(first_time, expected_start)
    
    def test_2hour_after_exit(self):
        """Test that window includes 2 hours after exit."""
        entry_time = datetime(2025, 3, 17, 10, 0)  # 10:00 AM
        bars_held = 30  # 30 minutes = exit at 10:30
        
        raw_ohlcv, warning = enforce_2hour_window(
            df_1m=self.df_1m,
            entry_time=entry_time,
            bars_held=bars_held
        )
        
        # Should have no warning
        self.assertIsNone(warning)
        
        # Parse last timestamp
        last_time = datetime.fromisoformat(raw_ohlcv[-1]['time'])
        
        # Should be approximately 2 hours after exit (12:30 PM)
        exit_time = entry_time + timedelta(minutes=bars_held)
        expected_end = exit_time + timedelta(hours=2)
        self.assertEqual(last_time, expected_end)
    
    def test_warning_when_data_missing_at_start(self):
        """Test that warning is issued when data missing at start."""
        # Entry at 8:30, need data from 6:30 but we only have from 8:00
        entry_time = datetime(2025, 3, 17, 8, 30)
        
        raw_ohlcv, warning = enforce_2hour_window(
            df_1m=self.df_1m,
            entry_time=entry_time,
            bars_held=30
        )
        
        # Should have warning about missing data
        self.assertIsNotNone(warning)
        self.assertIn('Missing', warning)
        self.assertIn('start', warning.lower())
    
    def test_warning_when_data_missing_at_end(self):
        """Test that warning is issued when data missing at end."""
        # Entry at 12:00, exit at 12:30, need data until 14:30 but we only have until 14:00
        entry_time = datetime(2025, 3, 17, 12, 0)
        
        raw_ohlcv, warning = enforce_2hour_window(
            df_1m=self.df_1m,
            entry_time=entry_time,
            bars_held=30
        )
        
        # Should have warning about missing data
        self.assertIsNotNone(warning)
        self.assertIn('Missing', warning)
        self.assertIn('end', warning.lower())
    
    def test_with_explicit_exit_time(self):
        """Test using explicit exit_time instead of bars_held."""
        entry_time = datetime(2025, 3, 17, 10, 0)
        exit_time = datetime(2025, 3, 17, 11, 0)
        
        raw_ohlcv, warning = enforce_2hour_window(
            df_1m=self.df_1m,
            entry_time=entry_time,
            exit_time=exit_time
        )
        
        # Should have no warning
        self.assertIsNone(warning)
        
        # Verify window bounds
        first_time = datetime.fromisoformat(raw_ohlcv[0]['time'])
        last_time = datetime.fromisoformat(raw_ohlcv[-1]['time'])
        
        expected_start = entry_time - timedelta(hours=2)
        expected_end = exit_time + timedelta(hours=2)
        
        self.assertEqual(first_time, expected_start)
        self.assertEqual(last_time, expected_end)
    
    def test_output_format(self):
        """Test that output format is correct."""
        entry_time = datetime(2025, 3, 17, 10, 0)
        
        raw_ohlcv, warning = enforce_2hour_window(
            df_1m=self.df_1m,
            entry_time=entry_time,
            bars_held=30
        )
        
        # Should be a list
        self.assertIsInstance(raw_ohlcv, list)
        
        # Should have bars
        self.assertGreater(len(raw_ohlcv), 0)
        
        # Each bar should be a dict with required keys
        bar = raw_ohlcv[0]
        self.assertIn('time', bar)
        self.assertIn('open', bar)
        self.assertIn('high', bar)
        self.assertIn('low', bar)
        self.assertIn('close', bar)
        self.assertIn('volume', bar)
        
        # Types should be correct
        self.assertIsInstance(bar['time'], str)
        self.assertIsInstance(bar['open'], float)
        self.assertIsInstance(bar['volume'], int)


class TestGetWindowBoundsFromTrades(unittest.TestCase):
    """Test get_window_bounds_from_trades function."""
    
    def test_with_single_trade(self):
        """Test window bounds computation with single trade."""
        trades = [{
            'entry_time': '2025-03-17T10:00:00-05:00',
            'exit_time': '2025-03-17T11:00:00-05:00',
        }]
        
        bounds = get_window_bounds_from_trades(trades)
        
        self.assertIsNotNone(bounds)
        self.assertIn('window_start', bounds)
        self.assertIn('window_end', bounds)
        self.assertIn('first_entry', bounds)
        self.assertIn('last_exit', bounds)
        
        # Parse timestamps
        first_entry = pd.Timestamp(bounds['first_entry'])
        last_exit = pd.Timestamp(bounds['last_exit'])
        window_start = pd.Timestamp(bounds['window_start'])
        window_end = pd.Timestamp(bounds['window_end'])
        
        # Verify 2-hour policy
        self.assertEqual(window_start, first_entry - timedelta(hours=2))
        self.assertEqual(window_end, last_exit + timedelta(hours=2))
    
    def test_with_multiple_trades(self):
        """Test window bounds computation with multiple trades."""
        trades = [
            {
                'entry_time': '2025-03-17T09:00:00-05:00',
                'exit_time': '2025-03-17T10:00:00-05:00',
            },
            {
                'entry_time': '2025-03-17T10:30:00-05:00',
                'exit_time': '2025-03-17T12:00:00-05:00',
            },
            {
                'entry_time': '2025-03-17T11:00:00-05:00',
                'exit_time': '2025-03-17T11:30:00-05:00',
            },
        ]
        
        bounds = get_window_bounds_from_trades(trades)
        
        # Parse timestamps
        first_entry = pd.Timestamp(bounds['first_entry'])
        last_exit = pd.Timestamp(bounds['last_exit'])
        
        # First entry should be earliest (09:00)
        self.assertEqual(first_entry.hour, 9)
        self.assertEqual(first_entry.minute, 0)
        
        # Last exit should be latest (12:00)
        self.assertEqual(last_exit.hour, 12)
        self.assertEqual(last_exit.minute, 0)
    
    def test_with_no_trades(self):
        """Test that None is returned when no trades."""
        bounds = get_window_bounds_from_trades([])
        self.assertIsNone(bounds)
    
    def test_with_missing_timestamps(self):
        """Test that None is returned when trades missing timestamps."""
        trades = [
            {'entry_time': None, 'exit_time': None},
        ]
        
        bounds = get_window_bounds_from_trades(trades)
        self.assertIsNone(bounds)


if __name__ == "__main__":
    unittest.main()
