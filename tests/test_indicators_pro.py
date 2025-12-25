"""
Tests for Professional Trading Indicators

Tests all 6 categories of trading primitives.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.indicators_pro import (
    # Bar Measurement
    calculate_heikin_ashi,
    calculate_bar_expansion,
    calculate_average_bar_size,
    # Time Series
    calculate_macd,
    calculate_stochastic,
    calculate_adx,
    calculate_ichimoku,
    # Volume
    calculate_obv,
    calculate_relative_volume,
    calculate_chaikin_money_flow,
    calculate_vwmacd,
    # Levels
    calculate_pivot_points,
    calculate_fibonacci_levels,
    calculate_round_levels,
    # Breakouts
    calculate_donchian_channels,
    detect_channel_breakout,
    detect_momentum_burst,
    # Filters/Risk
    filter_time_of_day,
    calculate_kelly_criterion,
    calculate_position_size,
    check_risk_reward_ratio,
    PivotLevels,
)


class TestBarMeasurement(unittest.TestCase):
    """Test bar measurement primitives."""
    
    def setUp(self):
        """Create sample OHLCV data."""
        dates = pd.date_range('2025-01-01', periods=100, freq='1min')
        np.random.seed(42)
        
        self.df = pd.DataFrame({
            'time': dates,
            'open': 5000 + np.cumsum(np.random.randn(100) * 0.5),
            'high': 5000 + np.cumsum(np.random.randn(100) * 0.5) + 2,
            'low': 5000 + np.cumsum(np.random.randn(100) * 0.5) - 2,
            'close': 5000 + np.cumsum(np.random.randn(100) * 0.5),
            'volume': np.random.randint(1000, 5000, 100)
        })
        self.df['high'] = self.df[['open', 'close', 'high']].max(axis=1) + 1
        self.df['low'] = self.df[['open', 'close', 'low']].min(axis=1) - 1
    
    def test_heikin_ashi(self):
        """Test Heikin-Ashi calculation."""
        ha = calculate_heikin_ashi(self.df)
        
        self.assertEqual(len(ha), len(self.df))
        self.assertIn('ha_open', ha.columns)
        self.assertIn('ha_close', ha.columns)
        self.assertIn('ha_high', ha.columns)
        self.assertIn('ha_low', ha.columns)
        
        # HA High should be >= HA Low
        self.assertTrue((ha['ha_high'] >= ha['ha_low']).all())
    
    def test_bar_expansion(self):
        """Test bar expansion detection."""
        expansion = calculate_bar_expansion(self.df, atr_period=14, threshold=1.5)
        
        self.assertEqual(len(expansion), len(self.df))
        self.assertIsInstance(expansion, pd.Series)
        # Should have some True values
        self.assertTrue(expansion.any())
    
    def test_average_bar_size(self):
        """Test average bar size calculation."""
        avg_size = calculate_average_bar_size(self.df, period=20)
        
        self.assertEqual(len(avg_size), len(self.df))
        # Should be positive
        self.assertTrue((avg_size[20:] > 0).all())


class TestTimeSeries(unittest.TestCase):
    """Test time series primitives."""
    
    def setUp(self):
        """Create sample price data."""
        np.random.seed(42)
        self.close = pd.Series(5000 + np.cumsum(np.random.randn(100) * 0.5))
        self.df = pd.DataFrame({
            'high': self.close + np.random.rand(100) * 2,
            'low': self.close - np.random.rand(100) * 2,
            'close': self.close,
            'volume': np.random.randint(1000, 5000, 100)
        })
    
    def test_macd(self):
        """Test MACD calculation."""
        macd, signal, hist = calculate_macd(self.close)
        
        self.assertEqual(len(macd), len(self.close))
        self.assertEqual(len(signal), len(self.close))
        self.assertEqual(len(hist), len(self.close))
        
        # Histogram = MACD - Signal
        np.testing.assert_array_almost_equal(
            hist.dropna().values,
            (macd - signal).dropna().values,
            decimal=10
        )
    
    def test_stochastic(self):
        """Test Stochastic Oscillator."""
        k, d = calculate_stochastic(self.df)
        
        self.assertEqual(len(k), len(self.df))
        self.assertEqual(len(d), len(self.df))
        
        # Values should be between 0 and 100
        self.assertTrue((k.dropna() >= 0).all())
        self.assertTrue((k.dropna() <= 100).all())
    
    def test_adx(self):
        """Test ADX calculation."""
        adx, plus_di, minus_di = calculate_adx(self.df)
        
        self.assertEqual(len(adx), len(self.df))
        # ADX should be 0-100
        self.assertTrue((adx.dropna() >= 0).all())
        self.assertTrue((adx.dropna() <= 100).all())
    
    def test_ichimoku(self):
        """Test Ichimoku Cloud calculation."""
        cloud = calculate_ichimoku(self.df)
        
        self.assertIn('tenkan', cloud)
        self.assertIn('kijun', cloud)
        self.assertIn('senkou_a', cloud)
        self.assertIn('senkou_b', cloud)
        self.assertIn('chikou', cloud)
        
        # All should have same length
        for key, series in cloud.items():
            self.assertEqual(len(series), len(self.df))


class TestVolume(unittest.TestCase):
    """Test volume primitives."""
    
    def setUp(self):
        """Create sample data."""
        np.random.seed(42)
        close = 5000 + np.cumsum(np.random.randn(100) * 0.5)
        self.df = pd.DataFrame({
            'close': close,
            'high': close + np.abs(np.random.randn(100)) + 1,
            'low': close - np.abs(np.random.randn(100)) - 1,
            'volume': np.random.randint(1000, 5000, 100)
        })
    
    def test_obv(self):
        """Test On-Balance Volume."""
        obv = calculate_obv(self.df)
        
        self.assertEqual(len(obv), len(self.df))
        # First value should be first volume
        self.assertEqual(obv.iloc[0], self.df.iloc[0]['volume'])
    
    def test_relative_volume(self):
        """Test relative volume calculation."""
        rel_vol = calculate_relative_volume(self.df, period=20)
        
        self.assertEqual(len(rel_vol), len(self.df))
        # Should have some values > 1
        self.assertTrue((rel_vol[20:] > 0).all())
    
    def test_chaikin_money_flow(self):
        """Test Chaikin Money Flow."""
        cmf = calculate_chaikin_money_flow(self.df, period=20)
        
        self.assertEqual(len(cmf), len(self.df))
        # CMF should be between -1 and 1
        self.assertTrue((cmf.dropna() >= -1).all())
        self.assertTrue((cmf.dropna() <= 1).all())
    
    def test_vwmacd(self):
        """Test Volume-Weighted MACD."""
        vwmacd, signal, hist = calculate_vwmacd(self.df)
        
        self.assertEqual(len(vwmacd), len(self.df))
        self.assertEqual(len(signal), len(self.df))
        self.assertEqual(len(hist), len(self.df))


class TestLevels(unittest.TestCase):
    """Test levels primitives."""
    
    def test_pivot_points_standard(self):
        """Test standard pivot points."""
        pivots = calculate_pivot_points(5010, 4990, 5000, method='standard')
        
        self.assertIsInstance(pivots, PivotLevels)
        # R levels should be above pivot
        self.assertGreater(pivots.r1, pivots.pivot)
        self.assertGreater(pivots.r2, pivots.r1)
        # S levels should be below pivot
        self.assertLess(pivots.s1, pivots.pivot)
        self.assertLess(pivots.s2, pivots.s1)
    
    def test_pivot_points_woodie(self):
        """Test Woodie pivot points."""
        pivots = calculate_pivot_points(5010, 4990, 5000, method='woodie')
        
        self.assertIsInstance(pivots, PivotLevels)
        self.assertIsNotNone(pivots.pivot)
    
    def test_pivot_points_camarilla(self):
        """Test Camarilla pivot points."""
        pivots = calculate_pivot_points(5010, 4990, 5000, method='camarilla')
        
        self.assertIsInstance(pivots, PivotLevels)
        # Camarilla levels are closer to price
        self.assertLess(abs(pivots.r1 - 5000), 20)
    
    def test_fibonacci_retracement(self):
        """Test Fibonacci retracement levels."""
        fibs = calculate_fibonacci_levels(5100, 5000, direction='retracement')
        
        self.assertIn('38.2', fibs)
        self.assertIn('61.8', fibs)
        # 50% level should be midpoint
        self.assertAlmostEqual(fibs['50.0'], 5050, places=2)
    
    def test_fibonacci_extension(self):
        """Test Fibonacci extension levels."""
        fibs = calculate_fibonacci_levels(5100, 5000, direction='extension')
        
        self.assertIn('161.8', fibs)
        self.assertIn('261.8', fibs)
        # Extension levels should be above swing high
        self.assertGreater(fibs['161.8'], 5100)
    
    def test_round_levels(self):
        """Test round level calculation."""
        levels = calculate_round_levels(5023, increment=50)
        
        self.assertEqual(len(levels), 5)
        # Should include 5000
        self.assertIn(5000, levels)
        # Should be multiples of 50
        for level in levels:
            self.assertEqual(level % 50, 0)


class TestBreakouts(unittest.TestCase):
    """Test breakout primitives."""
    
    def setUp(self):
        """Create sample data."""
        np.random.seed(42)
        close = 5000 + np.cumsum(np.random.randn(100) * 0.5)
        self.df = pd.DataFrame({
            'close': close,
            'high': close + np.abs(np.random.randn(100)) + 1,
            'low': close - np.abs(np.random.randn(100)) - 1,
            'volume': np.random.randint(1000, 5000, 100)
        })
    
    def test_donchian_channels(self):
        """Test Donchian channel calculation."""
        upper, lower, middle = calculate_donchian_channels(self.df, period=20)
        
        self.assertEqual(len(upper), len(self.df))
        # Upper should be >= lower
        self.assertTrue((upper[20:] >= lower[20:]).all())
        # Middle should be average
        np.testing.assert_array_almost_equal(
            middle[20:].values,
            ((upper + lower) / 2)[20:].values
        )
    
    def test_channel_breakout(self):
        """Test channel breakout detection."""
        breakouts = detect_channel_breakout(self.df, period=20)
        
        self.assertEqual(len(breakouts), len(self.df))
        # Should have values in {-1, 0, 1}
        unique_vals = breakouts.unique()
        for val in unique_vals:
            self.assertIn(val, [-1, 0, 1])
    
    def test_momentum_burst(self):
        """Test momentum burst detection."""
        bursts = detect_momentum_burst(self.df)
        
        self.assertEqual(len(bursts), len(self.df))
        self.assertIsInstance(bursts, pd.Series)


class TestFiltersRisk(unittest.TestCase):
    """Test filter and risk primitives."""
    
    def test_filter_time_of_day(self):
        """Test time-of-day filtering."""
        # Morning hours
        ts_morning = pd.Timestamp('2025-01-01 10:00:00', tz='America/New_York')
        allowed = [(9, 12), (14, 16)]
        
        self.assertTrue(filter_time_of_day(ts_morning, allowed))
        
        # Lunch hour (not allowed)
        ts_lunch = pd.Timestamp('2025-01-01 12:30:00', tz='America/New_York')
        self.assertFalse(filter_time_of_day(ts_lunch, allowed))
    
    def test_kelly_criterion(self):
        """Test Kelly Criterion calculation."""
        # 60% win rate, avg win 100, avg loss 50
        kelly = calculate_kelly_criterion(0.6, 100, 50)
        
        self.assertGreater(kelly, 0)
        self.assertLessEqual(kelly, 0.25)  # Capped at 25%
    
    def test_kelly_criterion_edge_cases(self):
        """Test Kelly edge cases."""
        # 0% win rate
        self.assertEqual(calculate_kelly_criterion(0.0, 100, 50), 0.0)
        
        # 100% win rate
        self.assertEqual(calculate_kelly_criterion(1.0, 100, 50), 0.0)
        
        # Zero avg loss
        self.assertEqual(calculate_kelly_criterion(0.6, 100, 0), 0.0)
    
    def test_position_size(self):
        """Test position size calculation."""
        contracts = calculate_position_size(
            account_balance=100000,
            risk_percent=1.0,
            entry_price=5000,
            stop_price=4990,
            contract_value=5.0
        )
        
        self.assertGreater(contracts, 0)
        self.assertIsInstance(contracts, int)
        
        # Risk = $1000, risk per contract = 10 * 5 = $50, contracts = 20
        self.assertEqual(contracts, 20)
    
    def test_risk_reward_ratio(self):
        """Test risk/reward ratio check."""
        # 2:1 RR (10 risk, 20 reward)
        self.assertTrue(check_risk_reward_ratio(5000, 4990, 5020, min_rr=2.0))
        
        # 1:1 RR (doesn't meet 2:1 minimum)
        self.assertFalse(check_risk_reward_ratio(5000, 4990, 5010, min_rr=2.0))


if __name__ == '__main__':
    unittest.main()
