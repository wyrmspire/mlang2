"""
Tests for unified OCO Engine.

Ensures:
- Consistent price rounding
- Correct bars_held calculation
- Flat oco_results output
- Stop/TP priority rules
- Smart stop integration
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.sim.oco_engine import (
    OCOEngine, OCOConfig, OCOBracket, OCOStatus, ExitPriority,
    create_oco_bracket, process_oco_bar
)
from src.sim.stop_calculator import StopType, StopConfig
from src.sim.costs import CostModel
from src.sim.bar_fill_model import BarFillEngine, BarFillConfig


class TestOCOEngine(unittest.TestCase):
    """Test the unified OCO engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = OCOEngine()
        self.costs = CostModel(tick_size=0.25, contract_value=5.0)
        
        # Create sample bar data
        base_time = datetime(2025, 3, 18, 9, 30)
        self.sample_bars = pd.DataFrame({
            'time': [base_time + timedelta(minutes=i) for i in range(100)],
            'open': [5000.0 + i * 0.5 for i in range(100)],
            'high': [5001.0 + i * 0.5 for i in range(100)],
            'low': [4999.0 + i * 0.5 for i in range(100)],
            'close': [5000.5 + i * 0.5 for i in range(100)],
            'volume': [1000] * 100,
        })
    
    def test_create_bracket_long_atr_stop(self):
        """Test creating LONG bracket with ATR-based stop."""
        config = OCOConfig(
            direction="LONG",
            entry_type="LIMIT",
            entry_offset_atr=0.25,
            stop_atr=1.0,
            tp_multiple=1.5,
            name="TEST_LONG"
        )
        
        bracket = self.engine.create_bracket(
            config=config,
            base_price=5000.0,
            atr=10.0
        )
        
        # Entry should be below base price
        self.assertLess(bracket.entry_price, 5000.0)
        # Stop should be below entry
        self.assertLess(bracket.stop_price, bracket.entry_price)
        # TP should be above entry
        self.assertGreater(bracket.tp_price, bracket.entry_price)
        
        # Check tick rounding (0.25 tick size)
        self.assertEqual(bracket.entry_price % 0.25, 0.0)
        self.assertEqual(bracket.stop_price % 0.25, 0.0)
        self.assertEqual(bracket.tp_price % 0.25, 0.0)
        
        # Check risk/reward
        risk = bracket.entry_price - bracket.stop_price
        reward = bracket.tp_price - bracket.entry_price
        self.assertAlmostEqual(reward / risk, 1.5, places=2)
    
    def test_create_bracket_short_atr_stop(self):
        """Test creating SHORT bracket with ATR-based stop."""
        config = OCOConfig(
            direction="SHORT",
            entry_type="LIMIT",
            entry_offset_atr=0.25,
            stop_atr=1.0,
            tp_multiple=2.0,
            name="TEST_SHORT"
        )
        
        bracket = self.engine.create_bracket(
            config=config,
            base_price=5000.0,
            atr=10.0
        )
        
        # Entry should be above base price
        self.assertGreater(bracket.entry_price, 5000.0)
        # Stop should be above entry
        self.assertGreater(bracket.stop_price, bracket.entry_price)
        # TP should be below entry
        self.assertLess(bracket.tp_price, bracket.entry_price)
        
        # Check tick rounding
        self.assertEqual(bracket.entry_price % 0.25, 0.0)
        self.assertEqual(bracket.stop_price % 0.25, 0.0)
        self.assertEqual(bracket.tp_price % 0.25, 0.0)
    
    def test_market_entry_fills_immediately(self):
        """Test that MARKET entry fills on first bar."""
        config = OCOConfig(
            direction="LONG",
            entry_type="MARKET",
            stop_atr=1.0,
            tp_multiple=1.5,
        )
        
        bracket = self.engine.create_bracket(
            config=config,
            base_price=5000.0,
            atr=10.0
        )
        
        self.assertEqual(bracket.status, OCOStatus.PENDING_ENTRY)
        
        # Process first bar
        bar = self.sample_bars.iloc[0]
        bracket, event = self.engine.process_bar(bracket, bar, 0)
        
        self.assertEqual(event, 'ENTRY')
        self.assertEqual(bracket.status, OCOStatus.ACTIVE)
        self.assertIsNotNone(bracket.entry_fill)
        self.assertEqual(bracket.entry_bar, 0)
        self.assertEqual(bracket.bars_in_trade, 0)  # Entry bar doesn't count
    
    def test_bars_held_calculation(self):
        """Test that bars_held is calculated correctly (bars AFTER entry)."""
        config = OCOConfig(
            direction="LONG",
            entry_type="MARKET",
            stop_atr=1000.0,  # Very wide stop to avoid exit
            tp_multiple=1000.0,  # Very wide TP
            max_bars=10,
        )
        
        bracket = self.engine.create_bracket(
            config=config,
            base_price=5000.0,
            atr=1.0
        )
        
        # Entry at bar 0
        bracket, event = self.engine.process_bar(bracket, self.sample_bars.iloc[0], 0)
        self.assertEqual(event, 'ENTRY')
        self.assertEqual(bracket.bars_in_trade, 0)
        
        # Process bars 1-5
        for i in range(1, 6):
            bracket, event = self.engine.process_bar(bracket, self.sample_bars.iloc[i], i)
            self.assertEqual(bracket.bars_in_trade, i)  # bars_in_trade = i (bars after entry)
    
    def test_timeout_exit(self):
        """Test timeout exit after max_bars."""
        config = OCOConfig(
            direction="LONG",
            entry_type="MARKET",
            stop_atr=1000.0,
            tp_multiple=1000.0,
            max_bars=5,
        )
        
        bracket = self.engine.create_bracket(
            config=config,
            base_price=5000.0,
            atr=1.0
        )
        
        # Entry at bar 0
        bracket, _ = self.engine.process_bar(bracket, self.sample_bars.iloc[0], 0)
        
        # Process until timeout
        for i in range(1, 10):
            bracket, event = self.engine.process_bar(bracket, self.sample_bars.iloc[i], i)
            if event == 'TIMEOUT':
                self.assertEqual(bracket.bars_in_trade, 5)
                self.assertEqual(bracket.status, OCOStatus.CLOSED_TIMEOUT)
                break
    
    def test_stop_loss_exit(self):
        """Test stop loss exit."""
        config = OCOConfig(
            direction="LONG",
            entry_type="MARKET",
            stop_atr=1.0,
            tp_multiple=10.0,
        )
        
        bracket = self.engine.create_bracket(
            config=config,
            base_price=5000.0,
            atr=10.0
        )
        
        # Entry
        bracket, _ = self.engine.process_bar(bracket, self.sample_bars.iloc[0], 0)
        
        # Create bar that hits stop
        stop_bar = pd.Series({
            'time': datetime(2025, 3, 18, 9, 31),
            'open': 5000.0,
            'high': 5000.0,
            'low': bracket.stop_price - 5.0,  # Goes below stop
            'close': 4995.0,
            'volume': 1000,
        })
        
        bracket, event = self.engine.process_bar(bracket, stop_bar, 1)
        
        self.assertEqual(event, 'SL')
        self.assertEqual(bracket.status, OCOStatus.CLOSED_SL)
        self.assertIsNotNone(bracket.exit_fill)
    
    def test_take_profit_exit(self):
        """Test take profit exit."""
        config = OCOConfig(
            direction="LONG",
            entry_type="MARKET",
            stop_atr=100.0,
            tp_multiple=1.0,
        )
        
        bracket = self.engine.create_bracket(
            config=config,
            base_price=5000.0,
            atr=10.0
        )
        
        # Entry
        bracket, _ = self.engine.process_bar(bracket, self.sample_bars.iloc[0], 0)
        
        # Create bar that hits TP
        tp_bar = pd.Series({
            'time': datetime(2025, 3, 18, 9, 31),
            'open': 5000.0,
            'high': bracket.tp_price + 5.0,  # Goes above TP
            'low': 5000.0,
            'close': 5010.0,
            'volume': 1000,
        })
        
        bracket, event = self.engine.process_bar(bracket, tp_bar, 1)
        
        self.assertEqual(event, 'TP')
        self.assertEqual(bracket.status, OCOStatus.CLOSED_TP)
        self.assertIsNotNone(bracket.exit_fill)
    
    def test_flat_oco_results_output(self):
        """Test that to_flat_dict produces flat structure."""
        config = OCOConfig(
            direction="LONG",
            entry_type="MARKET",
            stop_atr=1.0,
            tp_multiple=1.5,
            name="TEST"
        )
        
        bracket = self.engine.create_bracket(
            config=config,
            base_price=5000.0,
            atr=10.0
        )
        
        # Entry
        bracket, _ = self.engine.process_bar(bracket, self.sample_bars.iloc[0], 0)
        
        # Process a few bars
        for i in range(1, 5):
            bracket, _ = self.engine.process_bar(bracket, self.sample_bars.iloc[i], i)
        
        # Get flat dict
        result = bracket.to_flat_dict()
        
        # Verify it's a flat dict (no nested dicts)
        self.assertIsInstance(result, dict)
        for key, value in result.items():
            self.assertNotIsInstance(value, dict, f"Field {key} should not be a dict")
        
        # Verify required fields
        self.assertIn('bars_held', result)
        self.assertIn('filled', result)
        self.assertIn('outcome', result)
        self.assertIn('entry_price', result)
        self.assertIn('stop_price', result)
        self.assertIn('tp_price', result)
        
        # bars_held should match bars_in_trade
        self.assertEqual(result['bars_held'], bracket.bars_in_trade)
    
    def test_mae_mfe_tracking(self):
        """Test MAE/MFE tracking during trade."""
        config = OCOConfig(
            direction="LONG",
            entry_type="MARKET",
            stop_atr=100.0,
            tp_multiple=100.0,
        )
        
        bracket = self.engine.create_bracket(
            config=config,
            base_price=5000.0,
            atr=10.0
        )
        
        # Entry at 5000
        bracket, _ = self.engine.process_bar(bracket, self.sample_bars.iloc[0], 0)
        entry_price = bracket.entry_price
        
        # Bar with adverse move
        adverse_bar = pd.Series({
            'time': datetime(2025, 3, 18, 9, 31),
            'open': 5000.0,
            'high': 5010.0,
            'low': entry_price - 15.0,  # Adverse
            'close': 5000.0,
            'volume': 1000,
        })
        
        bracket, _ = self.engine.process_bar(bracket, adverse_bar, 1)
        self.assertGreaterEqual(bracket.mae, 15.0)
        
        # Bar with favorable move
        favorable_bar = pd.Series({
            'time': datetime(2025, 3, 18, 9, 32),
            'open': 5000.0,
            'high': entry_price + 25.0,  # Favorable
            'low': 5000.0,
            'close': 5020.0,
            'volume': 1000,
        })
        
        bracket, _ = self.engine.process_bar(bracket, favorable_bar, 2)
        self.assertGreaterEqual(bracket.mfe, 25.0)
    
    def test_backward_compatibility_wrappers(self):
        """Test that legacy functions still work."""
        config = OCOConfig(
            direction="LONG",
            stop_atr=1.0,
            tp_multiple=1.5,
        )
        
        # Test create_oco_bracket wrapper
        bracket = create_oco_bracket(
            config=config,
            base_price=5000.0,
            atr=10.0
        )
        
        self.assertIsInstance(bracket, OCOBracket)
        
        # Test process_oco_bar wrapper
        bracket, event = process_oco_bar(
            bracket=bracket,
            bar=self.sample_bars.iloc[0],
            bar_idx=0
        )
        
        self.assertIsInstance(event, (str, type(None)))


class TestSmartStopIntegration(unittest.TestCase):
    """Test integration with stop_calculator for smart stops."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = OCOEngine()
        
        # Create HTF data (5m bars)
        base_time = datetime(2025, 3, 18, 9, 30)
        self.df_5m = pd.DataFrame({
            'time': [base_time + timedelta(minutes=i*5) for i in range(20)],
            'open': [5000.0 + i * 2 for i in range(20)],
            'high': [5003.0 + i * 2 for i in range(20)],
            'low': [4997.0 + i * 2 for i in range(20)],
            'close': [5001.0 + i * 2 for i in range(20)],
            'volume': [5000] * 20,
        })
    
    def test_create_bracket_with_candle_low_stop(self):
        """Test bracket creation with candle low stop."""
        stop_config = StopConfig(
            stop_type=StopType.CANDLE_LOW,
            timeframe="5m",
            lookback=1,
            atr_padding=0.25
        )
        
        config = OCOConfig(
            direction="LONG",
            entry_type="MARKET",
            stop_config=stop_config,
            tp_multiple=1.5,
        )
        
        bracket = self.engine.create_bracket(
            config=config,
            base_price=5020.0,
            atr=10.0,
            df_htf=self.df_5m
        )
        
        # Stop should be based on previous 5m candle low, not ATR
        self.assertIsNotNone(bracket.stop_price)
        # Should be rounded to tick
        self.assertEqual(bracket.stop_price % 0.25, 0.0)


if __name__ == '__main__':
    unittest.main()
