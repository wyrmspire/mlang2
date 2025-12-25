"""
Property-based tests for OCO Engine.

Tests edge cases and invariants:
- Same-bar stop+TP hits
- Tick rounding for all price levels
- Price gaps and extreme movements
- Risk/reward ratio consistency
- Bars_held calculation correctness
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple

from src.sim.oco_engine import (
    OCOEngine, OCOConfig, OCOBracket, OCOStatus, ExitPriority
)
from src.sim.costs import CostModel


class TestOCOProperties(unittest.TestCase):
    """Property-based tests for OCO engine invariants."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = OCOEngine()
        self.costs = CostModel(tick_size=0.25, point_value=5.0)
    
    # =========================================================================
    # Property: Tick Rounding Invariant
    # =========================================================================
    
    def test_property_all_prices_tick_rounded(self):
        """Property: All prices must be rounded to tick size."""
        # Test various base prices and ATR values
        test_cases = [
            (5000.0, 10.0),
            (5000.13, 10.0),   # Non-aligned base price
            (5000.0, 10.37),   # Non-aligned ATR
            (4999.99, 9.99),   # Both non-aligned
            (10000.0, 50.0),   # Different scale
        ]
        
        for base_price, atr in test_cases:
            with self.subTest(base_price=base_price, atr=atr):
                # Test LONG
                config_long = OCOConfig(
                    direction="LONG",
                    entry_type="LIMIT",
                    entry_offset_atr=0.25,
                    stop_atr=1.0,
                    tp_multiple=1.5
                )
                bracket_long = self.engine.create_bracket(config_long, base_price, atr)
                
                # All prices must be multiples of tick size (0.25)
                self.assertEqual(bracket_long.entry_price % 0.25, 0.0,
                               f"Entry price {bracket_long.entry_price} not tick-aligned")
                self.assertEqual(bracket_long.stop_price % 0.25, 0.0,
                               f"Stop price {bracket_long.stop_price} not tick-aligned")
                self.assertEqual(bracket_long.tp_price % 0.25, 0.0,
                               f"TP price {bracket_long.tp_price} not tick-aligned")
                
                # Test SHORT
                config_short = OCOConfig(
                    direction="SHORT",
                    entry_type="LIMIT",
                    entry_offset_atr=0.25,
                    stop_atr=1.0,
                    tp_multiple=1.5
                )
                bracket_short = self.engine.create_bracket(config_short, base_price, atr)
                
                self.assertEqual(bracket_short.entry_price % 0.25, 0.0)
                self.assertEqual(bracket_short.stop_price % 0.25, 0.0)
                self.assertEqual(bracket_short.tp_price % 0.25, 0.0)
    
    # =========================================================================
    # Property: Risk/Reward Consistency
    # =========================================================================
    
    def test_property_risk_reward_ratio_preserved(self):
        """Property: TP/SL ratio should match configured tp_multiple."""
        test_multiples = [1.0, 1.5, 2.0, 2.5, 3.0]
        
        for tp_multiple in test_multiples:
            with self.subTest(tp_multiple=tp_multiple):
                # LONG
                config_long = OCOConfig(
                    direction="LONG",
                    stop_atr=1.0,
                    tp_multiple=tp_multiple
                )
                bracket_long = self.engine.create_bracket(config_long, 5000.0, 10.0)
                
                risk = bracket_long.entry_price - bracket_long.stop_price
                reward = bracket_long.tp_price - bracket_long.entry_price
                actual_multiple = reward / risk if risk > 0 else 0
                
                # Allow small tolerance due to tick rounding
                self.assertAlmostEqual(actual_multiple, tp_multiple, delta=0.1,
                                     msg=f"LONG: Expected {tp_multiple}, got {actual_multiple}")
                
                # SHORT
                config_short = OCOConfig(
                    direction="SHORT",
                    stop_atr=1.0,
                    tp_multiple=tp_multiple
                )
                bracket_short = self.engine.create_bracket(config_short, 5000.0, 10.0)
                
                risk = bracket_short.stop_price - bracket_short.entry_price
                reward = bracket_short.entry_price - bracket_short.tp_price
                actual_multiple = reward / risk if risk > 0 else 0
                
                self.assertAlmostEqual(actual_multiple, tp_multiple, delta=0.1,
                                     msg=f"SHORT: Expected {tp_multiple}, got {actual_multiple}")
    
    # =========================================================================
    # Property: Price Ordering Invariants
    # =========================================================================
    
    def test_property_price_ordering_long(self):
        """Property: For LONG, TP > Entry > Stop."""
        test_cases = [(1.0, 1.5), (0.5, 2.0), (2.0, 3.0)]
        
        for stop_atr, tp_multiple in test_cases:
            with self.subTest(stop_atr=stop_atr, tp_multiple=tp_multiple):
                config = OCOConfig(
                    direction="LONG",
                    stop_atr=stop_atr,
                    tp_multiple=tp_multiple
                )
                bracket = self.engine.create_bracket(config, 5000.0, 10.0)
                
                # Invariant: TP > Entry > Stop
                self.assertGreater(bracket.tp_price, bracket.entry_price,
                                 "LONG: TP should be > Entry")
                self.assertGreater(bracket.entry_price, bracket.stop_price,
                                 "LONG: Entry should be > Stop")
    
    def test_property_price_ordering_short(self):
        """Property: For SHORT, Stop > Entry > TP."""
        test_cases = [(1.0, 1.5), (0.5, 2.0), (2.0, 3.0)]
        
        for stop_atr, tp_multiple in test_cases:
            with self.subTest(stop_atr=stop_atr, tp_multiple=tp_multiple):
                config = OCOConfig(
                    direction="SHORT",
                    stop_atr=stop_atr,
                    tp_multiple=tp_multiple
                )
                bracket = self.engine.create_bracket(config, 5000.0, 10.0)
                
                # Invariant: Stop > Entry > TP
                self.assertGreater(bracket.stop_price, bracket.entry_price,
                                 "SHORT: Stop should be > Entry")
                self.assertGreater(bracket.entry_price, bracket.tp_price,
                                 "SHORT: Entry should be > TP")
    
    # =========================================================================
    # Property: Same-Bar Stop+TP Hits
    # =========================================================================
    
    def test_property_same_bar_both_hit_stop_first(self):
        """Property: When both SL and TP would hit, STOP_FIRST priority works correctly."""
        # Create bracket
        config = OCOConfig(
            direction="LONG",
            entry_type="MARKET",
            stop_atr=1.0,
            tp_multiple=1.5,
            exit_priority=ExitPriority.STOP_FIRST
        )
        bracket = self.engine.create_bracket(config, 5000.0, 10.0)
        bracket.status = OCOStatus.ACTIVE
        bracket.entry_bar = 0
        
        # Create bar that hits both stop and TP
        bar = pd.Series({
            'time': datetime(2025, 3, 18, 9, 30),
            'open': 5000.0,
            'high': bracket.tp_price + 10.0,  # Hits TP
            'low': bracket.stop_price - 10.0,  # Hits stop
            'close': 5000.0,
            'volume': 1000,
        })
        
        # Process bar
        updated_bracket, event = self.engine.process_bar(bracket, bar, 1)
        
        # Should close at stop with STOP_FIRST priority
        self.assertIsNotNone(event)
        self.assertEqual(updated_bracket.status, OCOStatus.CLOSED_SL)
        self.assertEqual(event, "SL")
    
    def test_property_same_bar_both_hit_tp_first(self):
        """Property: When both SL and TP would hit, TP_FIRST priority works correctly."""
        # TODO: Implement TP_FIRST priority in OCOEngine.process_bar()
        # Currently the engine always uses STOP_FIRST logic
        # This test is skipped until the feature is implemented
        self.skipTest("TP_FIRST priority not yet implemented in OCOEngine")
    
    # =========================================================================
    # Property: Price Gaps Handling
    # =========================================================================
    
    def test_property_gap_past_stop_long(self):
        """Property: Gap past stop should close at stop price (slippage model)."""
        config = OCOConfig(
            direction="LONG",
            entry_type="MARKET",
            stop_atr=1.0,
            tp_multiple=1.5
        )
        bracket = self.engine.create_bracket(config, 5000.0, 10.0)
        bracket.status = OCOStatus.ACTIVE
        bracket.entry_bar = 0
        
        # Create bar that gaps down past stop
        bar = pd.Series({
            'time': datetime(2025, 3, 18, 9, 30),
            'open': bracket.stop_price - 50.0,  # Gap down
            'high': bracket.stop_price - 40.0,
            'low': bracket.stop_price - 60.0,
            'close': bracket.stop_price - 50.0,
            'volume': 1000,
        })
        
        # Process bar
        updated_bracket, event = self.engine.process_bar(bracket, bar, 1)
        
        # Should close at stop
        self.assertIsNotNone(event)
        self.assertEqual(updated_bracket.status, OCOStatus.CLOSED_SL)
        self.assertEqual(event, "SL")
    
    def test_property_gap_past_tp_long(self):
        """Property: Gap past TP should close at TP price."""
        config = OCOConfig(
            direction="LONG",
            entry_type="MARKET",
            stop_atr=1.0,
            tp_multiple=1.5
        )
        bracket = self.engine.create_bracket(config, 5000.0, 10.0)
        bracket.status = OCOStatus.ACTIVE
        bracket.entry_bar = 0
        
        # Create bar that gaps up past TP
        bar = pd.Series({
            'time': datetime(2025, 3, 18, 9, 30),
            'open': bracket.tp_price + 50.0,  # Gap up
            'high': bracket.tp_price + 60.0,
            'low': bracket.tp_price + 40.0,
            'close': bracket.tp_price + 50.0,
            'volume': 1000,
        })
        
        # Process bar
        updated_bracket, event = self.engine.process_bar(bracket, bar, 1)
        
        # Should close at TP
        self.assertIsNotNone(event)
        self.assertEqual(updated_bracket.status, OCOStatus.CLOSED_TP)
        self.assertEqual(event, "TP")
    
    # =========================================================================
    # Property: Bars Held Calculation
    # =========================================================================
    
    def test_property_bars_held_counts_after_entry(self):
        """Property: bars_held should count bars AFTER entry bar."""
        config = OCOConfig(
            direction="LONG",
            entry_type="MARKET",
            stop_atr=1.0,
            tp_multiple=1.5,
            max_bars=10
        )
        bracket = self.engine.create_bracket(config, 5000.0, 10.0)
        bracket.status = OCOStatus.ACTIVE
        bracket.entry_bar = 5  # Entry at bar 5
        
        # Create bars that don't trigger exit
        base_time = datetime(2025, 3, 18, 9, 30)
        for bar_idx in range(6, 15):
            bar = pd.Series({
                'time': base_time + timedelta(minutes=bar_idx),
                'open': 5000.0,
                'high': 5001.0,
                'low': 4999.0,
                'close': 5000.0,
                'volume': 1000,
            })
            
            updated_bracket, event = self.engine.process_bar(bracket, bar, bar_idx)
            bracket = updated_bracket  # Update for next iteration
            
            expected_bars_held = bar_idx - bracket.entry_bar
            self.assertEqual(bracket.bars_in_trade, expected_bars_held,
                           f"At bar {bar_idx}: expected bars_held={expected_bars_held}, got {bracket.bars_in_trade}")
    
    def test_property_timeout_at_max_bars(self):
        """Property: Trade should timeout at max_bars."""
        config = OCOConfig(
            direction="LONG",
            entry_type="MARKET",
            stop_atr=1.0,
            tp_multiple=1.5,
            max_bars=5
        )
        bracket = self.engine.create_bracket(config, 5000.0, 10.0)
        bracket.status = OCOStatus.ACTIVE
        bracket.entry_bar = 10
        
        # Process bars until timeout
        base_time = datetime(2025, 3, 18, 9, 30)
        for i in range(1, 7):
            bar = pd.Series({
                'time': base_time + timedelta(minutes=i),
                'open': 5000.0,
                'high': 5001.0,
                'low': 4999.0,
                'close': 5000.0,
                'volume': 1000,
            })
            
            updated_bracket, event = self.engine.process_bar(bracket, bar, 10 + i)
            bracket = updated_bracket  # Update for next iteration
            
            if i <= 5:
                # Should still be active or just closed at i==5
                if bracket.status == OCOStatus.CLOSED_TIMEOUT:
                    # Timed out
                    self.assertEqual(i, 5, "Should timeout at bar 5")
                    self.assertEqual(event, "TIMEOUT")
                    break
        
        # Verify final state
        self.assertEqual(bracket.status, OCOStatus.CLOSED_TIMEOUT)
    
    # =========================================================================
    # Property: Flat OCO Results
    # =========================================================================
    
    def test_property_to_dict_produces_flat_oco_results(self):
        """Property: to_flat_dict() must produce flat oco_results (no nesting)."""
        config = OCOConfig(
            direction="LONG",
            stop_atr=1.0,
            tp_multiple=1.5
        )
        bracket = self.engine.create_bracket(config, 5000.0, 10.0)
        bracket.status = OCOStatus.ACTIVE
        bracket.entry_bar = 0
        
        # Get dictionary representation
        oco_dict = bracket.to_flat_dict()
        
        # Verify it's flat (all values are primitives or simple types)
        for key, value in oco_dict.items():
            self.assertNotIsInstance(value, dict,
                                   f"oco_results['{key}'] should not be a nested dict")
            if isinstance(value, list):
                for item in value:
                    self.assertNotIsInstance(item, dict,
                                           f"oco_results['{key}'] contains nested dict")
    
    def test_property_oco_results_has_required_fields(self):
        """Property: oco_results must have all required fields."""
        required_fields = [
            'entry_price', 'stop_price', 'tp_price', 'status',
            'entry_bar', 'bars_held', 'mae', 'mfe'
        ]
        
        config = OCOConfig(direction="LONG", stop_atr=1.0, tp_multiple=1.5)
        bracket = self.engine.create_bracket(config, 5000.0, 10.0)
        bracket.status = OCOStatus.ACTIVE
        
        oco_dict = bracket.to_flat_dict()
        
        for field in required_fields:
            self.assertIn(field, oco_dict,
                        f"oco_results missing required field: {field}")


if __name__ == '__main__':
    unittest.main()
