"""
Tests for Position Sizing Module

Validates that contract sizing is computed correctly and consistently.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sim.sizing import (
    calculate_contracts,
    calculate_pnl_dollars,
    calculate_reward_dollars,
    DEFAULT_MAX_RISK_DOLLARS,
    SizingResult
)
from src.sim.costs import CostModel


class TestCalculateContracts(unittest.TestCase):
    """Test contract sizing calculation."""
    
    def test_basic_sizing(self):
        """Test basic contract calculation."""
        # Entry at 5000, stop at 4990 = 10 points risk
        # Point value = $5 (MES default)
        # Risk per contract = 10 * $5 = $50
        # Max risk = $300
        # Contracts = floor(300 / 50) = 6
        result = calculate_contracts(
            entry_price=5000.0,
            stop_price=4990.0,
            max_risk_dollars=300.0
        )
        
        self.assertEqual(result.contracts, 6)
        self.assertEqual(result.risk_points, 10.0)
        self.assertAlmostEqual(result.risk_dollars, 300.0)
        self.assertEqual(result.max_risk_dollars, 300.0)
        self.assertEqual(result.point_value, 5.0)
    
    def test_minimum_one_contract(self):
        """Test that at least 1 contract is returned."""
        # Large stop = high risk per contract
        # Entry at 5000, stop at 4900 = 100 points risk
        # Risk per contract = 100 * $5 = $500
        # Max risk = $300
        # Contracts = floor(300 / 500) = 0 → min 1
        result = calculate_contracts(
            entry_price=5000.0,
            stop_price=4900.0,
            max_risk_dollars=300.0
        )
        
        self.assertEqual(result.contracts, 1)
        self.assertEqual(result.risk_points, 100.0)
        # Actual risk with 1 contract
        self.assertAlmostEqual(result.risk_dollars, 500.0)
    
    def test_short_position(self):
        """Test contract sizing for short positions."""
        # Entry at 5000, stop at 5010 = 10 points risk (abs value)
        result = calculate_contracts(
            entry_price=5000.0,
            stop_price=5010.0,
            max_risk_dollars=300.0
        )
        
        self.assertEqual(result.contracts, 6)
        self.assertEqual(result.risk_points, 10.0)
    
    def test_zero_risk_edge_case(self):
        """Test zero risk edge case (shouldn't happen, but be defensive)."""
        result = calculate_contracts(
            entry_price=5000.0,
            stop_price=5000.0,
            max_risk_dollars=300.0
        )
        
        self.assertEqual(result.contracts, 1)
        self.assertEqual(result.risk_points, 0.0)
        self.assertEqual(result.risk_dollars, 0.0)
    
    def test_custom_cost_model(self):
        """Test with custom cost model (different point value)."""
        # ES (big contract) has point value = $50
        es_costs = CostModel(point_value=50.0)
        
        # Entry at 5000, stop at 4990 = 10 points risk
        # Risk per contract = 10 * $50 = $500
        # Max risk = $300
        # Contracts = floor(300 / 500) = 0 → min 1
        result = calculate_contracts(
            entry_price=5000.0,
            stop_price=4990.0,
            max_risk_dollars=300.0,
            cost_model=es_costs
        )
        
        self.assertEqual(result.contracts, 1)  # Can't afford 1 full ES contract
        self.assertEqual(result.point_value, 50.0)
    
    def test_to_dict(self):
        """Test SizingResult.to_dict() for export."""
        result = calculate_contracts(
            entry_price=5000.0,
            stop_price=4990.0,
            max_risk_dollars=300.0
        )
        
        d = result.to_dict()
        
        self.assertIn('contracts', d)
        self.assertIn('risk_points', d)
        self.assertIn('risk_dollars', d)
        self.assertEqual(d['contracts'], 6)


class TestCalculatePnL(unittest.TestCase):
    """Test PnL calculation."""
    
    def test_long_win(self):
        """Test PnL for winning long trade."""
        # Long: Buy at 5000, sell at 5010
        # Profit = 10 points
        # 6 contracts * 10 points * $5 = $300
        # Commission = 6 * $1.25 * 2 = $15
        # Net = $300 - $15 = $285
        pnl_points, pnl_dollars = calculate_pnl_dollars(
            entry_price=5000.0,
            exit_price=5010.0,
            direction="LONG",
            contracts=6,
            include_commission=True
        )
        
        self.assertAlmostEqual(pnl_points, 10.0)
        self.assertAlmostEqual(pnl_dollars, 285.0)
    
    def test_long_loss(self):
        """Test PnL for losing long trade."""
        # Long: Buy at 5000, sell at 4990
        # Loss = -10 points
        # 6 contracts * -10 points * $5 = -$300
        # Commission = -$15
        # Net = -$315
        pnl_points, pnl_dollars = calculate_pnl_dollars(
            entry_price=5000.0,
            exit_price=4990.0,
            direction="LONG",
            contracts=6,
            include_commission=True
        )
        
        self.assertAlmostEqual(pnl_points, -10.0)
        self.assertAlmostEqual(pnl_dollars, -315.0)
    
    def test_short_win(self):
        """Test PnL for winning short trade."""
        # Short: Sell at 5000, buy back at 4990
        # Profit = 10 points
        pnl_points, pnl_dollars = calculate_pnl_dollars(
            entry_price=5000.0,
            exit_price=4990.0,
            direction="SHORT",
            contracts=6,
            include_commission=True
        )
        
        self.assertAlmostEqual(pnl_points, 10.0)
        self.assertAlmostEqual(pnl_dollars, 285.0)
    
    def test_short_loss(self):
        """Test PnL for losing short trade."""
        # Short: Sell at 5000, buy back at 5010
        # Loss = -10 points
        pnl_points, pnl_dollars = calculate_pnl_dollars(
            entry_price=5000.0,
            exit_price=5010.0,
            direction="SHORT",
            contracts=6,
            include_commission=True
        )
        
        self.assertAlmostEqual(pnl_points, -10.0)
        self.assertAlmostEqual(pnl_dollars, -315.0)
    
    def test_pnl_without_commission(self):
        """Test PnL calculation without commission."""
        pnl_points, pnl_dollars = calculate_pnl_dollars(
            entry_price=5000.0,
            exit_price=5010.0,
            direction="LONG",
            contracts=6,
            include_commission=False
        )
        
        self.assertAlmostEqual(pnl_points, 10.0)
        self.assertAlmostEqual(pnl_dollars, 300.0)  # No commission


class TestCalculateReward(unittest.TestCase):
    """Test reward calculation."""
    
    def test_long_reward(self):
        """Test reward for long position."""
        # Entry at 5000, TP at 5020 = 20 points reward
        # 6 contracts * 20 points * $5 = $600
        reward = calculate_reward_dollars(
            entry_price=5000.0,
            tp_price=5020.0,
            direction="LONG",
            contracts=6
        )
        
        self.assertAlmostEqual(reward, 600.0)
    
    def test_short_reward(self):
        """Test reward for short position."""
        # Entry at 5000, TP at 4980 = 20 points reward
        reward = calculate_reward_dollars(
            entry_price=5000.0,
            tp_price=4980.0,
            direction="SHORT",
            contracts=6
        )
        
        self.assertAlmostEqual(reward, 600.0)
    
    def test_reward_always_positive(self):
        """Test that reward is always positive."""
        # Even if TP is "wrong" direction, should be positive
        reward = calculate_reward_dollars(
            entry_price=5000.0,
            tp_price=4980.0,  # Below entry for long
            direction="LONG",
            contracts=1
        )
        
        self.assertGreater(reward, 0)


class TestSizingInvariant(unittest.TestCase):
    """Test sizing/PnL invariant: pnl_dollars == pnl_points * point_value * contracts."""
    
    def test_invariant_long(self):
        """Test invariant for long trade."""
        # Size the position
        sizing = calculate_contracts(5000.0, 4990.0, 300.0)
        
        # Calculate PnL (without commission for clean math)
        pnl_points, pnl_dollars = calculate_pnl_dollars(
            entry_price=5000.0,
            exit_price=5010.0,
            direction="LONG",
            contracts=sizing.contracts,
            include_commission=False
        )
        
        # Verify invariant
        expected_pnl = pnl_points * sizing.point_value * sizing.contracts
        self.assertAlmostEqual(pnl_dollars, expected_pnl)
    
    def test_invariant_short(self):
        """Test invariant for short trade."""
        sizing = calculate_contracts(5000.0, 5010.0, 300.0)
        
        pnl_points, pnl_dollars = calculate_pnl_dollars(
            entry_price=5000.0,
            exit_price=4990.0,
            direction="SHORT",
            contracts=sizing.contracts,
            include_commission=False
        )
        
        expected_pnl = pnl_points * sizing.point_value * sizing.contracts
        self.assertAlmostEqual(pnl_dollars, expected_pnl)


if __name__ == "__main__":
    unittest.main()
