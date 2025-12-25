"""
Position Sizing - Single Source of Truth

This module provides centralized position sizing calculations to ensure
consistent contract sizing across all strategies and exporters.

According to ARCHITECTURE_AGREEMENT.md:
- Never default to 1 contract without explicit sizing calculation
- contracts = floor(MAX_RISK_DOLLARS / (risk_points * point_value)), min 1
- All PnL calculations must use the same cost model
"""

import math
from typing import Tuple
from dataclasses import dataclass

from src.sim.costs import CostModel, DEFAULT_COSTS


# Default risk settings (can be overridden per strategy)
DEFAULT_MAX_RISK_DOLLARS = 300.0


@dataclass
class SizingResult:
    """Result of position sizing calculation."""
    contracts: int
    risk_points: float
    risk_dollars: float
    max_risk_dollars: float
    point_value: float
    
    def to_dict(self):
        """Export as dictionary for viz."""
        return {
            'contracts': self.contracts,
            'risk_points': self.risk_points,
            'risk_dollars': self.risk_dollars,
            'max_risk_dollars': self.max_risk_dollars,
            'point_value': self.point_value,
        }


def calculate_contracts(
    entry_price: float,
    stop_price: float,
    max_risk_dollars: float = DEFAULT_MAX_RISK_DOLLARS,
    cost_model: CostModel = None
) -> SizingResult:
    """
    Calculate number of contracts based on risk parameters.
    
    This is the SINGLE source of truth for contract sizing.
    
    Formula:
        risk_points = abs(entry_price - stop_price)
        contracts = floor(max_risk_dollars / (risk_points * point_value))
        contracts = max(1, contracts)  # minimum 1 contract
    
    Args:
        entry_price: Entry price for the trade
        stop_price: Stop loss price
        max_risk_dollars: Maximum dollar risk per trade (default: $300)
        cost_model: Cost model for point value (default: DEFAULT_COSTS)
    
    Returns:
        SizingResult with contracts and risk parameters
    
    Example:
        >>> result = calculate_contracts(5000.0, 4990.0, 300.0)
        >>> result.contracts
        6
        >>> result.risk_dollars
        300.0
    """
    if cost_model is None:
        cost_model = DEFAULT_COSTS
    
    # Calculate risk in points
    risk_points = abs(entry_price - stop_price)
    
    # Handle edge case: zero risk (shouldn't happen, but be defensive)
    if risk_points <= 0:
        return SizingResult(
            contracts=1,
            risk_points=0.0,
            risk_dollars=0.0,
            max_risk_dollars=max_risk_dollars,
            point_value=cost_model.point_value
        )
    
    # Calculate contracts
    # contracts = floor(max_risk / (risk_points * point_value))
    risk_per_contract = risk_points * cost_model.point_value
    contracts = int(math.floor(max_risk_dollars / risk_per_contract))
    
    # Minimum 1 contract
    contracts = max(1, contracts)
    
    # Calculate actual risk with rounded contracts
    actual_risk_dollars = contracts * risk_per_contract
    
    return SizingResult(
        contracts=contracts,
        risk_points=risk_points,
        risk_dollars=actual_risk_dollars,
        max_risk_dollars=max_risk_dollars,
        point_value=cost_model.point_value
    )


def calculate_reward_dollars(
    entry_price: float,
    tp_price: float,
    direction: str,
    contracts: int,
    cost_model: CostModel = None
) -> float:
    """
    Calculate potential reward in dollars.
    
    Args:
        entry_price: Entry price
        tp_price: Take profit price
        direction: "LONG" or "SHORT"
        contracts: Number of contracts
        cost_model: Cost model for point value
    
    Returns:
        Reward in dollars (always positive)
    """
    if cost_model is None:
        cost_model = DEFAULT_COSTS
    
    if direction == "LONG":
        reward_points = tp_price - entry_price
    else:
        reward_points = entry_price - tp_price
    
    reward_dollars = abs(reward_points * cost_model.point_value * contracts)
    return reward_dollars


def calculate_pnl_dollars(
    entry_price: float,
    exit_price: float,
    direction: str,
    contracts: int,
    cost_model: CostModel = None,
    include_commission: bool = True
) -> Tuple[float, float]:
    """
    Calculate trade PnL in points and dollars.
    
    This is the SINGLE source of truth for PnL calculation.
    MUST be consistent with OCOEngine and CostModel.
    
    Args:
        entry_price: Entry fill price
        exit_price: Exit fill price
        direction: "LONG" or "SHORT"
        contracts: Number of contracts
        cost_model: Cost model for point value and commission
        include_commission: Whether to subtract commission
    
    Returns:
        Tuple of (pnl_points, pnl_dollars)
    """
    if cost_model is None:
        cost_model = DEFAULT_COSTS
    
    # Calculate points
    if direction == "LONG":
        pnl_points = exit_price - entry_price
    else:
        pnl_points = entry_price - exit_price
    
    # Calculate dollars using cost model
    pnl_dollars = cost_model.calculate_pnl(
        entry_price=entry_price,
        exit_price=exit_price,
        direction=direction,
        contracts=contracts,
        include_commission=include_commission
    )
    
    return pnl_points, pnl_dollars
