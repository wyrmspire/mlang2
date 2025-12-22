"""
Trade Validation Rails

Safety checks for simulation integrity:
- Minimum distance between entry and stop/TP
- Prevents trades that can't be simulated on 1m data
- Flags suspicious fills

Usage:
    from src.sim.validation import validate_trade_distances, MIN_TRADE_DISTANCE
    
    if not validate_trade_distances(entry, stop, tp, candle_range):
        print("Trade too tight for simulation!")
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


# =============================================================================
# Minimum Trade Distance Rules
# =============================================================================

# Minimum distance in points - trades smaller than this can't be reliably simulated
MIN_TRADE_DISTANCE_POINTS = 1.0  # 1 point minimum (4 ticks on MES)

# Alternative: minimum as multiple of average candle range
MIN_DISTANCE_CANDLE_MULT = 0.5  # Stop/TP must be at least 0.5x average candle range


@dataclass
class ValidationResult:
    """Result of trade validation."""
    valid: bool
    reason: str = ""
    warnings: list = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def validate_trade_distances(
    entry: float,
    stop: float,
    tp: float,
    avg_candle_range: float,
    min_points: float = MIN_TRADE_DISTANCE_POINTS,
    min_candle_mult: float = MIN_DISTANCE_CANDLE_MULT,
) -> ValidationResult:
    """
    Validate that trade distances are large enough to simulate.
    
    Args:
        entry: Entry price
        stop: Stop loss price
        tp: Take profit price
        avg_candle_range: Average candle range (high - low) for the timeframe
        min_points: Minimum distance in points
        min_candle_mult: Minimum distance as multiple of candle range
    
    Returns:
        ValidationResult with valid flag and reason if invalid
    """
    stop_distance = abs(entry - stop)
    tp_distance = abs(entry - tp)
    min_candle_distance = avg_candle_range * min_candle_mult
    
    warnings = []
    
    # Check stop distance
    if stop_distance < min_points:
        return ValidationResult(
            valid=False,
            reason=f"Stop too tight: {stop_distance:.2f} pts < {min_points:.2f} pts minimum"
        )
    
    if stop_distance < min_candle_distance:
        return ValidationResult(
            valid=False,
            reason=f"Stop smaller than candle: {stop_distance:.2f} < {min_candle_distance:.2f} (0.5x avg candle)"
        )
    
    # Check TP distance
    if tp_distance < min_points:
        return ValidationResult(
            valid=False,
            reason=f"TP too tight: {tp_distance:.2f} pts < {min_points:.2f} pts minimum"
        )
    
    if tp_distance < min_candle_distance:
        return ValidationResult(
            valid=False,
            reason=f"TP smaller than candle: {tp_distance:.2f} < {min_candle_distance:.2f} (0.5x avg candle)"
        )
    
    # Warnings for marginal cases
    if stop_distance < avg_candle_range:
        warnings.append(f"Stop ({stop_distance:.2f}) < avg candle range ({avg_candle_range:.2f})")
    
    if tp_distance < avg_candle_range:
        warnings.append(f"TP ({tp_distance:.2f}) < avg candle range ({avg_candle_range:.2f})")
    
    return ValidationResult(valid=True, warnings=warnings)


def get_minimum_stop_distance(avg_candle_range: float, atr: float = None) -> float:
    """
    Calculate the minimum stop distance for reliable simulation.
    
    Returns the larger of:
    - MIN_TRADE_DISTANCE_POINTS
    - 0.5x average candle range
    - 0.5x ATR (if provided)
    
    Use this when placing stops to ensure simulation validity.
    """
    candidates = [MIN_TRADE_DISTANCE_POINTS]
    
    candidates.append(avg_candle_range * MIN_DISTANCE_CANDLE_MULT)
    
    if atr is not None:
        candidates.append(atr * 0.5)
    
    return max(candidates)


def check_same_bar_fill_risk(
    entry: float,
    stop: float,
    tp: float,
    bar_high: float,
    bar_low: float,
) -> Dict[str, Any]:
    """
    Check if both stop and TP could hit on the same bar (ambiguous).
    
    This happens when the bar's range contains both the stop and TP.
    When this occurs, we can't determine which hit first.
    
    Returns:
        Dict with 'at_risk' bool and 'details' string
    """
    bar_range = bar_high - bar_low
    stop_distance = abs(entry - stop)
    tp_distance = abs(entry - tp)
    
    # Check if bar range exceeds both distances
    both_in_range = (
        bar_range >= stop_distance and
        bar_range >= tp_distance
    )
    
    # Check if bar actually touched both
    stop_in_bar = bar_low <= stop <= bar_high or bar_low <= stop <= bar_high
    tp_in_bar = bar_low <= tp <= bar_high or bar_low <= tp <= bar_high
    
    at_risk = both_in_range or (stop_in_bar and tp_in_bar)
    
    return {
        'at_risk': at_risk,
        'bar_range': bar_range,
        'stop_distance': stop_distance,
        'tp_distance': tp_distance,
        'details': f"Bar range: {bar_range:.2f}, Stop: {stop_distance:.2f}, TP: {tp_distance:.2f}"
    }


# =============================================================================
# Helper for grid searches
# =============================================================================

def filter_valid_grid_params(
    stop_atr_range: list,
    tp_r_range: list,
    avg_candle_range: float,
    avg_atr: float,
) -> list:
    """
    Filter grid search parameters to only include valid combinations.
    
    Returns list of (stop_atr, tp_r) tuples that pass validation.
    """
    valid_combos = []
    min_stop = get_minimum_stop_distance(avg_candle_range, avg_atr)
    
    for stop_atr in stop_atr_range:
        stop_distance = avg_atr * stop_atr
        
        if stop_distance < min_stop:
            continue  # Stop too tight
        
        for tp_r in tp_r_range:
            tp_distance = stop_distance * tp_r
            
            if tp_distance < min_stop:
                continue  # TP too tight
            
            valid_combos.append((stop_atr, tp_r))
    
    return valid_combos


if __name__ == "__main__":
    # Quick test
    print("Trade Validation Rails")
    print("=" * 40)
    
    # Simulate typical MES values
    avg_candle_range = 2.5  # 2.5 points average 1m candle
    atr = 4.0  # 15m ATR
    
    print(f"Avg candle range: {avg_candle_range}")
    print(f"ATR: {atr}")
    print(f"Min stop distance: {get_minimum_stop_distance(avg_candle_range, atr):.2f}")
    print()
    
    # Test some trades
    test_cases = [
        (6000.0, 5999.5, 6001.0),  # Too tight
        (6000.0, 5998.0, 6004.0),  # OK
        (6000.0, 5997.0, 6006.0),  # Good
    ]
    
    for entry, stop, tp in test_cases:
        result = validate_trade_distances(entry, stop, tp, avg_candle_range)
        status = "✓ VALID" if result.valid else f"✗ INVALID: {result.reason}"
        print(f"Entry={entry}, Stop={stop}, TP={tp}: {status}")
