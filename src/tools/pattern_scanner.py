"""
Pattern Scanner - Scan historical data for pattern occurrences.

Helps agents discover when patterns occur and their success rates.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class PatternOccurrence:
    """Single pattern occurrence."""
    bar_idx: int
    timestamp: str
    pattern_type: str
    context: Dict[str, Any]
    forward_pnl: Optional[float] = None  # If we have outcome data
    

@dataclass
class PatternStats:
    """Statistics for a pattern."""
    pattern_type: str
    total_occurrences: int
    occurrences: List[PatternOccurrence]
    
    # If outcomes available
    success_rate: Optional[float] = None
    avg_forward_pnl: Optional[float] = None
    best_time_of_day: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_type': self.pattern_type,
            'total_occurrences': self.total_occurrences,
            'success_rate': self.success_rate,
            'avg_forward_pnl': self.avg_forward_pnl,
            'best_time_of_day': self.best_time_of_day,
            'sample_occurrences': [
                {
                    'bar_idx': occ.bar_idx,
                    'timestamp': occ.timestamp,
                    'context': occ.context,
                }
                for occ in self.occurrences[:5]  # Sample
            ]
        }


class PatternScanner:
    """
    Scan historical data for pattern occurrences.
    
    Useful for agents to:
    - Discover how often patterns occur
    - Find best times/contexts for patterns
    - Validate pattern usefulness before building strategies
    
    Usage:
        # Scan for candle patterns
        results = PatternScanner.scan(
            data_path="data/ES_1m_2025-03.parquet",
            pattern_type="candle_pattern",
            pattern_config={"patterns": ["hammer", "doji"]}
        )
        
        # Scan for indicator conditions
        results = PatternScanner.scan(
            data_path="data/ES_1m_2025-03.parquet",
            pattern_type="rsi_threshold",
            pattern_config={"threshold": 30, "direction": "below"}
        )
    """
    
    @staticmethod
    def scan(
        data_path: str,
        pattern_type: str,
        pattern_config: Dict[str, Any],
        include_outcomes: bool = False
    ) -> PatternStats:
        """
        Scan historical data for pattern occurrences.
        
        Args:
            data_path: Path to parquet data file
            pattern_type: Type of pattern (matches trigger types)
            pattern_config: Pattern configuration
            include_outcomes: Whether to calculate forward outcomes
            
        Returns:
            PatternStats with occurrence details
        """
        import pandas as pd
        from src.policy.triggers import trigger_from_dict
        from src.features.pipeline import FeaturePipeline
        from src.data.loader import load_market_data
        
        # Load data
        df = pd.read_parquet(data_path)
        
        # Create trigger
        trigger_config = {'type': pattern_type, **pattern_config}
        trigger = trigger_from_dict(trigger_config)
        
        # Scan through data
        occurrences = []
        
        # Simple scan (without full pipeline - just for discovery)
        # For production, would integrate with full feature pipeline
        for idx in range(len(df)):
            # Create minimal features for trigger
            # This is simplified - real implementation would use FeaturePipeline
            try:
                # Mock feature bundle (real implementation would be more robust)
                class MockFeatures:
                    def __init__(self, row, idx):
                        self.bar_idx = idx
                        self.timestamp = row.get('timestamp', '')
                        self.current_price = row.get('close', 0)
                        self.atr = row.get('atr', 5.0)  # Default
                        
                features = MockFeatures(df.iloc[idx], idx)
                
                result = trigger.check(features)
                
                if result.triggered:
                    occurrence = PatternOccurrence(
                        bar_idx=idx,
                        timestamp=str(df.iloc[idx].get('timestamp', '')),
                        pattern_type=pattern_type,
                        context=result.context,
                    )
                    
                    # Calculate forward outcome if requested
                    if include_outcomes and idx + 20 < len(df):
                        # Simple forward PnL (price change over next 20 bars)
                        entry_price = df.iloc[idx]['close']
                        future_prices = df.iloc[idx+1:idx+21]['close']
                        max_profit = (future_prices.max() - entry_price)
                        occurrence.forward_pnl = max_profit
                    
                    occurrences.append(occurrence)
                    
            except Exception as e:
                # Skip bars that fail
                continue
        
        # Calculate stats
        success_rate = None
        avg_pnl = None
        best_time = None
        
        if include_outcomes and occurrences:
            successful = [o for o in occurrences if o.forward_pnl and o.forward_pnl > 0]
            success_rate = len(successful) / len(occurrences)
            avg_pnl = sum(o.forward_pnl for o in occurrences if o.forward_pnl) / len(occurrences)
            
            # Find best time of day
            from datetime import datetime
            time_buckets = defaultdict(list)
            for occ in occurrences:
                try:
                    ts = pd.to_datetime(occ.timestamp)
                    hour = ts.hour
                    time_buckets[hour].append(occ.forward_pnl or 0)
                except:
                    pass
            
            if time_buckets:
                best_hour = max(time_buckets.items(), key=lambda x: sum(x[1])/len(x[1]) if x[1] else 0)
                best_time = f"{best_hour[0]:02d}:00"
        
        return PatternStats(
            pattern_type=pattern_type,
            total_occurrences=len(occurrences),
            occurrences=occurrences,
            success_rate=success_rate,
            avg_forward_pnl=avg_pnl,
            best_time_of_day=best_time,
        )
    
    @staticmethod
    def compare_patterns(
        data_path: str,
        pattern_configs: List[Dict[str, Any]],
        include_outcomes: bool = True
    ) -> Dict[str, PatternStats]:
        """
        Compare multiple patterns on same data.
        
        Returns dict mapping pattern description -> PatternStats
        """
        results = {}
        for config in pattern_configs:
            pattern_type = config.get('type', 'unknown')
            pattern_name = config.get('name', pattern_type)
            
            stats = PatternScanner.scan(
                data_path=data_path,
                pattern_type=pattern_type,
                pattern_config={k: v for k, v in config.items() if k not in ['type', 'name']},
                include_outcomes=include_outcomes
            )
            
            results[pattern_name] = stats
        
        return results
    
    @staticmethod
    def print_pattern_report(stats: PatternStats):
        """Print human-readable pattern report."""
        print(f"\nPattern Analysis: {stats.pattern_type}")
        print("=" * 60)
        print(f"Total Occurrences: {stats.total_occurrences}")
        
        if stats.success_rate is not None:
            print(f"Success Rate: {stats.success_rate:.1%}")
            print(f"Avg Forward PnL: ${stats.avg_forward_pnl:.2f}")
            print(f"Best Time of Day: {stats.best_time_of_day}")
        
        if stats.occurrences:
            print(f"\nSample Occurrences (showing first 5):")
            for i, occ in enumerate(stats.occurrences[:5], 1):
                print(f"  {i}. Bar {occ.bar_idx} at {occ.timestamp}")
                if occ.forward_pnl is not None:
                    print(f"     Forward PnL: ${occ.forward_pnl:.2f}")
    
    @staticmethod
    def print_comparison_report(results: Dict[str, PatternStats]):
        """Print comparison of multiple patterns."""
        print("\nPattern Comparison Report")
        print("=" * 80)
        
        # Sort by success rate if available
        sorted_patterns = sorted(
            results.items(),
            key=lambda x: x[1].success_rate if x[1].success_rate else 0,
            reverse=True
        )
        
        print(f"\n{'Pattern':<30} {'Count':<10} {'Success':<12} {'Avg PnL':<12} {'Best Time'}")
        print("-" * 80)
        
        for name, stats in sorted_patterns:
            success = f"{stats.success_rate:.1%}" if stats.success_rate else "N/A"
            pnl = f"${stats.avg_forward_pnl:.2f}" if stats.avg_forward_pnl else "N/A"
            time = stats.best_time_of_day or "N/A"
            
            print(f"{name:<30} {stats.total_occurrences:<10} {success:<12} {pnl:<12} {time}")


# Convenience function
def scan_pattern(**kwargs) -> PatternStats:
    """Quick pattern scan for agents."""
    return PatternScanner.scan(**kwargs)
