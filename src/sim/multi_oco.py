"""
Multi-OCO Grid Trading - Test multiple OCO brackets simultaneously.

Allows testing different risk/reward combinations on the same trade setup.
Supports limit order entries for better execution.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.sim.oco import OCOConfig


@dataclass
class MultiOCOConfig:
    """
    Configuration for multiple OCO brackets on same trigger.
    
    This allows testing different combinations of:
    - Entry offsets (limit order distances)
    - Stop loss sizes
    - Take profit targets
    
    All on the same trade trigger to compare performance.
    """
    
    # Base parameters
    direction: str = "LONG"
    use_limit_entry: bool = True  # Use limit orders instead of market
    
    # OCO bracket variations
    oco_configs: List[OCOConfig] = field(default_factory=list)
    
    # Naming
    grid_name: str = "multi_oco_grid"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'direction': self.direction,
            'use_limit_entry': self.use_limit_entry,
            'oco_configs': [oco.to_dict() for oco in self.oco_configs],
            'grid_name': self.grid_name,
        }
    
    @classmethod
    def create_standard_grid(
        cls,
        direction: str = "LONG",
        base_stop_atr: float = 1.0,
        tp_multiples: List[float] = None,
        entry_offsets: List[float] = None,
    ) -> 'MultiOCOConfig':
        """
        Create a standard multi-OCO grid.
        
        Args:
            direction: LONG or SHORT
            base_stop_atr: Base stop loss size in ATR
            tp_multiples: List of TP multiples to test (e.g., [1.0, 1.5, 2.0])
            entry_offsets: List of entry offsets in ATR (e.g., [0.1, 0.25, 0.5])
            
        Returns:
            MultiOCOConfig with all combinations
        """
        if tp_multiples is None:
            tp_multiples = [1.0, 1.5, 2.0]
        
        if entry_offsets is None:
            entry_offsets = [0.25]  # Single offset by default
        
        oco_configs = []
        
        for entry_offset in entry_offsets:
            for tp_mult in tp_multiples:
                oco = OCOConfig(
                    direction=direction,
                    entry_type="LIMIT",
                    entry_offset_atr=entry_offset,
                    stop_atr=base_stop_atr,
                    tp_multiple=tp_mult,
                    name=f"oco_offset{entry_offset}_tp{tp_mult}"
                )
                oco_configs.append(oco)
        
        return cls(
            direction=direction,
            use_limit_entry=True,
            oco_configs=oco_configs,
            grid_name=f"grid_{len(oco_configs)}ocos"
        )
    
    @classmethod
    def create_tight_medium_wide(
        cls,
        direction: str = "LONG",
        entry_offset: float = 0.25
    ) -> 'MultiOCOConfig':
        """
        Create 3-OCO grid: tight, medium, wide targets.
        
        Common pattern for testing different profit targets.
        """
        oco_configs = [
            OCOConfig(
                direction=direction,
                entry_type="LIMIT",
                entry_offset_atr=entry_offset,
                stop_atr=1.0,
                tp_multiple=1.0,  # Tight: 1R
                name="tight_1R"
            ),
            OCOConfig(
                direction=direction,
                entry_type="LIMIT",
                entry_offset_atr=entry_offset,
                stop_atr=1.0,
                tp_multiple=1.5,  # Medium: 1.5R
                name="medium_1.5R"
            ),
            OCOConfig(
                direction=direction,
                entry_type="LIMIT",
                entry_offset_atr=entry_offset,
                stop_atr=1.0,
                tp_multiple=2.0,  # Wide: 2R
                name="wide_2R"
            ),
        ]
        
        return cls(
            direction=direction,
            use_limit_entry=True,
            oco_configs=oco_configs,
            grid_name="tight_medium_wide"
        )
    
    @classmethod
    def create_entry_ladder(
        cls,
        direction: str = "LONG",
        entry_offsets: List[float] = None,
        tp_multiple: float = 1.5,
        stop_atr: float = 1.0
    ) -> 'MultiOCOConfig':
        """
        Create OCO grid with different entry levels (ladder).
        
        Tests different entry distances from trigger.
        """
        if entry_offsets is None:
            entry_offsets = [0.1, 0.25, 0.5]  # Close, medium, far
        
        oco_configs = []
        
        for offset in entry_offsets:
            oco = OCOConfig(
                direction=direction,
                entry_type="LIMIT",
                entry_offset_atr=offset,
                stop_atr=stop_atr,
                tp_multiple=tp_multiple,
                name=f"entry_offset{offset}"
            )
            oco_configs.append(oco)
        
        return cls(
            direction=direction,
            use_limit_entry=True,
            oco_configs=oco_configs,
            grid_name="entry_ladder"
        )


@dataclass
class MultiOCOResult:
    """Results from multi-OCO execution."""
    
    grid_name: str
    trigger_bar: int
    direction: str
    
    # Individual OCO results
    oco_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Summary stats
    total_ocos: int = 0
    filled_ocos: int = 0  # How many got filled
    winning_ocos: int = 0
    losing_ocos: int = 0
    
    total_pnl: float = 0.0
    best_oco: Optional[str] = None
    worst_oco: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'grid_name': self.grid_name,
            'trigger_bar': self.trigger_bar,
            'direction': self.direction,
            'total_ocos': self.total_ocos,
            'filled_ocos': self.filled_ocos,
            'winning_ocos': self.winning_ocos,
            'losing_ocos': self.losing_ocos,
            'total_pnl': self.total_pnl,
            'best_oco': self.best_oco,
            'worst_oco': self.worst_oco,
            'oco_results': self.oco_results,
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        output = [
            f"\nMulti-OCO Grid Result: {self.grid_name}",
            "=" * 60,
            f"Trigger Bar: {self.trigger_bar}",
            f"Direction: {self.direction}",
            f"Total OCOs: {self.total_ocos}",
            f"Filled: {self.filled_ocos} ({self.filled_ocos/self.total_ocos*100:.0f}%)" if self.total_ocos > 0 else "Filled: 0",
            f"Winners: {self.winning_ocos}",
            f"Losers: {self.losing_ocos}",
            f"Total PnL: ${self.total_pnl:.2f}",
        ]
        
        if self.best_oco:
            output.append(f"Best OCO: {self.best_oco}")
        
        if self.worst_oco:
            output.append(f"Worst OCO: {self.worst_oco}")
        
        output.append("\nIndividual OCO Results:")
        output.append("-" * 60)
        
        for oco in self.oco_results:
            name = oco.get('name', 'unknown')
            filled = "✓" if oco.get('filled', False) else "✗"
            pnl = oco.get('pnl', 0.0)
            status = oco.get('status', 'unknown')
            
            output.append(f"  {filled} {name:20s} PnL: ${pnl:6.2f}  Status: {status}")
        
        return "\n".join(output)


class MultiOCOHelper:
    """
    Helper functions for multi-OCO strategies.
    
    Usage in agent code:
        # Create grid
        grid = MultiOCOConfig.create_tight_medium_wide(direction="LONG")
        
        # Or custom
        grid = MultiOCOConfig.create_standard_grid(
            direction="LONG",
            base_stop_atr=1.0,
            tp_multiples=[1.0, 1.5, 2.0, 2.5]
        )
        
        # Use in strategy config
        strategy = StrategyBuilder.create(
            name="Multi-OCO Test",
            scanner_id="rsi_extreme",
            multi_oco_config=grid
        )
    """
    
    @staticmethod
    def analyze_grid_performance(
        results: List[MultiOCOResult]
    ) -> Dict[str, Any]:
        """
        Analyze performance across multiple grid executions.
        
        Returns:
            Analysis showing which OCO configs perform best overall
        """
        # Collect stats per OCO config
        oco_stats = {}
        
        for result in results:
            for oco in result.oco_results:
                name = oco.get('name', 'unknown')
                
                if name not in oco_stats:
                    oco_stats[name] = {
                        'trades': 0,
                        'wins': 0,
                        'losses': 0,
                        'total_pnl': 0.0,
                        'filled_count': 0,
                    }
                
                stats = oco_stats[name]
                stats['trades'] += 1
                
                if oco.get('filled', False):
                    stats['filled_count'] += 1
                    pnl = oco.get('pnl', 0.0)
                    stats['total_pnl'] += pnl
                    
                    if pnl > 0:
                        stats['wins'] += 1
                    else:
                        stats['losses'] += 1
        
        # Calculate metrics
        for name, stats in oco_stats.items():
            if stats['filled_count'] > 0:
                stats['fill_rate'] = stats['filled_count'] / stats['trades']
                stats['win_rate'] = stats['wins'] / stats['filled_count']
                stats['avg_pnl'] = stats['total_pnl'] / stats['filled_count']
            else:
                stats['fill_rate'] = 0.0
                stats['win_rate'] = 0.0
                stats['avg_pnl'] = 0.0
        
        # Find best
        best_by_pnl = max(oco_stats.items(), key=lambda x: x[1]['total_pnl'])
        best_by_wr = max(oco_stats.items(), key=lambda x: x[1]['win_rate'])
        
        return {
            'oco_stats': oco_stats,
            'best_by_total_pnl': best_by_pnl[0],
            'best_by_win_rate': best_by_wr[0],
            'total_grids_analyzed': len(results),
        }
    
    @staticmethod
    def print_analysis(analysis: Dict[str, Any]):
        """Print multi-OCO performance analysis."""
        print("\nMulti-OCO Performance Analysis")
        print("=" * 80)
        print(f"Total Grids Analyzed: {analysis['total_grids_analyzed']}")
        print(f"\nBest by Total PnL: {analysis['best_by_total_pnl']}")
        print(f"Best by Win Rate: {analysis['best_by_win_rate']}")
        
        print("\nDetailed OCO Statistics:")
        print("-" * 80)
        print(f"{'OCO Name':<25} {'Trades':<8} {'Fill%':<8} {'Win%':<8} {'Avg PnL':<10} {'Total PnL'}")
        print("-" * 80)
        
        stats = analysis['oco_stats']
        for name in sorted(stats.keys()):
            s = stats[name]
            print(
                f"{name:<25} "
                f"{s['trades']:<8} "
                f"{s['fill_rate']*100:<7.1f}% "
                f"{s['win_rate']*100:<7.1f}% "
                f"${s['avg_pnl']:<9.2f} "
                f"${s['total_pnl']:.2f}"
            )
