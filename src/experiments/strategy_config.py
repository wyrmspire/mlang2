"""
Strategy Configuration
Serializable configuration for strategy runs.

This allows strategies to be parameterized and run from the agent or UI
without needing code changes.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import json
from pathlib import Path


@dataclass
class StrategyConfig:
    """
    Complete strategy configuration for a run.
    
    This is the "public API" for configuring and running strategies.
    All parameters should be serializable and agent-controllable.
    """
    
    # Strategy identification
    strategy_id: str = "always"  # Scanner/strategy name
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    
    # Data range
    start_date: str = ""
    end_date: str = ""
    timeframe: str = "1m"
    
    # OCO Configuration
    oco_direction: str = "LONG"  # or "SHORT"
    oco_tp_multiple: float = 1.4
    oco_stop_atr: float = 1.0
    oco_max_bars: int = 200
    oco_entry_type: str = "LIMIT"  # or "MARKET"
    
    # Feature toggles
    use_1m_features: bool = True
    use_5m_features: bool = True
    use_15m_features: bool = True
    use_1h_features: bool = False
    use_4h_features: bool = False
    
    # Filter parameters
    enable_filters: bool = True
    filter_min_volume: Optional[float] = None
    filter_session_only: Optional[str] = None  # "rth", "overnight", None
    
    # Cooldown
    cooldown_bars: int = 10
    
    # Training
    train_model: bool = False
    model_epochs: int = 10
    model_batch_size: int = 64
    
    # Output
    output_name: Optional[str] = None
    enable_viz_export: bool = True
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StrategyConfig':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: Path):
        """Save to JSON file."""
        with open(path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, path: Path) -> 'StrategyConfig':
        """Load from JSON file."""
        with open(path) as f:
            return cls.from_json(f.read())
    
    def to_cli_args(self) -> list:
        """
        Convert to CLI arguments for backwards compatibility.
        
        This allows existing scripts to be called with this config.
        """
        args = [
            '--strategy', self.strategy_id,
            '--start-date', self.start_date,
            '--end-date', self.end_date,
            '--timeframe', self.timeframe,
            '--oco-tp', str(self.oco_tp_multiple),
            '--oco-stop', str(self.oco_stop_atr),
            '--seed', str(self.seed),
        ]
        
        if self.output_name:
            args.extend(['--out-name', self.output_name])
        
        if not self.enable_filters:
            args.append('--no-filters')
        
        if self.train_model:
            args.extend(['--train', '--epochs', str(self.model_epochs)])
        
        return args


# Preset configurations for common strategies
PRESET_CONFIGS = {
    "opening_range_default": StrategyConfig(
        strategy_id="opening_range",
        oco_direction="LONG",
        oco_tp_multiple=1.4,
        oco_stop_atr=1.0,
        use_1m_features=True,
        use_5m_features=True,
        use_15m_features=True,
    ),
    
    "opening_range_conservative": StrategyConfig(
        strategy_id="opening_range",
        oco_direction="LONG",
        oco_tp_multiple=1.0,
        oco_stop_atr=0.8,
        use_1m_features=True,
        use_5m_features=True,
        use_15m_features=True,
        filter_min_volume=1000.0,
    ),
    
    "opening_range_aggressive": StrategyConfig(
        strategy_id="opening_range",
        oco_direction="LONG",
        oco_tp_multiple=2.0,
        oco_stop_atr=1.2,
        use_1m_features=True,
        use_5m_features=True,
        use_15m_features=True,
    ),
    
    "always_default": StrategyConfig(
        strategy_id="always",
        oco_direction="LONG",
        oco_tp_multiple=1.4,
        oco_stop_atr=1.0,
        use_1m_features=True,
        use_5m_features=True,
        use_15m_features=True,
        use_1h_features=True,
        use_4h_features=True,
    ),
}


def get_preset_config(name: str) -> Optional[StrategyConfig]:
    """Get a preset configuration by name."""
    return PRESET_CONFIGS.get(name)
