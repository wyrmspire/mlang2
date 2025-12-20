"""
Strategy Builder - High-level tool for constructing complete strategies.

Helps agents build strategies from components without worrying about
low-level details.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.experiments.strategy_config import StrategyConfig
from src.policy.triggers import trigger_from_dict
from src.policy.brackets import bracket_from_dict


@dataclass
class StrategyTemplate:
    """Reusable strategy template."""
    template_id: str
    name: str
    description: str
    default_config: Dict[str, Any]
    required_params: List[str] = field(default_factory=list)
    recommended_bracket: str = "atr"


class StrategyBuilder:
    """
    Build complete strategies from modular components.
    
    Templates available:
    - mean_reversion: Trade RSI extremes near levels
    - opening_range: First bar breakout strategy
    - time_based: Fixed time entries
    - pattern_following: Candle pattern entries
    - level_bounce: Price rejection at key levels
    
    Usage:
        # From template
        strategy = StrategyBuilder.from_template(
            "mean_reversion",
            oversold=25,
            overbought=75,
            stop_atr=1.0,
            tp_multiple=1.5
        )
        
        # Custom build
        strategy = StrategyBuilder.create(
            name="my_strategy",
            scanner_id="level_proximity",
            scanner_params={"atr_threshold": 0.3},
            trigger={"type": "rsi_threshold", "threshold": 30, "direction": "below"},
            bracket={"type": "atr", "stop_atr": 1.0, "tp_atr": 2.0}
        )
    """
    
    # Strategy templates
    TEMPLATES = {
        'mean_reversion': StrategyTemplate(
            template_id='mean_reversion',
            name='Mean Reversion',
            description='Trade RSI extremes with level confluence',
            default_config={
                'scanner_id': 'rsi_extreme',
                'scanner_params': {
                    'oversold': 30,
                    'overbought': 70,
                },
                'oco_tp_multiple': 1.5,
                'oco_stop_atr': 1.0,
            },
            recommended_bracket='atr',
        ),
        'opening_range': StrategyTemplate(
            template_id='opening_range',
            name='Opening Range Breakout',
            description='Trade breakouts from first bar range',
            default_config={
                'scanner_id': 'openingrange',
                'scanner_params': {
                    'or_minutes': 30,
                    'atr_threshold': 0.5,
                },
                'oco_tp_multiple': 1.4,
                'oco_stop_atr': 0.8,
            },
            recommended_bracket='atr',
        ),
        'time_based': StrategyTemplate(
            template_id='time_based',
            name='Time-Based Entry',
            description='Enter at specific times with model confirmation',
            default_config={
                'scanner_id': 'simpletime',
                'scanner_params': {
                    'entry_hour': 10,
                    'entry_minute': 0,
                },
                'oco_tp_multiple': 1.5,
                'oco_stop_atr': 1.0,
            },
            recommended_bracket='atr',
        ),
        'pattern_following': StrategyTemplate(
            template_id='pattern_following',
            name='Pattern Following',
            description='Enter on candle patterns',
            default_config={
                'scanner_id': 'modular',
                'trigger_config': {
                    'type': 'candle_pattern',
                    'patterns': ['hammer', 'engulfing'],
                },
                'oco_tp_multiple': 1.5,
                'oco_stop_atr': 1.0,
            },
            recommended_bracket='atr',
        ),
        'level_bounce': StrategyTemplate(
            template_id='level_bounce',
            name='Level Bounce',
            description='Trade rejections at key levels',
            default_config={
                'scanner_id': 'level_proximity',
                'scanner_params': {
                    'atr_threshold': 0.5,
                    'level_types': ['1h', '4h', 'pdh', 'pdl'],
                },
                'oco_tp_multiple': 1.5,
                'oco_stop_atr': 1.0,
            },
            recommended_bracket='atr',
        ),
    }
    
    @classmethod
    def list_templates(cls) -> List[StrategyTemplate]:
        """List all available strategy templates."""
        return list(cls.TEMPLATES.values())
    
    @classmethod
    def get_template(cls, template_id: str) -> StrategyTemplate:
        """Get a specific template."""
        if template_id not in cls.TEMPLATES:
            raise ValueError(f"Unknown template: {template_id}. Available: {list(cls.TEMPLATES.keys())}")
        return cls.TEMPLATES[template_id]
    
    @classmethod
    def from_template(
        cls,
        template_id: str,
        start_date: str = "",
        end_date: str = "",
        **overrides
    ) -> StrategyConfig:
        """
        Create strategy from template with optional parameter overrides.
        
        Args:
            template_id: Template to use
            start_date: Start date for backtest
            end_date: End date for backtest
            **overrides: Override any default parameters
            
        Returns:
            StrategyConfig ready for simulation
        """
        template = cls.get_template(template_id)
        
        # Merge defaults with overrides
        config = template.default_config.copy()
        
        # Handle nested scanner_params
        if 'scanner_params' in overrides:
            if 'scanner_params' in config:
                config['scanner_params'].update(overrides.pop('scanner_params'))
            else:
                config['scanner_params'] = overrides.pop('scanner_params')
        
        config.update(overrides)
        
        # Create strategy config
        return StrategyConfig(
            strategy_id=template_id,
            name=template.name,
            description=template.description,
            start_date=start_date,
            end_date=end_date,
            **config
        )
    
    @classmethod
    def create(
        cls,
        name: str,
        scanner_id: str,
        start_date: str = "",
        end_date: str = "",
        scanner_params: Optional[Dict[str, Any]] = None,
        trigger: Optional[Dict[str, Any]] = None,
        bracket: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> StrategyConfig:
        """
        Create custom strategy from components.
        
        Args:
            name: Strategy name
            scanner_id: Scanner to use
            start_date: Start date
            end_date: End date
            scanner_params: Scanner configuration
            trigger: Trigger configuration (for modular scanner)
            bracket: Bracket configuration
            **kwargs: Additional StrategyConfig parameters
            
        Returns:
            StrategyConfig
        """
        config = {
            'strategy_id': name.lower().replace(' ', '_'),
            'name': name,
            'scanner_id': scanner_id,
            'scanner_params': scanner_params or {},
            'start_date': start_date,
            'end_date': end_date,
        }
        
        # If using modular scanner, add trigger config
        if scanner_id == 'modular' and trigger:
            config['scanner_params']['trigger_config'] = trigger
        
        # Add bracket if specified
        if bracket:
            # Extract bracket type and params
            bracket_type = bracket.get('type', 'atr')
            if bracket_type == 'atr':
                config['oco_stop_atr'] = bracket.get('stop_atr', 1.0)
                config['oco_tp_multiple'] = bracket.get('tp_atr', 1.5) / bracket.get('stop_atr', 1.0)
        
        config.update(kwargs)
        
        return StrategyConfig(**config)
    
    @classmethod
    def validate_trigger(cls, trigger_config: Dict[str, Any]) -> bool:
        """
        Validate trigger configuration.
        
        Returns True if valid, raises ValueError if not.
        """
        try:
            trigger = trigger_from_dict(trigger_config)
            return True
        except Exception as e:
            raise ValueError(f"Invalid trigger config: {e}")
    
    @classmethod
    def validate_bracket(cls, bracket_config: Dict[str, Any]) -> bool:
        """
        Validate bracket configuration.
        
        Returns True if valid, raises ValueError if not.
        """
        try:
            bracket = bracket_from_dict(bracket_config)
            return True
        except Exception as e:
            raise ValueError(f"Invalid bracket config: {e}")
    
    @classmethod
    def explain_template(cls, template_id: str) -> str:
        """Get detailed explanation of a template."""
        template = cls.get_template(template_id)
        
        output = [
            f"Strategy Template: {template.name}",
            "=" * 60,
            f"\nDescription: {template.description}",
            f"\nTemplate ID: {template.template_id}",
            f"\nRecommended Bracket: {template.recommended_bracket}",
            f"\nDefault Configuration:",
        ]
        
        import json
        output.append(json.dumps(template.default_config, indent=2))
        
        output.append("\nUsage:")
        output.append(f"  strategy = StrategyBuilder.from_template('{template_id}',")
        output.append(f"      start_date='2025-03-01',")
        output.append(f"      end_date='2025-03-15')")
        
        return "\n".join(output)
    
    @classmethod
    def print_all_templates(cls):
        """Print all available templates."""
        print("\nAvailable Strategy Templates")
        print("=" * 60)
        
        for template in cls.list_templates():
            print(f"\n• {template.name} ({template.template_id})")
            print(f"  {template.description}")
            print(f"  Scanner: {template.default_config.get('scanner_id')}")
            print(f"  Bracket: {template.recommended_bracket}")
