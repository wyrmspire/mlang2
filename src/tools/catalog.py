"""
Tool Catalog - Central discovery system for all agent capabilities.

Provides a comprehensive, searchable catalog of all available tools,
triggers, scanners, models, and utilities that agents can use.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import inspect


@dataclass
class ToolInfo:
    """Metadata about a tool."""
    tool_id: str
    name: str
    category: str  # 'trigger', 'scanner', 'model', 'bracket', 'indicator', 'utility'
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    related_tools: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tool_id': self.tool_id,
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'parameters': self.parameters,
            'examples': self.examples,
            'related_tools': self.related_tools,
        }


class ToolCatalog:
    """
    Comprehensive catalog of all agent tools.
    
    Usage:
        catalog = ToolCatalog()
        
        # List all tools
        all_tools = catalog.list_all()
        
        # Find tools by category
        triggers = catalog.list_by_category('trigger')
        
        # Search for tools
        results = catalog.search('time')
        
        # Get detailed info
        info = catalog.get_info('time_trigger')
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolInfo] = {}
        self._register_all_tools()
    
    def _register_all_tools(self):
        """Auto-discover and register all available tools."""
        
        # Register Triggers
        self._register_triggers()
        
        # Register Scanners
        self._register_scanners()
        
        # Register Brackets
        self._register_brackets()
        
        # Register Models
        self._register_models()
        
        # Register Indicators
        self._register_indicators()
        
        # Register Utilities
        self._register_utilities()
    
    def _register_triggers(self):
        """Register all available triggers."""
        from src.policy.triggers import TRIGGER_REGISTRY
        
        trigger_docs = {
            'time': {
                'name': 'Time Trigger',
                'description': 'Fires at specific time(s) of day. Useful for session-based strategies.',
                'parameters': {
                    'hour': {'type': 'int', 'description': 'Hour (0-23)', 'required': False},
                    'minute': {'type': 'int', 'description': 'Minute (0-59)', 'default': 0},
                    'hours': {'type': 'list[int]', 'description': 'Multiple hours', 'required': False},
                },
                'examples': [
                    '{"type": "time", "hour": 10, "minute": 0}  # Market open',
                    '{"type": "time", "hours": [10, 14], "minute": 30}  # Multiple times',
                ],
                'related_tools': ['simple_time_scanner', 'session_filters']
            },
            'candle_pattern': {
                'name': 'Candle Pattern Trigger',
                'description': 'Detects candlestick patterns (hammer, doji, engulfing, etc.)',
                'parameters': {
                    'patterns': {'type': 'list[str]', 'description': 'Patterns to detect', 'required': True},
                    'lookback': {'type': 'int', 'description': 'Candles to check', 'default': 1},
                },
                'examples': [
                    '{"type": "candle_pattern", "patterns": ["hammer"]}',
                    '{"type": "candle_pattern", "patterns": ["doji", "engulfing"]}',
                ],
                'related_tools': ['pattern_scanner']
            },
            'ema_cross': {
                'name': 'EMA Crossover Trigger',
                'description': 'Fires on EMA crossover events (fast crosses slow)',
                'parameters': {
                    'fast': {'type': 'int', 'description': 'Fast EMA period', 'default': 9},
                    'slow': {'type': 'int', 'description': 'Slow EMA period', 'default': 21},
                },
                'examples': [
                    '{"type": "ema_cross", "fast": 9, "slow": 21}',
                    '{"type": "ema_cross", "fast": 12, "slow": 26}',
                ],
                'related_tools': ['indicator_triggers', 'trend_scanners']
            },
            'rsi_threshold': {
                'name': 'RSI Threshold Trigger',
                'description': 'Fires when RSI crosses threshold (overbought/oversold)',
                'parameters': {
                    'threshold': {'type': 'float', 'description': 'RSI threshold', 'required': True},
                    'direction': {'type': 'str', 'description': 'above or below', 'required': True},
                    'oversold': {'type': 'float', 'description': 'Oversold level', 'default': 30},
                    'overbought': {'type': 'float', 'description': 'Overbought level', 'default': 70},
                },
                'examples': [
                    '{"type": "rsi_threshold", "threshold": 30, "direction": "below"}',
                    '{"type": "rsi_threshold", "oversold": 25, "overbought": 75}',
                ],
                'related_tools': ['rsi_extreme_scanner', 'mean_reversion']
            },
        }
        
        for trigger_id, trigger_class in TRIGGER_REGISTRY.items():
            docs = trigger_docs.get(trigger_id, {})
            self._tools[f'{trigger_id}_trigger'] = ToolInfo(
                tool_id=f'{trigger_id}_trigger',
                name=docs.get('name', f'{trigger_id.title()} Trigger'),
                category='trigger',
                description=docs.get('description', f'{trigger_id} trigger'),
                parameters=docs.get('parameters', {}),
                examples=docs.get('examples', []),
                related_tools=docs.get('related_tools', []),
            )
    
    def _register_scanners(self):
        """Register all available scanners."""
        scanners = [
            {
                'tool_id': 'always_scanner',
                'name': 'Always Scanner',
                'description': 'Triggers on every bar. Useful for testing or fixed-interval strategies.',
                'parameters': {},
                'examples': ['{"scanner_id": "always"}'],
                'related_tools': ['interval_scanner'],
            },
            {
                'tool_id': 'interval_scanner',
                'name': 'Interval Scanner',
                'description': 'Triggers every N bars.',
                'parameters': {
                    'interval': {'type': 'int', 'description': 'Bars between triggers', 'default': 5},
                },
                'examples': ['{"scanner_id": "interval", "interval": 10}'],
                'related_tools': ['always_scanner'],
            },
            {
                'tool_id': 'level_proximity_scanner',
                'name': 'Level Proximity Scanner',
                'description': 'Triggers when price approaches key levels (1h/4h highs/lows, PDH/PDL)',
                'parameters': {
                    'atr_threshold': {'type': 'float', 'description': 'Distance in ATR', 'default': 0.5},
                    'level_types': {'type': 'list[str]', 'description': 'Levels to check', 'default': ['1h', '4h', 'pdh', 'pdl']},
                },
                'examples': ['{"scanner_id": "level_proximity", "atr_threshold": 0.3}'],
                'related_tools': ['structure_break_scanner'],
            },
            {
                'tool_id': 'rsi_extreme_scanner',
                'name': 'RSI Extreme Scanner',
                'description': 'Triggers at RSI extremes (overbought/oversold)',
                'parameters': {
                    'oversold': {'type': 'float', 'description': 'Oversold level', 'default': 30},
                    'overbought': {'type': 'float', 'description': 'Overbought level', 'default': 70},
                },
                'examples': ['{"scanner_id": "rsi_extreme", "oversold": 25, "overbought": 75}'],
                'related_tools': ['rsi_threshold_trigger', 'mean_reversion'],
            },
            {
                'tool_id': 'modular_scanner',
                'name': 'Modular Scanner',
                'description': 'Wraps any trigger in a scanner interface with cooldown.',
                'parameters': {
                    'trigger_config': {'type': 'dict', 'description': 'Trigger configuration', 'required': True},
                    'cooldown_bars': {'type': 'int', 'description': 'Bars before next trigger', 'default': 20},
                },
                'examples': [
                    '{"trigger_config": {"type": "time", "hour": 10}, "cooldown_bars": 30}',
                ],
                'related_tools': ['trigger_composer'],
            },
        ]
        
        for scanner in scanners:
            self._tools[scanner['tool_id']] = ToolInfo(**scanner, category='scanner')
    
    def _register_brackets(self):
        """Register all bracket/risk management tools."""
        from src.policy.brackets import BRACKET_REGISTRY
        
        bracket_docs = {
            'atr': {
                'name': 'ATR Bracket',
                'description': 'Stop loss and take profit based on ATR multiples. Adapts to volatility.',
                'parameters': {
                    'stop_atr': {'type': 'float', 'description': 'Stop distance in ATR', 'default': 1.0},
                    'tp_atr': {'type': 'float', 'description': 'Take profit in ATR', 'default': 1.5},
                },
                'examples': [
                    '{"type": "atr", "stop_atr": 1.0, "tp_atr": 2.0}  # 1:2 risk/reward',
                    '{"type": "atr", "stop_atr": 0.5, "tp_atr": 1.5}  # Tight stop',
                ],
                'related_tools': ['oco_config', 'risk_calculator'],
            },
            'percent': {
                'name': 'Percent Bracket',
                'description': 'Stop and target as percentage of entry price.',
                'parameters': {
                    'stop_pct': {'type': 'float', 'description': 'Stop loss %', 'required': True},
                    'tp_pct': {'type': 'float', 'description': 'Take profit %', 'required': True},
                },
                'examples': [
                    '{"type": "percent", "stop_pct": 0.5, "tp_pct": 1.0}',
                ],
                'related_tools': ['fixed_bracket'],
            },
            'fixed': {
                'name': 'Fixed Point Bracket',
                'description': 'Fixed point stop and target (absolute price levels).',
                'parameters': {
                    'stop_points': {'type': 'float', 'description': 'Stop distance in points', 'required': True},
                    'tp_points': {'type': 'float', 'description': 'Target in points', 'required': True},
                },
                'examples': [
                    '{"type": "fixed", "stop_points": 10, "tp_points": 20}',
                ],
                'related_tools': ['percent_bracket'],
            },
        }
        
        for bracket_id, bracket_class in BRACKET_REGISTRY.items():
            docs = bracket_docs.get(bracket_id, {})
            self._tools[f'{bracket_id}_bracket'] = ToolInfo(
                tool_id=f'{bracket_id}_bracket',
                name=docs.get('name', f'{bracket_id.title()} Bracket'),
                category='bracket',
                description=docs.get('description', f'{bracket_id} bracket'),
                parameters=docs.get('parameters', {}),
                examples=docs.get('examples', []),
                related_tools=docs.get('related_tools', []),
            )
    
    def _register_models(self):
        """Register available models."""
        models = [
            {
                'tool_id': 'fusion_cnn',
                'name': 'Fusion CNN Model',
                'description': 'CNN for price patterns + MLP for context. Makes trade/no-trade decisions.',
                'parameters': {
                    'model_path': {'type': 'str', 'description': 'Path to trained model', 'required': True},
                    'role': {'type': 'ModelRole', 'description': 'Model role (TRAINING_ONLY, REPLAY_ONLY, etc.)', 'default': 'REPLAY_ONLY'},
                },
                'examples': [
                    'model_path="runs/exp_001/model.pt", role=ModelRole.REPLAY_ONLY',
                ],
                'related_tools': ['train_model', 'simulation_runner'],
            },
        ]
        
        for model in models:
            self._tools[model['tool_id']] = ToolInfo(**model, category='model')
    
    def _register_indicators(self):
        """Register available indicators."""
        indicators = [
            {
                'tool_id': 'ema',
                'name': 'Exponential Moving Average',
                'description': 'Smooth trend indicator',
                'parameters': {
                    'period': {'type': 'int', 'description': 'EMA period', 'default': 20},
                },
                'examples': [],
                'related_tools': ['ema_cross_trigger'],
            },
            {
                'tool_id': 'rsi',
                'name': 'Relative Strength Index',
                'description': 'Momentum oscillator (0-100)',
                'parameters': {
                    'period': {'type': 'int', 'description': 'RSI period', 'default': 14},
                },
                'examples': [],
                'related_tools': ['rsi_threshold_trigger', 'rsi_extreme_scanner'],
            },
            {
                'tool_id': 'atr',
                'name': 'Average True Range',
                'description': 'Volatility indicator',
                'parameters': {
                    'period': {'type': 'int', 'description': 'ATR period', 'default': 14},
                },
                'examples': [],
                'related_tools': ['atr_bracket', 'level_proximity_scanner'],
            },
        ]
        
        for indicator in indicators:
            self._tools[indicator['tool_id']] = ToolInfo(**indicator, category='indicator')
    
    def _register_utilities(self):
        """Register utility tools."""
        utilities = [
            {
                'tool_id': 'strategy_builder',
                'name': 'Strategy Builder',
                'description': 'High-level tool to construct complete strategies from components.',
                'parameters': {
                    'strategy_type': {'type': 'str', 'description': 'Strategy template', 'required': True},
                },
                'examples': [
                    'StrategyBuilder.create("mean_reversion", entry_trigger={...}, bracket={...})',
                ],
                'related_tools': ['trigger_composer', 'strategy_validator'],
            },
            {
                'tool_id': 'trigger_composer',
                'name': 'Trigger Composer',
                'description': 'Combine multiple triggers with AND/OR logic.',
                'parameters': {},
                'examples': [
                    'TriggerComposer.AND([time_trigger, rsi_trigger])',
                    'TriggerComposer.OR([pattern_trigger, level_trigger])',
                ],
                'related_tools': ['strategy_builder'],
            },
            {
                'tool_id': 'pattern_scanner',
                'name': 'Pattern Scanner',
                'description': 'Scan historical data for pattern occurrences.',
                'parameters': {
                    'pattern_type': {'type': 'str', 'description': 'Pattern to scan', 'required': True},
                },
                'examples': [],
                'related_tools': ['candle_pattern_trigger'],
            },
            {
                'tool_id': 'strategy_validator',
                'name': 'Strategy Validator',
                'description': 'Validate strategy configuration before execution.',
                'parameters': {},
                'examples': [],
                'related_tools': ['strategy_builder'],
            },
            {
                'tool_id': 'simulation_runner',
                'name': 'Simulation Runner',
                'description': 'Run strategy with model inference in simulation mode against playback data.',
                'parameters': {
                    'strategy_config': {'type': 'dict', 'description': 'Strategy configuration', 'required': True},
                    'model_path': {'type': 'str', 'description': 'Path to trained model for inference', 'required': True},
                    'data_range': {'type': 'dict', 'description': 'Start/end dates', 'required': True},
                    'multi_oco_config': {'type': 'dict', 'description': 'Multiple OCO brackets to test', 'required': False},
                },
                'examples': [
                    'SimulationRunner.run(strategy_config={...}, model_path="model.pt", data_range={...})',
                ],
                'related_tools': ['strategy_builder', 'fusion_cnn', 'multi_oco_grid'],
            },
            {
                'tool_id': 'multi_oco_grid',
                'name': 'Multi-OCO Grid',
                'description': 'Test multiple OCO brackets simultaneously with limit order entries.',
                'parameters': {
                    'direction': {'type': 'str', 'description': 'LONG or SHORT', 'default': 'LONG'},
                    'tp_multiples': {'type': 'list[float]', 'description': 'Target profit multiples', 'default': [1.0, 1.5, 2.0]},
                    'entry_offsets': {'type': 'list[float]', 'description': 'Limit entry offsets in ATR', 'default': [0.25]},
                },
                'examples': [
                    'MultiOCOConfig.create_tight_medium_wide(direction="LONG", entry_offset=0.25)',
                    'MultiOCOConfig.create_standard_grid(tp_multiples=[1.0, 1.5, 2.0], entry_offsets=[0.1, 0.25])',
                ],
                'related_tools': ['simulation_runner', 'atr_bracket'],
            },
        ]
        
        for utility in utilities:
            self._tools[utility['tool_id']] = ToolInfo(**utility, category='utility')
    
    def list_all(self) -> List[ToolInfo]:
        """List all available tools."""
        return list(self._tools.values())
    
    def list_by_category(self, category: str) -> List[ToolInfo]:
        """List tools in a specific category."""
        return [t for t in self._tools.values() if t.category == category]
    
    def search(self, query: str) -> List[ToolInfo]:
        """Search for tools by name or description."""
        query = query.lower()
        results = []
        for tool in self._tools.values():
            if (query in tool.name.lower() or 
                query in tool.description.lower() or 
                query in tool.tool_id.lower()):
                results.append(tool)
        return results
    
    def get_info(self, tool_id: str) -> Optional[ToolInfo]:
        """Get detailed info for a specific tool."""
        return self._tools.get(tool_id)
    
    def print_catalog(self, category: Optional[str] = None):
        """Print formatted catalog."""
        if category:
            tools = self.list_by_category(category)
            print(f"\n=== {category.upper()} TOOLS ===\n")
        else:
            tools = self.list_all()
            print("\n=== ALL AGENT TOOLS ===\n")
        
        by_category = {}
        for tool in tools:
            if tool.category not in by_category:
                by_category[tool.category] = []
            by_category[tool.category].append(tool)
        
        for cat, cat_tools in sorted(by_category.items()):
            if not category:
                print(f"\n{cat.upper()}:")
            for tool in sorted(cat_tools, key=lambda t: t.tool_id):
                print(f"  • {tool.name} ({tool.tool_id})")
                print(f"    {tool.description}")
                if tool.examples:
                    print(f"    Example: {tool.examples[0]}")
                print()


# Global catalog instance
catalog = ToolCatalog()


def list_all_tools() -> str:
    """Helper for agents to discover all tools."""
    output = ["MLang2 Agent Tools Catalog", "=" * 50, ""]
    
    for category in ['trigger', 'scanner', 'bracket', 'model', 'indicator', 'utility']:
        tools = catalog.list_by_category(category)
        if tools:
            output.append(f"\n{category.upper()}S ({len(tools)}):")
            for tool in sorted(tools, key=lambda t: t.tool_id):
                output.append(f"  • {tool.name}")
                output.append(f"    ID: {tool.tool_id}")
                output.append(f"    {tool.description}")
                if tool.examples:
                    output.append(f"    Example: {tool.examples[0]}")
                output.append("")
    
    return "\n".join(output)
