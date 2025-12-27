"""
Pattern Recognition Skills
Atomic tools for identifying chart patterns (Flags, Wedges, Pullbacks).
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.core.tool_registry import ToolRegistry, ToolCategory
from src.features import patterns

@ToolRegistry.register(
    tool_id="detect_chart_patterns",
    category=ToolCategory.SCANNER,
    name="Detect Chart Patterns",
    description="Identify flags and wedges in recent price action",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol to check (default: continuous)",
                "default": "continuous"
            },
            "lookback_bars": {
                "type": "integer",
                "description": "Number of bars to analyze (default: 100)",
                "default": 100
            },
            "pattern_type": {
                "type": "string",
                "enum": ["ALL", "FLAG", "WEDGE"],
                "default": "ALL"
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "patterns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "direction": {"type": "string"},
                        "start_idx": {"type": "integer"},
                        "end_idx": {"type": "integer"},
                        "entry": {"type": "number"},
                        "stop": {"type": "number"},
                        "target": {"type": "number"},
                        "confidence": {"type": "number"}
                    }
                }
            }
        }
    }
)
class DetectChartPatternsTool:
    def execute(self, symbol: str = "continuous", lookback_bars: int = 100, pattern_type: str = "ALL", **kwargs) -> Dict[str, Any]:
        """Detect chart patterns."""
        from src.data.loader import load_continuous_contract

        df = load_continuous_contract()
        # Ensure enough data
        if len(df) > lookback_bars + 50:
            df = df.tail(lookback_bars + 50) # Add buffer for lookback window within feature

        found_patterns = []

        # Detect Flags
        if pattern_type in ["ALL", "FLAG"]:
            flags = patterns.detect_flags(df, lookback=30)
            found_patterns.extend(flags)

        # Detect Wedges
        if pattern_type in ["ALL", "WEDGE"]:
            wedges = patterns.detect_wedges(df, lookback=30)
            found_patterns.extend(wedges)

        # Sort by confidence
        found_patterns.sort(key=lambda x: x.confidence, reverse=True)

        # Convert to dict
        result = []
        for p in found_patterns:
            result.append({
                "type": p.pattern_type,
                "direction": p.direction,
                "start_idx": int(p.start_idx) if hasattr(p.start_idx, '__int__') else str(p.start_idx),
                "end_idx": int(p.end_idx) if hasattr(p.end_idx, '__int__') else str(p.end_idx),
                "entry": round(p.entry_price, 2),
                "stop": round(p.stop_loss, 2),
                "target": round(p.target_price, 2),
                "confidence": p.confidence
            })

        return {"patterns": result}


@ToolRegistry.register(
    tool_id="analyze_pullback",
    category=ToolCategory.SCANNER,
    name="Analyze Pullback",
    description="Analyze historical pullbacks to EMA or key levels",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol to check",
                "default": "continuous"
            },
            "ema_period": {
                "type": "integer",
                "default": 20
            },
            "lookback_bars": {
                "type": "integer",
                "description": "Number of bars to analyze (default: 500)",
                "default": 500
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "pullbacks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "direction": {"type": "string"},
                        "idx": {"type": "integer"},
                        "entry": {"type": "number"},
                        "stop": {"type": "number"},
                        "target": {"type": "number"},
                        "confidence": {"type": "number"}
                    }
                }
            },
            "count": {"type": "integer"},
            "is_current_pullback": {"type": "boolean"}
        }
    }
)
class AnalyzePullbackTool:
    def execute(self, symbol: str = "continuous", ema_period: int = 20, lookback_bars: int = 500, **kwargs) -> Dict[str, Any]:
        """Analyze pullback."""
        from src.data.loader import load_continuous_contract

        df = load_continuous_contract()
        if len(df) > lookback_bars + ema_period:
            df = df.tail(lookback_bars + ema_period)

        pullbacks = patterns.detect_pullback(df, ema_period=ema_period)

        # Check if current bar is pullback
        is_current = False
        if pullbacks:
            last_idx = df.index[-1]
            if pullbacks[-1].end_idx == last_idx:
                is_current = True

        # Format results
        result_list = []
        for p in pullbacks:
            result_list.append({
                "direction": p.direction,
                "idx": int(p.end_idx) if hasattr(p.end_idx, '__int__') else str(p.end_idx),
                "entry": round(p.entry_price, 2),
                "stop": round(p.stop_loss, 2),
                "target": round(p.target_price, 2),
                "confidence": p.confidence
            })

        return {
            "pullbacks": result_list,
            "count": len(result_list),
            "is_current_pullback": is_current
        }
