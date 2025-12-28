        df = load_continuous_contract()
        
        # Filter last N days
        cutoff = df['time'].iloc[-1] - timedelta(days=lookback_days)
        df = df[df['time'] >= cutoff].copy()
        
        df['hour'] = df['time'].dt.hour
        df['range'] = df['high'] - df['low']
        
        stats = df.groupby('hour').agg({
            'range': 'mean',
            'volume': 'mean',
            'close': 'std' # Simple volatility proxy
        }).reset_index()
        
        hourly_stats = []
        for _, row in stats.iterrows():
            hourly_stats.append({
                "hour": int(row['hour']),
                "avg_range": round(float(row['range']), 2),
                "avg_volume": round(float(row['volume']), 0),
                "volatility": round(float(row['close']), 2)
            })
            
        return {"hourly_stats": hourly_stats}

```

### src/skills/indicator_skills.py

```python
"""
Indicator Skills
Atomic tools for calculating technical indicators.
These skills wrap the core features library for the Agent's use during Research.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from src.core.tool_registry import ToolRegistry, ToolCategory
from src.features import indicators
from src.features import indicators_pro
from src.features import fvg


@ToolRegistry.register(
    tool_id="get_rsi",
    category=ToolCategory.INDICATOR,
    name="Get RSI",
    description="Calculate RSI (Relative Strength Index) for a list of prices",
    input_schema={
        "type": "object",
        "properties": {
            "prices": {
                "type": "array",
                "items": {"type": "number"},
                "description": "List of price values"
            },
            "period": {
                "type": "integer",
                "description": "RSI period (default: 14)",
                "default": 14
            }
        },
        "required": ["prices"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "rsi_values": {
                "type": "array",
                "items": {"type": "number"}
            }
        }
    }
)
class GetRSITool:
    def execute(self, prices: List[float], period: int = 14, **kwargs) -> Dict[str, Any]:
        """Calculate RSI for a list of prices."""
        series = pd.Series(prices)
        rsi = indicators.calculate_rsi(series, period)
        return {"rsi_values": rsi.tolist()}


@ToolRegistry.register(
    tool_id="check_ema_cross",
    category=ToolCategory.INDICATOR,
    name="Check EMA Cross",
    description="Check if fast EMA crossed slow EMA on the most recent bar",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol to check (default: MES)",
                "default": "continuous"
            },
            "fast": {
                "type": "integer",
                "description": "Fast EMA period",
                "default": 9
            },
            "slow": {
                "type": "integer",
                "description": "Slow EMA period",
                "default": 21
            },
            "lookback_bars": {
                "type": "integer",
                "description": "Number of bars to analyze",
                "default": 100
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "cross_type": {
                "type": "string",
                "enum": ["BULLISH", "BEARISH", "NONE"]
            },
            "fast_value": {"type": "number"},
            "slow_value": {"type": "number"}
        }
    }
)
class CheckEMACrossTool:
    def execute(self, symbol: str = "continuous", fast: int = 9, slow: int = 21, lookback_bars: int = 100, **kwargs) -> Dict[str, Any]:
        """Check if EMA cross occurred."""
        from src.data.loader import load_continuous_contract
        
        df = load_continuous_contract()
        if len(df) > lookback_bars:
            df = df.tail(lookback_bars)
        
        ema_fast = indicators.calculate_ema(df['close'], fast)
        ema_slow = indicators.calculate_ema(df['close'], slow)
        
        if len(df) < 2:
            return {"cross_type": "NONE", "fast_value": 0.0, "slow_value": 0.0}
            
        curr_fast = float(ema_fast.iloc[-1])
        curr_slow = float(ema_slow.iloc[-1])
        prev_fast = float(ema_fast.iloc[-2])
        prev_slow = float(ema_slow.iloc[-2])
        
        cross_type = "NONE"
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            cross_type = "BULLISH"
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            cross_type = "BEARISH"
            
        return {
            "cross_type": cross_type,
            "fast_value": curr_fast,
            "slow_value": curr_slow
        }


@ToolRegistry.register(
    tool_id="get_current_rsi",
    category=ToolCategory.INDICATOR,
    name="Get Current RSI",
    description="Get the current RSI value for a symbol",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol (default: continuous)",
                "default": "continuous"
            },
            "period": {
                "type": "integer",
                "description": "RSI period",
                "default": 14
            },
            "lookback_bars": {
                "type": "integer",
                "description": "Number of bars to use for calculation",
                "default": 50
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "rsi": {"type": "number"},
            "timestamp": {"type": "string"}
        }
    }
)
class GetCurrentRSITool:
    def execute(self, symbol: str = "continuous", period: int = 14, lookback_bars: int = 50, **kwargs) -> Dict[str, Any]:
        """Get current RSI value."""
        from src.data.loader import load_continuous_contract
        
        df = load_continuous_contract()
        if len(df) > lookback_bars:
            df = df.tail(lookback_bars)
        
        if len(df) < period + 1:
            return {"rsi": 50.0, "timestamp": ""}
            
        rsi_series = indicators.calculate_rsi(df['close'], period)
        current_rsi = float(rsi_series.iloc[-1])
        timestamp = str(df['time'].iloc[-1])
        
        return {
            "rsi": current_rsi,
            "timestamp": timestamp
        }
@ToolRegistry.register(
    tool_id="get_atr",
    category=ToolCategory.INDICATOR,
    name="Get ATR",
    description="Calculate Average True Range (volatility) for a symbol",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol (default: continuous)",
                "default": "continuous"
            },
            "period": {
                "type": "integer",
                "description": "ATR period (default: 14)",
                "default": 14
            },
            "lookback_bars": {
                "type": "integer",
                "description": "Number of bars to use (default: 100)",
                "default": 100
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "atr_values": {
                "type": "array",
                "items": {"type": "number"}
            },
            "current_atr": {"type": "number"}
        }
    }
)
class GetATRTool:
    def execute(self, symbol: str = "continuous", period: int = 14, lookback_bars: int = 100, **kwargs) -> Dict[str, Any]:
        """Calculate ATR."""
        from src.data.loader import load_continuous_contract
        df = load_continuous_contract()
        if len(df) > lookback_bars + period:
            df = df.tail(lookback_bars + period)
        
        atr_series = indicators.calculate_atr(df, period)
        # Shifted back by 1 because calculate_atr usually shifts to be causal, 
        # but for a point-in-time tool we might want the last calculated value
        atr_values = atr_series.tail(lookback_bars).dropna().tolist()
        current_atr = atr_values[-1] if atr_values else 0.0
        
        return {
            "atr_values": atr_values,
            "current_atr": current_atr
        }


@ToolRegistry.register(
    tool_id="get_vwap",
    category=ToolCategory.INDICATOR,
    name="Get VWAP",
    description="Calculate Volume Weighted Average Price for a symbol",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol (default: continuous)",
                "default": "continuous"
            },
            "period": {
                "type": "string",
                "enum": ["session", "daily", "weekly"],
                "description": "VWAP anchor period",
                "default": "session"
            },
            "lookback_bars": {
                "type": "integer",
                "description": "Number of bars to return",
                "default": 50
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "vwap_values": {
                "type": "array",
                "items": {"type": "number"}
            },
            "current_vwap": {"type": "number"}
        }
    }
)
class GetVWAPTool:
    def execute(self, symbol: str = "continuous", period: str = "session", lookback_bars: int = 50, **kwargs) -> Dict[str, Any]:
        """Calculate VWAP."""
        from src.data.loader import load_continuous_contract
        df = load_continuous_contract()
        # We need enough data for the session/period
        df = df.tail(max(500, lookback_bars))
        
        vwap_series = indicators.calculate_vwap(df, period=period)
        vwap_values = vwap_series.tail(lookback_bars).tolist()
        current_vwap = vwap_values[-1] if vwap_values else 0.0
        
        return {
            "vwap_values": vwap_values,
            "current_vwap": current_vwap
        }


@ToolRegistry.register(
    tool_id="detect_support_resistance",
    category=ToolCategory.INDICATOR,
    name="Detect Support & Resistance",
    description="Identify major support and resistance levels based on price clustering",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol (default: continuous)",
                "default": "continuous"
            },
            "lookback_bars": {
                "type": "integer",
                "description": "Number of bars to analyze (default: 500)",
                "default": 500
            },
            "sensitivity": {
                "type": "number",
                "description": "Cluster sensitivity (default: 1.0)",
                "default": 1.0
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "levels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "price": {"type": "number"},
                        "strength": {"type": "number"},
                        "type": {"type": "string", "enum": ["SUPPORT", "RESISTANCE", "ZONE"]}
                    }
                }
            }
        }
    }
)
class DetectSupportResistanceTool:
    def execute(self, symbol: str = "continuous", lookback_bars: int = 500, sensitivity: float = 1.0, **kwargs) -> Dict[str, Any]:
        """Detect S&R levels."""
        from src.data.loader import load_continuous_contract
        df = load_continuous_contract().tail(lookback_bars)
        
        # Simple clustering: histogram of highs and lows
        prices = pd.concat([df['high'], df['low']])
        # Bin size ~ 0.5 points (typical for ES/MES)
        bins = int((prices.max() - prices.min()) / (0.5 * sensitivity))
        if bins < 5: bins = 5
        if bins > 100: bins = 100
        
        hist, edges = np.histogram(prices, bins=bins)
        
        # Find peaks in histogram
        levels = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > (lookback_bars * 0.05):
                price = float((edges[i] + edges[i+1]) / 2)
                # Determine if it's above or below current price
                current_price = df['close'].iloc[-1]
                ltype = "RESISTANCE" if price > current_price else "SUPPORT"
                
                levels.append({
                    "price": round(price, 2),
                    "strength": int(hist[i]),
                    "type": ltype
                })
        
        # Sort by strength
        levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return {"levels": levels[:10]}


@ToolRegistry.register(
    tool_id="get_volume_profile",
    category=ToolCategory.INDICATOR,
    name="Get Volume Profile",
    description="Calculate volume-at-price profile over a period",
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol (default: continuous)",
                "default": "continuous"
            },
            "lookback_bars": {
                "type": "integer",
                "description": "Number of bars to analyze (default: 500)",
                "default": 500
            },
            "bins": {
                "type": "integer",
                "description": "Number of price bins (default: 50)",
                "default": 50
            }
        }
    },
    output_schema={
        "type": "object",
        "properties": {
            "bins": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "price": {"type": "number"},
                        "volume": {"type": "number"},
                        "is_poc": {"type": "boolean"}
                    }
                }
            },
            "poc_price": {"type": "number"}
        }
    }
)
class GetVolumeProfileTool:
    def execute(self, symbol: str = "continuous", lookback_bars: int = 500, bins: int = 50, **kwargs) -> Dict[str, Any]:
        """Calculate volume profile."""
        from src.data.loader import load_continuous_contract
        df = load_continuous_contract().tail(lookback_bars)
        
        # Determine price ranges and bin width
        p_min = df['low'].min()
        p_max = df['high'].max()
        if p_max == p_min:
            return {"bins": [], "poc_price": p_min}
            
        bin_width = (p_max - p_min) / bins
        
        # Accumulate volume for each bin
        # We simplify by using the close price or distributing between high/low
        # For a simple tool, we'll use close price
        # More advanced would use OHLC interpolation
        hist, edges = np.histogram(df['close'], bins=bins, weights=df['volume'])
        
        poc_idx = np.argmax(hist)
        poc_price = float((edges[poc_idx] + edges[poc_idx+1]) / 2)
        
        result_bins = []
        for i in range(len(hist)):
            price = float((edges[i] + edges[i+1]) / 2)
            result_bins.append({
                "price": round(price, 2),
                "volume": int(hist[i]),
                "is_poc": i == poc_idx
            })
            
        return {
            "bins": result_bins,
            "poc_price": round(poc_price, 2)
        }



```

### src/skills/pattern_skills.py

```python
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

```

### src/storage/__init__.py

```python
"""
Storage module for MLang2.

Provides experiment database and result management.
"""

from src.storage.experiments_db import ExperimentDB, ExperimentRecord

__all__ = ['ExperimentDB', 'ExperimentRecord']

```

### src/storage/experiments_db.py

```python
"""
Experiment Database

SQLite-backed storage for experiment results.
Enables the agent to query past performance and find optimal configurations.

Usage:
    from src.storage.experiments_db import ExperimentDB
    
    db = ExperimentDB()
    db.store_run(run_id, metrics)
    best = db.query_best("win_rate", top_k=5)
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExperimentRecord:
    """Single experiment record."""
    run_id: str
    created_at: str
    strategy: str
    model_path: Optional[str]
    config: Dict[str, Any]
    
    # Key metrics
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    
    # Optional extended metrics
    sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None
    profit_factor: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'run_id': self.run_id,
            'created_at': self.created_at,
            'strategy': self.strategy,
            'model_path': self.model_path,
            'config': self.config,
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'avg_pnl_per_trade': self.avg_pnl_per_trade,
            'sharpe': self.sharpe,
            'max_drawdown': self.max_drawdown,
            'profit_factor': self.profit_factor,
        }


class ExperimentDB:
    """
    SQLite database for experiment results.
    
    Allows the agent to:
    - Store run results
    - Query best configurations by metric
    - Compare strategies
    - Learn from history
    """
    
    def __init__(self, db_path: str = "results/experiments.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                strategy TEXT NOT NULL,
                model_path TEXT,
                config_json TEXT NOT NULL,
                
                total_trades INTEGER NOT NULL,
                wins INTEGER NOT NULL,
                losses INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                total_pnl REAL NOT NULL,
                avg_pnl_per_trade REAL NOT NULL,
                
                sharpe REAL,
                max_drawdown REAL,
                profit_factor REAL
            )
        """)
        
        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy ON experiments(strategy)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_win_rate ON experiments(win_rate DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_total_pnl ON experiments(total_pnl DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created ON experiments(created_at DESC)")
        
        conn.commit()
        conn.close()
    
    def store_run(
        self,
        run_id: str,
        strategy: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        model_path: Optional[str] = None
    ) -> bool:
        """
        Store experiment results.
        
        Args:
            run_id: Unique run identifier
            strategy: Strategy name (e.g., "opening_range", "modular")
            config: Run configuration dict
            metrics: Must include: total_trades, wins, losses, win_rate, total_pnl
            model_path: Path to model file if applicable
        
        Returns:
            True if stored successfully
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO experiments (
                    run_id, created_at, strategy, model_path, config_json,
                    total_trades, wins, losses, win_rate, total_pnl, avg_pnl_per_trade,
                    sharpe, max_drawdown, profit_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                datetime.utcnow().isoformat() + 'Z',
                strategy,
                model_path,
                json.dumps(config),
                metrics.get('total_trades', metrics.get('trades', 0)),
                metrics.get('wins', 0),
                metrics.get('losses', 0),
                metrics.get('win_rate', 0.0),
                metrics.get('total_pnl', 0.0),
                metrics.get('avg_pnl_per_trade', 0.0),
                metrics.get('sharpe'),
                metrics.get('max_drawdown'),
                metrics.get('profit_factor'),
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error storing experiment: {e}")
            return False
        finally:
            conn.close()
    
    def query_best(
        self,
        metric: str = "win_rate",
        strategy: Optional[str] = None,
        min_trades: int = 10,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Query best experiments by metric.
        
        Args:
            metric: Column to sort by (win_rate, total_pnl, sharpe, etc.)
            strategy: Filter by strategy name
            min_trades: Minimum trades required
            top_k: Number of results
        
        Returns:
            List of experiment dicts
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = f"""
            SELECT * FROM experiments
            WHERE total_trades >= ?
            {"AND strategy = ?" if strategy else ""}
            ORDER BY {metric} DESC
            LIMIT ?
        """
        
        params = [min_trades]
        if strategy:
            params.append(strategy)
        params.append(top_k)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            record = dict(row)
            record['config'] = json.loads(record['config_json'])
            del record['config_json']
            results.append(record)
        
        return results
    
    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get a specific run by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM experiments WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            record = dict(row)
            record['config'] = json.loads(record['config_json'])
            del record['config_json']
            return record
        return None
    
    def list_strategies(self) -> List[Dict]:
        """List all strategies with their stats."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                strategy,
                COUNT(*) as run_count,
                AVG(win_rate) as avg_win_rate,
                AVG(total_pnl) as avg_pnl,
                MAX(win_rate) as best_win_rate,
                MAX(total_pnl) as best_pnl
            FROM experiments
            GROUP BY strategy
            ORDER BY run_count DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'strategy': row[0],
                'run_count': row[1],
                'avg_win_rate': row[2],
                'avg_pnl': row[3],
                'best_win_rate': row[4],
                'best_pnl': row[5],
            }
            for row in rows
        ]
    
    def count(self) -> int:
        """Get total number of experiments."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM experiments")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def delete_run(self, run_id: str) -> bool:
        """Delete a run by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM experiments WHERE run_id = ?", (run_id,))
            conn.commit()
            return cursor.rowcount > 0

    def delete_all(self):
        """Delete ALL runs from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM experiments")
            conn.commit()
        # The following lines are redundant when using 'with sqlite3.connect(...) as conn:'
        # conn.commit()
        # conn.close()
        # The variable 'deleted' is not defined in this scope.
        # A more appropriate return would be True for success, or the number of rows deleted.
        # For faithfulness to the instruction, I'll make it return True as a placeholder for success.
        return True

```

### src/strategy/__init__.py

```python
"""
Strategy Package

Provides the canonical way to run strategy scans.
"""

from src.strategy.scan import run_strategy_scan, ScanResult, RTHFilter, MinATRFilter

```

### src/strategy/scan.py

```python
"""
Strategy Scan Runner - Single Entry Point

This is THE way to run strategy scans. All required outputs are built-in.
Like a car factory - wheels are part of the assembly line, not an afterthought.

Usage:
    from src.strategy.scan import run_strategy_scan
    
    result = run_strategy_scan(
        trigger=FakeoutTrigger(level="pdh"),
        bracket=FixedBracket(stop_points=2, tp_points=4),
        start_date="2025-08-18",
        weeks=4,
        filters=[RTHFilter(), MinATRFilter()],
        run_name="fakeout_pdh_august"
    )
    
    # result.manifest_path → load in UI
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable
import uuid

from src.config import RESULTS_DIR, NY_TZ
from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes
from src.features.indicators import calculate_atr
from src.features.pipeline import compute_features, FeatureConfig
from src.labels.counterfactual import compute_smart_stop_counterfactual

from src.policy.triggers.base import Trigger, TriggerResult
from src.policy.brackets import Bracket
from src.policy.actions import Action, SkipReason

from src.datasets.decision_record import DecisionRecord
from src.datasets.trade_record import TradeRecord
from src.sim.oco_engine import OCOBracket, OCOConfig, OCOEngine, DEFAULT_OCO_ENGINE
from src.sim.sizing import calculate_contracts, calculate_pnl_dollars, calculate_reward_dollars, DEFAULT_MAX_RISK_DOLLARS

from src.viz.export import Exporter
from src.viz.config import VizConfig
from src.viz.window_utils import enforce_2hour_window


# =============================================================================
# FILTERS - Composable pre-trade filters
# =============================================================================

class PreTradeFilter:
    """Base class for pre-trade filters."""
    def __init__(self, name: str):
        self.name = name
    
    def check(self, timestamp: pd.Timestamp, atr: float, bar: pd.Series) -> tuple[bool, str]:
        """Returns (passed, reason) - reason is empty if passed."""
        raise NotImplementedError


class RTHFilter(PreTradeFilter):
    """Only trade during Regular Trading Hours (9:30-16:00 ET)."""
    def __init__(self):
        super().__init__("session_rth")
    
    def check(self, timestamp: pd.Timestamp, atr: float, bar: pd.Series) -> tuple[bool, str]:
        if timestamp.tzinfo is None:
            ts = timestamp.tz_localize(NY_TZ)
        else:
            ts = timestamp.tz_convert(NY_TZ)
        
        in_rth = 9 <= ts.hour < 16 or (ts.hour == 9 and ts.minute >= 30)
        return (in_rth, "" if in_rth else "Outside RTH")


class MinATRFilter(PreTradeFilter):
    """Require minimum ATR for volatility."""
    def __init__(self, threshold: float = 2.0):
        super().__init__("min_atr")
        self.threshold = threshold
    
    def check(self, timestamp: pd.Timestamp, atr: float, bar: pd.Series) -> tuple[bool, str]:
        passed = atr >= self.threshold
        return (passed, "" if passed else f"ATR {atr:.2f} < {self.threshold}")


# =============================================================================
# SCAN RESULT
# =============================================================================

@dataclass
class ScanResult:
    """Result of a strategy scan - all paths for UI consumption."""
    run_name: str
    manifest_path: Path
    decisions_path: Path
    trades_path: Path
    run_path: Path
    filter_failures_path: Optional[Path]
    
    total_decisions: int
    total_trades: int
    total_filtered: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "manifest_path": str(self.manifest_path),
            "total_decisions": self.total_decisions,
            "total_trades": self.total_trades,
            "total_filtered": self.total_filtered
        }


# =============================================================================
# THE MAIN FUNCTION - SINGLE ENTRY POINT
# =============================================================================

def run_strategy_scan(
    trigger: Trigger,
    bracket: Bracket,
    start_date: str,
    weeks: int,
    filters: Optional[List[PreTradeFilter]] = None,
    run_name: Optional[str] = None,
    timeframe: str = "5m",
    lookback_bars: int = 60,
    lookahead_bars: int = 30,
    cooldown_bars: int = 20,
    extra_context_fn: Optional[Callable] = None,  # For custom per-bar context
    compute_cf: bool = True,  # Compute counterfactual outcomes
) -> ScanResult:
    """
    Run a strategy scan with ALL outputs guaranteed.
    
    This is the ONLY way to run scans. It internally calls:
    - exporter.on_decision()
    - exporter.on_bracket_created()
    - exporter.on_trade_closed()
    
    You CANNOT forget any of these - they're built into this function.
    
    Args:
        trigger: Trigger instance (e.g., FakeoutTrigger, EMACrossTrigger)
        bracket: Bracket instance (e.g., FixedBracket, ATRBracket)
        start_date: Start date string "YYYY-MM-DD"
        weeks: Number of weeks to scan
        filters: List of PreTradeFilter instances (default: [RTHFilter(), MinATRFilter()])
        run_name: Custom run name (auto-generated if None)
        timeframe: Timeframe for scanning ("1m", "5m", "15m")
        lookback_bars: Bars of history for chart viz
        lookahead_bars: Bars of future for chart viz
        cooldown_bars: Bars to wait between trades
        extra_context_fn: Optional function(bar, features) -> dict for custom context
        
    Returns:
        ScanResult with all artifact paths
    """
    
    # Use default filters if none provided
    if filters is None:
        filters = [RTHFilter(), MinATRFilter()]
    
    # Setup
    start = pd.Timestamp(start_date)
    end = start + pd.Timedelta(weeks=weeks, unit='W')
    run_name = run_name or f"{trigger.trigger_id}_{start_date.replace('-', '')}"
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"STRATEGY SCAN: {trigger.trigger_id}")
    print(f"Period: {start.date()} to {end.date()} ({weeks} weeks)")
    print(f"Bracket: {bracket.bracket_type}")
    print(f"Filters: {[f.name for f in filters]}")
    print("=" * 60)
    
    # 1. Load Data (with date filtering for performance)
    print("\n[1/5] Loading data...")
    df_1m = load_continuous_contract(start_date=str(start.date()), end_date=str(end.date()))
    
    # Compute VWAP for triggers that need it
    from src.features.indicators import calculate_vwap
    if 'vwap_session' not in df_1m.columns:
        df_1m['vwap_session'] = calculate_vwap(df_1m, period='session')
    
    print(f"  Loaded {len(df_1m)} 1m bars")
    
    # 2. Resample
    print("\n[2/5] Resampling...")
    htf_data = resample_all_timeframes(df_1m)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    
    tf_map = {'1m': df_1m, '5m': df_5m, '15m': df_15m}
    df_scan = tf_map.get(timeframe, df_5m)
    
    if df_scan is not None and len(df_scan) > 14:
        df_scan = df_scan.copy()
        df_scan['atr'] = calculate_atr(df_scan, 14)
        avg_atr = df_scan['atr'].dropna().mean()
        
        # Add EMA calculations for indicator-based triggers
        from src.features.indicators import calculate_ema
        for period in [9, 20, 21, 50, 200]:
            col_name = f'ema_{timeframe}_{period}'
            df_scan[col_name] = calculate_ema(df_scan['close'], period)
    else:
        avg_atr = 5.0
    
    print(f"  Scanning on {timeframe}: {len(df_scan)} bars, avg ATR: {avg_atr:.2f}")
    
    # 3. Setup Exporter (REQUIRED - all hooks will be called)
    print("\n[3/5] Initializing exporter...")
    viz_config = VizConfig(include_windows=True, include_full_series=False)
    exporter = Exporter(
        config=viz_config,
        run_id=run_name,
        experiment_config={
            "trigger": trigger.to_dict(),
            "bracket": bracket.to_dict(),
            "start_date": str(start.date()),
            "weeks": weeks,
            "timeframe": timeframe,
            "filters": [f.name for f in filters]
        }
    )
    
    # Tracking
    filter_failures = []
    decision_idx = 0
    last_trigger_bar = -cooldown_bars - 1
    
    # 4. Run Scan Loop
    print("\n[4/5] Scanning...")
    
    for bar_idx in range(lookback_bars, len(df_scan) - lookahead_bars):
        bar = df_scan.iloc[bar_idx]
        bar_start_time = pd.Timestamp(bar['time'])
        
        # CRITICAL FIX: For market-on-close entry, timestamp should be bar CLOSE time
        # not bar START time. This aligns timestamp with entry_price.
        # For 5m bar: start=09:30, close=09:34 (we enter at 09:34)
        timeframe_minutes = {'1m': 1, '5m': 5, '15m': 15}[timeframe]
        bar_time = bar_start_time + pd.Timedelta(minutes=timeframe_minutes - 1)
        
        atr_value = bar.get('atr', avg_atr)
        if pd.isna(atr_value):
            atr_value = avg_atr
        
        # Cooldown check
        if bar_idx - last_trigger_bar < cooldown_bars:
            continue
        
        # Build features for trigger (mock bundle for now)
        class MockFeatures:
            pass
        class MockIndicators:
            pass
        
        features = MockFeatures()
        features.current_price = bar['close']
        features.bar_high = bar['high']
        features.bar_low = bar['low']
        features.bar_close = bar['close']
        features.timestamp = bar_time
        features.atr = atr_value
        
        # Add indicators for EMA-based triggers
        indicators = MockIndicators()
        for period in [9, 20, 21, 50, 200]:
            col_name = f'ema_{timeframe}_{period}'
            if col_name in bar.index:
                setattr(indicators, col_name, bar[col_name])
            else:
                setattr(indicators, col_name, 0)
        features.indicators = indicators
        
        # Add extra context if provided
        if extra_context_fn:
            extra = extra_context_fn(bar, features)
            for k, v in extra.items():
                setattr(features, k, v)
        
        # Check trigger
        result = trigger.check(features)
        
        if not result.triggered:
            continue
        
        direction = result.direction.value
        entry_price = bar['close']
        
        # === APPLY FILTERS ===
        filtered = False
        for f in filters:
            passed, reason = f.check(bar_time, atr_value, bar)
            if not passed:
                filter_failures.append({
                    "bar_idx": bar_idx,
                    "timestamp": str(bar_time),
                    "filter": f.name,
                    "reason": reason
                })
                filtered = True
                break
        
        if filtered:
            continue
        
        # === PASSED - RECORD DECISION ===
        last_trigger_bar = bar_idx
        
        # Compute bracket levels
        levels = bracket.compute(entry_price, direction, atr_value)
        
        # Compute counterfactual outcome
        cf_outcome = ""
        cf_pnl_dollars = 0.0
        cf_bars_held = 0
        
        if compute_cf:
            cf_mask = df_1m['time'] <= bar_time
            cf_entry_idx = cf_mask.sum() - 1 if cf_mask.any() else 0
            
            try:
                cf = compute_smart_stop_counterfactual(
                    df=df_1m,
                    entry_idx=cf_entry_idx,
                    direction=direction,
                    stop_price=levels.stop_price,
                    tp_multiple=levels.r_multiple,
                    atr=atr_value,
                    oco_name="strategy"
                )
                cf_outcome = cf.outcome
                cf_pnl_dollars = cf.pnl_dollars
                cf_bars_held = cf.bars_held
            except Exception as e:
                print(f"Error computing counterfactual: {e}")
                cf_outcome = "ERROR"
        else:
            cf_outcome = "SKIPPED"
            # Estimate bars held based on TP/SL ratio logic or just default
            cf_bars_held = 60  # Default estimate for window viz
        
        # Get raw OHLCV window for chart - ENFORCING 2-HOUR POLICY
        # According to ARCHITECTURE_AGREEMENT.md Section 3:
        # - 2 hours before entry
        # - 2 hours after exit (estimated from bars_held)
        raw_ohlcv, window_warning = enforce_2hour_window(
            df_1m=df_1m,
            entry_time=bar_time,
            bars_held=int(cf_bars_held)
        )
        
        # Record warning if data is missing
        if window_warning:
            # Will be added to exporter._window_warnings
            pass
        
        # Future bars from 1m data (for counterfactual viz)
        # Keep existing future bars for backward compatibility
        entry_idx_1m = (df_1m['time'] <= bar_time).sum() - 1
        future_bars_1m = 120
        future_slice = df_1m.iloc[entry_idx_1m+1:entry_idx_1m+future_bars_1m+1]
        future_bars = [
            {
                "time": row['time'].isoformat() if hasattr(row['time'], 'isoformat') else str(row['time']),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row['volume'])
            }
            for _, row in future_slice.iterrows()
        ] if len(future_slice) > 0 else []
        
        # === POSITION SIZING: Use centralized sizing function ===
        sizing_result = calculate_contracts(
            entry_price=entry_price,
            stop_price=levels.stop_price,
            max_risk_dollars=DEFAULT_MAX_RISK_DOLLARS
        )
        contracts = sizing_result.contracts
        risk_points = sizing_result.risk_points
        risk_dollars = sizing_result.risk_dollars
        
        # Calculate reward using same cost model
        reward_dollars = calculate_reward_dollars(
            entry_price=entry_price,
            tp_price=levels.tp_price,
            direction=direction,
            contracts=contracts
        )
        
        # Create decision record
        # CRITICAL: Include direction in scanner_context for UI position boxes
        context_with_direction = {**result.context, 'direction': direction}
        
        decision_id = f"{trigger.trigger_id}_{decision_idx:04d}"
        decision = DecisionRecord(
            decision_id=decision_id,
            timestamp=bar_time,
            bar_idx=bar_idx,
            scanner_id=trigger.trigger_id,
            scanner_context=context_with_direction,
            action=Action.PLACE_ORDER,
            current_price=entry_price,
            atr=atr_value,
            cf_outcome=cf_outcome,
            cf_pnl_dollars=cf_pnl_dollars
        )
        
        # === REQUIRED EXPORTER HOOK 1: on_decision ===
        class FeatBundle:
            x_price_1m = None
            x_price_5m = None
            x_price_15m = None
            x_context = None
        
        exporter.on_decision(
            decision=decision,
            features=FeatBundle(),
            raw_ohlcv=raw_ohlcv,
            future_1m=future_bars,
            indicators={
                "entry_price": entry_price,
                "stop_price": levels.stop_price,
                "tp_price": levels.tp_price,
                "atr": atr_value,
                "direction": direction,
                "contracts": contracts,
                "risk_dollars": risk_dollars,
                "reward_dollars": reward_dollars,
                "risk_points": risk_points
            }
        )
        
        # === REQUIRED EXPORTER HOOK 2: on_bracket_created ===
        oco_config = OCOConfig(
            direction=direction,
            entry_type="MARKET",
            stop_atr=levels.risk_points / atr_value if atr_value > 0 else 1.0,
            tp_multiple=levels.r_multiple
        )
        oco_bracket = OCOBracket(
            entry_price=entry_price,
            stop_price=levels.stop_price,
            tp_price=levels.tp_price,
            atr_at_creation=atr_value,
            config=oco_config
        )
        # CRITICAL: Pass contracts to exporter (not defaulted to 1)
        exporter.on_bracket_created(decision_id, oco_bracket, contracts=contracts)
        
        # === REQUIRED EXPORTER HOOK 3: on_trade_closed ===
        # NOTE: cf.bars_held is in 1-minute bars (since counterfactual runs on df_1m)
        exit_bar = bar_idx + int(cf.bars_held)
        exit_time = bar_time + pd.Timedelta(minutes=int(cf.bars_held))  # 1 min per bar since CF uses 1m data
        
        # Use centralized PnL calculation (SINGLE source of truth)
        pnl_points, pnl_dollars = calculate_pnl_dollars(
            entry_price=entry_price,
            exit_price=cf.exit_price,
            direction=direction,
            contracts=contracts,
            include_commission=True
        )
        
        trade = TradeRecord(
            trade_id=str(uuid.uuid4())[:8],
            decision_id=decision_id,
            entry_time=bar_time,
            entry_bar=bar_idx,
            entry_price=entry_price,
            direction=direction,
            exit_time=exit_time,
            exit_bar=exit_bar,
            exit_price=cf.exit_price,
            exit_reason=cf.outcome,
            outcome=cf.outcome,
            pnl_points=pnl_points,
            pnl_dollars=pnl_dollars,
            r_multiple=pnl_points / risk_points if risk_points > 0 else 0,
            bars_held=int(cf.bars_held),
            mae=0,  # Would need to compute via OCOEngine
            mfe=0,  # Would need to compute via OCOEngine
            scanner_id=trigger.trigger_id,
            entry_atr=atr_value
        )
        exporter.on_trade_closed(trade)
        
        print(f"  [{decision_idx}] {direction} @ {bar_time.strftime('%Y-%m-%d %H:%M')} | "
              f"Entry: {entry_price:.2f} SL: {levels.stop_price:.2f} TP: {levels.tp_price:.2f} | {cf.outcome}")
        decision_idx += 1
    
    # 5. Finalize - writes manifest + all artifacts
    print(f"\n[5/5] Finalizing...")
    exporter.finalize(out_dir)
    
    # Save filter failures
    filter_failures_path = None
    if filter_failures:
        import json
        filter_failures_path = out_dir / "filter_failures.jsonl"
        with open(filter_failures_path, "w") as f:
            for ff in filter_failures:
                f.write(json.dumps(ff) + "\n")
        print(f"  Saved {len(filter_failures)} filter failures")
    
    # Build result
    result = ScanResult(
        run_name=run_name,
        manifest_path=out_dir / "manifest.json",
        decisions_path=out_dir / "decisions.jsonl",
        trades_path=out_dir / "trades.jsonl",
        run_path=out_dir / "run.json",
        filter_failures_path=filter_failures_path,
        total_decisions=decision_idx,
        total_trades=len(exporter.trades),
        total_filtered=len(filter_failures)
    )
    
    print(f"\n{'=' * 60}")
    print(f"SCAN COMPLETE")
    print(f"  Decisions: {result.total_decisions}")
    print(f"  Trades: {result.total_trades}")
    print(f"  Filtered: {result.total_filtered}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}")
    
    return result

```

### src/strategy/spec.py

```python
"""
Declarative Strategy Specification

Replaces ad-hoc strategy scripts with a unified, declarative specification.
Agents create StrategySpec objects instead of writing random Python files.

This ensures:
- Consistent strategy definition
- Reproducible runs (stored in manifest)
- Validation before execution
- No "strategy snowflakes"

Usage:
    from src.strategy.spec import StrategySpec, TriggerConfig, BracketConfig
    
    spec = StrategySpec(
        strategy_id="ema_cross_2025",
        trigger=TriggerConfig(
            type="ema_cross",
            params={"fast": 9, "slow": 21}
        ),
        bracket=BracketConfig(
            type="atr",
            stop_atr=2.0,
            tp_atr=3.0
        ),
        sizing=SizingConfig(
            risk_percent=0.02,
            max_contracts=5
        )
    )
    
    # Stored in manifest
    manifest = {
        'strategy_spec': spec.to_dict(),
        ...
    }
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import hashlib
import json


class TriggerType(Enum):
    """Supported trigger types."""
    EMA_CROSS = "ema_cross"
    EMA_BOUNCE = "ema_bounce"
    RSI_THRESHOLD = "rsi_threshold"
    IFVG = "ifvg"
    ORB = "orb"
    CANDLE_PATTERN = "candle_pattern"
    TIME = "time"
    MODEL = "model"  # ML model prediction


class BracketType(Enum):
    """Supported bracket types."""
    ATR = "atr"
    PERCENT = "percent"
    FIXED = "fixed"
    RISK_REWARD = "risk_reward"


class SizingMethod(Enum):
    """Position sizing methods."""
    FIXED_CONTRACTS = "fixed_contracts"
    FIXED_RISK = "fixed_risk"  # % of account
    KELLY = "kelly"
    VOLATILITY_SCALED = "volatility_scaled"


@dataclass
class TriggerConfig:
    """
    Trigger configuration.
    Defines when to enter a trade.
    """
    type: TriggerType
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Optional ML model for prediction
    model_id: Optional[str] = None
    
    # Filters (AND logic)
    filters: List[str] = field(default_factory=list)  # ["session:rth", "time:09:30-15:30"]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value if isinstance(self.type, TriggerType) else self.type,
            'params': self.params,
            'model_id': self.model_id,
            'filters': self.filters,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TriggerConfig':
        """Create from dictionary."""
        return cls(
            type=TriggerType(data['type']) if isinstance(data['type'], str) else data['type'],
            params=data.get('params', {}),
            model_id=data.get('model_id'),
            filters=data.get('filters', []),
        )


@dataclass
class BracketConfig:
    """
    Bracket configuration.
    Defines stop loss and take profit.
    """
    type: BracketType
    
    # For ATR brackets
    stop_atr: Optional[float] = None
    tp_atr: Optional[float] = None
    
    # For percent brackets
    stop_percent: Optional[float] = None
    tp_percent: Optional[float] = None
    
    # For fixed brackets
    stop_points: Optional[float] = None
    tp_points: Optional[float] = None
    
    # For risk/reward brackets
    risk_reward_ratio: Optional[float] = None
    
    # Entry type
    entry_type: str = "LIMIT"  # or "MARKET"
    entry_offset_atr: float = 0.25
    
    # Max bars in trade
    max_bars: int = 200
    
    # Reference (for indicator-based levels)
    reference: str = "PRICE"  # or "EMA_5M", "VWAP", etc.
    reference_offset_atr: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value if isinstance(self.type, BracketType) else self.type,
            'stop_atr': self.stop_atr,
            'tp_atr': self.tp_atr,
            'stop_percent': self.stop_percent,
            'tp_percent': self.tp_percent,
            'stop_points': self.stop_points,
            'tp_points': self.tp_points,
            'risk_reward_ratio': self.risk_reward_ratio,
            'entry_type': self.entry_type,
            'entry_offset_atr': self.entry_offset_atr,
            'max_bars': self.max_bars,
            'reference': self.reference,
            'reference_offset_atr': self.reference_offset_atr,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BracketConfig':
        """Create from dictionary."""
        return cls(
            type=BracketType(data['type']) if isinstance(data['type'], str) else data['type'],
            stop_atr=data.get('stop_atr'),
            tp_atr=data.get('tp_atr'),
            stop_percent=data.get('stop_percent'),
            tp_percent=data.get('tp_percent'),
            stop_points=data.get('stop_points'),
            tp_points=data.get('tp_points'),
            risk_reward_ratio=data.get('risk_reward_ratio'),
            entry_type=data.get('entry_type', 'LIMIT'),
            entry_offset_atr=data.get('entry_offset_atr', 0.25),
            max_bars=data.get('max_bars', 200),
            reference=data.get('reference', 'PRICE'),
            reference_offset_atr=data.get('reference_offset_atr', 0.0),
        )


@dataclass
class SizingConfig:
    """
    Position sizing configuration.
    Defines how many contracts to trade.
    """
    method: SizingMethod
    
    # For fixed contracts
    contracts: Optional[int] = None
    
    # For risk-based sizing
    risk_percent: Optional[float] = None  # e.g., 0.02 = 2% risk
    max_contracts: Optional[int] = None
    
    # Account size (optional, can be passed at runtime)
    account_size: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method.value if isinstance(self.method, SizingMethod) else self.method,
            'contracts': self.contracts,
            'risk_percent': self.risk_percent,
            'max_contracts': self.max_contracts,
            'account_size': self.account_size,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SizingConfig':
        """Create from dictionary."""
        return cls(
            method=SizingMethod(data['method']) if isinstance(data['method'], str) else data['method'],
            contracts=data.get('contracts'),
            risk_percent=data.get('risk_percent'),
            max_contracts=data.get('max_contracts'),
            account_size=data.get('account_size'),
        )


@dataclass
class FilterConfig:
    """
    Entry filter configuration.
    Additional conditions that must be met.
    """
    filter_type: str  # "session", "time", "trend", "volatility"
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'filter_type': self.filter_type,
            'params': self.params,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FilterConfig':
        return cls(
            filter_type=data['filter_type'],
            params=data.get('params', {}),
        )


@dataclass
class StrategySpec:
    """
    Complete strategy specification.
    
    This is the declarative definition that replaces ad-hoc scripts.
    Stored in run manifest for reproducibility.
    """
    strategy_id: str
    trigger: TriggerConfig
    bracket: BracketConfig
    sizing: SizingConfig
    
    # Optional components
    filters: List[FilterConfig] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)  # Indicator IDs to compute
    
    # Metadata
    name: Optional[str] = None
    description: Optional[str] = None
    version: str = "1.0"
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Date range
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Walk-forward settings
    walk_forward: bool = False
    train_weeks: Optional[int] = None
    test_weeks: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'strategy_id': self.strategy_id,
            'trigger': self.trigger.to_dict(),
            'bracket': self.bracket.to_dict(),
            'sizing': self.sizing.to_dict(),
            'filters': [f.to_dict() for f in self.filters],
            'indicators': self.indicators,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'author': self.author,
            'tags': self.tags,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'walk_forward': self.walk_forward,
            'train_weeks': self.train_weeks,
            'test_weeks': self.test_weeks,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategySpec':
        """Create from dictionary."""
        return cls(
            strategy_id=data['strategy_id'],
            trigger=TriggerConfig.from_dict(data['trigger']),
            bracket=BracketConfig.from_dict(data['bracket']),
            sizing=SizingConfig.from_dict(data['sizing']),
            filters=[FilterConfig.from_dict(f) for f in data.get('filters', [])],
            indicators=data.get('indicators', []),
            name=data.get('name'),
            description=data.get('description'),
            version=data.get('version', '1.0'),
            author=data.get('author'),
            tags=data.get('tags', []),
            start_date=data.get('start_date'),
            end_date=data.get('end_date'),
            walk_forward=data.get('walk_forward', False),
            train_weeks=data.get('train_weeks'),
            test_weeks=data.get('test_weeks'),
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StrategySpec':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def fingerprint(self) -> str:
        """
        Generate deterministic fingerprint for the strategy.
        
        Used for:
        - Run identification
        - Deduplication
        - Caching
        
        Returns:
            SHA256 hash of canonical JSON representation
        """
        # Canonical JSON (sorted keys, no whitespace)
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    
    def validate(self) -> List[str]:
        """
        Validate strategy specification.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate trigger
        if not self.trigger.type:
            errors.append("Trigger type is required")
        
        # Validate bracket based on type
        if self.bracket.type == BracketType.ATR:
            if self.bracket.stop_atr is None or self.bracket.tp_atr is None:
                errors.append("ATR bracket requires stop_atr and tp_atr")
        elif self.bracket.type == BracketType.PERCENT:
            if self.bracket.stop_percent is None or self.bracket.tp_percent is None:
                errors.append("Percent bracket requires stop_percent and tp_percent")
        elif self.bracket.type == BracketType.FIXED:
            if self.bracket.stop_points is None or self.bracket.tp_points is None:
                errors.append("Fixed bracket requires stop_points and tp_points")
        
        # Validate sizing
        if self.sizing.method == SizingMethod.FIXED_CONTRACTS:
            if self.sizing.contracts is None:
                errors.append("Fixed contracts sizing requires contracts value")
        elif self.sizing.method == SizingMethod.FIXED_RISK:
            if self.sizing.risk_percent is None:
                errors.append("Fixed risk sizing requires risk_percent")
        
        # Validate dates if provided
        if self.start_date and self.end_date:
            if self.start_date >= self.end_date:
                errors.append("start_date must be before end_date")
        
        return errors


# =============================================================================
# Convenience Functions
# =============================================================================

def create_ema_cross_strategy(
    fast: int = 9,
    slow: int = 21,
    stop_atr: float = 2.0,
    tp_atr: float = 3.0,
    risk_percent: float = 0.02,
    **kwargs
) -> StrategySpec:
    """Convenience function to create EMA cross strategy."""
    return StrategySpec(
        strategy_id=f"ema_cross_{fast}_{slow}",
        trigger=TriggerConfig(
            type=TriggerType.EMA_CROSS,
            params={'fast': fast, 'slow': slow}
        ),
        bracket=BracketConfig(
            type=BracketType.ATR,
            stop_atr=stop_atr,
            tp_atr=tp_atr
        ),
        sizing=SizingConfig(
            method=SizingMethod.FIXED_RISK,
            risk_percent=risk_percent
        ),
        **kwargs
    )


def create_ifvg_strategy(
    stop_atr: float = 2.0,
    tp_atr: float = 3.0,
    risk_percent: float = 0.02,
    **kwargs
) -> StrategySpec:
    """Convenience function to create IFVG strategy."""
    return StrategySpec(
        strategy_id="ifvg",
        trigger=TriggerConfig(
            type=TriggerType.IFVG,
            params={}
        ),
        bracket=BracketConfig(
            type=BracketType.ATR,
            stop_atr=stop_atr,
            tp_atr=tp_atr
        ),
        sizing=SizingConfig(
            method=SizingMethod.FIXED_RISK,
            risk_percent=risk_percent
        ),
        **kwargs
    )


def create_model_strategy(
    model_id: str,
    stop_atr: float = 2.0,
    tp_atr: float = 3.0,
    risk_percent: float = 0.02,
    **kwargs
) -> StrategySpec:
    """Convenience function to create ML model strategy."""
    return StrategySpec(
        strategy_id=f"model_{model_id}",
        trigger=TriggerConfig(
            type=TriggerType.MODEL,
            model_id=model_id
        ),
        bracket=BracketConfig(
            type=BracketType.ATR,
            stop_atr=stop_atr,
            tp_atr=tp_atr
        ),
        sizing=SizingConfig(
            method=SizingMethod.FIXED_RISK,
            risk_percent=risk_percent
        ),
        **kwargs
    )

```

### src/tools/__init__.py

```python

```

### src/tools/agent_tools.py

```python
"""
Agent Tools for MLang2

Registered tools that agents can use for strategy creation, navigation, and analysis.
These replace the hardcoded AGENT_TOOLS and LAB_TOOLS definitions.
"""

from typing import Dict, Any, List
from src.core.tool_registry import ToolRegistry, ToolCategory


# =============================================================================
# Strategy Execution Tools
# =============================================================================

@ToolRegistry.register(
    tool_id="run_strategy",
    category=ToolCategory.STRATEGY,
    name="Run Strategy Scan",
    description="Run a modular strategy scan on historical data. Creates a new run that appears in the run list for visualization.",
    input_schema={
        "type": "object",
        "properties": {
            "strategy": {
                "type": "string",
                "enum": ["modular", "opening_range"],
                "description": "Strategy type. Use 'modular' for custom trigger/bracket configs."
            },
            "start_date": {
                "type": "string",
                "description": "Start date in YYYY-MM-DD format. Data available: 2025-03-18 to 2025-09-17."
            },
            "weeks": {
                "type": "integer",
                "description": "Number of weeks to scan.",
                "minimum": 1,
                "maximum": 26
            },
            "run_name": {
                "type": "string",
                "description": "Optional custom name for the run."
            },
            "trigger_type": {
                "type": "string",
                "enum": ["ema_cross", "ema_bounce", "rsi_threshold", "ifvg", "orb", "candle_pattern", "time"],
                "description": "Type of entry trigger."
            },
            "trigger_params": {
                "type": "object",
                "description": "Parameters for the trigger (e.g., {fast: 9, slow: 21} for ema_cross)."
            },
            "bracket_type": {
                "type": "string",
                "enum": ["atr", "percent", "fixed"],
                "description": "Type of stop/take-profit bracket."
            },
            "stop_atr": {
                "type": "number",
                "description": "Stop loss in ATR multiples (for atr bracket).",
                "default": 2.0
            },
            "tp_atr": {
                "type": "number",
                "description": "Take profit in ATR multiples (for atr bracket).",
                "default": 3.0
            }
        },
        "required": ["strategy", "start_date", "weeks", "trigger_type", "bracket_type"]
    },
    produces_artifacts=True,
    artifact_spec={
        "outputs": ["manifest.json", "decisions.jsonl", "trades.jsonl"],
        "format": "run_artifact_v1"
    }
)
class RunStrategyTool:
    """Tool for running strategy scans."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """Execute strategy scan - handled by server/UI."""
        return {
            "status": "queued",
            "message": "Strategy run initiated",
            "inputs": inputs
        }


@ToolRegistry.register(
    tool_id="run_modular_strategy",
    category=ToolCategory.STRATEGY,
    name="Run Modular Strategy",
    description="Run a modular strategy scan on historical data with custom trigger and bracket configuration.",
    input_schema={
        "type": "object",
        "properties": {
            "trigger_type": {
                "type": "string",
                "enum": ["ema_cross", "ema_bounce", "rsi_threshold", "ifvg", "orb", "candle_pattern", "time"],
                "description": "Type of entry trigger"
            },
            "trigger_params": {
                "type": "object",
                "description": "Parameters for the trigger (e.g., {fast: 9, slow: 21} for ema_cross)"
            },
            "bracket_type": {
                "type": "string",
                "enum": ["atr", "percent", "fixed"],
                "description": "Type of stop/take-profit bracket"
            },
            "stop_atr": {
                "type": "number",
                "description": "Stop loss in ATR multiples",
                "default": 2.0
            },
            "tp_atr": {
                "type": "number",
                "description": "Take profit in ATR multiples",
                "default": 3.0
            },
            "start_date": {
                "type": "string",
                "description": "Start date YYYY-MM-DD (data: 2025-03-18 to 2025-09-17)"
            },
            "weeks": {
                "type": "integer",
                "description": "Number of weeks to scan",
                "minimum": 1,
                "maximum": 26
            },
            "run_name": {
                "type": "string",
                "description": "Optional custom name for the run"
            },
            "silent": {
                "type": "boolean",
                "description": "If true, run silently without forcing visualization UI (default: false)",
                "default": False
            }
        },
        "required": ["trigger_type", "bracket_type", "start_date", "weeks"]
    },
    produces_artifacts=True,
    artifact_spec={
        "outputs": ["manifest.json", "decisions.jsonl", "trades.jsonl"],
        "format": "run_artifact_v1"
    }
)
class RunModularStrategyTool:
    """Tool for running modular strategy scans."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """Execute modular strategy scan - handled by server/UI."""
        return {
            "status": "queued",
            "message": "Modular strategy run initiated",
            "inputs": inputs
        }


# =============================================================================
# Navigation Tools
# =============================================================================

@ToolRegistry.register(
    tool_id="set_index",
    category=ToolCategory.UTILITY,
    name="Set Index",
    description="Navigate to a specific decision or trade by index number.",
    input_schema={
        "type": "object",
        "properties": {
            "index": {
                "type": "integer",
                "description": "The index to navigate to."
            }
        },
        "required": ["index"]
    }
)
class SetIndexTool:
    """Tool for navigating to a specific index."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """Navigate to index - handled by UI."""
        return {
            "status": "success",
            "index": inputs.get("index", 0)
        }


@ToolRegistry.register(
    tool_id="set_mode",
    category=ToolCategory.UTILITY,
    name="Set View Mode",
    description="Switch between viewing decisions or trades.",
    input_schema={
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["DECISION", "TRADE"],
                "description": "The view mode to switch to."
            }
        },
        "required": ["mode"]
    }
)
class SetModeTool:
    """Tool for switching view modes."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """Switch view mode - handled by UI."""
        return {
            "status": "success",
            "mode": inputs.get("mode", "DECISION")
        }


# =============================================================================
# Data Access Tools
# =============================================================================

@ToolRegistry.register(
    tool_id="load_run",
    category=ToolCategory.UTILITY,
    name="Load Run",
    description="Load an existing run for visualization.",
    input_schema={
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run ID to load."
            }
        },
        "required": ["run_id"]
    }
)
class LoadRunTool:
    """Tool for loading existing runs."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """Load run - handled by UI."""
        return {
            "status": "success",
            "run_id": inputs.get("run_id")
        }


@ToolRegistry.register(
    tool_id="list_runs",
    category=ToolCategory.UTILITY,
    name="List Runs",
    description="List all available runs that can be loaded.",
    input_schema={
        "type": "object",
        "properties": {}
    }
)
class ListRunsTool:
    """Tool for listing available runs."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """List runs - handled by server."""
        return {
            "status": "success",
            "runs": []  # Populated by server
        }


# =============================================================================
# Lab-Specific Tools
# =============================================================================

@ToolRegistry.register(
    tool_id="start_live_mode",
    category=ToolCategory.UTILITY,
    name="Start Live Mode",
    description="Start live trading simulation with real-time YFinance data.",
    input_schema={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "enum": ["MES=F", "ES=F", "NQ=F", "SPY"],
                "description": "Ticker symbol"
            },
            "strategy": {
                "type": "string",
                "enum": ["ema_cross", "ifvg", "orb"],
                "description": "Strategy to use"
            }
        },
        "required": ["ticker", "strategy"]
    }
)
class StartLiveModeTool:
    """Tool for starting live trading simulation."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """Start live mode - handled by server."""
        return {
            "status": "started",
            "ticker": inputs.get("ticker"),
            "strategy": inputs.get("strategy")
        }


@ToolRegistry.register(
    tool_id="query_experiments",
    category=ToolCategory.UTILITY,
    name="Query Experiments",
    description="Query the experiment database for past strategy results.",
    input_schema={
        "type": "object",
        "properties": {
            "sort_by": {
                "type": "string",
                "enum": ["win_rate", "total_pnl", "total_trades"],
                "description": "Metric to sort by"
            },
            "min_trades": {
                "type": "integer",
                "description": "Minimum number of trades required to include experiment in results",
                "default": 1
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 5
            }
        },
        "required": ["sort_by"]
    }
)
class QueryExperimentsTool:
    """Tool for querying experiment history."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """Query experiments - handled by server."""
        return {
            "status": "success",
            "sort_by": inputs.get("sort_by"),
            "results": []  # Populated by server
        }
@ToolRegistry.register(
    tool_id="compare_runs",
    category=ToolCategory.UTILITY,
    name="Compare Runs",
    description="Compare results of two or more runs side-by-side",
    input_schema={
        "type": "object",
        "properties": {
            "run_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of run IDs to compare"
            }
        },
        "required": ["run_ids"]
    }
)
class CompareRunsTool:
    def execute(self, **inputs) -> Dict[str, Any]:
        """Compare runs - handled by server."""
        return {"status": "success", "run_ids": inputs.get("run_ids")}


@ToolRegistry.register(
    tool_id="get_run_config",
    category=ToolCategory.UTILITY,
    name="Get Run Config",
    description="Get the configuration/recipe used for a specific run",
    input_schema={
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run ID to inspect"
            }
        },
        "required": ["run_id"]
    }
)
class GetRunConfigTool:
    def execute(self, **inputs) -> Dict[str, Any]:
        """Get run config - handled by server."""
        return {"status": "success", "run_id": inputs.get("run_id")}


@ToolRegistry.register(
    tool_id="create_variation",
    category=ToolCategory.STRATEGY,
    name="Create Variation",
    description="Create a new run by modifying an existing run's configuration",
    input_schema={
        "type": "object",
        "properties": {
            "base_run_id": {
                "type": "string",
                "description": "The run ID to use as a base"
            },
            "modifications": {
                "type": "object",
                "description": "Changes to apply to the base config (e.g., {'tp_atr': 4.0})"
            },
            "run_name": {
                "type": "string",
                "description": "Optional custom name for the new run"
            }
        },
        "required": ["base_run_id", "modifications"]
    }
)
class CreateVariationTool:
    def execute(self, **inputs) -> Dict[str, Any]:
        """Create variation - handled by server."""
        return {"status": "queued", "base_run_id": inputs.get("base_run_id")}


@ToolRegistry.register(
    tool_id="save_to_tradeviz",
    category=ToolCategory.UTILITY,
    name="Save to Trade Viz",
    description="Move a successful experiment from the Lab to the main Trade Viz view",
    input_schema={
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run ID to save"
            }
        },
        "required": ["run_id"]
    }
)
class SaveToTradeVizTool:
    def execute(self, **inputs) -> Dict[str, Any]:
        """Save to Trade Viz - handled by server."""
        return {"status": "success", "run_id": inputs.get("run_id")}


@ToolRegistry.register(
    tool_id="delete_run",
    category=ToolCategory.UTILITY,
    name="Delete Run",
    description="Delete a run and its associated data files",
    input_schema={
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run ID to delete"
            }
        },
        "required": ["run_id"]
    }
)
class DeleteRunTool:
    def execute(self, **inputs) -> Dict[str, Any]:
        """Delete run - handled by server."""
        return {"status": "success", "run_id": inputs.get("run_id")}
@ToolRegistry.register(
    tool_id="train_model",
    category=ToolCategory.STRATEGY,
    name="Train ML Model",
    description="Train a machine learning model (XGBoost, CNN, etc.) on historical data",
    input_schema={
        "type": "object",
        "properties": {
            "model_type": {
                "type": "string",
                "enum": ["xgboost", "cnn", "lstm"],
                "description": "Type of model to train"
            },
            "target": {
                "type": "string",
                "description": "Training target (e.g., 'next_bar_direction', 'atr_cross')"
            },
            "start_date": {
                "type": "string",
                "description": "Training start date (YYYY-MM-DD)"
            },
            "end_date": {
                "type": "string",
                "description": "Training end date (YYYY-MM-DD)"
            },
            "params": {
                "type": "object",
                "description": "Training hyperparameters"
            }
        },
        "required": ["model_type", "target", "start_date", "end_date"]
    }
)
class TrainModelTool:
    def execute(self, **inputs) -> Dict[str, Any]:
        """Train model - handled by server."""
        return {"status": "queued", "model_type": inputs.get("model_type")}

```

### src/tools/analysis_tools.py

```python
"""
Analysis Tools for MLang2
Tools for deep analysis of runs, trades, and price context.
Designed to help agents "Diagnose Failures" and "Understand Context".
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from src.core.tool_registry import ToolRegistry, ToolCategory
from src.config import RESULTS_DIR, NY_TZ, CONTINUOUS_CONTRACT_PATH
from src.data import loader


@ToolRegistry.register(
    tool_id="diagnose_run",
    category=ToolCategory.UTILITY,
    name="Diagnose Run Performance",
    description="Analyze a completed strategy run to find patterns in wins vs losses (by hour, day, duration, etc.)",
    input_schema={
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run ID to analyze"
            }
        },
        "required": ["run_id"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "summary": {"type": "object"},
            "hourly_performance": {"type": "array"},
            "daily_performance": {"type": "array"},
            "duration_stats": {"type": "object"},
            "worst_drawdown": {"type": "number"},
            "consecutive_losses": {"type": "integer"}
        }
    }
)
class DiagnoseRunTool:
    def execute(self, run_id: str, **kwargs) -> Dict[str, Any]:
        """Analyze run performance patterns."""
        # Locate run directory
        run_dir = None
        # Check viz directory first
        viz_path = RESULTS_DIR / "viz" / run_id
        if viz_path.exists():
            run_dir = viz_path
        else:
            # Check direct in results
            direct_path = RESULTS_DIR / run_id
            if direct_path.exists():
                run_dir = direct_path

        if not run_dir:
            return {"error": f"Run {run_id} not found"}

        # Load trades
        trades = []
        trades_file = run_dir / "trades.jsonl"

        if trades_file.exists():
            with open(trades_file) as f:
                for line in f:
                    if line.strip():
                        trades.append(json.loads(line))
        else:
            # Try records/decisions fallback
            records_file = run_dir / "records.jsonl"
            if not records_file.exists():
                records_file = run_dir / "decisions.jsonl"

            if records_file.exists():
                with open(records_file) as f:
                    for i, line in enumerate(f):
                        if line.strip():
                            r = json.loads(line)
                            # Only if triggered trade
                            if 'best_oco' in r or r.get('scanner_context', {}).get('triggered', False):
                                trades.append({
                                    'trade_id': r.get('decision_id', f"tr_{i}"),
                                    'entry_time': r.get('timestamp', r.get('time')),
                                    'pnl_dollars': r.get('best_pnl', 0.0),
                                    'direction': r.get('scanner_context', {}).get('direction', 'LONG'),
                                    # Try to infer duration if available
                                    'bars_held': r.get('bars_held', 0)
                                })

        if not trades:
            return {"error": "No trades found in run"}

        # Convert to DataFrame
        df = pd.DataFrame(trades)

        # Ensure timestamps
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            if df['entry_time'].dt.tz is None:
                df['entry_time'] = df['entry_time'].dt.tz_localize(NY_TZ)
            else:
                df['entry_time'] = df['entry_time'].dt.tz_convert(NY_TZ)

        df['pnl'] = pd.to_numeric(df['pnl_dollars'], errors='coerce').fillna(0.0)
        df['win'] = df['pnl'] > 0

        # --- Analysis ---

        # 1. Summary
        summary = {
            "total_trades": len(df),
            "win_rate": float(df['win'].mean()),
            "total_pnl": float(df['pnl'].sum()),
            "avg_win": float(df[df['win']]['pnl'].mean()) if not df[df['win']].empty else 0.0,
            "avg_loss": float(df[~df['win']]['pnl'].mean()) if not df[~df['win']].empty else 0.0
        }

        # 2. Hourly Performance
        if 'entry_time' in df.columns:
            df['hour'] = df['entry_time'].dt.hour
            hourly = df.groupby('hour').agg({
                'pnl': ['count', 'sum', 'mean'],
                'win': 'mean'
            })
            hourly.columns = ['trades', 'total_pnl', 'avg_pnl', 'win_rate']
            hourly = hourly.reset_index()
            hourly_perf = hourly.to_dict('records')
        else:
            hourly_perf = []

        # 3. Daily Performance (Day of Week)
        if 'entry_time' in df.columns:
            df['day_name'] = df['entry_time'].dt.day_name()
            df['day_idx'] = df['entry_time'].dt.dayofweek
            daily = df.groupby(['day_idx', 'day_name']).agg({
                'pnl': ['count', 'sum', 'mean'],
                'win': 'mean'
            })
            daily.columns = ['trades', 'total_pnl', 'avg_pnl', 'win_rate']
            daily = daily.reset_index().sort_values('day_idx')
            daily_perf = daily[['day_name', 'trades', 'total_pnl', 'avg_pnl', 'win_rate']].to_dict('records')
        else:
            daily_perf = []

        # 4. Duration Stats (if available)
        duration_stats = {}
        if 'bars_held' in df.columns:
            duration_stats = {
                "avg_bars_win": float(df[df['win']]['bars_held'].mean()) if not df[df['win']].empty else 0.0,
                "avg_bars_loss": float(df[~df['win']]['bars_held'].mean()) if not df[~df['win']].empty else 0.0
            }

        # 5. Streaks
        # Identify streaks of wins/losses
        streaks = df['win'].ne(df['win'].shift()).cumsum()
        df['streak_id'] = streaks
        streak_counts = df.groupby('streak_id').size()
        streak_types = df.groupby('streak_id')['win'].first()

        max_win_streak = streak_counts[streak_types].max() if any(streak_types) else 0
        max_loss_streak = streak_counts[~streak_types].max() if any(~streak_types) else 0

        return {
            "summary": summary,
            "hourly_performance": hourly_perf,
            "daily_performance": daily_perf,
            "duration_stats": duration_stats,
            "consecutive_losses": int(max_loss_streak),
            "consecutive_wins": int(max_win_streak)
        }


@ToolRegistry.register(
    tool_id="get_price_context",
    category=ToolCategory.UTILITY,
    name="Get Price Context",
    description="Get OHLCV bars around a specific timestamp to understand what happened before/after a trade.",
    input_schema={
        "type": "object",
        "properties": {
            "timestamp": {
                "type": "string",
                "description": "Center timestamp (ISO format)"
            },
            "range_minutes": {
                "type": "integer",
                "description": "Total range in minutes (e.g. 60 = 30 min before, 30 min after)",
                "default": 60
            },
            "symbol": {
                "type": "string",
                "default": "continuous"
            }
        },
        "required": ["timestamp"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "bars": {"type": "array"}
        }
    }
)
class GetPriceContextTool:
    def execute(self, timestamp: str, range_minutes: int = 60, symbol: str = "continuous", **kwargs) -> Dict[str, Any]:
        """Fetch bars surrounding a timestamp."""
        try:
            center_dt = pd.to_datetime(timestamp)
            if center_dt.tzinfo is None:
                center_dt = center_dt.tz_localize(NY_TZ)
        except Exception as e:
            return {"error": f"Invalid timestamp: {str(e)}"}

        start_dt = center_dt - timedelta(minutes=range_minutes // 2)
        end_dt = center_dt + timedelta(minutes=range_minutes // 2)

        # Load data efficiently
        # We use the loader directly to get the DataFrame
        df = loader.load_continuous_contract(
            start_date=start_dt.strftime('%Y-%m-%d'),
            end_date=(end_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        )

        # Filter exact time range
        mask = (df['time'] >= start_dt) & (df['time'] <= end_dt)
        df_slice = df.loc[mask].copy()

        # Convert to dict
        records = []
        for _, row in df_slice.iterrows():
            records.append({
                "time": row['time'].isoformat(),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume'])
            })

        return {
            "center_time": center_dt.isoformat(),
            "count": len(records),
            "bars": records
        }

```

### src/tools/contract_linter.py

```python
"""
Contract Linter - Validates Run Artifacts

This linter enforces the "Single Run Artifact Contract" defined in ARCHITECTURE_AGREEMENT.md.

Checks:
1. Required files exist (manifest.json, decisions.jsonl, trades.jsonl if applicable)
2. Manifest schema is valid
3. Timestamps are ISO 8601 with timezone
4. oco_results is flat (not nested)
5. Required fields are present
6. Contracts field exists for decisions
7. Strategy spec is present (if using StrategySpec)

Usage:
    python -m src.tools.contract_linter <run_dir>
    
Integration:
    - Called by build/CI to validate artifacts before merge
    - Can be used in exporter to validate during write
    - Part of golden run validation
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime


class ContractLinterError:
    """Represents a linting error."""
    
    def __init__(self, severity: str, category: str, message: str, file: str = "", line: int = 0):
        self.severity = severity  # ERROR, WARNING, INFO
        self.category = category  # MISSING_FILE, SCHEMA, TIMESTAMP, etc.
        self.message = message
        self.file = file
        self.line = line
    
    def __str__(self):
        location = f"{self.file}:{self.line}" if self.file else ""
        return f"[{self.severity}] {self.category}: {self.message} ({location})"


class ContractLinter:
    """
    Linter for run artifact contracts.
    
    Validates that run outputs conform to the canonical schema.
    """
    
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.errors: List[ContractLinterError] = []
    
    def lint(self) -> Tuple[bool, List[ContractLinterError]]:
        """
        Run all linting checks.
        
        Returns:
            (is_valid, errors)
        """
        self.errors = []
        
        # Check directory exists
        if not self.run_dir.exists():
            self.errors.append(ContractLinterError(
                "ERROR", "MISSING_DIR",
                f"Run directory does not exist: {self.run_dir}"
            ))
            return False, self.errors
        
        # Run checks
        self._check_required_files()
        self._check_manifest()
        self._check_decisions()
        self._check_trades()
        self._check_windows()
        
        # Determine if valid (no ERRORs)
        has_errors = any(e.severity == "ERROR" for e in self.errors)
        return not has_errors, self.errors
    
    def _check_required_files(self):
        """Check that required files exist."""
        required = ["manifest.json"]
        for filename in required:
            if not (self.run_dir / filename).exists():
                self.errors.append(ContractLinterError(
                    "ERROR", "MISSING_FILE",
                    f"Required file missing: {filename}"
                ))
        
        # decisions.jsonl should exist if there are any decisions
        decisions_file = self.run_dir / "decisions.jsonl"
        if not decisions_file.exists():
            self.errors.append(ContractLinterError(
                "WARNING", "MISSING_FILE",
                "decisions.jsonl not found - run may have no decisions"
            ))
    
    def _check_manifest(self):
        """Validate manifest.json structure."""
        manifest_path = self.run_dir / "manifest.json"
        if not manifest_path.exists():
            return  # Already reported in required files
        
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(ContractLinterError(
                "ERROR", "SCHEMA",
                f"manifest.json is not valid JSON: {e}",
                file="manifest.json"
            ))
            return
        
        # Check required fields
        required_fields = ["run_id", "created_at", "strategy"]
        for field in required_fields:
            if field not in manifest:
                self.errors.append(ContractLinterError(
                    "ERROR", "SCHEMA",
                    f"manifest.json missing required field: {field}",
                    file="manifest.json"
                ))
        
        # Check timestamp format
        if "created_at" in manifest:
            self._check_timestamp(manifest["created_at"], "manifest.json", "created_at")
        
        # Check if strategy_spec exists (recommended)
        if "strategy_spec" not in manifest:
            self.errors.append(ContractLinterError(
                "WARNING", "SCHEMA",
                "manifest.json missing strategy_spec field - recommended for reproducibility",
                file="manifest.json"
            ))
    
    def _check_decisions(self):
        """Validate decisions.jsonl structure."""
        decisions_path = self.run_dir / "decisions.jsonl"
        if not decisions_path.exists():
            return  # Already reported
        
        try:
            with open(decisions_path) as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        decision = json.loads(line)
                    except json.JSONDecodeError as e:
                        self.errors.append(ContractLinterError(
                            "ERROR", "SCHEMA",
                            f"Invalid JSON: {e}",
                            file="decisions.jsonl",
                            line=line_num
                        ))
                        continue
                    
                    # Check required fields
                    required = ["decision_id", "timestamp", "action"]
                    for field in required:
                        if field not in decision:
                            self.errors.append(ContractLinterError(
                                "ERROR", "SCHEMA",
                                f"Decision missing required field: {field}",
                                file="decisions.jsonl",
                                line=line_num
                            ))
                    
                    # Check timestamp
                    if "timestamp" in decision:
                        self._check_timestamp(
                            decision["timestamp"],
                            "decisions.jsonl",
                            f"decision.timestamp (line {line_num})"
                        )
                    
                    # Check oco_results is flat if present
                    if "oco_results" in decision:
                        self._check_oco_results_flat(
                            decision["oco_results"],
                            "decisions.jsonl",
                            line_num
                        )
                    
                    # Check contracts field if action is PLACE_ORDER
                    if decision.get("action") == "PLACE_ORDER":
                        if "contracts" not in decision:
                            self.errors.append(ContractLinterError(
                                "WARNING", "SCHEMA",
                                "PLACE_ORDER decision missing 'contracts' field",
                                file="decisions.jsonl",
                                line=line_num
                            ))
        
        except Exception as e:
            self.errors.append(ContractLinterError(
                "ERROR", "FILE_READ",
                f"Failed to read decisions.jsonl: {e}",
                file="decisions.jsonl"
            ))
    
    def _check_trades(self):
        """Validate trades.jsonl if it exists."""
        trades_path = self.run_dir / "trades.jsonl"
        if not trades_path.exists():
            return  # Optional file
        
        try:
            with open(trades_path) as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        trade = json.loads(line)
                    except json.JSONDecodeError as e:
                        self.errors.append(ContractLinterError(
                            "ERROR", "SCHEMA",
                            f"Invalid JSON: {e}",
                            file="trades.jsonl",
                            line=line_num
                        ))
                        continue
                    
                    # Check required fields
                    required = ["trade_id", "entry_time", "exit_time", "outcome"]
                    for field in required:
                        if field not in trade:
                            self.errors.append(ContractLinterError(
                                "ERROR", "SCHEMA",
                                f"Trade missing required field: {field}",
                                file="trades.jsonl",
                                line=line_num
                            ))
                    
                    # Check timestamps
                    for ts_field in ["entry_time", "exit_time"]:
                        if ts_field in trade:
                            self._check_timestamp(
                                trade[ts_field],
                                "trades.jsonl",
                                f"trade.{ts_field} (line {line_num})"
                            )
        
        except Exception as e:
            self.errors.append(ContractLinterError(
                "ERROR", "FILE_READ",
                f"Failed to read trades.jsonl: {e}",
                file="trades.jsonl"
            ))
    
    def _check_windows(self):
        """Validate windows.json if it exists."""
        windows_path = self.run_dir / "windows.json"
        if not windows_path.exists():
            return  # Optional file
        
        try:
            with open(windows_path) as f:
                windows = json.load(f)
            
            # Check structure
            if not isinstance(windows, dict):
                self.errors.append(ContractLinterError(
                    "ERROR", "SCHEMA",
                    "windows.json must be a dictionary",
                    file="windows.json"
                ))
                return
            
            if "windows" not in windows:
                self.errors.append(ContractLinterError(
                    "ERROR", "SCHEMA",
                    "windows.json missing 'windows' field",
                    file="windows.json"
                ))
        
        except json.JSONDecodeError as e:
            self.errors.append(ContractLinterError(
                "ERROR", "SCHEMA",
                f"windows.json is not valid JSON: {e}",
                file="windows.json"
            ))
        except Exception as e:
            self.errors.append(ContractLinterError(
                "ERROR", "FILE_READ",
                f"Failed to read windows.json: {e}",
                file="windows.json"
            ))
    
    def _check_timestamp(self, ts: str, file: str, field: str):
        """Check timestamp is ISO 8601 with timezone."""
        if not isinstance(ts, str):
            self.errors.append(ContractLinterError(
                "ERROR", "TIMESTAMP",
                f"{field} must be a string, got {type(ts)}",
                file=file
            ))
            return
        
        # Check for timezone indicator
        if not (ts.endswith('Z') or '+' in ts or ts.endswith('00:00')):
            self.errors.append(ContractLinterError(
                "ERROR", "TIMESTAMP",
                f"{field} missing timezone (must end with Z or have +offset): {ts}",
                file=file
            ))
        
        # Try to parse as ISO 8601
        try:
            datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except ValueError as e:
            self.errors.append(ContractLinterError(
                "ERROR", "TIMESTAMP",
                f"{field} is not valid ISO 8601: {ts} ({e})",
                file=file
            ))
    
    def _check_oco_results_flat(self, oco_results: Any, file: str, line: int):
        """
        Check that oco_results is a flat dict, not nested.
        
        According to ARCHITECTURE_AGREEMENT.md:
        oco_results must be a flat dictionary with keys like:
        - filled, outcome, exit_price, bars_held, etc.
        
        NOT nested like: {"LONG_1R": {...}, "SHORT_2R": {...}}
        """
        if not isinstance(oco_results, dict):
            self.errors.append(ContractLinterError(
                "ERROR", "OCO_RESULTS",
                f"oco_results must be a dict, got {type(oco_results)}",
                file=file,
                line=line
            ))
            return
        
        # Check for nested structure (values are dicts)
        for key, value in oco_results.items():
            if isinstance(value, dict):
                self.errors.append(ContractLinterError(
                    "ERROR", "OCO_RESULTS",
                    f"oco_results must be flat, found nested dict at key '{key}'",
                    file=file,
                    line=line
                ))
                return
        
        # Check for required fields
        required = ["filled", "outcome"]
        for field in required:
            if field not in oco_results:
                self.errors.append(ContractLinterError(
                    "WARNING", "OCO_RESULTS",
                    f"oco_results missing recommended field: {field}",
                    file=file,
                    line=line
                ))


def lint_run_artifact(run_dir: Path, verbose: bool = False) -> bool:
    """
    Lint a run artifact directory.
    
    Args:
        run_dir: Path to run directory
        verbose: Print all errors/warnings
    
    Returns:
        True if valid (no errors)
    """
    linter = ContractLinter(run_dir)
    is_valid, errors = linter.lint()
    
    if verbose or not is_valid:
        print(f"\n{'='*60}")
        print(f"Contract Linter: {run_dir}")
        print(f"{'='*60}\n")
        
        if errors:
            for error in errors:
                print(error)
            
            error_count = sum(1 for e in errors if e.severity == "ERROR")
            warning_count = sum(1 for e in errors if e.severity == "WARNING")
            
            print(f"\n{'='*60}")
            print(f"Total: {error_count} errors, {warning_count} warnings")
            print(f"{'='*60}\n")
        else:
            print("✓ No issues found\n")
    
    return is_valid


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.tools.contract_linter <run_dir>")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    is_valid = lint_run_artifact(run_dir, verbose=True)
    
    sys.exit(0 if is_valid else 1)

```

### src/tools/exploration_tools.py

```python
"""
Exploration Tools for MLang2

Safe exploration tools that write ONLY to results/exploration/.
These tools are non-promotable by default and cannot break TradeViz.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List

from src.core.tool_registry import ToolRegistry, ToolCategory
from src.config import RESULTS_DIR


EXPLORATION_DIR = RESULTS_DIR / "exploration"


# =============================================================================
# Core Exploration Tools
# =============================================================================

@ToolRegistry.register(
    tool_id="explore_strategy",
    category=ToolCategory.UTILITY,
    name="Explore Strategy (Safe)",
    description="Run parameter sweeps WITHOUT generating TradeViz artifacts. Output goes to results/exploration/ only. NOTE: Counterfactuals are disabled for speed - this is for parameter optimization only, not detailed trade analysis.",
    input_schema={
        "type": "object",
        "properties": {
            "recipe": {
                "type": "object",
                "description": "Base recipe configuration (entry_trigger, oco, etc.)"
            },
            "param_grid": {
                "type": "object",
                "description": "Parameter grid for sweep. Keys are dot-paths, values are lists. E.g. {'oco.take_profit.multiple': [2, 3, 4]}"
            },
            "exploration_name": {
                "type": "string",
                "description": "Name for this exploration run"
            },
            "start_date": {
                "type": "string",
                "description": "Start date YYYY-MM-DD (default: 2025-04-01)"
            },
            "end_date": {
                "type": "string",
                "description": "End date YYYY-MM-DD (default: 2025-04-30)"
            }
        },
        "required": ["recipe", "param_grid", "exploration_name"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "exploration_id": {"type": "string"},
            "best_config": {"type": "object"},
            "total_configs": {"type": "integer"},
            "output_path": {"type": "string"}
        }
    }
)
class ExploreStrategyTool:
    """Safe sweep tool - writes to exploration dir only."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        recipe = inputs.get("recipe", {})
        param_grid = inputs.get("param_grid", {})
        exploration_name = inputs.get("exploration_name", "unnamed")
        start_date = inputs.get("start_date", "2025-04-01")
        end_date = inputs.get("end_date", "2025-04-30")
        
        # Write recipe to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(recipe, f, indent=2)
            recipe_path = f.name
        
        try:
            cmd = [
                sys.executable,
                "-m", "scripts.explore_strategy",
                "--recipe", recipe_path,
                "--grid", json.dumps(param_grid),
                "--out", exploration_name,
                "--start-date", start_date,
                "--end-date", end_date
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 min timeout for sweeps
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr
                }
            
            # Load results
            output_path = EXPLORATION_DIR / f"{exploration_name}.json"
            if output_path.exists():
                with open(output_path) as f:
                    summary = json.load(f)
                
                return {
                    "success": True,
                    "exploration_id": exploration_name,
                    "best_config": summary.get("best_config"),
                    "total_configs": summary.get("total_configs", 0),
                    "output_path": str(output_path)
                }
            else:
                return {
                    "success": False,
                    "error": "Output file not created"
                }
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Exploration timed out after 10 minutes"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            Path(recipe_path).unlink(missing_ok=True)


@ToolRegistry.register(
    tool_id="compare_explorations",
    category=ToolCategory.UTILITY,
    name="Compare Explorations",
    description="Compare multiple exploration runs side-by-side. Returns dominance table and trade-offs.",
    input_schema={
        "type": "object",
        "properties": {
            "exploration_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of exploration IDs to compare"
            }
        },
        "required": ["exploration_ids"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "comparison": {"type": "array"},
            "best_overall": {"type": "object"}
        }
    }
)
class CompareExplorationsTool:
    """Compare exploration results."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        exploration_ids = inputs.get("exploration_ids", [])
        
        results = []
        for exp_id in exploration_ids:
            path = EXPLORATION_DIR / f"{exp_id}.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                    best = data.get("best_config", {})
                    results.append({
                        "exploration_id": exp_id,
                        "win_rate": best.get("win_rate", 0),
                        "total_pnl": best.get("total_pnl", 0),
                        "total_trades": best.get("total_trades", 0),
                        "config_summary": str(best.get("recipe", {}))[:200]
                    })
            else:
                results.append({
                    "exploration_id": exp_id,
                    "error": "Not found"
                })
        
        # Sort by win_rate, then pnl
        valid = [r for r in results if "error" not in r]
        valid.sort(key=lambda r: (r["win_rate"], r["total_pnl"]), reverse=True)
        
        return {
            "comparison": results,
            "best_overall": valid[0] if valid else None
        }


@ToolRegistry.register(
    tool_id="diagnose_exploration_run",
    category=ToolCategory.UTILITY,
    name="Diagnose Exploration Run",
    description="Analyze an exploration run to find patterns in wins vs losses (by hour, day, etc.)",
    input_schema={
        "type": "object",
        "properties": {
            "exploration_id": {
                "type": "string",
                "description": "The exploration ID to analyze"
            }
        },
        "required": ["exploration_id"]
    }
)
class DiagnoseExplorationRunTool:
    """Diagnose exploration run - alias for diagnose_run but exploration-scoped."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        exploration_id = inputs.get("exploration_id", "")
        
        path = EXPLORATION_DIR / f"{exploration_id}.json"
        if not path.exists():
            return {"error": f"Exploration {exploration_id} not found"}
        
        with open(path) as f:
            data = json.load(f)
        
        all_results = data.get("all_results", [])
        
        # Aggregate across all configs
        total_trades = sum(r.get("total_trades", 0) for r in all_results if "error" not in r)
        total_wins = sum(r.get("wins", 0) for r in all_results if "error" not in r)
        total_pnl = sum(r.get("total_pnl", 0) for r in all_results if "error" not in r)
        
        # Find best and worst
        valid = [r for r in all_results if "error" not in r and r.get("total_trades", 0) > 0]
        valid.sort(key=lambda r: r.get("win_rate", 0), reverse=True)
        
        return {
            "exploration_id": exploration_id,
            "total_configs_run": len(all_results),
            "successful_configs": len(valid),
            "aggregate_trades": total_trades,
            "aggregate_wins": total_wins,
            "aggregate_pnl": total_pnl,
            "best_config": valid[0] if valid else None,
            "worst_config": valid[-1] if valid else None
        }


# =============================================================================
# Phase 2: Context & Explanation Tools
# =============================================================================

@ToolRegistry.register(
    tool_id="get_session_context",
    category=ToolCategory.UTILITY,
    name="Get Session Context",
    description="Get session context at a timestamp: RTH/Globex, ORH/ORL, PDH/PDL, VWAP location",
    input_schema={
        "type": "object",
        "properties": {
            "timestamp": {
                "type": "string",
                "description": "ISO timestamp to analyze"
            }
        },
        "required": ["timestamp"]
    }
)
class GetSessionContextTool:
    """Get session-aware context."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        import pandas as pd
        from datetime import timedelta
        from src.data.loader import load_continuous_contract
        from src.config import NY_TZ
        
        ts_str = inputs.get("timestamp", "")
        try:
            ts = pd.to_datetime(ts_str)
            if ts.tzinfo is None:
                ts = ts.tz_localize(NY_TZ)
        except Exception as e:
            return {"error": f"Invalid timestamp: {e}"}
        
        # Load data around timestamp
        start = (ts - timedelta(days=2)).strftime('%Y-%m-%d')
        end = (ts + timedelta(days=1)).strftime('%Y-%m-%d')
        df = load_continuous_contract(start_date=start, end_date=end)
        
        if df.empty:
            return {"error": "No data for date range"}
        
        # Determine session
        hour = ts.hour
        minute = ts.minute
        time_of_day = hour * 60 + minute
        
        # RTH = 9:30 - 16:00 ET (570 - 960 minutes)
        is_rth = 570 <= time_of_day <= 960
        session = "RTH" if is_rth else "GLOBEX"
        
        # Get today's data
        today = ts.date()
        today_mask = df['time'].dt.date == today
        today_df = df[today_mask]
        
        # Previous day
        yesterday = today - timedelta(days=1)
        yesterday_mask = df['time'].dt.date == yesterday
        yesterday_df = df[yesterday_mask]
        
        # PDH/PDL
        pdh = float(yesterday_df['high'].max()) if not yesterday_df.empty else 0.0
        pdl = float(yesterday_df['low'].min()) if not yesterday_df.empty else 0.0
        
        # ORH/ORL (first 30 min of RTH)
        rth_start = pd.Timestamp(f"{today} 09:30:00", tz=NY_TZ)
        rth_30 = rth_start + timedelta(minutes=30)
        or_mask = (df['time'] >= rth_start) & (df['time'] < rth_30)
        or_df = df[or_mask]
        
        orh = float(or_df['high'].max()) if not or_df.empty else 0.0
        orl = float(or_df['low'].min()) if not or_df.empty else 0.0
        
        # Current price at timestamp
        at_ts = df[df['time'] <= ts]
        current_price = float(at_ts['close'].iloc[-1]) if not at_ts.empty else 0.0
        
        # VWAP (simplified: session VWAP from RTH start)
        if is_rth:
            vwap_df = df[(df['time'] >= rth_start) & (df['time'] <= ts)]
        else:
            vwap_df = today_df[today_df['time'] <= ts]
        
        if not vwap_df.empty and vwap_df['volume'].sum() > 0:
            vwap = float((vwap_df['close'] * vwap_df['volume']).sum() / vwap_df['volume'].sum())
        else:
            vwap = current_price
        
        return {
            "timestamp": ts.isoformat(),
            "session": session,
            "is_rth": is_rth,
            "current_price": current_price,
            "pdh": pdh,
            "pdl": pdl,
            "orh": orh,
            "orl": orl,
            "vwap": vwap,
            "price_vs_vwap": "ABOVE" if current_price > vwap else "BELOW",
            "price_vs_pdh": "ABOVE" if current_price > pdh else "BELOW",
            "price_in_or": orl <= current_price <= orh if orh > 0 else None
        }


@ToolRegistry.register(
    tool_id="explain_scan_fire",
    category=ToolCategory.UTILITY,
    name="Explain Scan Fire",
    description="Explain why a scan fired at a specific decision. Shows which conditions were true/false.",
    input_schema={
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run ID containing the decision"
            },
            "decision_index": {
                "type": "integer",
                "description": "Index of the decision to explain"
            }
        },
        "required": ["run_id", "decision_index"]
    }
)
class ExplainScanFireTool:
    """Explain why a scan fired."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        run_id = inputs.get("run_id", "")
        decision_index = inputs.get("decision_index", 0)
        
        # Find run directory
        run_dir = RESULTS_DIR / "viz" / run_id
        if not run_dir.exists():
            run_dir = RESULTS_DIR / run_id
        if not run_dir.exists():
            return {"error": f"Run {run_id} not found"}
        
        # Load decisions
        decisions_file = run_dir / "decisions.jsonl"
        if not decisions_file.exists():
            decisions_file = run_dir / "records.jsonl"
        if not decisions_file.exists():
            return {"error": "No decisions file found"}
        
        decisions = []
        with open(decisions_file) as f:
            for line in f:
                if line.strip():
                    decisions.append(json.loads(line))
        
        if decision_index >= len(decisions):
            return {"error": f"Decision index {decision_index} out of range (max: {len(decisions)-1})"}
        
        decision = decisions[decision_index]
        
        # Extract scanner context
        scanner_ctx = decision.get("scanner_context", {})
        trigger_info = scanner_ctx.get("trigger_conditions", {})
        
        # Get features at decision time
        features = decision.get("features", {})
        
        return {
            "decision_index": decision_index,
            "timestamp": decision.get("timestamp", decision.get("time")),
            "direction": scanner_ctx.get("direction", "UNKNOWN"),
            "scanner_id": scanner_ctx.get("scanner_id", "UNKNOWN"),
            "trigger_conditions": trigger_info,
            "features_at_fire": {
                k: v for k, v in features.items() 
                if isinstance(v, (int, float, str, bool))
            },
            "entry_price": decision.get("current_price", 0),
            "outcome": decision.get("outcome", "UNKNOWN")
        }


@ToolRegistry.register(
    tool_id="scan_coverage_report",
    category=ToolCategory.UTILITY,
    name="Scan Coverage Report",
    description="Analyze scan trigger frequency, clustering, and dead zones over a date range.",
    input_schema={
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run ID to analyze"
            }
        },
        "required": ["run_id"]
    }
)
class ScanCoverageReportTool:
    """Analyze scan coverage."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        import pandas as pd
        
        run_id = inputs.get("run_id", "")
        
        # Find run
        run_dir = RESULTS_DIR / "viz" / run_id
        if not run_dir.exists():
            run_dir = RESULTS_DIR / run_id
        if not run_dir.exists():
            # Try exploration dir
            exp_path = EXPLORATION_DIR / f"{run_id}.json"
            if exp_path.exists():
                with open(exp_path) as f:
                    data = json.load(f)
                return {
                    "type": "exploration",
                    "total_configs": data.get("total_configs", 0),
                    "successful": data.get("successful_configs", 0),
                    "note": "This is an exploration run, not a viz run"
                }
            return {"error": f"Run {run_id} not found"}
        
        # Load decisions
        decisions_file = run_dir / "decisions.jsonl"
        if not decisions_file.exists():
            decisions_file = run_dir / "records.jsonl"
        
        if not decisions_file.exists():
            return {"error": "No decisions file"}
        
        decisions = []
        with open(decisions_file) as f:
            for line in f:
                if line.strip():
                    decisions.append(json.loads(line))
        
        if not decisions:
            return {"total_triggers": 0, "note": "No decisions in run"}
        
        # Parse timestamps
        times = []
        for d in decisions:
            ts = d.get("timestamp", d.get("time"))
            if ts:
                times.append(pd.to_datetime(ts))
        
        if not times:
            return {"total_triggers": len(decisions), "timestamps_parsed": 0}
        
        df = pd.DataFrame({"time": times})
        df['hour'] = df['time'].dt.hour
        df['day'] = df['time'].dt.day_name()
        
        # Hourly distribution
        hourly = df['hour'].value_counts().sort_index().to_dict()
        
        # Day distribution
        daily = df['day'].value_counts().to_dict()
        
        # Clustering: time between triggers
        df = df.sort_values('time')
        df['gap_minutes'] = df['time'].diff().dt.total_seconds() / 60
        
        avg_gap = float(df['gap_minutes'].mean()) if len(df) > 1 else 0
        max_gap = float(df['gap_minutes'].max()) if len(df) > 1 else 0
        
        # Dead zones (hours with 0 triggers during RTH)
        rth_hours = set(range(9, 16))
        active_hours = set(hourly.keys())
        dead_hours = list(rth_hours - active_hours)
        
        return {
            "total_triggers": len(decisions),
            "hourly_distribution": hourly,
            "daily_distribution": daily,
            "avg_gap_minutes": round(avg_gap, 1),
            "max_gap_minutes": round(max_gap, 1),
            "dead_rth_hours": dead_hours,
            "coverage_quality": "GOOD" if len(dead_hours) <= 2 else "SPARSE"
        }


# =============================================================================
# Phase 3: Counterfactual Tools
# =============================================================================

@ToolRegistry.register(
    tool_id="counterfactual_entry_shift",
    category=ToolCategory.UTILITY,
    name="Counterfactual Entry Shift",
    description="Test what-if scenarios: what if entry was N bars earlier or later? Returns P&L delta, MFE, MAE per shift.",
    input_schema={
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "The run ID containing the trade"
            },
            "decision_index": {
                "type": "integer",
                "description": "Index of the decision to analyze"
            },
            "shifts": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of bar shifts to test (e.g., [-3, -2, -1, 0, 1, 2, 3])",
                "default": [-3, -2, -1, 0, 1, 2, 3]
            }
        },
        "required": ["run_id", "decision_index"]
    }
)
class CounterfactualEntryShiftTool:
    """Test entry timing alternatives."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        import pandas as pd
        from datetime import timedelta
        from src.data.loader import load_continuous_contract
        from src.config import NY_TZ
        
        run_id = inputs.get("run_id", "")
        decision_index = inputs.get("decision_index", 0)
        shifts = inputs.get("shifts", [-3, -2, -1, 0, 1, 2, 3])
        
        # Find run
        run_dir = RESULTS_DIR / "viz" / run_id
        if not run_dir.exists():
            run_dir = RESULTS_DIR / run_id
        if not run_dir.exists():
            return {"error": f"Run {run_id} not found"}
        
        # Load decisions
        decisions_file = run_dir / "decisions.jsonl"
        if not decisions_file.exists():
            decisions_file = run_dir / "records.jsonl"
        if not decisions_file.exists():
            return {"error": "No decisions file"}
        
        decisions = []
        with open(decisions_file) as f:
            for line in f:
                if line.strip():
                    decisions.append(json.loads(line))
        
        if decision_index >= len(decisions):
            return {"error": f"Decision index {decision_index} out of range"}
        
        decision = decisions[decision_index]
        
        # Get entry info
        ts_str = decision.get("timestamp", decision.get("time"))
        if not ts_str:
            return {"error": "Decision has no timestamp"}
        
        entry_ts = pd.to_datetime(ts_str)
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize(NY_TZ)
        
        entry_price = decision.get("current_price", 0)
        direction = decision.get("scanner_context", {}).get("direction", "LONG")
        
        # Get OCO levels
        oco = decision.get("oco", decision.get("best_oco", {}))
        tp_price = oco.get("tp_price", 0)
        sl_price = oco.get("sl_price", 0)
        
        if not (entry_price and tp_price and sl_price):
            return {"error": "Missing price data in decision"}
        
        # Load bars around entry
        start = (entry_ts - timedelta(hours=1)).strftime('%Y-%m-%d')
        end = (entry_ts + timedelta(hours=4)).strftime('%Y-%m-%d')
        df = load_continuous_contract(start_date=start, end_date=end)
        
        if df.empty:
            return {"error": "No data for date range"}
        
        # Find entry bar index
        df = df.reset_index(drop=True)
        time_diffs = (df['time'] - entry_ts).abs()
        entry_idx = time_diffs.idxmin()
        
        results = []
        for shift in shifts:
            shifted_idx = entry_idx + shift
            
            if shifted_idx < 0 or shifted_idx >= len(df):
                results.append({
                    "shift": shift,
                    "error": "Out of range"
                })
                continue
            
            shifted_entry_price = float(df.loc[shifted_idx, 'close'])
            
            # Recalculate TP/SL based on same distance
            if direction == "LONG":
                tp_distance = tp_price - entry_price
                sl_distance = entry_price - sl_price
                new_tp = shifted_entry_price + tp_distance
                new_sl = shifted_entry_price - sl_distance
            else:
                tp_distance = entry_price - tp_price
                sl_distance = sl_price - entry_price
                new_tp = shifted_entry_price - tp_distance
                new_sl = shifted_entry_price + sl_distance
            
            # Simulate outcome (simple: check next 60 bars)
            future_bars = df.loc[shifted_idx+1:shifted_idx+60]
            
            mfe = 0.0
            mae = 0.0
            outcome = "TIMEOUT"
            exit_bar = 0
            pnl = 0.0
            
            for i, (_, bar) in enumerate(future_bars.iterrows()):
                if direction == "LONG":
                    excursion = bar['high'] - shifted_entry_price
                    adverse = shifted_entry_price - bar['low']
                    mfe = max(mfe, excursion)
                    mae = max(mae, adverse)
                    
                    if bar['high'] >= new_tp:
                        outcome = "WIN"
                        pnl = tp_distance
                        exit_bar = i + 1
                        break
                    if bar['low'] <= new_sl:
                        outcome = "LOSS"
                        pnl = -sl_distance
                        exit_bar = i + 1
                        break
                else:
                    excursion = shifted_entry_price - bar['low']
                    adverse = bar['high'] - shifted_entry_price
                    mfe = max(mfe, excursion)
                    mae = max(mae, adverse)
                    
                    if bar['low'] <= new_tp:
                        outcome = "WIN"
                        pnl = tp_distance
                        exit_bar = i + 1
                        break
                    if bar['high'] >= new_sl:
                        outcome = "LOSS"
                        pnl = -sl_distance
                        exit_bar = i + 1
                        break
            
            results.append({
                "shift": shift,
                "shifted_entry_price": round(shifted_entry_price, 2),
                "outcome": outcome,
                "pnl": round(pnl, 2),
                "mfe": round(mfe, 2),
                "mae": round(mae, 2),
                "exit_bar": exit_bar
            })
        
        # Find best shift
        valid_results = [r for r in results if "error" not in r]
        if valid_results:
            best = max(valid_results, key=lambda r: r["pnl"])
        else:
            best = None
        
        return {
            "run_id": run_id,
            "decision_index": decision_index,
            "original_direction": direction,
            "original_entry": entry_price,
            "results": results,
            "best_shift": best
        }



```

### src/tools/price_analysis_tools.py

```python
"""
Price Analysis Tools for MLang2

These tools analyze RAW PRICE DATA to find opportunities.
They are PRICE-FIRST, not scanner-dependent.

The agent should use these as the PRIMARY source of trade ideas,
falling back to scanners only when explicitly asked.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import timedelta

from src.core.tool_registry import ToolRegistry, ToolCategory
from src.config import RESULTS_DIR, NY_TZ
from src.data.loader import load_continuous_contract


# =============================================================================
# Price-First Analysis Tools
# =============================================================================

@ToolRegistry.register(
    tool_id="find_price_opportunities",
    category=ToolCategory.UTILITY,
    name="Find Price Opportunities",
    description="Analyze raw price data to find clean trading opportunities (swing lows, breakouts, pullbacks). This is PRICE-FIRST analysis - does NOT depend on scanners.",
    input_schema={
        "type": "object",
        "properties": {
            "start_date": {
                "type": "string",
                "description": "Start date YYYY-MM-DD"
            },
            "end_date": {
                "type": "string",
                "description": "End date YYYY-MM-DD"
            },
            "direction": {
                "type": "string",
                "enum": ["LONG", "SHORT", "BOTH"],
                "description": "Direction to look for",
                "default": "BOTH"
            },
            "min_move_atr": {
                "type": "number",
                "description": "Minimum move size in ATR multiples to consider 'clean'",
                "default": 2.0
            },
            "timeframe": {
                "type": "string",
                "enum": ["1m", "5m", "15m"],
                "description": "Timeframe to analyze",
                "default": "5m"
            }
        },
        "required": ["start_date", "end_date"]
    }
)
class FindPriceOpportunitiesTool:
    """Find opportunities from raw price - no scanner dependency."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        start_date = inputs.get("start_date")
        end_date = inputs.get("end_date")
        direction = inputs.get("direction", "BOTH")
        min_move_atr = inputs.get("min_move_atr", 2.0)
        timeframe = inputs.get("timeframe", "5m")
        
        # Load data
        df = load_continuous_contract(start_date=start_date, end_date=end_date)
        if df.empty:
            return {"error": "No data for date range"}
        
        # Resample if needed
        if timeframe != "1m":
            df = df.set_index('time')
            rule = {'5m': '5T', '15m': '15T'}[timeframe]
            df = df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna().reset_index()
        
        # Calculate ATR for context
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        
        # Find swing points (local min/max over 5 bars)
        df['swing_low'] = (
            (df['low'] < df['low'].shift(1)) & 
            (df['low'] < df['low'].shift(2)) &
            (df['low'] < df['low'].shift(-1)) &
            (df['low'] < df['low'].shift(-2))
        )
        df['swing_high'] = (
            (df['high'] > df['high'].shift(1)) & 
            (df['high'] > df['high'].shift(2)) &
            (df['high'] > df['high'].shift(-1)) &
            (df['high'] > df['high'].shift(-2))
        )
        
        opportunities = []
        
        # Find LONG opportunities (swing lows followed by upward move)
        if direction in ["LONG", "BOTH"]:
            swing_low_indices = df[df['swing_low']].index.tolist()
            for idx in swing_low_indices:
                if idx + 20 >= len(df):
                    continue
                    
                entry_bar = df.iloc[idx]
                entry_price = float(entry_bar['low'])
                atr = float(entry_bar['atr']) if pd.notna(entry_bar['atr']) else 2.0
                
                # Look forward 20 bars for move
                future = df.iloc[idx+1:idx+21]
                max_high = future['high'].max()
                min_low = future['low'].min()
                
                mfe = max_high - entry_price  # max favorable
                mae = entry_price - min_low   # max adverse
                
                # "Clean" = good MFE, low MAE
                if mfe >= min_move_atr * atr and mae < atr:
                    opportunities.append({
                        "direction": "LONG",
                        "timestamp": entry_bar['time'].isoformat() if hasattr(entry_bar['time'], 'isoformat') else str(entry_bar['time']),
                        "entry_price": round(entry_price, 2),
                        "suggested_stop": round(entry_price - atr, 2),
                        "suggested_target": round(entry_price + 2*atr, 2),
                        "mfe": round(mfe, 2),
                        "mae": round(mae, 2),
                        "mfe_atr": round(mfe/atr, 1),
                        "quality": "CLEAN" if mae < 0.5*atr else "GOOD",
                        "reason": "Swing low with strong follow-through, minimal drawdown"
                    })
        
        # Find SHORT opportunities (swing highs followed by downward move)
        if direction in ["SHORT", "BOTH"]:
            swing_high_indices = df[df['swing_high']].index.tolist()
            for idx in swing_high_indices:
                if idx + 20 >= len(df):
                    continue
                    
                entry_bar = df.iloc[idx]
                entry_price = float(entry_bar['high'])
                atr = float(entry_bar['atr']) if pd.notna(entry_bar['atr']) else 2.0
                
                future = df.iloc[idx+1:idx+21]
                min_low = future['low'].min()
                max_high = future['high'].max()
                
                mfe = entry_price - min_low
                mae = max_high - entry_price
                
                if mfe >= min_move_atr * atr and mae < atr:
                    opportunities.append({
                        "direction": "SHORT",
                        "timestamp": entry_bar['time'].isoformat() if hasattr(entry_bar['time'], 'isoformat') else str(entry_bar['time']),
                        "entry_price": round(entry_price, 2),
                        "suggested_stop": round(entry_price + atr, 2),
                        "suggested_target": round(entry_price - 2*atr, 2),
                        "mfe": round(mfe, 2),
                        "mae": round(mae, 2),
                        "mfe_atr": round(mfe/atr, 1),
                        "quality": "CLEAN" if mae < 0.5*atr else "GOOD",
                        "reason": "Swing high with strong follow-through, minimal drawdown"
                    })
        
        # Sort by quality (MFE/MAE ratio)
        opportunities.sort(key=lambda x: x['mfe'] / max(x['mae'], 0.1), reverse=True)
        
        return {
            "date_range": f"{start_date} to {end_date}",
            "timeframe": timeframe,
            "direction_filter": direction,
            "total_opportunities": len(opportunities),
            "top_opportunities": opportunities[:10],  # Top 10
            "note": "These are PRICE-DERIVED opportunities, not scanner signals"
        }


@ToolRegistry.register(
    tool_id="describe_price_action",
    category=ToolCategory.UTILITY,
    name="Describe Price Action",
    description="Generate a narrative description of what price did during a time window. Useful for understanding context before proposing trades.",
    input_schema={
        "type": "object",
        "properties": {
            "start_date": {
                "type": "string",
                "description": "Start date YYYY-MM-DD"
            },
            "end_date": {
                "type": "string",
                "description": "End date YYYY-MM-DD"
            },
            "timeframe": {
                "type": "string",
                "enum": ["1m", "5m", "15m", "1h"],
                "default": "5m"
            }
        },
        "required": ["start_date", "end_date"]
    }
)
class DescribePriceActionTool:
    """Generate narrative description of price action."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        start_date = inputs.get("start_date")
        end_date = inputs.get("end_date")
        timeframe = inputs.get("timeframe", "5m")
        
        df = load_continuous_contract(start_date=start_date, end_date=end_date)
        if df.empty:
            return {"error": "No data for date range"}
        
        # Resample if needed
        if timeframe != "1m":
            df = df.set_index('time')
            rule = {'5m': '5T', '15m': '15T', '1h': '1H'}[timeframe]
            df = df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna().reset_index()
        
        if df.empty:
            return {"error": "No data after resampling"}
        
        # Calculate stats
        open_price = float(df['open'].iloc[0])
        close_price = float(df['close'].iloc[-1])
        high_price = float(df['high'].max())
        low_price = float(df['low'].min())
        
        net_change = close_price - open_price
        net_pct = (net_change / open_price) * 100
        total_range = high_price - low_price
        
        # Daily breakdown
        df['date'] = pd.to_datetime(df['time']).dt.date
        daily = df.groupby('date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        
        daily_summaries = []
        for date, row in daily.iterrows():
            day_change = row['close'] - row['open']
            direction = "UP" if day_change > 0 else "DOWN" if day_change < 0 else "FLAT"
            daily_summaries.append({
                "date": str(date),
                "direction": direction,
                "open": round(row['open'], 2),
                "close": round(row['close'], 2),
                "high": round(row['high'], 2),
                "low": round(row['low'], 2),
                "range": round(row['high'] - row['low'], 2)
            })
        
        # Determine overall trend
        if net_pct > 1:
            trend = "BULLISH"
        elif net_pct < -1:
            trend = "BEARISH"
        else:
            trend = "CHOPPY/RANGE-BOUND"
        
        # Narrative
        narrative = f"From {start_date} to {end_date}, MES moved from {open_price:.2f} to {close_price:.2f} "
        narrative += f"(net {'+' if net_change > 0 else ''}{net_change:.2f} points, {net_pct:.2f}%). "
        narrative += f"The period high was {high_price:.2f}, low was {low_price:.2f}, total range {total_range:.2f} points. "
        narrative += f"Overall character: {trend}."
        
        return {
            "date_range": f"{start_date} to {end_date}",
            "timeframe": timeframe,
            "bars_analyzed": len(df),
            "open": open_price,
            "close": close_price,
            "high": high_price,
            "low": low_price,
            "net_change": round(net_change, 2),
            "net_pct": round(net_pct, 2),
            "overall_trend": trend,
            "narrative": narrative,
            "daily_breakdown": daily_summaries
        }


@ToolRegistry.register(
    tool_id="propose_trade",
    category=ToolCategory.UTILITY,
    name="Propose Trade",
    description="Given a timestamp, propose a specific trade with entry, stop, and target based on surrounding price structure.",
    input_schema={
        "type": "object",
        "properties": {
            "timestamp": {
                "type": "string",
                "description": "ISO timestamp for the trade entry"
            },
            "direction": {
                "type": "string",
                "enum": ["LONG", "SHORT"],
                "description": "Trade direction"
            },
            "risk_atr": {
                "type": "number",
                "description": "Stop distance in ATR multiples",
                "default": 1.0
            },
            "reward_atr": {
                "type": "number",
                "description": "Target distance in ATR multiples",
                "default": 2.0
            }
        },
        "required": ["timestamp", "direction"]
    }
)
class ProposeTradePool:
    """Propose a specific trade with levels."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        timestamp = inputs.get("timestamp")
        direction = inputs.get("direction")
        risk_atr = inputs.get("risk_atr", 1.0)
        reward_atr = inputs.get("reward_atr", 2.0)
        
        ts = pd.to_datetime(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize(NY_TZ)
        
        # Load surrounding data
        start = (ts - timedelta(days=1)).strftime('%Y-%m-%d')
        end = (ts + timedelta(days=1)).strftime('%Y-%m-%d')
        df = load_continuous_contract(start_date=start, end_date=end)
        
        if df.empty:
            return {"error": "No data for date range"}
        
        # Calculate ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        
        # Find entry bar
        df = df.reset_index(drop=True)
        time_diffs = (df['time'] - ts).abs()
        entry_idx = time_diffs.idxmin()
        entry_bar = df.iloc[entry_idx]
        
        entry_price = float(entry_bar['close'])
        atr = float(entry_bar['atr']) if pd.notna(entry_bar['atr']) else 2.0
        
        if direction == "LONG":
            stop = entry_price - (risk_atr * atr)
            target = entry_price + (reward_atr * atr)
        else:
            stop = entry_price + (risk_atr * atr)
            target = entry_price - (reward_atr * atr)
        
        # Check outcome if we have future data
        future = df.iloc[entry_idx+1:entry_idx+61]  # Next hour
        outcome = "UNKNOWN"
        exit_price = None
        bars_to_exit = None
        
        for i, (_, bar) in enumerate(future.iterrows()):
            if direction == "LONG":
                if bar['high'] >= target:
                    outcome = "WIN"
                    exit_price = target
                    bars_to_exit = i + 1
                    break
                if bar['low'] <= stop:
                    outcome = "LOSS"
                    exit_price = stop
                    bars_to_exit = i + 1
                    break
            else:
                if bar['low'] <= target:
                    outcome = "WIN"
                    exit_price = target
                    bars_to_exit = i + 1
                    break
                if bar['high'] >= stop:
                    outcome = "LOSS"
                    exit_price = stop
                    bars_to_exit = i + 1
                    break
        
        return {
            "timestamp": ts.isoformat(),
            "direction": direction,
            "entry_price": round(entry_price, 2),
            "stop_loss": round(stop, 2),
            "take_profit": round(target, 2),
            "risk_points": round(abs(entry_price - stop), 2),
            "reward_points": round(abs(target - entry_price), 2),
            "rr_ratio": round(reward_atr / risk_atr, 1),
            "atr_at_entry": round(atr, 2),
            "outcome": outcome,
            "exit_price": round(exit_price, 2) if exit_price else None,
            "bars_to_exit": bars_to_exit
        }


@ToolRegistry.register(
    tool_id="study_obvious_trades",
    category=ToolCategory.UTILITY,
    name="Study Obvious Winners",
    description="Find 'obvious in hindsight' trades, analyze what they had in common, and emit a candidate scan spec. This is a complete research workflow.",
    input_schema={
        "type": "object",
        "properties": {
            "start_date": {
                "type": "string",
                "description": "Start date YYYY-MM-DD"
            },
            "end_date": {
                "type": "string",
                "description": "End date YYYY-MM-DD"
            },
            "direction": {
                "type": "string",
                "enum": ["LONG", "SHORT", "BOTH"],
                "default": "BOTH"
            },
            "min_move_atr": {
                "type": "number",
                "description": "Minimum move in ATR to qualify as 'obvious'",
                "default": 3.0
            },
            "timeframe": {
                "type": "string",
                "enum": ["1m", "5m", "15m"],
                "default": "5m"
            },
            "top_n": {
                "type": "integer",
                "description": "How many top trades to analyze",
                "default": 10
            }
        },
        "required": ["start_date", "end_date"]
    }
)
class StudyObviousTradesTool:
    """Complete 'Obvious Winners Study' workflow."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        from collections import Counter
        
        start_date = inputs.get("start_date")
        end_date = inputs.get("end_date")
        direction = inputs.get("direction", "BOTH")
        min_move_atr = inputs.get("min_move_atr", 3.0)
        timeframe = inputs.get("timeframe", "5m")
        top_n = inputs.get("top_n", 10)
        
        # Step 1: Find opportunities
        finder = FindPriceOpportunitiesTool()
        opps_result = finder.execute(
            start_date=start_date,
            end_date=end_date,
            direction=direction,
            min_move_atr=min_move_atr,
            timeframe=timeframe
        )
        
        if "error" in opps_result:
            return opps_result
        
        all_opps = opps_result.get("top_opportunities", [])
        if not all_opps:
            return {"error": "No opportunities found matching criteria"}
        
        # Take top N by MFE/MAE ratio
        top_trades = all_opps[:top_n]
        
        # Step 2: Analyze context for each
        from src.tools.exploration_tools import GetSessionContextTool
        session_tool = GetSessionContextTool()
        
        contexts = []
        for trade in top_trades:
            ctx = session_tool.execute(timestamp=trade["timestamp"])
            if "error" not in ctx:
                # Add regime tags
                ctx["trade_direction"] = trade["direction"]
                ctx["mfe"] = trade["mfe"]
                ctx["mae"] = trade["mae"]
                ctx["mfe_mae_ratio"] = trade["mfe"] / max(abs(trade["mae"]), 0.1)
                ctx["quality"] = trade["quality"]
                contexts.append(ctx)
        
        if not contexts:
            return {"error": "Could not get context for any trades"}
        
        # Step 3: Aggregate patterns
        session_counts = Counter(c["session"] for c in contexts)
        vwap_counts = Counter(c["price_vs_vwap"] for c in contexts)
        pdh_counts = Counter(c["price_vs_pdh"] for c in contexts)
        or_counts = Counter(str(c.get("price_in_or", "?")) for c in contexts)
        
        # Separate by direction
        long_trades = [c for c in contexts if c["trade_direction"] == "LONG"]
        short_trades = [c for c in contexts if c["trade_direction"] == "SHORT"]
        
        long_vwap = Counter(c["price_vs_vwap"] for c in long_trades) if long_trades else {}
        short_vwap = Counter(c["price_vs_vwap"] for c in short_trades) if short_trades else {}
        
        # Step 4: Generate candidate scan spec
        def most_common(counter):
            return counter.most_common(1)[0][0] if counter else None
        
        scan_spec = {
            "name": f"obvious_winners_{start_date}_{end_date}",
            "derived_from": f"Top {len(contexts)} trades by MFE/MAE ratio",
            "direction_logic": {}
        }
        
        if long_trades:
            scan_spec["direction_logic"]["LONG"] = {
                "primary_session": most_common(Counter(c["session"] for c in long_trades)),
                "price_vs_vwap": most_common(long_vwap),
                "price_vs_pdh": most_common(Counter(c["price_vs_pdh"] for c in long_trades)),
                "in_opening_range": most_common(Counter(c.get("price_in_or") for c in long_trades)),
                "sample_size": len(long_trades)
            }
        
        if short_trades:
            scan_spec["direction_logic"]["SHORT"] = {
                "primary_session": most_common(Counter(c["session"] for c in short_trades)),
                "price_vs_vwap": most_common(short_vwap),
                "price_vs_pdh": most_common(Counter(c["price_vs_pdh"] for c in short_trades)),
                "in_opening_range": most_common(Counter(c.get("price_in_or") for c in short_trades)),
                "sample_size": len(short_trades)
            }
        
        scan_spec["min_move_atr"] = min_move_atr
        scan_spec["timeframe"] = timeframe
        
        return {
            "date_range": f"{start_date} to {end_date}",
            "total_obvious_trades": opps_result.get("total_opportunities", 0),
            "analyzed_count": len(contexts),
            "top_trades": [
                {
                    "timestamp": t["timestamp"],
                    "direction": t["direction"],
                    "entry_price": t["entry_price"],
                    "mfe": t["mfe"],
                    "mae": t["mae"],
                    "quality": t["quality"]
                }
                for t in top_trades[:5]  # Top 5 summary
            ],
            "aggregated_context": {
                "session_distribution": dict(session_counts),
                "vwap_relation": dict(vwap_counts),
                "pdh_relation": dict(pdh_counts),
                "opening_range_relation": dict(or_counts),
                "long_vwap": dict(long_vwap),
                "short_vwap": dict(short_vwap)
            },
            "candidate_scan_spec": scan_spec,
            "key_insight": self._generate_insight(contexts, long_trades, short_trades)
        }
    
    def _generate_insight(self, contexts, long_trades, short_trades) -> str:
        """Generate a human-readable insight."""
        insights = []
        
        total = len(contexts)
        
        # Check for dominant patterns
        below_pdh = sum(1 for c in contexts if c.get("price_vs_pdh") == "BELOW")
        if below_pdh >= total * 0.7:
            insights.append(f"{below_pdh}/{total} trades entered BELOW previous day high")
        
        outside_or = sum(1 for c in contexts if c.get("price_in_or") == False)
        if outside_or >= total * 0.7:
            insights.append(f"{outside_or}/{total} trades were OUTSIDE opening range")
        
        # Direction-specific
        if long_trades:
            long_below_vwap = sum(1 for c in long_trades if c.get("price_vs_vwap") == "BELOW")
            if long_below_vwap >= len(long_trades) * 0.6:
                insights.append(f"LONG entries favored BELOW VWAP ({long_below_vwap}/{len(long_trades)})")
        
        if short_trades:
            short_above_vwap = sum(1 for c in short_trades if c.get("price_vs_vwap") == "ABOVE")
            if short_above_vwap >= len(short_trades) * 0.6:
                insights.append(f"SHORT entries favored ABOVE VWAP ({short_above_vwap}/{len(short_trades)})")
        
        if not insights:
            return "No dominant pattern detected - trades were distributed across various contexts"
        
        return " | ".join(insights)


# =============================================================================
# Priority 1: Core Analysis Tools
# =============================================================================

@ToolRegistry.register(
    tool_id="cluster_trades",
    category=ToolCategory.UTILITY,
    name="Cluster Trades",
    description="Group trades by time of day, session, volatility state, or VWAP relation. Enables 'morning vs afternoon' comparisons.",
    input_schema={
        "type": "object",
        "properties": {
            "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
            "cluster_by": {
                "type": "string",
                "enum": ["time_of_day", "session", "day_of_week"],
                "default": "time_of_day"
            },
            "min_move_atr": {"type": "number", "default": 2.0}
        },
        "required": ["start_date", "end_date"]
    }
)
class TradeClusterTool:
    """Group trades by various dimensions."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        from collections import defaultdict
        
        start_date = inputs.get("start_date")
        end_date = inputs.get("end_date")
        cluster_by = inputs.get("cluster_by", "time_of_day")
        min_move_atr = inputs.get("min_move_atr", 2.0)
        
        # Get all opportunities
        finder = FindPriceOpportunitiesTool()
        result = finder.execute(
            start_date=start_date,
            end_date=end_date,
            direction="BOTH",
            min_move_atr=min_move_atr,
            timeframe="5m"
        )
        
        if "error" in result:
            return result
        
        all_opps = result.get("top_opportunities", [])
        
        # Cluster
        clusters = defaultdict(list)
        
        for opp in all_opps:
            ts = pd.to_datetime(opp["timestamp"])
            
            if cluster_by == "time_of_day":
                hour = ts.hour
                if 9 <= hour < 12:
                    key = "MORNING (9:30-12)"
                elif 12 <= hour < 14:
                    key = "MIDDAY (12-14)"
                elif 14 <= hour < 16:
                    key = "AFTERNOON (14-16)"
                else:
                    key = "GLOBEX"
            elif cluster_by == "session":
                hour = ts.hour
                key = "RTH" if 9 <= hour < 16 else "GLOBEX"
            elif cluster_by == "day_of_week":
                key = ts.strftime("%A")
            else:
                key = "ALL"
            
            clusters[key].append(opp)
        
        # Aggregate stats
        cluster_stats = []
        for name, trades in clusters.items():
            if not trades:
                continue
            avg_mfe = sum(t["mfe"] for t in trades) / len(trades)
            avg_mae = sum(abs(t["mae"]) for t in trades) / len(trades)
            clean_pct = sum(1 for t in trades if t["quality"] == "CLEAN") / len(trades) * 100
            long_pct = sum(1 for t in trades if t["direction"] == "LONG") / len(trades) * 100
            
            cluster_stats.append({
                "cluster": name,
                "count": len(trades),
                "avg_mfe": round(avg_mfe, 2),
                "avg_mae": round(avg_mae, 2),
                "mfe_mae_ratio": round(avg_mfe / max(avg_mae, 0.1), 1),
                "clean_pct": round(clean_pct, 1),
                "long_pct": round(long_pct, 1)
            })
        
        cluster_stats.sort(key=lambda x: x["mfe_mae_ratio"], reverse=True)
        
        return {
            "date_range": f"{start_date} to {end_date}",
            "cluster_by": cluster_by,
            "total_trades": len(all_opps),
            "clusters": cluster_stats,
            "best_cluster": cluster_stats[0]["cluster"] if cluster_stats else None
        }


@ToolRegistry.register(
    tool_id="compare_trade_pools",
    category=ToolCategory.UTILITY,
    name="Compare Trade Pools",
    description="Compare two clusters of trades and output structured differences in MFE, MAE, win rate.",
    input_schema={
        "type": "object",
        "properties": {
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "pool_a": {"type": "string", "description": "First pool name (e.g., 'MORNING')"},
            "pool_b": {"type": "string", "description": "Second pool name (e.g., 'AFTERNOON')"},
            "cluster_by": {"type": "string", "default": "time_of_day"}
        },
        "required": ["start_date", "end_date", "pool_a", "pool_b"]
    }
)
class TradeBehaviorCompareTool:
    """Compare two trade pools."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        start_date = inputs.get("start_date")
        end_date = inputs.get("end_date")
        pool_a = inputs.get("pool_a")
        pool_b = inputs.get("pool_b")
        cluster_by = inputs.get("cluster_by", "time_of_day")
        
        # Get clusters
        cluster_tool = TradeClusterTool()
        result = cluster_tool.execute(
            start_date=start_date,
            end_date=end_date,
            cluster_by=cluster_by
        )
        
        if "error" in result:
            return result
        
        clusters = {c["cluster"]: c for c in result.get("clusters", [])}
        
        if pool_a not in clusters and pool_b not in clusters:
            return {"error": f"Neither {pool_a} nor {pool_b} found in clusters"}
        
        a = clusters.get(pool_a, {"count": 0, "avg_mfe": 0, "avg_mae": 0, "mfe_mae_ratio": 0})
        b = clusters.get(pool_b, {"count": 0, "avg_mfe": 0, "avg_mae": 0, "mfe_mae_ratio": 0})
        
        return {
            "pool_a": {"name": pool_a, **a},
            "pool_b": {"name": pool_b, **b},
            "comparison": {
                "count_delta": a.get("count", 0) - b.get("count", 0),
                "mfe_delta": round(a.get("avg_mfe", 0) - b.get("avg_mfe", 0), 2),
                "mae_delta": round(a.get("avg_mae", 0) - b.get("avg_mae", 0), 2),
                "ratio_delta": round(a.get("mfe_mae_ratio", 0) - b.get("mfe_mae_ratio", 0), 1)
            },
            "winner": pool_a if a.get("mfe_mae_ratio", 0) > b.get("mfe_mae_ratio", 0) else pool_b,
            "insight": self._generate_insight(pool_a, pool_b, a, b)
        }
    
    def _generate_insight(self, name_a, name_b, a, b) -> str:
        ratio_a = a.get("mfe_mae_ratio", 0)
        ratio_b = b.get("mfe_mae_ratio", 0)
        
        if ratio_a > ratio_b * 1.5:
            return f"{name_a} significantly outperforms {name_b} ({ratio_a}x vs {ratio_b}x MFE/MAE)"
        elif ratio_b > ratio_a * 1.5:
            return f"{name_b} significantly outperforms {name_a} ({ratio_b}x vs {ratio_a}x MFE/MAE)"
        else:
            return f"{name_a} and {name_b} have similar performance ({ratio_a}x vs {ratio_b}x MFE/MAE)"


@ToolRegistry.register(
    tool_id="detect_regime",
    category=ToolCategory.UTILITY,
    name="Detect Market Regime",
    description="Identify if a day was TREND_UP, TREND_DOWN, RANGE, or SPIKE_CHANNEL.",
    input_schema={
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "Date YYYY-MM-DD to analyze"}
        },
        "required": ["date"]
    }
)
class RegimeDetectionTool:
    """Detect market regime for a day."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        date = inputs.get("date")
        
        df = load_continuous_contract(start_date=date, end_date=date)
        if df.empty:
            return {"error": f"No data for {date}"}
        
        # Basic stats
        open_price = float(df['open'].iloc[0])
        close_price = float(df['close'].iloc[-1])
        high = float(df['high'].max())
        low = float(df['low'].min())
        
        net_change = close_price - open_price
        total_range = high - low
        
        # Calculate ATR (need previous data for context)
        prev_date = (pd.to_datetime(date) - timedelta(days=7)).strftime('%Y-%m-%d')
        df_context = load_continuous_contract(start_date=prev_date, end_date=date)
        
        if len(df_context) > 14:
            df_context['tr'] = np.maximum(
                df_context['high'] - df_context['low'],
                np.maximum(
                    abs(df_context['high'] - df_context['close'].shift(1)),
                    abs(df_context['low'] - df_context['close'].shift(1))
                )
            )
            avg_atr = df_context['tr'].rolling(14).mean().iloc[-1]
        else:
            avg_atr = total_range
        
        # Regime detection
        change_pct = abs(net_change / open_price) * 100
        range_vs_avg = total_range / max(avg_atr, 0.1)
        
        if change_pct > 0.75 and net_change > 0:
            regime = "TREND_UP"
            confidence = min(change_pct / 1.5, 1.0)
        elif change_pct > 0.75 and net_change < 0:
            regime = "TREND_DOWN"
            confidence = min(change_pct / 1.5, 1.0)
        elif range_vs_avg > 1.5 and change_pct < 0.3:
            regime = "SPIKE_CHANNEL"
            confidence = min(range_vs_avg / 2, 1.0)
        else:
            regime = "RANGE"
            confidence = 1 - min(change_pct / 1.5, 0.8)
        
        return {
            "date": date,
            "regime": regime,
            "confidence": round(confidence, 2),
            "open": open_price,
            "close": close_price,
            "high": high,
            "low": low,
            "net_change": round(net_change, 2),
            "total_range": round(total_range, 2),
            "range_vs_avg_atr": round(range_vs_avg, 2),
            "recommendation": self._get_recommendation(regime)
        }
    
    def _get_recommendation(self, regime: str) -> str:
        recs = {
            "TREND_UP": "Favor longs, use trailing stops, avoid counter-trend shorts",
            "TREND_DOWN": "Favor shorts, use trailing stops, avoid counter-trend longs",
            "RANGE": "Use mean reversion, tighter targets, avoid breakout entries",
            "SPIKE_CHANNEL": "Wait for retest of spike levels, careful with stops"
        }
        return recs.get(regime, "Unknown regime")


# =============================================================================
# Priority 2: Trade Optimization Tools
# =============================================================================

@ToolRegistry.register(
    tool_id="trade_fingerprint",
    category=ToolCategory.UTILITY,
    name="Trade Fingerprint",
    description="Build a state vector for a trade timestamp: PDH/PDL distance, VWAP position, OR context, volatility percentile.",
    input_schema={
        "type": "object",
        "properties": {
            "timestamp": {"type": "string", "description": "ISO timestamp"}
        },
        "required": ["timestamp"]
    }
)
class TradeFingerprintTool:
    """Build a fingerprint for pattern matching."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        from src.tools.exploration_tools import GetSessionContextTool
        
        timestamp = inputs.get("timestamp")
        
        # Get session context
        ctx_tool = GetSessionContextTool()
        ctx = ctx_tool.execute(timestamp=timestamp)
        
        if "error" in ctx:
            return ctx
        
        # Calculate additional metrics
        ts = pd.to_datetime(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize(NY_TZ)
        
        start = (ts - timedelta(days=5)).strftime('%Y-%m-%d')
        end = (ts + timedelta(days=1)).strftime('%Y-%m-%d')
        df = load_continuous_contract(start_date=start, end_date=end)
        
        if df.empty:
            return {"error": "No data"}
        
        # Current price
        current_price = ctx.get("current_price", 0)
        pdh = ctx.get("pdh", 0)
        pdl = ctx.get("pdl", 0)
        orh = ctx.get("orh", 0)
        orl = ctx.get("orl", 0)
        vwap = ctx.get("vwap", current_price)
        
        # ATR percentile
        df['tr'] = df['high'] - df['low']
        atr_series = df['tr'].rolling(14).mean()
        current_atr = atr_series.iloc[-1] if len(atr_series) > 0 else 2.0
        atr_percentile = (atr_series < current_atr).sum() / max(len(atr_series), 1) * 100
        
        # Volume Z-score (last bar vs rolling mean)
        vol_mean = df['volume'].rolling(50).mean().iloc[-1]
        vol_std = df['volume'].rolling(50).std().iloc[-1]
        last_vol = df['volume'].iloc[-1]
        vol_z = (last_vol - vol_mean) / max(vol_std, 1)
        
        return {
            "timestamp": timestamp,
            "fingerprint": {
                "pdh_distance": round((current_price - pdh) / max(current_atr, 0.1), 2),
                "pdl_distance": round((current_price - pdl) / max(current_atr, 0.1), 2),
                "vwap_distance": round((current_price - vwap) / max(current_atr, 0.1), 2),
                "or_position": "INSIDE" if orl <= current_price <= orh else "ABOVE" if current_price > orh else "BELOW",
                "atr_percentile": round(atr_percentile, 1),
                "volume_z": round(vol_z, 2),
                "session": ctx.get("session"),
                "is_rth": ctx.get("is_rth")
            }
        }


@ToolRegistry.register(
    tool_id="indicator_impact",
    category=ToolCategory.UTILITY,
    name="Indicator Impact Analysis",
    description="Would adding an RSI or VWAP filter have improved results? Test filter impact on a pool of trades.",
    input_schema={
        "type": "object",
        "properties": {
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "indicator": {"type": "string", "enum": ["rsi", "vwap", "ema"]},
            "threshold": {"type": "number", "description": "Filter threshold (e.g., RSI < 30 for longs)"}
        },
        "required": ["start_date", "end_date", "indicator"]
    }
)
class IndicatorImpactTool:
    """Analyze impact of adding an indicator filter."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        start_date = inputs.get("start_date")
        end_date = inputs.get("end_date")
        indicator = inputs.get("indicator", "vwap")
        threshold = inputs.get("threshold")
        
        # Get trades and analyze with/without filter
        finder = FindPriceOpportunitiesTool()
        result = finder.execute(
            start_date=start_date,
            end_date=end_date,
            direction="BOTH",
            min_move_atr=2.0
        )
        
        if "error" in result:
            return result
        
        all_trades = result.get("top_opportunities", [])
        if not all_trades:
            return {"error": "No trades to analyze"}
        
        # Get session context for VWAP filtering
        from src.tools.exploration_tools import GetSessionContextTool
        session_tool = GetSessionContextTool()
        
        kept = []
        filtered = []
        
        for trade in all_trades:
            ctx = session_tool.execute(timestamp=trade["timestamp"])
            if "error" in ctx:
                continue
            
            passes_filter = False
            if indicator == "vwap":
                if trade["direction"] == "LONG":
                    passes_filter = ctx.get("price_vs_vwap") == "BELOW"
                else:
                    passes_filter = ctx.get("price_vs_vwap") == "ABOVE"
            elif indicator == "rsi":
                # Would need RSI calculation - simplified
                passes_filter = True  # Placeholder
            elif indicator == "ema":
                # Would need EMA calculation - simplified
                passes_filter = True  # Placeholder
            
            if passes_filter:
                kept.append(trade)
            else:
                filtered.append(trade)
        
        # Compare stats
        def calc_stats(trades):
            if not trades:
                return {"count": 0, "avg_mfe": 0, "avg_mae": 0}
            return {
                "count": len(trades),
                "avg_mfe": round(sum(t["mfe"] for t in trades) / len(trades), 2),
                "avg_mae": round(sum(abs(t["mae"]) for t in trades) / len(trades), 2),
                "clean_pct": round(sum(1 for t in trades if t["quality"] == "CLEAN") / len(trades) * 100, 1)
            }
        
        before = calc_stats(all_trades)
        after = calc_stats(kept)
        removed = calc_stats(filtered)
        
        return {
            "indicator": indicator,
            "before_filter": before,
            "after_filter": after,
            "removed_trades": removed,
            "filter_impact": {
                "trades_removed": len(filtered),
                "mfe_improvement": round(after["avg_mfe"] - before["avg_mfe"], 2) if after["count"] else 0,
                "mae_reduction": round(before["avg_mae"] - after["avg_mae"], 2) if after["count"] else 0
            },
            "recommendation": "ADD" if after.get("clean_pct", 0) > before.get("clean_pct", 0) + 5 else "SKIP"
        }


# =============================================================================
# Priority 3: Pattern Discovery Tools
# =============================================================================

@ToolRegistry.register(
    tool_id="find_killer_moves",
    category=ToolCategory.UTILITY,
    name="Find Killer Moves",
    description="Find the biggest, cleanest price moves in a date range - the opportunities you'd hate to miss.",
    input_schema={
        "type": "object",
        "properties": {
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "top_n": {"type": "integer", "default": 5}
        },
        "required": ["start_date", "end_date"]
    }
)
class KillerMoveDetectorTool:
    """Find the biggest opportunities."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        start_date = inputs.get("start_date")
        end_date = inputs.get("end_date")
        top_n = inputs.get("top_n", 5)
        
        df = load_continuous_contract(start_date=start_date, end_date=end_date)
        if df.empty:
            return {"error": "No data"}
        
        # Resample to 5m
        df = df.set_index('time')
        df = df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        
        # Find big moves (20-bar windows)
        moves = []
        for i in range(len(df) - 20):
            window = df.iloc[i:i+20]
            start_price = window['open'].iloc[0]
            max_up = window['high'].max() - start_price
            max_down = start_price - window['low'].min()
            
            if max_up > max_down:
                direction = "LONG"
                move_size = max_up
                entry = float(window['open'].iloc[0])
                target = float(window['high'].max())
            else:
                direction = "SHORT"
                move_size = max_down
                entry = float(window['open'].iloc[0])
                target = float(window['low'].min())
            
            moves.append({
                "timestamp": window['time'].iloc[0].isoformat(),
                "direction": direction,
                "entry_price": round(entry, 2),
                "best_exit": round(target, 2),
                "points": round(move_size, 2),
                "duration_bars": 20
            })
        
        # Sort by move size
        moves.sort(key=lambda x: x["points"], reverse=True)
        
        return {
            "date_range": f"{start_date} to {end_date}",
            "killer_moves": moves[:top_n],
            "insight": f"Top move was {moves[0]['points']} points {moves[0]['direction']} on {moves[0]['timestamp'][:10]}" if moves else "No significant moves found"
        }


@ToolRegistry.register(
    tool_id="synthesize_scan",
    category=ToolCategory.UTILITY,
    name="Synthesize Scanner",
    description="Given a pool of good trades, auto-generate a candidate scanner spec based on common patterns.",
    input_schema={
        "type": "object",
        "properties": {
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "min_mfe_atr": {"type": "number", "default": 3.0},
            "max_mae_atr": {"type": "number", "default": 1.0}
        },
        "required": ["start_date", "end_date"]
    }
)
class ScanSynthesizerTool:
    """Auto-generate scanner from good trades."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        start_date = inputs.get("start_date")
        end_date = inputs.get("end_date")
        min_mfe_atr = inputs.get("min_mfe_atr", 3.0)
        max_mae_atr = inputs.get("max_mae_atr", 1.0)
        
        # Use study_obvious_trades as foundation
        study_tool = StudyObviousTradesTool()
        result = study_tool.execute(
            start_date=start_date,
            end_date=end_date,
            direction="BOTH",
            min_move_atr=min_mfe_atr,
            top_n=20
        )
        
        if "error" in result:
            return result
        
        # Extract scan spec and enhance
        base_spec = result.get("candidate_scan_spec", {})
        
        # Add OCO suggestions based on observed MFE/MAE
        top_trades = result.get("top_trades", [])
        if top_trades:
            avg_mfe = sum(t["mfe"] for t in top_trades) / len(top_trades)
            suggested_tp = round(avg_mfe * 0.6, 1)  # Target 60% of avg MFE
            suggested_sl = round(max_mae_atr, 1)
        else:
            suggested_tp = 6.0
            suggested_sl = 3.0
        
        enhanced_spec = {
            **base_spec,
            "oco_suggestion": {
                "tp_points": suggested_tp,
                "sl_points": suggested_sl,
                "rr_ratio": round(suggested_tp / max(suggested_sl, 0.1), 1)
            },
            "confidence": "HIGH" if result.get("analyzed_count", 0) >= 10 else "MEDIUM",
            "sample_size": result.get("analyzed_count", 0)
        }
        
        return {
            "synthesized_scan": enhanced_spec,
            "key_insight": result.get("key_insight"),
            "usage": "Feed this spec to explore_strategy for validation"
        }


# =============================================================================
# Reusable Scan Filters
# =============================================================================

def filter_by_session(signals_df: pd.DataFrame, session: str = "RTH") -> pd.DataFrame:
    """Filter signals to only include RTH or GLOBEX.
    
    Args:
        signals_df: DataFrame with 'time' column
        session: "RTH" (9:30-16:00) or "GLOBEX" (all other hours)
    
    Returns:
        Filtered DataFrame
    """
    df = signals_df.copy()
    df['_hour'] = pd.to_datetime(df['time']).dt.hour
    
    if session == "RTH":
        mask = (df['_hour'] >= 9) & (df['_hour'] < 16)
    else:
        mask = (df['_hour'] < 9) | (df['_hour'] >= 16)
    
    return df[mask].drop(columns=['_hour'])


def filter_by_prevolatility(signals_df: pd.DataFrame, 
                            full_df: pd.DataFrame,
                            threshold: float = 4.8,
                            lookback_bars: int = 6,
                            above: bool = True) -> pd.DataFrame:
    """Filter signals by pre-entry volatility.
    
    Args:
        signals_df: DataFrame with signal rows
        full_df: Full price DataFrame for lookback
        threshold: Volatility threshold (pts/bar)
        lookback_bars: How many bars to look back
        above: If True, keep signals where pre-vol >= threshold
    
    Returns:
        Filtered DataFrame
    """
    keep_indices = []
    
    for idx in signals_df.index:
        if idx < lookback_bars:
            continue
        
        pre_bars = full_df.iloc[idx-lookback_bars:idx]
        pre_vol = (pre_bars['high'] - pre_bars['low']).mean()
        
        if above and pre_vol >= threshold:
            keep_indices.append(idx)
        elif not above and pre_vol < threshold:
            keep_indices.append(idx)
    
    return signals_df.loc[keep_indices]


def filter_by_regime(signals_df: pd.DataFrame,
                     full_df: pd.DataFrame,
                     regime: str = "TREND") -> pd.DataFrame:
    """Filter signals by day regime (TREND or RANGE).
    
    Args:
        signals_df: DataFrame with signal rows
        full_df: Full price DataFrame
        regime: "TREND" or "RANGE"
    """
    keep_indices = []
    
    for idx in signals_df.index:
        date = full_df.loc[idx, 'time'].date()
        day_data = full_df[full_df['time'].dt.date == date]
        
        if len(day_data) < 10:
            continue
        
        day_open = float(day_data['open'].iloc[0])
        day_close = float(day_data['close'].iloc[-1])
        day_range = float(day_data['high'].max() - day_data['low'].min())
        
        net_pct = abs(day_close - day_open) / day_open * 100
        
        is_trend = net_pct > 0.5
        
        if regime == "TREND" and is_trend:
            keep_indices.append(idx)
        elif regime == "RANGE" and not is_trend:
            keep_indices.append(idx)
    
    return signals_df.loc[keep_indices]


# =============================================================================
# Scan Evaluation Tool
# =============================================================================

@ToolRegistry.register(
    tool_id="evaluate_scan",
    category=ToolCategory.UTILITY,
    name="Evaluate Scan",
    description="Realistically backtest any scan with proper stops, entry at close, and win rate breakdown by session/volatility.",
    input_schema={
        "type": "object",
        "properties": {
            "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
            "scan_type": {
                "type": "string",
                "enum": ["swing_low", "swing_high", "ema_cross"],
                "default": "swing_low"
            },
            "direction": {
                "type": "string",
                "enum": ["LONG", "SHORT"],
                "default": "LONG"
            },
            "tp_points": {"type": "number", "default": 6.0, "description": "Take profit in points"},
            "sl_points": {"type": "number", "default": 3.0, "description": "Stop loss in points"},
            "filters": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filters to apply: 'rth_only', 'high_volatility', 'trend_days'",
                "default": []
            }
        },
        "required": ["start_date", "end_date"]
    }
)
class ScanEvaluationTool:
    """Realistically evaluate any scan."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        start_date = inputs.get("start_date")
        end_date = inputs.get("end_date")
        scan_type = inputs.get("scan_type", "swing_low")
        direction = inputs.get("direction", "LONG")
        tp_points = inputs.get("tp_points", 6.0)
        sl_points = inputs.get("sl_points", 3.0)
        filters = inputs.get("filters", [])
        
        # Load and resample data
        df = load_continuous_contract(start_date=start_date, end_date=end_date)
        if df.empty:
            return {"error": "No data"}
        
        df = df.set_index('time').resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()
        
        # Generate scan signals
        if scan_type == "swing_low":
            df['signal'] = (
                (df['low'] < df['low'].shift(1)) &
                (df['low'] < df['low'].shift(2)) &
                (df['low'] < df['low'].shift(-1)) &
                (df['low'] < df['low'].shift(-2))
            )
        elif scan_type == "swing_high":
            df['signal'] = (
                (df['high'] > df['high'].shift(1)) &
                (df['high'] > df['high'].shift(2)) &
                (df['high'] > df['high'].shift(-1)) &
                (df['high'] > df['high'].shift(-2))
            )
        elif scan_type == "ema_cross":
            df['ema9'] = df['close'].ewm(span=9).mean()
            df['ema21'] = df['close'].ewm(span=21).mean()
            if direction == "LONG":
                df['signal'] = (df['ema9'] > df['ema21']) & (df['ema9'].shift(1) <= df['ema21'].shift(1))
            else:
                df['signal'] = (df['ema9'] < df['ema21']) & (df['ema9'].shift(1) >= df['ema21'].shift(1))
        
        signals_df = df[df['signal']].copy()
        
        # Apply filters
        if "rth_only" in filters:
            signals_df = filter_by_session(signals_df, "RTH")
        if "globex_only" in filters:
            signals_df = filter_by_session(signals_df, "GLOBEX")
        if "high_volatility" in filters:
            signals_df = filter_by_prevolatility(signals_df, df, threshold=4.8, above=True)
        if "low_volatility" in filters:
            signals_df = filter_by_prevolatility(signals_df, df, threshold=4.8, above=False)
        if "trend_days" in filters:
            signals_df = filter_by_regime(signals_df, df, "TREND")
        if "range_days" in filters:
            signals_df = filter_by_regime(signals_df, df, "RANGE")
        
        # Evaluate each signal
        results = []
        for idx in signals_df.index:
            if idx + 30 >= len(df) or idx < 6:
                continue
            
            entry = float(df.loc[idx, 'close'])
            
            if direction == "LONG":
                target = entry + tp_points
                stop = entry - sl_points
            else:
                target = entry - tp_points
                stop = entry + sl_points
            
            # Get context
            pre_bars = df.iloc[max(0, idx-6):idx]
            pre_vol = (pre_bars['high'] - pre_bars['low']).mean() if len(pre_bars) > 0 else 0
            hour = df.loc[idx, 'time'].hour
            session = "RTH" if 9 <= hour < 16 else "GLOBEX"
            
            # Find outcome
            outcome = "TIMEOUT"
            bars_held = 30
            
            for i in range(idx + 1, min(idx + 31, len(df))):
                bar = df.iloc[i]
                
                if direction == "LONG":
                    if bar['low'] <= stop:
                        outcome = "LOSS"
                        bars_held = i - idx
                        break
                    if bar['high'] >= target:
                        outcome = "WIN"
                        bars_held = i - idx
                        break
                else:
                    if bar['high'] >= stop:
                        outcome = "LOSS"
                        bars_held = i - idx
                        break
                    if bar['low'] <= target:
                        outcome = "WIN"
                        bars_held = i - idx
                        break
            
            results.append({
                "outcome": outcome,
                "session": session,
                "hour": hour,
                "pre_vol": pre_vol,
                "bars_held": bars_held
            })
        
        if not results:
            return {"error": "No signals after filtering"}
        
        # Aggregate results
        rdf = pd.DataFrame(results)
        wins = rdf[rdf['outcome'] == 'WIN']
        losses = rdf[rdf['outcome'] == 'LOSS']
        timeouts = rdf[rdf['outcome'] == 'TIMEOUT']
        
        total = len(rdf)
        win_rate = len(wins) / total * 100 if total > 0 else 0
        
        # Expected value calculation
        avg_win = tp_points
        avg_loss = sl_points
        ev = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
        
        # Session breakdown
        session_stats = {}
        for sess in ["RTH", "GLOBEX"]:
            subset = rdf[rdf['session'] == sess]
            if len(subset) > 0:
                sess_wins = len(subset[subset['outcome'] == 'WIN'])
                session_stats[sess] = {
                    "signals": len(subset),
                    "win_rate": round(sess_wins / len(subset) * 100, 1)
                }
        
        return {
            "scan_type": scan_type,
            "direction": direction,
            "tp_sl": f"{tp_points}/{sl_points}",
            "filters_applied": filters,
            "total_signals": total,
            "wins": len(wins),
            "losses": len(losses),
            "timeouts": len(timeouts),
            "win_rate": round(win_rate, 1),
            "expected_value_per_trade": round(ev, 2),
            "profitable": ev > 0,
            "session_breakdown": session_stats,
            "avg_bars_held": round(rdf['bars_held'].mean(), 1)
        }



```

### src/types/viz.ts

```typescript
// Continuous contract data for base chart
export interface BarData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface ContinuousData {
  timeframe: string;
  count: number;
  bars: BarData[];
}

export interface VizWindow {
  x_price_1m: number[][]; // [open, high, low, close, volume]
  x_price_5m: number[][];
  x_price_15m: number[][];
  x_context: number[];
  norm_method: string;
  norm_params: Record<string, any>;
}

export interface VizOCO {
  entry_price: number;
  stop_price: number;
  tp_price: number;
  entry_type: string;
  direction: string;
  reference_type: string;
  reference_value: number;
  atr_at_creation: number;
  max_bars: number;
  stop_atr: number;
  tp_multiple: number;
}

export interface VizDecision {
  decision_id: string;
  timestamp: string | null;
  bar_idx: number;
  index: number;
  scanner_id: string;
  scanner_context: Record<string, any>;
  action: string;
  skip_reason: string;
  current_price: number;
  atr: number;
  cf_outcome: string;
  cf_pnl_dollars: number;
  window?: VizWindow | null;
  oco?: VizOCO | null;
  oco_results?: Record<string, { outcome?: string; pnl_dollars?: number; bars_held?: number; exit_price?: number }>;
  contracts?: number;
  risk_dollars?: number;
  reward_dollars?: number;
}

export interface VizFill {
  order_id: string;
  fill_type: string;
  price: number;
  bar_idx: number;
  timestamp?: string | null;
}

export interface VizTrade {
  trade_id: string;
  decision_id: string;
  index: number;
  direction: string;
  size: number;
  entry_time: string | null;
  entry_bar: number;
  entry_price: number;
  exit_time: string | null;
  exit_bar: number;
  exit_price: number;
  exit_reason: string;
  outcome: string;
  pnl_points: number;
  pnl_dollars: number;
  r_multiple: number;
  bars_held: number;
  mae: number;
  mfe: number;
  fills: VizFill[];
}

export interface RunManifest {
  run_id: string;
  start_time: string;
  config: any;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export type UIActionType = 'SET_INDEX' | 'SET_FILTER' | 'SET_MODE' | 'LOAD_RUN' | 'RUN_STRATEGY' | 'START_REPLAY' | 'TRAIN_FROM_SCAN';

export interface UIAction {
  type: UIActionType;
  payload: any;
}

export interface AgentResponse {
  reply: string;
  ui_action?: UIAction;
}
```

### src/viz/__init__.py

```python
"""
Viz Package
Export pipelines for React UI visualization.
"""

```

### src/viz/config.py

```python
"""
Viz Config
Configuration for visualization export.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VizConfig:
    """Configuration for viz export."""
    
    # What to include
    include_full_series: bool = False  # Full OHLCV for overview mode
    include_windows: bool = True       # x_price windows at decision time
    include_model_outputs: bool = True # Logits/probabilities
    
    # Window settings
    window_lookback_1m: int = 120
    window_lookback_5m: int = 24
    window_lookback_15m: int = 8
    window_lookback_1h: int = 24   # 24 hours of 1h bars
    window_lookback_4h: int = 12   # 48 hours of 4h bars
    
    # Output format
    output_format: str = "jsonl"  # 'json' or 'jsonl'
    compress: bool = False
    
    def to_dict(self) -> dict:
        return {
            'include_full_series': self.include_full_series,
            'include_windows': self.include_windows,
            'include_model_outputs': self.include_model_outputs,
            'window_lookback_1m': self.window_lookback_1m,
            'window_lookback_5m': self.window_lookback_5m,
            'window_lookback_15m': self.window_lookback_15m,
            'window_lookback_1h': self.window_lookback_1h,
            'window_lookback_4h': self.window_lookback_4h,
            'output_format': self.output_format,
            'compress': self.compress,
        }

```

### src/viz/export.py

```python
"""
Viz Export
Exporter class that collects events during simulation and writes artifacts.
"""

import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from src.viz.schema import (
    VizRun, VizSplit, VizDecision, VizTrade, VizOCO, VizFill, VizWindow, VizBarSeries
)
from src.viz.config import VizConfig
from src.datasets.decision_record import DecisionRecord
from src.datasets.trade_record import TradeRecord
from src.sim.oco_engine import OCOBracket
from src.features.pipeline import FeatureBundle


class Exporter:
    """
    Collects events during backtest/simulation for viz export.
    
    Usage:
        exporter = Exporter(config, run_id="my_run")
        # During simulation:
        exporter.on_decision(decision, features)
        exporter.on_bracket_created(decision_id, bracket)
        exporter.on_order_fill(decision_id, fill_type, price, bar_idx, timestamp)
        exporter.on_trade_closed(trade)
        # At end:
        exporter.finalize(out_dir)
    """
    
    def __init__(
        self,
        config: VizConfig,
        run_id: Optional[str] = None,
        experiment_config: Optional[Dict[str, Any]] = None
    ):
        self.config = config
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.experiment_config = experiment_config or {}
        
        # Storage
        self.decisions: List[VizDecision] = []
        self.trades: List[VizTrade] = []
        self.splits: List[VizSplit] = []
        self.full_series: Optional[VizBarSeries] = None
        
        # Tracking
        self._decision_idx = 0
        self._trade_idx = 0
        self._current_split_id: Optional[str] = None
        
        # Temp storage for linking
        self._pending_ocos: Dict[str, VizOCO] = {}  # decision_id -> oco
        self._pending_fills: Dict[str, List[VizFill]] = {}  # decision_id -> fills
        
        # Window bounds tracking (for 2-hour policy enforcement)
        self._window_warnings: List[str] = []
    
    def set_split(self, split_id: str, split_idx: int, train_start: str, train_end: str, test_start: str, test_end: str):
        """Start a new split."""
        self._current_split_id = split_id
        self.splits.append(VizSplit(
            split_id=split_id,
            split_idx=split_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        ))
    
    def set_full_series(self, df, timeframe: str = "1m"):
        """
        Set the full OHLCV series for global timeline view.
        
        Args:
            df: DataFrame with time, open, high, low, close, volume columns
            timeframe: Timeframe string (e.g., "1m")
