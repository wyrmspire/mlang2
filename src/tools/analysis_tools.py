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
