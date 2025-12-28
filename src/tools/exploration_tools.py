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
    description="Run parameter sweeps WITHOUT generating TradeViz artifacts. Output goes to results/exploration/ only. Use this for brute-force optimization.",
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


