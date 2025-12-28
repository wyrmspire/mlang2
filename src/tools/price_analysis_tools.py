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


