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
