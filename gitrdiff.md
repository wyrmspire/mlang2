# Git Diff Report

**Generated**: Sun, Dec 28, 2025  2:32:37 AM

**Local Branch**: master

**Comparing Against**: origin/master

---

## Uncommitted Changes (working directory)

### Modified/Staged Files

```
 M agents.md
 M src/tools/price_analysis_tools.py
?? gitrdiff.md
```

### Uncommitted Diff

```diff
diff --git a/agents.md b/agents.md
index 8136b67..1864f04 100644
--- a/agents.md
+++ b/agents.md
@@ -37,6 +37,14 @@ This prevents Jules from “optimizing” the wrong things.
    - `find_price_opportunities` - Find clean swing trades from raw OHLCV
    - `describe_price_action` - Narrative of price behavior
    - `propose_trade` - Entry/stop/target from structure
+   - `study_obvious_trades` - Complete "obvious winners" workflow
+   - `cluster_trades` - Group by time of day, session, day of week
+   - `compare_trade_pools` - Morning vs afternoon comparisons
+   - `detect_regime` - TREND_UP/DOWN, RANGE, SPIKE_CHANNEL
+   - `trade_fingerprint` - State vector for pattern matching
+   - `indicator_impact` - "Would VWAP filter help?"
+   - `find_killer_moves` - Biggest opportunities in a range
+   - `synthesize_scan` - Auto-generate scanner spec from trades
 
 ### Workflow for "Find Opportunities" Requests
 1. `describe_price_action` for wide date range (e.g., full month)
@@ -45,6 +53,11 @@ This prevents Jules from “optimizing” the wrong things.
 4. Present narrative: "Price did X, cleanest trades were Y"
 5. **Optionally** correlate with scanners if relevant
 
+### Workflow for "Compare X vs Y" Requests
+1. `cluster_trades` to group by the relevant dimension
+2. `compare_trade_pools` for structured comparison
+3. Present insights with winner and reason
+
 ### Never Block Analysis
 If asked about trading opportunities, you MUST provide analysis. Fallback chain:
 1. Try raw price analysis
diff --git a/src/tools/price_analysis_tools.py b/src/tools/price_analysis_tools.py
index 24fa0d0..9fe7b8c 100644
--- a/src/tools/price_analysis_tools.py
+++ b/src/tools/price_analysis_tools.py
@@ -624,3 +624,598 @@ class StudyObviousTradesTool:
             return "No dominant pattern detected - trades were distributed across various contexts"
         
         return " | ".join(insights)
+
+
+# =============================================================================
+# Priority 1: Core Analysis Tools
+# =============================================================================
+
+@ToolRegistry.register(
+    tool_id="cluster_trades",
+    category=ToolCategory.UTILITY,
+    name="Cluster Trades",
+    description="Group trades by time of day, session, volatility state, or VWAP relation. Enables 'morning vs afternoon' comparisons.",
+    input_schema={
+        "type": "object",
+        "properties": {
+            "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
+            "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
+            "cluster_by": {
+                "type": "string",
+                "enum": ["time_of_day", "session", "day_of_week"],
+                "default": "time_of_day"
+            },
+            "min_move_atr": {"type": "number", "default": 2.0}
+        },
+        "required": ["start_date", "end_date"]
+    }
+)
+class TradeClusterTool:
+    """Group trades by various dimensions."""
+    
+    def execute(self, **inputs) -> Dict[str, Any]:
+        from collections import defaultdict
+        
+        start_date = inputs.get("start_date")
+        end_date = inputs.get("end_date")
+        cluster_by = inputs.get("cluster_by", "time_of_day")
+        min_move_atr = inputs.get("min_move_atr", 2.0)
+        
+        # Get all opportunities
+        finder = FindPriceOpportunitiesTool()
+        result = finder.execute(
+            start_date=start_date,
+            end_date=end_date,
+            direction="BOTH",
+            min_move_atr=min_move_atr,
+            timeframe="5m"
+        )
+        
+        if "error" in result:
+            return result
+        
+        all_opps = result.get("top_opportunities", [])
+        
+        # Cluster
+        clusters = defaultdict(list)
+        
+        for opp in all_opps:
+            ts = pd.to_datetime(opp["timestamp"])
+            
+            if cluster_by == "time_of_day":
+                hour = ts.hour
+                if 9 <= hour < 12:
+                    key = "MORNING (9:30-12)"
+                elif 12 <= hour < 14:
+                    key = "MIDDAY (12-14)"
+                elif 14 <= hour < 16:
+                    key = "AFTERNOON (14-16)"
+                else:
+                    key = "GLOBEX"
+            elif cluster_by == "session":
+                hour = ts.hour
+                key = "RTH" if 9 <= hour < 16 else "GLOBEX"
+            elif cluster_by == "day_of_week":
+                key = ts.strftime("%A")
+            else:
+                key = "ALL"
+            
+            clusters[key].append(opp)
+        
+        # Aggregate stats
+        cluster_stats = []
+        for name, trades in clusters.items():
+            if not trades:
+                continue
+            avg_mfe = sum(t["mfe"] for t in trades) / len(trades)
+            avg_mae = sum(abs(t["mae"]) for t in trades) / len(trades)
+            clean_pct = sum(1 for t in trades if t["quality"] == "CLEAN") / len(trades) * 100
+            long_pct = sum(1 for t in trades if t["direction"] == "LONG") / len(trades) * 100
+            
+            cluster_stats.append({
+                "cluster": name,
+                "count": len(trades),
+                "avg_mfe": round(avg_mfe, 2),
+                "avg_mae": round(avg_mae, 2),
+                "mfe_mae_ratio": round(avg_mfe / max(avg_mae, 0.1), 1),
+                "clean_pct": round(clean_pct, 1),
+                "long_pct": round(long_pct, 1)
+            })
+        
+        cluster_stats.sort(key=lambda x: x["mfe_mae_ratio"], reverse=True)
+        
+        return {
+            "date_range": f"{start_date} to {end_date}",
+            "cluster_by": cluster_by,
+            "total_trades": len(all_opps),
+            "clusters": cluster_stats,
+            "best_cluster": cluster_stats[0]["cluster"] if cluster_stats else None
+        }
+
+
+@ToolRegistry.register(
+    tool_id="compare_trade_pools",
+    category=ToolCategory.UTILITY,
+    name="Compare Trade Pools",
+    description="Compare two clusters of trades and output structured differences in MFE, MAE, win rate.",
+    input_schema={
+        "type": "object",
+        "properties": {
+            "start_date": {"type": "string"},
+            "end_date": {"type": "string"},
+            "pool_a": {"type": "string", "description": "First pool name (e.g., 'MORNING')"},
+            "pool_b": {"type": "string", "description": "Second pool name (e.g., 'AFTERNOON')"},
+            "cluster_by": {"type": "string", "default": "time_of_day"}
+        },
+        "required": ["start_date", "end_date", "pool_a", "pool_b"]
+    }
+)
+class TradeBehaviorCompareTool:
+    """Compare two trade pools."""
+    
+    def execute(self, **inputs) -> Dict[str, Any]:
+        start_date = inputs.get("start_date")
+        end_date = inputs.get("end_date")
+        pool_a = inputs.get("pool_a")
+        pool_b = inputs.get("pool_b")
+        cluster_by = inputs.get("cluster_by", "time_of_day")
+        
+        # Get clusters
+        cluster_tool = TradeClusterTool()
+        result = cluster_tool.execute(
+            start_date=start_date,
+            end_date=end_date,
+            cluster_by=cluster_by
+        )
+        
+        if "error" in result:
+            return result
+        
+        clusters = {c["cluster"]: c for c in result.get("clusters", [])}
+        
+        if pool_a not in clusters and pool_b not in clusters:
+            return {"error": f"Neither {pool_a} nor {pool_b} found in clusters"}
+        
+        a = clusters.get(pool_a, {"count": 0, "avg_mfe": 0, "avg_mae": 0, "mfe_mae_ratio": 0})
+        b = clusters.get(pool_b, {"count": 0, "avg_mfe": 0, "avg_mae": 0, "mfe_mae_ratio": 0})
+        
+        return {
+            "pool_a": {"name": pool_a, **a},
+            "pool_b": {"name": pool_b, **b},
+            "comparison": {
+                "count_delta": a.get("count", 0) - b.get("count", 0),
+                "mfe_delta": round(a.get("avg_mfe", 0) - b.get("avg_mfe", 0), 2),
+                "mae_delta": round(a.get("avg_mae", 0) - b.get("avg_mae", 0), 2),
+                "ratio_delta": round(a.get("mfe_mae_ratio", 0) - b.get("mfe_mae_ratio", 0), 1)
+            },
+            "winner": pool_a if a.get("mfe_mae_ratio", 0) > b.get("mfe_mae_ratio", 0) else pool_b,
+            "insight": self._generate_insight(pool_a, pool_b, a, b)
+        }
+    
+    def _generate_insight(self, name_a, name_b, a, b) -> str:
+        ratio_a = a.get("mfe_mae_ratio", 0)
+        ratio_b = b.get("mfe_mae_ratio", 0)
+        
+        if ratio_a > ratio_b * 1.5:
+            return f"{name_a} significantly outperforms {name_b} ({ratio_a}x vs {ratio_b}x MFE/MAE)"
+        elif ratio_b > ratio_a * 1.5:
+            return f"{name_b} significantly outperforms {name_a} ({ratio_b}x vs {ratio_a}x MFE/MAE)"
+        else:
+            return f"{name_a} and {name_b} have similar performance ({ratio_a}x vs {ratio_b}x MFE/MAE)"
+
+
+@ToolRegistry.register(
+    tool_id="detect_regime",
+    category=ToolCategory.UTILITY,
+    name="Detect Market Regime",
+    description="Identify if a day was TREND_UP, TREND_DOWN, RANGE, or SPIKE_CHANNEL.",
+    input_schema={
+        "type": "object",
+        "properties": {
+            "date": {"type": "string", "description": "Date YYYY-MM-DD to analyze"}
+        },
+        "required": ["date"]
+    }
+)
+class RegimeDetectionTool:
+    """Detect market regime for a day."""
+    
+    def execute(self, **inputs) -> Dict[str, Any]:
+        date = inputs.get("date")
+        
+        df = load_continuous_contract(start_date=date, end_date=date)
+        if df.empty:
+            return {"error": f"No data for {date}"}
+        
+        # Basic stats
+        open_price = float(df['open'].iloc[0])
+        close_price = float(df['close'].iloc[-1])
+        high = float(df['high'].max())
+        low = float(df['low'].min())
+        
+        net_change = close_price - open_price
+        total_range = high - low
+        
+        # Calculate ATR (need previous data for context)
+        prev_date = (pd.to_datetime(date) - timedelta(days=7)).strftime('%Y-%m-%d')
+        df_context = load_continuous_contract(start_date=prev_date, end_date=date)
+        
+        if len(df_context) > 14:
+            df_context['tr'] = np.maximum(
+                df_context['high'] - df_context['low'],
+                np.maximum(
+                    abs(df_context['high'] - df_context['close'].shift(1)),
+                    abs(df_context['low'] - df_context['close'].shift(1))
+                )
+            )
+            avg_atr = df_context['tr'].rolling(14).mean().iloc[-1]
+        else:
+            avg_atr = total_range
+        
+        # Regime detection
+        change_pct = abs(net_change / open_price) * 100
+        range_vs_avg = total_range / max(avg_atr, 0.1)
+        
+        if change_pct > 0.75 and net_change > 0:
+            regime = "TREND_UP"
+            confidence = min(change_pct / 1.5, 1.0)
+        elif change_pct > 0.75 and net_change < 0:
+            regime = "TREND_DOWN"
+            confidence = min(change_pct / 1.5, 1.0)
+        elif range_vs_avg > 1.5 and change_pct < 0.3:
+            regime = "SPIKE_CHANNEL"
+            confidence = min(range_vs_avg / 2, 1.0)
+        else:
+            regime = "RANGE"
+            confidence = 1 - min(change_pct / 1.5, 0.8)
+        
+        return {
+            "date": date,
+            "regime": regime,
+            "confidence": round(confidence, 2),
+            "open": open_price,
+            "close": close_price,
+            "high": high,
+            "low": low,
+            "net_change": round(net_change, 2),
+            "total_range": round(total_range, 2),
+            "range_vs_avg_atr": round(range_vs_avg, 2),
+            "recommendation": self._get_recommendation(regime)
+        }
+    
+    def _get_recommendation(self, regime: str) -> str:
+        recs = {
+            "TREND_UP": "Favor longs, use trailing stops, avoid counter-trend shorts",
+            "TREND_DOWN": "Favor shorts, use trailing stops, avoid counter-trend longs",
+            "RANGE": "Use mean reversion, tighter targets, avoid breakout entries",
+            "SPIKE_CHANNEL": "Wait for retest of spike levels, careful with stops"
+        }
+        return recs.get(regime, "Unknown regime")
+
+
+# =============================================================================
+# Priority 2: Trade Optimization Tools
+# =============================================================================
+
+@ToolRegistry.register(
+    tool_id="trade_fingerprint",
+    category=ToolCategory.UTILITY,
+    name="Trade Fingerprint",
+    description="Build a state vector for a trade timestamp: PDH/PDL distance, VWAP position, OR context, volatility percentile.",
+    input_schema={
+        "type": "object",
+        "properties": {
+            "timestamp": {"type": "string", "description": "ISO timestamp"}
+        },
+        "required": ["timestamp"]
+    }
+)
+class TradeFingerprintTool:
+    """Build a fingerprint for pattern matching."""
+    
+    def execute(self, **inputs) -> Dict[str, Any]:
+        from src.tools.exploration_tools import GetSessionContextTool
+        
+        timestamp = inputs.get("timestamp")
+        
+        # Get session context
+        ctx_tool = GetSessionContextTool()
+        ctx = ctx_tool.execute(timestamp=timestamp)
+        
+        if "error" in ctx:
+            return ctx
+        
+        # Calculate additional metrics
+        ts = pd.to_datetime(timestamp)
+        if ts.tzinfo is None:
+            ts = ts.tz_localize(NY_TZ)
+        
+        start = (ts - timedelta(days=5)).strftime('%Y-%m-%d')
+        end = (ts + timedelta(days=1)).strftime('%Y-%m-%d')
+        df = load_continuous_contract(start_date=start, end_date=end)
+        
+        if df.empty:
+            return {"error": "No data"}
+        
+        # Current price
+        current_price = ctx.get("current_price", 0)
+        pdh = ctx.get("pdh", 0)
+        pdl = ctx.get("pdl", 0)
+        orh = ctx.get("orh", 0)
+        orl = ctx.get("orl", 0)
+        vwap = ctx.get("vwap", current_price)
+        
+        # ATR percentile
+        df['tr'] = df['high'] - df['low']
+        atr_series = df['tr'].rolling(14).mean()
+        current_atr = atr_series.iloc[-1] if len(atr_series) > 0 else 2.0
+        atr_percentile = (atr_series < current_atr).sum() / max(len(atr_series), 1) * 100
+        
+        # Volume Z-score (last bar vs rolling mean)
+        vol_mean = df['volume'].rolling(50).mean().iloc[-1]
+        vol_std = df['volume'].rolling(50).std().iloc[-1]
+        last_vol = df['volume'].iloc[-1]
+        vol_z = (last_vol - vol_mean) / max(vol_std, 1)
+        
+        return {
+            "timestamp": timestamp,
+            "fingerprint": {
+                "pdh_distance": round((current_price - pdh) / max(current_atr, 0.1), 2),
+                "pdl_distance": round((current_price - pdl) / max(current_atr, 0.1), 2),
+                "vwap_distance": round((current_price - vwap) / max(current_atr, 0.1), 2),
+                "or_position": "INSIDE" if orl <= current_price <= orh else "ABOVE" if current_price > orh else "BELOW",
+                "atr_percentile": round(atr_percentile, 1),
+                "volume_z": round(vol_z, 2),
+                "session": ctx.get("session"),
+                "is_rth": ctx.get("is_rth")
+            }
+        }
+
+
+@ToolRegistry.register(
+    tool_id="indicator_impact",
+    category=ToolCategory.UTILITY,
+    name="Indicator Impact Analysis",
+    description="Would adding an RSI or VWAP filter have improved results? Test filter impact on a pool of trades.",
+    input_schema={
+        "type": "object",
+        "properties": {
+            "start_date": {"type": "string"},
+            "end_date": {"type": "string"},
+            "indicator": {"type": "string", "enum": ["rsi", "vwap", "ema"]},
+            "threshold": {"type": "number", "description": "Filter threshold (e.g., RSI < 30 for longs)"}
+        },
+        "required": ["start_date", "end_date", "indicator"]
+    }
+)
+class IndicatorImpactTool:
+    """Analyze impact of adding an indicator filter."""
+    
+    def execute(self, **inputs) -> Dict[str, Any]:
+        start_date = inputs.get("start_date")
+        end_date = inputs.get("end_date")
+        indicator = inputs.get("indicator", "vwap")
+        threshold = inputs.get("threshold")
+        
+        # Get trades and analyze with/without filter
+        finder = FindPriceOpportunitiesTool()
+        result = finder.execute(
+            start_date=start_date,
+            end_date=end_date,
+            direction="BOTH",
+            min_move_atr=2.0
+        )
+        
+        if "error" in result:
+            return result
+        
+        all_trades = result.get("top_opportunities", [])
+        if not all_trades:
+            return {"error": "No trades to analyze"}
+        
+        # Get session context for VWAP filtering
+        from src.tools.exploration_tools import GetSessionContextTool
+        session_tool = GetSessionContextTool()
+        
+        kept = []
+        filtered = []
+        
+        for trade in all_trades:
+            ctx = session_tool.execute(timestamp=trade["timestamp"])
+            if "error" in ctx:
+                continue
+            
+            passes_filter = False
+            if indicator == "vwap":
+                if trade["direction"] == "LONG":
+                    passes_filter = ctx.get("price_vs_vwap") == "BELOW"
+                else:
+                    passes_filter = ctx.get("price_vs_vwap") == "ABOVE"
+            elif indicator == "rsi":
+                # Would need RSI calculation - simplified
+                passes_filter = True  # Placeholder
+            elif indicator == "ema":
+                # Would need EMA calculation - simplified
+                passes_filter = True  # Placeholder
+            
+            if passes_filter:
+                kept.append(trade)
+            else:
+                filtered.append(trade)
+        
+        # Compare stats
+        def calc_stats(trades):
+            if not trades:
+                return {"count": 0, "avg_mfe": 0, "avg_mae": 0}
+            return {
+                "count": len(trades),
+                "avg_mfe": round(sum(t["mfe"] for t in trades) / len(trades), 2),
+                "avg_mae": round(sum(abs(t["mae"]) for t in trades) / len(trades), 2),
+                "clean_pct": round(sum(1 for t in trades if t["quality"] == "CLEAN") / len(trades) * 100, 1)
+            }
+        
+        before = calc_stats(all_trades)
+        after = calc_stats(kept)
+        removed = calc_stats(filtered)
+        
+        return {
+            "indicator": indicator,
+            "before_filter": before,
+            "after_filter": after,
+            "removed_trades": removed,
+            "filter_impact": {
+                "trades_removed": len(filtered),
+                "mfe_improvement": round(after["avg_mfe"] - before["avg_mfe"], 2) if after["count"] else 0,
+                "mae_reduction": round(before["avg_mae"] - after["avg_mae"], 2) if after["count"] else 0
+            },
+            "recommendation": "ADD" if after.get("clean_pct", 0) > before.get("clean_pct", 0) + 5 else "SKIP"
+        }
+
+
+# =============================================================================
+# Priority 3: Pattern Discovery Tools
+# =============================================================================
+
+@ToolRegistry.register(
+    tool_id="find_killer_moves",
+    category=ToolCategory.UTILITY,
+    name="Find Killer Moves",
+    description="Find the biggest, cleanest price moves in a date range - the opportunities you'd hate to miss.",
+    input_schema={
+        "type": "object",
+        "properties": {
+            "start_date": {"type": "string"},
+            "end_date": {"type": "string"},
+            "top_n": {"type": "integer", "default": 5}
+        },
+        "required": ["start_date", "end_date"]
+    }
+)
+class KillerMoveDetectorTool:
+    """Find the biggest opportunities."""
+    
+    def execute(self, **inputs) -> Dict[str, Any]:
+        start_date = inputs.get("start_date")
+        end_date = inputs.get("end_date")
+        top_n = inputs.get("top_n", 5)
+        
+        df = load_continuous_contract(start_date=start_date, end_date=end_date)
+        if df.empty:
+            return {"error": "No data"}
+        
+        # Resample to 5m
+        df = df.set_index('time')
+        df = df.resample('5min').agg({
+            'open': 'first',
+            'high': 'max',
+            'low': 'min',
+            'close': 'last',
+            'volume': 'sum'
+        }).dropna().reset_index()
+        
+        # Find big moves (20-bar windows)
+        moves = []
+        for i in range(len(df) - 20):
+            window = df.iloc[i:i+20]
+            start_price = window['open'].iloc[0]
+            max_up = window['high'].max() - start_price
+            max_down = start_price - window['low'].min()
+            
+            if max_up > max_down:
+                direction = "LONG"
+                move_size = max_up
+                entry = float(window['open'].iloc[0])
+                target = float(window['high'].max())
+            else:
+                direction = "SHORT"
+                move_size = max_down
+                entry = float(window['open'].iloc[0])
+                target = float(window['low'].min())
+            
+            moves.append({
+                "timestamp": window['time'].iloc[0].isoformat(),
+                "direction": direction,
+                "entry_price": round(entry, 2),
+                "best_exit": round(target, 2),
+                "points": round(move_size, 2),
+                "duration_bars": 20
+            })
+        
+        # Sort by move size
+        moves.sort(key=lambda x: x["points"], reverse=True)
+        
+        return {
+            "date_range": f"{start_date} to {end_date}",
+            "killer_moves": moves[:top_n],
+            "insight": f"Top move was {moves[0]['points']} points {moves[0]['direction']} on {moves[0]['timestamp'][:10]}" if moves else "No significant moves found"
+        }
+
+
+@ToolRegistry.register(
+    tool_id="synthesize_scan",
+    category=ToolCategory.UTILITY,
+    name="Synthesize Scanner",
+    description="Given a pool of good trades, auto-generate a candidate scanner spec based on common patterns.",
+    input_schema={
+        "type": "object",
+        "properties": {
+            "start_date": {"type": "string"},
+            "end_date": {"type": "string"},
+            "min_mfe_atr": {"type": "number", "default": 3.0},
+            "max_mae_atr": {"type": "number", "default": 1.0}
+        },
+        "required": ["start_date", "end_date"]
+    }
+)
+class ScanSynthesizerTool:
+    """Auto-generate scanner from good trades."""
+    
+    def execute(self, **inputs) -> Dict[str, Any]:
+        start_date = inputs.get("start_date")
+        end_date = inputs.get("end_date")
+        min_mfe_atr = inputs.get("min_mfe_atr", 3.0)
+        max_mae_atr = inputs.get("max_mae_atr", 1.0)
+        
+        # Use study_obvious_trades as foundation
+        study_tool = StudyObviousTradesTool()
+        result = study_tool.execute(
+            start_date=start_date,
+            end_date=end_date,
+            direction="BOTH",
+            min_move_atr=min_mfe_atr,
+            top_n=20
+        )
+        
+        if "error" in result:
+            return result
+        
+        # Extract scan spec and enhance
+        base_spec = result.get("candidate_scan_spec", {})
+        
+        # Add OCO suggestions based on observed MFE/MAE
+        top_trades = result.get("top_trades", [])
+        if top_trades:
+            avg_mfe = sum(t["mfe"] for t in top_trades) / len(top_trades)
+            suggested_tp = round(avg_mfe * 0.6, 1)  # Target 60% of avg MFE
+            suggested_sl = round(max_mae_atr, 1)
+        else:
+            suggested_tp = 6.0
+            suggested_sl = 3.0
+        
+        enhanced_spec = {
+            **base_spec,
+            "oco_suggestion": {
+                "tp_points": suggested_tp,
+                "sl_points": suggested_sl,
+                "rr_ratio": round(suggested_tp / max(suggested_sl, 0.1), 1)
+            },
+            "confidence": "HIGH" if result.get("analyzed_count", 0) >= 10 else "MEDIUM",
+            "sample_size": result.get("analyzed_count", 0)
+        }
+        
+        return {
+            "synthesized_scan": enhanced_spec,
+            "key_insight": result.get("key_insight"),
+            "usage": "Feed this spec to explore_strategy for validation"
+        }
+
```

### New Untracked Files

#### `gitrdiff.md` (661 lines - truncated)

```
# Git Diff Report

**Generated**: Sun, Dec 28, 2025  2:32:37 AM

**Local Branch**: master

**Comparing Against**: origin/master

---

## Uncommitted Changes (working directory)

### Modified/Staged Files

```
 M agents.md
 M src/tools/price_analysis_tools.py
?? gitrdiff.md
```

### Uncommitted Diff

```diff
diff --git a/agents.md b/agents.md
index 8136b67..1864f04 100644
--- a/agents.md
+++ b/agents.md
@@ -37,6 +37,14 @@ This prevents Jules from “optimizing” the wrong things.
    - `find_price_opportunities` - Find clean swing trades from raw OHLCV
    - `describe_price_action` - Narrative of price behavior
    - `propose_trade` - Entry/stop/target from structure
+   - `study_obvious_trades` - Complete "obvious winners" workflow
+   - `cluster_trades` - Group by time of day, session, day of week
+   - `compare_trade_pools` - Morning vs afternoon comparisons
+   - `detect_regime` - TREND_UP/DOWN, RANGE, SPIKE_CHANNEL
+   - `trade_fingerprint` - State vector for pattern matching
+   - `indicator_impact` - "Would VWAP filter help?"
+   - `find_killer_moves` - Biggest opportunities in a range
+   - `synthesize_scan` - Auto-generate scanner spec from trades
 
 ### Workflow for "Find Opportunities" Requests
 1. `describe_price_action` for wide date range (e.g., full month)
@@ -45,6 +53,11 @@ This prevents Jules from “optimizing” the wrong things.
 4. Present narrative: "Price did X, cleanest trades were Y"
 5. **Optionally** correlate with scanners if relevant
 
+### Workflow for "Compare X vs Y" Requests
+1. `cluster_trades` to group by the relevant dimension
+2. `compare_trade_pools` for structured comparison
+3. Present insights with winner and reason
+
 ### Never Block Analysis
 If asked about trading opportunities, you MUST provide analysis. Fallback chain:
 1. Try raw price analysis
diff --git a/src/tools/price_analysis_tools.py b/src/tools/price_analysis_tools.py
index 24fa0d0..9fe7b8c 100644
--- a/src/tools/price_analysis_tools.py
+++ b/src/tools/price_analysis_tools.py
@@ -624,3 +624,598 @@ class StudyObviousTradesTool:
             return "No dominant pattern detected - trades were distributed across various contexts"
         
         return " | ".join(insights)
+
+
+# =============================================================================
+# Priority 1: Core Analysis Tools
+# =============================================================================
+
+@ToolRegistry.register(
+    tool_id="cluster_trades",
+    category=ToolCategory.UTILITY,
+    name="Cluster Trades",
+    description="Group trades by time of day, session, volatility state, or VWAP relation. Enables 'morning vs afternoon' comparisons.",
+    input_schema={
+        "type": "object",
+        "properties": {
+            "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
+            "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
+            "cluster_by": {
+                "type": "string",
+                "enum": ["time_of_day", "session", "day_of_week"],
+                "default": "time_of_day"
+            },
+            "min_move_atr": {"type": "number", "default": 2.0}
+        },
+        "required": ["start_date", "end_date"]
+    }
+)
+class TradeClusterTool:
+    """Group trades by various dimensions."""
+    
+    def execute(self, **inputs) -> Dict[str, Any]:
+        from collections import defaultdict
+        
+        start_date = inputs.get("start_date")
+        end_date = inputs.get("end_date")
+        cluster_by = inputs.get("cluster_by", "time_of_day")
+        min_move_atr = inputs.get("min_move_atr", 2.0)
+        
+        # Get all opportunities
... (12 total lines)
```

---

## Commits Ahead (local changes not on remote)

```
```

## Commits Behind (remote changes not pulled)

```
```

---

## File Changes (YOUR UNPUSHED CHANGES)

```
```

---

## Full Diff of Your Unpushed Changes

Green (+) = lines you ADDED locally
Red (-) = lines you REMOVED locally

```diff
```
