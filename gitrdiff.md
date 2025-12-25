# Git Diff Report

**Generated**: Wed, Dec 24, 2025  9:14:38 PM

**Local Branch**: master

**Comparing Against**: origin/master

---

## Uncommitted Changes (working directory)

### Modified/Staged Files

```
 M src/components/CandleChart.tsx
 M src/strategy/scan.py
 M src/viz/export.py
 M src/viz/schema.py
?? gitrdiff.md
?? problems_doing_maths.md
```

### Uncommitted Diff

```diff
diff --git a/src/components/CandleChart.tsx b/src/components/CandleChart.tsx
index 0c5c0c5..106f788 100644
--- a/src/components/CandleChart.tsx
+++ b/src/components/CandleChart.tsx
@@ -293,23 +293,33 @@ export const CandleChart: React.FC<CandleChartProps> = ({
             const oco = decision.oco;
             if (!oco?.entry_price || !oco?.stop_price || !oco?.tp_price) return;
 
-            // Snap start time to current timeframe interval
-            const rawStartIdx = findBarIndex(continuousData.bars, decision.timestamp);
-            const snappedStartIdx = Math.floor(rawStartIdx / interval);
-            const snappedStartBar = aggregatedBars[Math.min(snappedStartIdx, aggregatedBars.length - 1)];
-            if (!snappedStartBar) return;
-            const startTime = parseTime(snappedStartBar.time) as Time;
+            // ========================================
+            // FIX: Use decision's timestamp directly, not continuousData lookup
+            // This ensures alignment even when continuousData doesn't include premkt
+            // ========================================
+            const startTime = parseTime(decision.timestamp) as Time;
 
             // Calculate end time using bars_held from oco_results if available
-            const ocoResults = decision.oco_results || {};
-            const bestOco = Object.values(ocoResults)[0] as { bars_held?: number } | undefined;
-            const barsHeld = bestOco?.bars_held || 30; // Fallback to 30 mins if not available
-
-            // Convert bars_held (which is in 1m bars) to current timeframe bars
-            const barsInTimeframe = Math.max(1, Math.ceil(barsHeld / interval));
-            const endIdx = Math.min(snappedStartIdx + barsInTimeframe, aggregatedBars.length - 1);
-            const endBar = aggregatedBars[endIdx];
-            const endTime = endBar ? parseTime(endBar.time) as Time : startTime;
+            // Support both flat format (ifvg_debug) and nested format (older scans)
+            const ocoResults = decision.oco_results as any || {};
+            let barsHeld = 30; // Fallback
+
+            if (typeof ocoResults.bars_held === 'number') {
+                // Flat format (ifvg_debug, new scans)
+                barsHeld = ocoResults.bars_held;
+            } else if (typeof ocoResults === 'object') {
+                // Nested format - check first value
+                const firstVal = Object.values(ocoResults)[0];
+                if (firstVal && typeof firstVal === 'object' && 'bars_held' in (firstVal as object)) {
+                    barsHeld = (firstVal as { bars_held: number }).bars_held;
+                }
+            }
+
+            // Compute endTime by adding bars_held minutes to startTime
+            // bars_held is in 1-minute bars, so add that many minutes
+            const startDate = new Date(decision.timestamp);
+            const endDate = new Date(startDate.getTime() + barsHeld * 60 * 1000);
+            const endTime = parseTime(endDate.toISOString()) as Time;
 
             const direction = (decision.scanner_context?.direction || oco.direction || 'LONG') as 'LONG' | 'SHORT';
 
diff --git a/src/strategy/scan.py b/src/strategy/scan.py
index 7c2d959..4df7075 100644
--- a/src/strategy/scan.py
+++ b/src/strategy/scan.py
@@ -283,10 +283,14 @@ def run_strategy_scan(
         # Compute bracket levels
         levels = bracket.compute(entry_price, direction, atr_value)
         
+        # Find entry_idx in df_1m using time-based lookup (same pattern as raw_ohlcv)
+        cf_mask = df_1m['time'] <= bar_time
+        cf_entry_idx = cf_mask.sum() - 1 if cf_mask.any() else 0
+        
         # Compute counterfactual outcome
         cf = compute_smart_stop_counterfactual(
             df=df_1m,
-            entry_idx=bar_idx * (5 if timeframe == '5m' else 15 if timeframe == '15m' else 1),
+            entry_idx=cf_entry_idx,
             direction=direction,
             stop_price=levels.stop_price,
             tp_multiple=levels.r_multiple,
@@ -294,15 +298,28 @@ def run_strategy_scan(
             oco_name="strategy"
         )
         
-        # Get raw OHLCV window for chart - MUST include timestamps for arrow alignment
-        start_raw_idx = max(0, bar_idx - lookback_bars)
-        end_raw_idx = min(len(df_scan), bar_idx + lookahead_bars + 1)
-        raw_slice = df_scan.iloc[start_raw_idx:end_raw_idx]
+        # Get raw OHLCV window for chart - MUST match ifvg_debug pattern:
+        # 1. Use df_1m (not df_scan)
+        # 2. Time-based lookup centered on entry_time
+        # 3. 60 bars history, 120 bars future on 1m data
+        history_bars_1m = 60
+        future_bars_1m = 120
+        
+        # Find entry index in df_1m by time
+        mask = df_1m['time'] <= bar_time
+        if not mask.any():
+            entry_idx_1m = 0
+        else:
+            entry_idx_1m = mask.sum() - 1
+        
+        start_raw_idx = max(0, entry_idx_1m - history_bars_1m)
+        end_raw_idx = min(len(df_1m), entry_idx_1m + future_bars_1m)
+        raw_slice = df_1m.iloc[start_raw_idx:end_raw_idx]
         
         # Format as objects with timestamps (like ifvg_debug) - UI requires this
         raw_ohlcv = [
             {
-                "time": str(row['time']),
+                "time": row['time'].isoformat() if hasattr(row['time'], 'isoformat') else str(row['time']),
                 "open": float(row['open']),
                 "high": float(row['high']),
                 "low": float(row['low']),
@@ -312,11 +329,11 @@ def run_strategy_scan(
             for _, row in raw_slice.iterrows()
         ]
         
-        # Future bars (also with timestamps)
-        future_slice = df_scan.iloc[bar_idx+1:bar_idx+lookahead_bars+1]
+        # Future bars from 1m data (for counterfactual viz)
+        future_slice = df_1m.iloc[entry_idx_1m+1:entry_idx_1m+future_bars_1m+1]
         future_bars = [
             {
-                "time": str(row['time']),
+                "time": row['time'].isoformat() if hasattr(row['time'], 'isoformat') else str(row['time']),
                 "open": float(row['open']),
                 "high": float(row['high']),
                 "low": float(row['low']),
@@ -408,7 +425,7 @@ def run_strategy_scan(
             exit_reason=cf.outcome,
             outcome=cf.outcome,
             pnl_points=cf.pnl_dollars / 50,  # Approx conversion
-            pnl_dollars=cf.pnl_dollars,
+            pnl_dollars=cf.pnl_dollars * contracts,  # Scale by contracts for $300 risk
             r_multiple=cf.pnl_dollars / (levels.risk_points * 50) if levels.risk_points > 0 else 0,
             bars_held=int(cf.bars_held),
             mae=0,  # Would need to compute
diff --git a/src/viz/export.py b/src/viz/export.py
index e7616f9..1c29167 100644
--- a/src/viz/export.py
+++ b/src/viz/export.py
@@ -132,13 +132,14 @@ class Exporter:
             viz_decision.model_logits = model_logits
             viz_decision.model_probs = model_probs
         
-        # Add window data
-        if self.config.include_windows and features:
+        # Add window data - output even if features is sparse/dummy
+        # The key fields for UI are raw_ohlcv_1m and indicators
+        if self.config.include_windows and (raw_ohlcv or indicators):
             viz_decision.window = VizWindow(
-                x_price_1m=features.x_price_1m.tolist() if features.x_price_1m is not None else [],
-                x_price_5m=features.x_price_5m.tolist() if features.x_price_5m is not None else [],
-                x_price_15m=features.x_price_15m.tolist() if features.x_price_15m is not None else [],
-                x_context=features.x_context.tolist() if features.x_context is not None else [],
+                x_price_1m=features.x_price_1m.tolist() if features and features.x_price_1m is not None else [],
+                x_price_5m=features.x_price_5m.tolist() if features and features.x_price_5m is not None else [],
+                x_price_15m=features.x_price_15m.tolist() if features and features.x_price_15m is not None else [],
+                x_context=features.x_context.tolist() if features and features.x_context is not None else [],
                 raw_ohlcv_1m=raw_ohlcv or [],
                 future_price_1m=future_1m or [],
                 indicators=indicators or {}
@@ -229,16 +230,16 @@ class Exporter:
         
         # === CRITICAL: Update the matching decision with oco_results ===
         # This is required for position box duration in the UI
+        # Format MUST match ifvg_debug: flat dict, not nested
         for d in reversed(self.decisions):
             if d.decision_id == trade.decision_id:
                 d.oco_results = {
-                    "strategy": {
-                        "outcome": trade.outcome,
-                        "pnl_dollars": trade.pnl_dollars,
-                        "bars_held": trade.bars_held,
-                        "exit_price": trade.exit_price,
-                        "r_multiple": trade.r_multiple
-                    }
+                    "filled": True,
+                    "outcome": trade.outcome,
+                    "exit_price": trade.exit_price,
+                    "bars_held": trade.bars_held,
+                    "pnl_points": trade.pnl_points,
+                    "pnl_dollars": trade.pnl_dollars
                 }
                 break
         
diff --git a/src/viz/schema.py b/src/viz/schema.py
index 76397cc..f0dde0d 100644
--- a/src/viz/schema.py
+++ b/src/viz/schema.py
@@ -62,6 +62,7 @@ class VizOCO:
     tp_price: float = 0.0
     entry_type: str = "LIMIT"
     direction: str = "LONG"
+    contracts: int = 1  # Number of contracts for position sizing
     
     reference_type: str = "PRICE"
     reference_value: float = 0.0
@@ -79,6 +80,7 @@ class VizOCO:
             'tp_price': self.tp_price,
             'entry_type': self.entry_type,
             'direction': self.direction,
+            'contracts': self.contracts,
             'reference_type': self.reference_type,
             'reference_value': self.reference_value,
             'atr_at_creation': self.atr_at_creation,
```

### New Untracked Files

#### `gitrdiff.md`

```
```

#### `problems_doing_maths.md`

```
# Position Box / Output Regression Problems

## ✅ FIX APPLIED

**Root cause identified by user:** Chart's "render all trades" path used `continuousData.bars` for time lookup, but decision's `window.raw_ohlcv_1m` may include different time range (premkt/RTH mismatch).

**Fix in `CandleChart.tsx` (line 296-323):**
- Now uses `decision.timestamp` directly for `startTime` (not continuousData lookup)
- Computes `endTime` by adding `bars_held` minutes to startTime
- No longer depends on aggregatedBars containing the decision's time range

**Test by loading NEW_test_fixed4 in UI** - position boxes should now align correctly!

---

## Current Issues

### 1. Position Box Not Rendering Correctly
- ifvg_debug works fine, but new scans (NEW_test_fixed4, etc.) have position box issues
- User reports: "box shows to start at 9:30 but price is above the box"
- User reports: "second trade doesn't even have a box"
- User reports: "price gets cut off before it could even trigger"

### 2. $300 Risk Not Applied
- Scans should size positions for $300 max risk
- Current output shows low pnl_dollars values (-66.25) instead of scaled by contracts

---

## Files Modified This Session

| File | Changes |
|------|---------|
| `src/viz/schema.py` | Added `contracts: int = 1` to VizOCO |
| `src/viz/export.py` | Changed oco_results from nested format to flat format |
| `src/strategy/scan.py` | Fixed raw_ohlcv to use df_1m with time-based lookup, fixed counterfactual entry_idx |
| `src/components/CandleChart.tsx` | Fixed to handle flat oco_results.bars_held |

---

## What ifvg_debug Does Correctly

The working run is at `results/ifvg_debug/records.jsonl`:

```json
{
  "oco": {
    "entry_price": 5839.5,
    "contracts": 1,
    "stop_price": 5837.5,
    "tp_price": 5845.5,
    "direction": "LONG",
    "order_type": "LIMIT"
  },
  "oco_results": {
    "filled": true,
    "outcome": "WIN",
    "exit_price": 5845.5,
    "bars_held": 14,
    "pnl_points": 6.0,
    "pnl_dollars": 300.0
  },
  "window": {
    "raw_ohlcv_1m": [...]  // 180 bars centered on entry time
  }
}
```

Key points:
- `window.raw_ohlcv_1m` has 180 bars (60 before, 120 after entry)
- Timestamps use `.isoformat()` format
- `oco.contracts` is present
- `oco_results` is flat (not nested)

---

## What New Scans Output

```json
{
  "oco": {
    "entry_price": 6675.0,
    "stop_price": 6671.0,
    "tp_price": 6683.0,
    "contracts": 1,  // Now present after fix
    ...
  },
  "oco_results": {
    "filled": true,
    "outcome": "LOSS",
    "bars_held": 78,
    "pnl_dollars": -66.25
  }
}
```

---

## UI Code (CandleChart.tsx line ~303-315)

The UI reads `oco_results.bars_held` to compute position box duration:

```typescript
const ocoResults = decision.oco_results as any || {};
let barsHeld = 30; // Fallback

if (typeof ocoResults.bars_held === 'number') {
    barsHeld = ocoResults.bars_held;
} else if (typeof ocoResults === 'object') {
    const firstVal = Object.values(ocoResults)[0];
    if (firstVal && typeof firstVal === 'object' && 'bars_held' in (firstVal as object)) {
        barsHeld = (firstVal as { bars_held: number }).bars_held;
    }
}
```

---

## Verified Facts

1. Data file `NEW_test_fixed4/decisions.jsonl` has correct format:
   - `oco_results.bars_held: 78`
   - `timestamp: 2025-08-19T09:30:00-04:00`
   - `window.raw_ohlcv_1m` first bar: `2025-08-19T08:30:00-04:00` (1 hour before)

2. Counterfactual is CORRECT - stop at 6671 was not hit until 10:48 (78 mins after 09:30 entry)

---

## Possible Remaining Issues

1. Pre-market bars (08:30-09:30) might not be showing on chart
2. Chart timeframe mismatch (5m chart but 1m raw data?)
3. Something in the visualization pipeline not using the raw_ohlcv window correctly
4. Server might need full restart (not just Vite HMR)

---

## Test Runs Created

- `NEW_test_fixed` / `NEW_test_fixed2` / `NEW_test_fixed3` / `NEW_test_fixed4`
- Compare against working: `ifvg_debug`
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
