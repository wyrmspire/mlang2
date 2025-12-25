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
