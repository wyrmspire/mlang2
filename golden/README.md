# Golden Runs - Regression Anchor

This directory contains **golden reference runs** that serve as regression anchors for the MLang2 platform.

## Purpose

Golden runs establish structural expectations that all future runs must meet:
- Correct file inventory (manifest.json, decisions.jsonl, trades.jsonl)
- Proper viz schema compliance (flat oco_results, contracts present)
- Valid timestamp formats (ISO 8601 with timezone)
- 2-hour context windows
- Consistent OCO execution results

## Golden Runs

### `ifvg_reference/` (Primary Reference)

**Strategy:** IFVG (Imbalance/Fair Value Gap) detection  
**Date Range:** 2025-03-18 to 2025-04-15 (4 weeks)  
**Characteristics:**
- ✅ Flat `oco_results` in decisions
- ✅ `contracts` field present in all OCO brackets
- ✅ ISO 8601 timestamps with timezone
- ✅ 2-hour pre/post trade context
- ✅ Complete file inventory

**Why this run?**
- Representative of pattern-based strategies
- Known-good structure
- Multiple trades for validation
- All viz schema requirements met

## Usage

### For Regression Testing

```python
# tests/test_golden_runs.py
from golden.validator import validate_run_structure

def test_new_run_matches_golden_structure():
    golden_path = Path("golden/ifvg_reference")
    new_run_path = Path("results/viz/my_new_run")
    
    issues = validate_run_structure(new_run_path, golden_path)
    assert len(issues) == 0, f"Structure violations: {issues}"
```

### For Development

When creating a new strategy or modifying the exporter:

1. Run your strategy
2. Validate against golden structure: `python golden/validate_run.py results/viz/your_run`
3. Fix any violations
4. Re-run until clean

## Validation Rules

The golden run validator checks:

1. **File Inventory**
   - [ ] manifest.json exists
   - [ ] decisions.jsonl exists
   - [ ] trades.jsonl exists
   - [ ] Files listed in manifest match actual files

2. **Manifest Structure**
   - [ ] Contains run_id, fingerprint, created_at
   - [ ] created_at is ISO 8601 with timezone
   - [ ] config is present and valid dict
   - [ ] file_inventory is present and valid list

3. **Decision Structure** (each decision in decisions.jsonl)
   - [ ] Contains decision_id, timestamp, bar_idx, action
   - [ ] timestamp is ISO 8601 with timezone
   - [ ] If action == "PLACE_ORDER", oco is present
   - [ ] If oco is present, oco_results is FLAT dict (not nested)
   - [ ] If oco is present, contracts field exists

4. **Trade Structure** (each trade in trades.jsonl)
   - [ ] Contains trade_id, decision_id
   - [ ] Contains entry_price, exit_price, pnl_dollars, outcome
   - [ ] Contains bars_held, entry_time, exit_time

5. **OCO Results Format**
   ```json
   // ✅ CORRECT (flat)
   {
     "decision_id": "abc",
     "oco_results": {
       "filled": true,
       "bars_held": 23,
       ...
     }
   }
   
   // ❌ WRONG (nested)
   {
     "decision_id": "abc",
     "oco": {
       "results": {...}
     }
   }
   ```

6. **Contracts Field**
   ```json
   // ✅ CORRECT
   {
     "oco": {
       "contracts": 2,
       "entry_price": 5000.0,
       ...
     }
   }
   
   // ❌ WRONG (missing)
   {
     "oco": {
       "entry_price": 5000.0,
       ...
     }
   }
   ```

## Creating a New Golden Run

If you need to establish a new golden run (e.g., for a new strategy type):

1. **Run the strategy** with known-good configuration
2. **Manually verify** the output meets all requirements
3. **Validate** with the golden run validator
4. **Copy** to `golden/your_strategy_name/`
5. **Document** in this README
6. **Add tests** in `tests/test_golden_runs.py`

Example:
```bash
# Run strategy
python scripts/backtest_ict_ifvg.py --start-date 2025-03-18 --weeks 4

# Validate
python golden/validate_run.py results/viz/ict_ifvg_20250318

# If clean, copy to golden
cp -r results/viz/ict_ifvg_20250318 golden/ifvg_reference/

# Add to tests
# Edit tests/test_golden_runs.py
```

## Maintenance

- **Review Cycle:** After each phase that changes artifact structure
- **Update Trigger:** If ARCHITECTURE_AGREEMENT.md changes schema requirements
- **Deprecation:** Old golden runs archived when incompatible with new schema

## Archive

Old golden runs that are no longer compatible with current schema:

- `archive/pre_phase5_oco/` - Before OCO engine unification (nested oco_results)
- `archive/pre_phase6_windows/` - Before exporter window policy

---

**Next Steps (Phase 2):**
1. Generate `ifvg_reference/` golden run
2. Implement `golden/validator.py`
3. Add `tests/test_golden_runs.py`
4. Add to CI pipeline
