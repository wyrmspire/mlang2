# Compatibility Matrix

**Version:** 1.0  
**Date:** 2025-12-25

This document tracks compatibility between different components of the MLang2 system.

---

## Schema Version Tracking

### Current Versions

| Component | Version | Last Updated | Status |
|-----------|---------|--------------|--------|
| VizSchema | 1.0 | 2025-12-25 | Baseline |
| ToolRegistry | 0.9 (pre-unified) | 2025-12-25 | In transition |
| OCO Engine | 0.9 (multiple impls) | 2025-12-25 | Needs consolidation |
| Exporter | 1.0 | 2025-12-25 | Stable |

---

## Backend ↔ Frontend Compatibility

### API Endpoints

| Endpoint | Method | Response Schema | Frontend Version | Backend Version | Status |
|----------|--------|----------------|------------------|-----------------|---------|
| `/runs` | GET | `VizRun[]` | 1.0+ | 1.0+ | ✅ Stable |
| `/runs/{id}/decisions` | GET | `VizDecision[]` | 1.0+ | 1.0+ | ✅ Stable |
| `/runs/{id}/trades` | GET | `VizTrade[]` | 1.0+ | 1.0+ | ✅ Stable |
| `/runs/{id}/series` | GET | `VizBarSeries` | 1.0+ | 1.0+ | ✅ Stable |
| `/agent/chat` | POST | `AgentResponse` | 1.0+ | 1.0+ | ⚠️ Tool schemas hardcoded |
| `/replay/start` | POST | `ReplaySession` | 1.0+ | 1.0+ | ✅ Stable |

### Known Issues

1. **OCO Results Format Drift**
   - **Issue**: Some code paths produce nested `oco_results` instead of flat
   - **Affected Versions**: Pre-1.0
   - **Fix**: Phase 5 (single OCO engine)
   - **Workaround**: Frontend has defensive checks

2. **Missing Contracts Field**
   - **Issue**: Some runs missing `contracts` in VizOCO
   - **Affected Versions**: Pre-1.0
   - **Fix**: Phase 5 (enforce in linter)
   - **Workaround**: Frontend defaults to 1

3. **Timestamp Format Inconsistency**
   - **Issue**: Some timestamps lack timezone info
   - **Affected Versions**: Pre-1.0
   - **Fix**: Phase 9 (linter enforcement)
   - **Workaround**: Frontend assumes America/New_York

4. **Window Sizing Issues**
   - **Issue**: "2h context" not guaranteed; UI has to hunt for bars
   - **Affected Versions**: Pre-1.0
   - **Fix**: Phase 6 (exporter owns windows)
   - **Workaround**: UI performs fallback lookups

---

## Tool System Compatibility

### Current Tool Systems

| System | Location | Purpose | Status |
|--------|----------|---------|--------|
| ScannerRegistry | `src/core/registries.py` | Scanner plugins | ✅ Good pattern |
| ModelRegistry | `src/core/registries.py` | Model plugins | ✅ Good pattern |
| IndicatorRegistry | `src/core/registries.py` | Indicator plugins | ✅ Good pattern |
| SkillRegistry | `src/skills/registry.py` | Agent skills | ⚠️ Different contract |
| AGENT_TOOLS | `src/server/main.py` | Hardcoded tool schemas | ❌ Should be auto-generated |

### Compatibility During Migration

**Phase 3 Migration Plan:**

1. Create `ToolRegistry` with unified contract
2. Adapt existing registries to work with `ToolRegistry`
3. Make `SkillRegistry` a thin wrapper over `ToolRegistry`
4. Auto-generate `AGENT_TOOLS` from `ToolRegistry`
5. Deprecate direct use of old registries (but keep for backward compat)
6. Hard cutoff in Phase 10

**Compatibility Table:**

| Old System | New System | Transition Period | Full Migration |
|------------|-----------|-------------------|----------------|
| ScannerRegistry.register() | ToolRegistry.register(category="scan") | Phase 3-9 | Phase 10 |
| SkillRegistry.register() | ToolRegistry.register(category="skill") | Phase 3-9 | Phase 10 |
| Hardcoded AGENT_TOOLS | /tools/catalog endpoint | Phase 3-9 | Phase 9 |

---

## OCO Engine Compatibility

### Current Implementations

Multiple OCO fill engines exist with different behaviors:

| Implementation | Location | Fill Model | Stop/TP Priority | Tick Rounding | Contracts Calc |
|----------------|----------|------------|------------------|---------------|----------------|
| OCOBracket | `src/sim/oco.py` | Bar-based | ⚠️ Inconsistent | ❌ Missing | ⚠️ Sometimes |
| Strategy-specific | Various scripts | Custom | ❌ Varies | ❌ Varies | ❌ Varies |

### Target State (Phase 5)

Single `OCOEngine` in `src/sim/oco_engine.py`:
- Consistent fill rules
- Documented stop/TP priority
- Tick size rounding enforced
- Contracts always calculated
- Flat `oco_results` output

---

## Indicator System Compatibility

### Current State

| Component | Version | Indicator Support |
|-----------|---------|-------------------|
| IndicatorRegistry | 1.0 | ✅ Generic IndicatorSeries |
| Exporter | 0.9 | ⚠️ Partial indicator export |
| Frontend | 0.9 | ⚠️ Some hardcoded overlays |

### Target State (Phase 7)

- Strategy declares `indicator_ids` in StrategySpec
- Exporter computes all requested indicators
- Frontend renders generically from IndicatorSeries type
- No hardcoded indicator logic in frontend

---

## File Format Compatibility

### Run Artifacts

| File | Format | Required | Schema Version |
|------|--------|----------|----------------|
| manifest.json | JSON | ✅ Yes | 1.0 |
| decisions.jsonl | JSONL | ✅ Yes | 1.0 |
| trades.jsonl | JSONL | ✅ Yes | 1.0 |
| full_series.json | JSON | ❌ Optional | 1.0 |
| events.jsonl | JSONL | ❌ Optional | 1.0 |

### Backward Compatibility

**Phase 2-10 Migration:**
- Old runs (pre-Phase 1) may lack manifest.json → graceful degradation
- Old runs may have nested `oco_results` → linter warns, UI handles
- Old runs may lack `contracts` → UI defaults to 1
- New runs (post-Phase 5) must be fully compliant → enforced by linter

**Cutoff:** After Phase 10, non-compliant runs fail validation.

---

## Strategy Definition Compatibility

### Current State

| Strategy Type | Definition Format | Status |
|---------------|-------------------|--------|
| Hardcoded scripts | Python files in `scripts/` | ⚠️ Ad-hoc |
| Modular strategies | Partial declarative (via agent tools) | ⚠️ Moving toward StrategySpec |

### Target State (Phase 4)

| Strategy Type | Definition Format | Status |
|---------------|-------------------|--------|
| All strategies | StrategySpec (declarative) | ✅ Unified |
| Ad-hoc scripts | ❌ Not allowed | Agent blocked |

**Migration Path:**
- Phase 4: Create StrategySpec
- Phase 4-9: Allow both old scripts and StrategySpec
- Phase 10: Enforce StrategySpec only

---

## Python Dependencies

### Core Dependencies Version Matrix

| Package | Minimum Version | Tested Version | Used For |
|---------|----------------|----------------|----------|
| pandas | 2.0.0 | 2.1.0 | Data manipulation |
| numpy | 1.24.0 | 1.26.0 | Numerical operations |
| torch | 2.0.0 | 2.1.0 | ML models |
| fastapi | 0.100.0 | 0.104.0 | Backend API |
| pydantic | 2.0.0 | 2.5.0 | Data validation |

### Breaking Changes

- **pandas 2.0** introduced timezone-aware datetime handling → Required for our timestamp invariants
- **pydantic 2.0** changed validation API → May need updates in Phase 4 (StrategySpec)

---

## Frontend Dependencies

### React/TypeScript Version Matrix

| Package | Minimum Version | Tested Version | Used For |
|---------|----------------|----------------|----------|
| react | 18.0.0 | 18.2.0 | UI framework |
| typescript | 5.0.0 | 5.3.0 | Type safety |
| lightweight-charts | 4.0.0 | 4.1.0 | Candlestick charts |
| vite | 5.0.0 | 5.0.0 | Build tool |

### Known Issues

- **lightweight-charts**: Position box rendering requires `contracts` field from backend

---

## CI/CD Pipeline Compatibility

### Linter Versions

| Linter | Version | Used For |
|--------|---------|----------|
| ruff (Python) | Latest | Code style |
| eslint (TypeScript) | Latest | Code style |
| Tool contract linter | TBD (Phase 9) | Schema validation |

### Test Framework Versions

| Framework | Version | Used For |
|-----------|---------|----------|
| unittest (Python) | Built-in | Unit tests |
| Golden file tests | TBD (Phase 2) | Regression tests |
| Property tests | TBD (Phase 10) | OCO engine validation |

---

## Rollback Strategy

### If Issues Arise Post-Deployment

**Phase 1-2 (Documentation):**
- Low risk, no rollback needed

**Phase 3-4 (Tool unification):**
- Keep old registries functional during transition
- Feature flag for new ToolRegistry
- Rollback: disable feature flag

**Phase 5 (OCO engine):**
- Keep old OCO implementations temporarily
- Feature flag for unified engine
- Rollback: disable feature flag, fall back to old engines

**Phase 6-7 (Exporter/Indicators):**
- Versioned exporter output
- Frontend supports both old and new formats during transition
- Rollback: revert exporter version

**Phase 9-10 (Enforcement):**
- Linter warnings before hard errors
- Grace period for migration
- Rollback: disable linter enforcement

---

## Version Bump Guidelines

### When to Bump Schema Version

**Patch Version (1.0.X):**
- Bug fixes
- Documentation updates
- Internal refactors

**Minor Version (1.X.0):**
- Adding optional fields
- Adding new tool categories
- Adding new indicator types
- Backward compatible changes

**Major Version (X.0.0):**
- Removing required fields
- Changing field types
- Changing oco_results structure
- Breaking timestamp format
- Any incompatible change

### Current Plan

- Phase 1-9: Stay on v1.0
- Phase 10: Bump to v2.0 (hard cutoff, full enforcement)

---

## Document Control

- **Owner**: Architecture Team
- **Review Cycle**: After each phase completion
- **Approval Required**: Yes (1+ reviewer)
- **Next Review**: After Phase 2

---

**END OF COMPATIBILITY MATRIX**
