# MLang2 Software Plan — Path to 2.0

> **Status**: ~60–70% aligned with target architecture  
> **Priority**: Hardening, not invention

---

## 1️⃣ What's Already Done Right

### ✅ Time Causality
`MarketStepper` has **no future access**. Future peeking quarantined in `labels/` only.
> This prevents 90% of future leakage bugs.

### ✅ Decision ≠ Trade ≠ Label
Distinct artifacts:
- `DecisionRecord`
- `TradeRecord`
- Counterfactual labels
- Viz exports

Essential for "when NOT to trade", comparing OCOs, training vs replay.

### ✅ Feature Split
Correct separation:
- CNN inputs (`x_price_*`)
- MLP context (`x_context`)
- HTF levels (1H / 4H)
- Time/session logic

### ✅ OCO Logic is Modular
OCO construction, processing, labeling are **isolated and parameterized**.

---

## 2️⃣ Structural Risks to Address

### 🔴 Risk #1: Training/Decision Models Not Separated
Nothing prevents trained model from being used during labeling or walk-forward.
Relying on discipline, not architecture.

### 🔴 Risk #2: Policy/Model Blend
Current: `scanner → trade`
Target: `scanner → signals → policies → action → execution → viz`

### 🔴 Risk #3: Replay is Implicit
No explicit **Replay Engine** concept. Needed for:
- Simulated real-time stepping
- Agent speed/pause/resume control
- OCO zones animating bar-by-bar

### 🔴 Risk #4: Viz Schema Not Future-Proofed
Assumes one scanner, one decision source, one model.
Need slots for: multiple model votes, confidence bands, HTF overlays.

---

## 3️⃣ Non-Negotiable Boundaries

### 🧱 A: Model Roles Must Be Explicit
```python
ModelRole = {
    TRAINING_ONLY,
    FROZEN_EVAL,
    REPLAY_ONLY,
    SCAN_ASSIST
}
```
**Rule**: Model with role ≠ REPLAY_ONLY cannot fire during replay.

### 🧱 B: Decisions Are Immutable
Once created, `DecisionRecord` never changes. Can be annotated, not rewritten.

### 🧱 C: Explicit Run Modes

| Mode   | Peek Future | Can Learn | Can Trade |
|--------|-------------|-----------|-----------|
| TRAIN  | ✅          | ✅        | ❌        |
| REPLAY | ❌          | ❌        | ✅ (sim)  |
| SCAN   | ❌          | ❌        | ❌        |

### 🧱 D: Viz is Truth
Always show: what model saw, what policies blocked, what OCO was constructed, what would have happened.

---

## 4️⃣ 20-Phase Path to 2.0

### Phase 0.x — Hardening (NOW)
- [ ] 0.1 — Introduce `RunMode` enum (TRAIN / REPLAY / SCAN)
- [ ] 0.2 — Tag all models with `ModelRole`
- [ ] 0.3 — Enforce role checks at inference time
- [ ] 0.4 — Make `DecisionRecord` immutable (frozen dataclass)
- [ ] 0.5 — Explicit `ReplayConfig` (speed, start, end)

### Phase 1.x — Visualization Spine
- [ ] 1.0 — Unified timeline (decisions + trades + fills)
- [ ] 1.1 — OCO rendered as **zones**, not infinite lines
- [ ] 1.2 — Zoom: single trade ↔ full history
- [ ] 1.3 — Step-forward replay (1m bars)
- [ ] 1.4 — HTF overlays (1H / 4H)
- [ ] 1.5 — Policy-block reasons visualized

### Phase 1.9 — Stability Gate
- [ ] 1.9 — Deterministic replay checksum (no new logic without passing)

### Phase 2.x — Policy-First Architecture
- [ ] 2.0 — Decision → Signal → Policy → Action graph
- [ ] 2.1 — Multiple models voting (no learning yet)
- [ ] 2.2 — "When NOT to trade" as explicit policy
- [ ] 2.3 — Time-of-day / session policies
- [ ] 2.4 — HTF-context policy layer
- [ ] 2.5 — Agent allowed to toggle policies, not code

---

## Files to Modify (Phase 0.x)

| Phase | File | Change |
|-------|------|--------|
| 0.1 | `src/experiments/config.py` | Add `RunMode` enum |
| 0.1 | `src/experiments/runner.py` | Accept and enforce `RunMode` |
| 0.2 | `src/models/fusion.py` | Add `role: ModelRole` field |
| 0.3 | `src/models/fusion.py` | Check role before `forward()` |
| 0.4 | `src/datasets/decision_record.py` | `@dataclass(frozen=True)` |
| 0.5 | `src/experiments/config.py` | Add `ReplayConfig` dataclass |

---

## Not Needed Yet
- Live trading
- RL / online learning
- Production infrastructure
- External data feeds

---

## Next Steps
1. Implement Phase 0.1–0.5 (hardening)
2. Design Replay Engine as first-class object
3. Define policy graph interface
