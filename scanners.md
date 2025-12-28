# Scanners Architecture - Dev Notes

> This document captures insights on how scanners work and what's needed to make "verified scans" automatically available in Replay Mode.

## Current State

**With the code as it stands, "turning a trade into a scan" will NOT automatically make that scan show up as an available scanner in Replay Mode.**

### Why it doesn't auto-appear (today)

#### 1. Replay Mode's "scanner list" is not driven by past scan artifacts

`RunManifest` explicitly separates **SCAN** vs **REPLAY** runs:
- `create_for_scan(...)` stores `scanners=[ScannerConfig(...)]`
- `create_for_replay(...)` stores `models=[ModelConfig(...)]` and **does not include scanners**

Even if a scan run produced a good scanner config, Replay isn't reading that manifest field because Replay manifests don't carry scanners.

#### 2. New scanners created by `scripts/create_strategy.py` are just files — not auto-registered

- `create_strategy.py` scaffolds a new scanner class into `src/policy/library/*.py`
- But `src/policy/scanner_registry_init.py` only registers a couple built-ins (`always`, `interval`, `modular`) and doesn't import/discover the library folder

**Meaning:** Unless something imports that new module (or you add discovery), the Replay UI won't "see" it as an option.

---

## What it would take (clean path)

Two missing bridges: **(A) discovery/registration** and **(B) replay consumption**.

### A) Make scanners "discoverable" automatically

| Option | Approach |
|--------|----------|
| **A1 (simple/robust)** | Plugin discovery on startup. Scan `src/policy/library/` and import every module so scanner classes exist. Add a "register decorator" pattern for library scanners. |
| **A2 (cleaner long-term)** | Treat "verified scans" as data, not code. Store a "scanner recipe" (e.g., `trigger_config`) and load dynamically via `ScannerRegistry.create("modular", trigger_config=...)`. Avoids generating python files for every idea. |

### B) Let Replay Mode run a scanner (not just a model)

Right now Replay's manifest factory is model-only, but Replay UI expects a "scanner selection" concept. The missing wiring:

1. Add `scanner_id` + `scanner_params` to Replay "start" request (backend + frontend).
2. In replay engine startup, instantiate a scanner from the registry (or from a recipe).
3. Optionally: allow "Load Scanner From Run" where UI reads a prior SCAN run's manifest and offers **"Use this scanner in Replay"**.

---

## Minimal "Works Fast" Implementation

1. **Verified scan writes a `scanner_recipe.json`** into the run artifact folder.

2. **Replay "scanner dropdown" gets a second tab:**
   - **Built-ins** (always/interval/modular)
   - **Verified scans** (pulled from `results/viz/*/manifest.json` where `run_mode == SCAN` and it has `scanners[]`)

3. **Replay start uses:**
   - `modular` scanner + the saved recipe (fast, no import games), OR
   - a registry-created scanner if you implement discovery

**Result:** Ideation can "backcheat" and be instant, then **Verify** produces:
- A persistent artifact (scan run + recipe)
- A replay-usable scanner (immediately selectable)

---

## Files to Wire

| Component | File | Change Needed |
|-----------|------|---------------|
| Replay Start Request | `src/server/replay_routes.py` | Add `scanner_id`, `scanner_params` fields |
| Replay Engine | `scripts/session_replay.py` | Instantiate scanner from registry |
| Scanner Registry | `src/policy/scanner_registry_init.py` | Add discovery for library folder |
| Frontend Dropdown | `src/components/LiveSessionView.tsx` | Add "Verified Scans" tab to scanner picker |
| Run Manifest | `src/viz/manifest.py` | Store `scanner_recipe` in verified runs |
