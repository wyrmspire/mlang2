# 🚨 DEFINITIVE GOLDEN PATHS — DO NOT DEVIATE

The system has **two explicit golden paths**.
All work **must preserve both**. If behavior falls outside these paths, it is considered **deprecated or incorrect**.

---

## Golden Path A — User Workflow (Product Behavior)

This defines how the system is **used and experienced**.

1. **Start the app** (`./start.sh`)
   Backend and frontend must start cleanly.

2. **User actions (exactly one of the following):**

   * **View existing results**
     Select a run in the UI and inspect decisions / trades.
   * **Run a new strategy via Lab**
     Use the Lab to run a scan or experiment, then **load the resulting run for visualization**.

3. **Replay / Simulation**

   * **Replay/Simulation is for stepping/experience only.**
   * **Backtest pipeline is the only thing allowed to mint viz artifacts.**
   * Prefer **Simulation (JSON-backed)** for reproducibility.
   * User configures **model + scanner + OCO**, then presses Play.
   * Replay must never recompute history or introduce lookahead.

Any UX change must preserve this flow.

---

## Golden Path B — Engineering Workflow (Artifact Contract)

This defines what constitutes **correct output**.
Implementation details are irrelevant as long as this contract is met.

### 1) Canonical Reference Run

* `golden/reference_scan/` is the **authoritative reference**
* All visualization-capable runs must conform to this structure

---

### 2) Required Artifacts (Non-Negotiable)

All viz runs **must** include:

* `manifest.json`
* `decisions.jsonl`
* `trades.jsonl`

With the following invariants:

* ISO 8601 timestamps **with timezone**
* `oco_results` is **flat**, never nested
* Orders include `contracts` when applicable

If a change breaks this, it is **incorrect**, even if the UI “looks fine.”

---

### 3) Mandatory Dev Loop

All engineering changes must follow this loop:

1. Run a strategy to produce artifacts
2. Validate with:
   `python golden/validate_run.py results/viz/<run>`
3. **All changes must keep the validator passing.**
4. Fix violations and re-run until clean

No exceptions.

---

## Enforcement Rule

* **User Path** defines what the product must do.
* **Engineering Path** defines what outputs must look like.
* If the two conflict, **the artifact contract wins**.
* Any code, tool, or script outside these paths is **deprecated or experimental**.
