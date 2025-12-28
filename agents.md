A. What this project is

Deterministic market research + simulation platform

Not a live trading bot

Not an auto-execution system

Focused on learning from price behavior via replay and analysis

This sets scope boundaries immediately.

B. Core invariants (non-negotiable)

Spell these out explicitly:

No future leakage

All conclusions must be grounded in artifacts (runs, trades, metrics)

Replay/OCO logic is authoritative

Indicators describe context, not signals by default

Models annotate decisions; they do not “decide trades” autonomously

This prevents Jules from “optimizing” the wrong things.

## Safe Exploration Directives

> **RULE: Exploration runs are non-promotable by default.**

### Three-Layer Architecture
| Layer | Can Touch? | Example |
|-------|------------|---------|
| Exploration | ✅ Yes | `explore_strategy`, `compare_explorations` |
| Pipeline | ❌ Call only | `run_experiment` internals |
| Presentation (TradeViz) | ❌ Never | `results/viz/`, position boxes |

### Safe Tools (use freely)
- `explore_strategy` - Sweeps, writes to `results/exploration/`
- `compare_explorations` - Rank sweep results
- `diagnose_exploration_run` - Analyze exploration metrics
- `get_session_context` - RTH/Globex, ORH/ORL, PDH/PDL
- `explain_scan_fire` - Why a scan fired
- `scan_coverage_report` - Trigger frequency analysis
- `counterfactual_entry_shift` - "What if entry N bars earlier?"
- `get_price_context` - OHLCV around timestamp

### Gated Tools (require user intent)
- `run_strategy` - Writes TradeViz artifacts
- `run_modular_strategy` - Writes TradeViz artifacts

### Output Directories
- **Safe**: `results/exploration/` (metrics only, no viz)
- **Gated**: `results/viz/` (full artifacts, affects UI)

---

## Don't Touch

- Heavy ML training code (not used right now)
- CNN/LSTM model training pipelines
- Position box rendering logic
- TradeViz schema definitions

D. Tool intent (high level)

Describe tools by purpose, not implementation:

Scanning tools: generate candidate opportunities

Replay tools: simulate execution truthfully

Analysis tools: explain performance patterns

Indicator tools: provide contextual signals

Counterfactual tools: test “what if” changes

This helps Jules choose tools intelligently.

E. What agents should NOT do

This is critical for safety:

Do not invent new execution rules

Do not bypass replay logic

Do not assume indicators are predictive

Do not refactor core mechanics without explicit instruction

Do not optimize for win rate alone

This avoids “helpful but destructive” changes.

F. How to validate changes

Give Jules a checklist mindset:

Does this preserve determinism?

Does this maintain artifact compatibility?

Does replay still produce identical results?

Can this be explained to a trader clearly?