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

## Price-First Behavior (CRITICAL)

> **RULE: Analyze RAW PRICE first, not scanner signals.**

### Guardrails
1. **Never say "no scanner fired" as a final answer.** If no strategy triggered, analyze raw price.
2. **Default to wide date ranges.** If user says "around May 12", load May 1-31, not just that day.
3. **Primary tools are price-based:**
   - `find_price_opportunities` - Find clean swing trades from raw OHLCV
   - `describe_price_action` - Narrative of price behavior
   - `propose_trade` - Entry/stop/target from structure
   - `study_obvious_trades` - Complete "obvious winners" workflow
   - `cluster_trades` - Group by time of day, session, day of week
   - `compare_trade_pools` - Morning vs afternoon comparisons
   - `detect_regime` - TREND_UP/DOWN, RANGE, SPIKE_CHANNEL
   - `trade_fingerprint` - State vector for pattern matching
   - `indicator_impact` - "Would VWAP filter help?"
   - `find_killer_moves` - Biggest opportunities in a range
   - `synthesize_scan` - Auto-generate scanner spec from trades

### Workflow for "Find Opportunities" Requests
1. `describe_price_action` for wide date range (e.g., full month)
2. `find_price_opportunities` to identify clean trades
3. `propose_trade` on the best 2-3 setups
4. Present narrative: "Price did X, cleanest trades were Y"
5. **Optionally** correlate with scanners if relevant

### Workflow for "Compare X vs Y" Requests
1. `cluster_trades` to group by the relevant dimension
2. `compare_trade_pools` for structured comparison
3. Present insights with winner and reason

### Never Block Analysis
If asked about trading opportunities, you MUST provide analysis. Fallback chain:
1. Try raw price analysis
2. Try existing run artifacts
3. Propose hypothetical trades
4. **Never** end with "nothing to say because no scanner fired"

---

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

---

## How to Interact with Agents

### 1. Terminal Chat (Direct Interaction & Role-play)
The primary way to interact with agents outside the UI is via the `scripts/agent_chat.py` script. This allows you to chat with either the **Lab Agent** or the **TradeViz Agent** directly from your terminal.

*   **To chat with the Lab Agent (Analysis Expert):**
    ```bash
    python scripts/agent_chat.py --agent lab
    ```
*   **To chat with the TradeViz Agent (Strategy Executor):**
    ```bash
    python scripts/agent_chat.py --agent tradeviz
    ```

**Role-play Tip:** You can pipe instructions to test intuition:
```bash
echo "Run a simple trend strategy" | python scripts/agent_chat.py --agent tradeviz
```

### 2. Backend API Interaction
The agents reside in the FastAPI backend and can be reached via `curl` or any HTTP client. This is how the UI communicates with them.

*   **TradeViz Agent Contextual Chat:**
    ```bash
    curl -X POST http://localhost:8000/agent/chat \
         -H "Content-Type: application/json" \
         -d '{"messages": [{"role": "user", "content": "Run a trend strategy"}], "context": {"runId": "", "currentIndex": 0, "currentMode": "exploration"}}'
    ```

*   **Lab Agent Research Chat:**
    ```bash
    curl -X POST http://localhost:8000/lab/agent \
         -H "Content-Type: application/json" \
         -d '{"messages": [{"role": "user", "content": "Analyze June volatility"}]}'
    ```

### 3. Role-play Files & Structure
If you want to understand how the agents "think" or modify their behavior, these are the key files:
- **Prompt Logic:** `src/server/main.py` (Functions: `build_agent_system_prompt` and `GENERIC_LAB_PROMPT`)
- **Tool Catalog:** `src/tools/agent_tools.py` (TradeViz tools) and `src/tools/price_analysis_tools.py` (Lab tools)
- **Execution Script:** `scripts/agent_chat.py` (The terminal interface used for role-playing)
- **Intuition Defaults:** Described in `chat.md` stress test results.
