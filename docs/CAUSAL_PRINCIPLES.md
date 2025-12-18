# Causal Principles in MLang2

## Core Principle: Time Causality

**MLang2 maintains strict separation between CAUSAL simulation and FUTURE labeling.**

This separation is fundamental to preventing future leakage bugs and ensuring valid backtesting.

---

## Summary

| Component          | Can See Future? | Used In        | Run Mode      |
|--------------------|----------------|----------------|---------------|
| MarketStepper      | ❌ No           | Simulation     | REPLAY, SCAN  |
| Scanner            | ❌ No           | Simulation     | REPLAY, SCAN  |
| Feature Pipeline   | ❌ No           | Simulation     | All modes     |
| Labeler            | ✅ Yes          | Training only  | TRAIN only    |
| TradeOutcome       | ✅ Yes          | Training only  | TRAIN only    |
| Model (REPLAY)     | ❌ No           | Replay         | REPLAY only   |
| Model (TRAINING)   | N/A            | Training       | TRAIN only    |

**Key Insight:** By keeping simulation (CAUSAL) and labeling (FUTURE) completely separate, we prevent 90% of future leakage bugs.

See full documentation in this file for details on RunMode, ModelRole, and best practices.
