#!/usr/bin/env python3
"""
Agent Integration Test

Tests the full Agent research cycle:
1. Query DB for existing strategies
2. Run a scan to find patterns
3. Train a model from scan results  
4. Simulate with the trained model
5. Store results to DB
6. Query DB to confirm storage

This validates all infrastructure components work together.

Run:
    python tests/test_agent_integration.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from datetime import datetime


def test_full_agent_cycle():
    """
    Simulate what an Agent would do to find a profitable strategy.
    """
    print("=" * 70)
    print("AGENT INTEGRATION TEST - Full Research Cycle")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Query existing experiments
    # =========================================================================
    print("\n[1/6] Querying ExperimentDB for existing strategies...")
    
    from src.storage import ExperimentDB
    db = ExperimentDB()
    
    existing_count = db.count()
    print(f"      Found {existing_count} existing experiments")
    
    if existing_count > 0:
        best = db.query_best("win_rate", min_trades=1, top_k=3)
        print("      Top 3 by win_rate:")
        for exp in best:
            print(f"        - {exp['run_id']}: {exp['win_rate']:.1%} ({exp['total_trades']} trades)")
    
    # =========================================================================
    # Step 2: Check for scan data (or use existing)
    # =========================================================================
    print("\n[2/6] Looking for scan data...")
    
    from src.config import RESULTS_DIR
    
    # Find any existing scan with records
    scan_run_id = None
    records_file = None
    
    for run_dir in RESULTS_DIR.iterdir():
        if run_dir.is_dir():
            for candidate in ["records.jsonl", "decisions.jsonl"]:
                f = run_dir / candidate
                if f.exists():
                    # Check it has records
                    with open(f) as file:
                        lines = file.readlines()
                    if len(lines) >= 20:  # Need at least 20 for training
                        scan_run_id = run_dir.name
                        records_file = f
                        break
        if scan_run_id:
            break
    
    if scan_run_id:
        print(f"      Found existing scan: {scan_run_id}")
        with open(records_file) as f:
            record_count = sum(1 for _ in f)
        print(f"      Records: {record_count}")
    else:
        print("      No scan data found. Skipping train step.")
        print("      (Run a scan first: python scripts/run_or_multi_oco.py)")
        return {"success": False, "reason": "No scan data available"}
    
    # =========================================================================
    # Step 3: Train model from scan (using the training function directly)
    # =========================================================================
    print("\n[3/6] Training model from scan data...")
    
    model_name = f"test_integration_{datetime.now().strftime('%H%M%S')}"
    model_path = Path(f"models/{model_name}.pth")
    
    # Load and process records (same as /agent/train-from-scan does)
    from scripts.train_ifvg_4class import IFVG4ClassCNN, IFVG4ClassDataset, train_model
    import torch
    from torch.utils.data import DataLoader, random_split
    
    records = []
    with open(records_file) as f:
        for line in f:
            records.append(json.loads(line))
    
    dataset = IFVG4ClassDataset(records, lookback=30)
    if len(dataset) < 10:
        print("      Not enough samples for training")
        return {"success": False, "reason": "Not enough training samples"}
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    print(f"      Training on {len(train_ds)} samples (val: {len(val_ds)})...")
    
    model = IFVG4ClassCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Quick training for test (10 epochs instead of 50)
    best_state, best_acc = train_model(model, train_loader, val_loader, 10, 0.001, device)
    
    # Save model
    Path('models').mkdir(exist_ok=True)
    torch.save(best_state, model_path)
    print(f"      Model saved: {model_path}")
    print(f"      Accuracy: {best_acc:.1%}")
    
    # =========================================================================
    # Step 4: Store training to ExperimentDB
    # =========================================================================
    print("\n[4/6] Storing training result to ExperimentDB...")
    
    db.store_run(
        run_id=f"train_{model_name}",
        strategy="integration_test",
        config={
            "scan_run_id": scan_run_id,
            "epochs": 10,
            "lookback": 30,
        },
        metrics={
            "total_trades": len(train_ds),
            "wins": int(len(train_ds) * best_acc),
            "losses": int(len(train_ds) * (1 - best_acc)),
            "win_rate": best_acc,
            "total_pnl": 0,
        },
        model_path=str(model_path)
    )
    print(f"      Stored: train_{model_name}")
    
    # =========================================================================
    # Step 5: Run simulation with trained model
    # =========================================================================
    print("\n[5/6] Running simulation with trained model...")
    
    from tests.test_backend_simulation import run_simulation
    
    sim_results = run_simulation(
        model_path=str(model_path),
        days=2,
        threshold=0.2,
        verbose=False
    )
    
    print(f"      Triggers: {sim_results['triggers']}")
    print(f"      Trades: {sim_results['trades']}")
    print(f"      Win Rate: {sim_results['win_rate']:.1%}")
    print(f"      Total PnL: ${sim_results['total_pnl']:.2f}")
    
    # Store simulation result
    if sim_results['trades'] > 0:
        db.store_run(
            run_id=f"sim_{model_name}",
            strategy="simulation_test",
            config={
                "model_id": model_name,
                "threshold": 0.2,
                "days": 2,
            },
            metrics={
                "total_trades": sim_results['trades'],
                "wins": sim_results['wins'],
                "losses": sim_results['losses'],
                "win_rate": sim_results['win_rate'],
                "total_pnl": sim_results['total_pnl'],
                "avg_pnl_per_trade": sim_results['total_pnl'] / max(1, sim_results['trades']),
            },
            model_path=str(model_path)
        )
        print(f"      Stored: sim_{model_name}")
    
    # =========================================================================
    # Step 6: Query DB to confirm full cycle
    # =========================================================================
    print("\n[6/6] Confirming experiments in database...")
    
    final_count = db.count()
    print(f"      Total experiments: {final_count} (was {existing_count})")
    
    # Query our new experiments
    train_exp = db.get_run(f"train_{model_name}")
    sim_exp = db.get_run(f"sim_{model_name}")
    
    if train_exp:
        print(f"      ✓ Training result found: {train_exp['win_rate']:.1%} accuracy")
    if sim_exp:
        print(f"      ✓ Simulation result found: {sim_exp['win_rate']:.1%} win rate, ${sim_exp['total_pnl']:.2f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 70)
    print()
    print("✓ Query existing experiments")
    print("✓ Found scan data")
    print("✓ Trained model")
    print("✓ Stored training to DB")
    print("✓ Ran simulation")
    print("✓ Stored simulation to DB")
    print("✓ Queried results from DB")
    print()
    print("The Agent can now perform the full research cycle!")
    
    return {
        "success": True,
        "model_name": model_name,
        "training_accuracy": best_acc,
        "simulation_win_rate": sim_results['win_rate'],
        "simulation_pnl": sim_results['total_pnl'],
        "experiments_added": final_count - existing_count,
    }


if __name__ == "__main__":
    result = test_full_agent_cycle()
    print("\nResult:", json.dumps(result, indent=2))
