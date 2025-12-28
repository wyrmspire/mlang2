import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.run_recipe import main
import argparse
from unittest.mock import patch

def test_full_scan():
    """Test full scan (creates files, no counterfactuals)."""
    print("\n\n=== TESTING FULL SCAN (NO LIGHT MODE, NO CF) ===")
    
    # Create temp recipe
    recipe = {
        "name": "test_verification",
        "entry_trigger": {
            "type": "ema_cross",
            "fast": 10,
            "slow": 20
        },
        "oco": {
            "entry": "MARKET",
            "take_profit": {"multiple": 2.0},
            "stop_loss": {"multiple": 1.0}
        }
    }
    
    with open("test_recipe.json", "w") as f:
        json.dump(recipe, f)
    
    # Run command
    args = [
        "scripts/run_recipe.py",
        "--recipe", "test_recipe.json",
        "--out", "test_verify_full",
        "--start-date", "2025-03-18",
        "--days", "2",
        "--no-cf"  # Explicitly disable CF
    ]
    
    with patch.object(sys, 'argv', args):
        main()
        
    # Verify outputs
    out_dir = Path("results/viz/test_verify_full")
    if (out_dir / "trades.jsonl").exists():
        print("✅ SUCCESS: trades.jsonl created")
    else:
        print("❌ FAIL: trades.jsonl missing")

if __name__ == "__main__":
    test_full_scan()
