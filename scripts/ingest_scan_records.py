"""
Ingest Scan Records
Converts JSONL records from a Scan into a Sharded Dataset for Training.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.datasets.decision_record import DecisionRecord
from src.datasets.writer import ShardWriter


def parse_record(data: dict) -> DecisionRecord:
    """
    Parse a dictionary (from JSONL) into a DecisionRecord object.
    Reconstructs numpy arrays from lists.
    """
    # Extract window data
    window = data.get('window', {})
    
    x_price_1m = np.array(window.get('x_price_1m', []))
    x_context = np.array(window.get('x_context', []))
    
    # Extract outcomes
    oco_res = data.get('oco_results', {}).get('swing_breakout', {})
    outcome = oco_res.get('outcome', 'TIMEOUT')
    pnl = oco_res.get('pnl_dollars', 0.0)
    
    # Parse timestamp
    ts_str = data.get('timestamp')
    timestamp = pd.Timestamp(ts_str) if ts_str else None
    
    return DecisionRecord(
        timestamp=timestamp,
        bar_idx=data.get('bar_idx', 0),
        decision_id=data.get('decision_id'),
        scanner_id=data.get('scanner_id'),
        
        # Features
        x_price_1m=x_price_1m,
        x_price_5m=None, # Not explicitly in simple scan output yet
        x_price_15m=None,
        x_context=x_context,
        
        # Labels
        cf_outcome=outcome,
        cf_pnl=pnl,
        cf_mae=0.0, # Populated if available
        cf_mfe=0.0,
        cf_bars_held=0,
        
        # Metadata
        current_price=data.get('current_price', 0.0),
        atr=data.get('atr', 0.0)
    )

def main():
    parser = argparse.ArgumentParser(description="Ingest Scan Records to Shards")
    parser.add_argument("--input", type=str, required=True, help="Path to records.jsonl")
    parser.add_argument("--out", type=str, required=True, help="Output shard directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of records")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.out)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Ingesting from: {input_path}")
    print(f"Writing to:     {output_dir}")
    
    count = 0
    with ShardWriter(output_dir, records_per_shard=1000) as writer:
        with open(input_path, 'r') as f:
            for line in tqdm(f):
                if not line.strip():
                    continue
                    
                try:
                    data = json.loads(line)
                    record = parse_record(data)
                    writer.write(record)
                    count += 1
                    
                    if args.limit > 0 and count >= args.limit:
                        break
                        
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line")
                except Exception as e:
                    print(f"Error parsing record: {e}")
                    # raise e # Uncomment to debug
                    
    print(f"Done. Ingested {count} records.")

if __name__ == "__main__":
    main()
