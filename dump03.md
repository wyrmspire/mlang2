    Input: (batch, seq_len, features)
    Output: (batch, num_classes)
    """
    def __init__(
        self,
        input_size: int = 1,      # Just close prices
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,     # LONG_WIN, LONG_LOSS, SHORT_WIN, SHORT_LOSS
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,  # Causal - only look back
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)
        
        return self.fc(last_hidden)


# =============================================================================
# Dataset
# =============================================================================

class LSTMDataset(Dataset):
    """Dataset for LSTM training using close price sequences."""
    
    LABEL_MAP = {
        'LONG_WIN': 0, 'LONG_LOSS': 1,
        'SHORT_WIN': 2, 'SHORT_LOSS': 3,
    }
    
    def __init__(self, records: List[Dict], lookback: int = 60):
        self.samples = []
        self.labels = []
        self.lookback = lookback
        
        for rec in records:
            # Get label
            direction = rec.get('direction', 'LONG')
            outcome = rec.get('label', rec.get('outcome', 'LOSS'))
            label_str = f"{direction}_{outcome}"
            
            if label_str not in self.LABEL_MAP:
                continue
            
            # Get close prices from window
            window = rec.get('window', {})
            ohlcv = window.get('raw_ohlcv_1m', [])
            
            if len(ohlcv) < lookback:
                continue
            
            # Extract close prices - handle both dict and array format
            closes = []
            for bar in ohlcv[-lookback:]:
                if isinstance(bar, dict):
                    closes.append(bar.get('close', bar.get('Close', 0)))
                else:
                    closes.append(bar[3])  # Index 3 = close in array format
            
            closes = np.array(closes, dtype=np.float32)
            
            # Normalize: Z-score
            mean = np.mean(closes)
            std = np.std(closes) + 1e-8
            closes_norm = (closes - mean) / std
            
            self.samples.append(closes_norm)
            self.labels.append(self.LABEL_MAP[label_str])
        
        print(f"Dataset: {len(self.samples)} samples")
        for label, idx in self.LABEL_MAP.items():
            count = sum(1 for l in self.labels if l == idx)
            print(f"  {label} ({idx}): {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32).unsqueeze(-1)  # (seq, 1)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# =============================================================================
# Training
# =============================================================================

def train_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cuda',
) -> Tuple[dict, float]:
    """Train LSTM and return best state dict and accuracy."""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, pred = out.max(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = correct / total if total > 0 else 0
        
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {acc:.1%}")
    
    return best_state, best_acc


# =============================================================================
# Main
# =============================================================================

def run_comparison(input_path: str, lookback: int = 60) -> Dict[str, Any]:
    """
    Train LSTM and compare to CNN baseline.
    """
    print("=" * 60)
    print("LSTM vs CNN COMPARISON")
    print("=" * 60)
    print(f"Lookback: {lookback} bars (close prices only)")
    print("=" * 60)
    
    # Load records
    print(f"\n[1] Loading records from {input_path}...")
    records = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"    Loaded {len(records)} records")
    
    # Create dataset
    print(f"\n[2] Creating LSTM dataset...")
    dataset = LSTMDataset(records, lookback=lookback)
    
    if len(dataset) < 20:
        print("ERROR: Not enough samples for training")
        return {'lstm_acc': 0, 'cnn_acc': 0}
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    print(f"    Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Train LSTM
    print(f"\n[3] Training LSTM...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    Device: {device}")
    
    lstm_model = PriceLSTM(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        num_classes=4,
    )
    
    lstm_state, lstm_acc = train_lstm(
        lstm_model, train_loader, val_loader,
        epochs=50, lr=0.001, device=device
    )
    
    # Save LSTM
    lstm_path = Path("models/lstm_price_seq.pth")
    lstm_path.parent.mkdir(exist_ok=True)
    torch.save(lstm_state, lstm_path)
    print(f"\n    LSTM saved to {lstm_path}")
    print(f"    LSTM Val Accuracy: {lstm_acc:.1%}")
    
    # Load CNN baseline if exists
    cnn_acc = 0.0
    cnn_path = Path("models/ifvg_4class_cnn.pth")
    if cnn_path.exists():
        print(f"\n[4] Loading CNN baseline from {cnn_path}...")
        # We'd need to evaluate CNN on same data - for now use stored metrics
        db = ExperimentDB()
        cnn_runs = db.query_best("win_rate", strategy="cnn_training", top_k=1)
        if cnn_runs:
            cnn_acc = cnn_runs[0].get('win_rate', 0)
            print(f"    CNN baseline accuracy: {cnn_acc:.1%}")
    
    # Compare
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"  LSTM Accuracy: {lstm_acc:.1%}")
    print(f"  CNN Accuracy:  {cnn_acc:.1%}")
    
    if lstm_acc > cnn_acc:
        diff = (lstm_acc - cnn_acc) * 100
        print(f"\n  ✓ LSTM wins by {diff:.1f} percentage points!")
        print("  -> Sequence/flow matters more than shape for this data")
    elif cnn_acc > lstm_acc:
        diff = (cnn_acc - lstm_acc) * 100
        print(f"\n  ✓ CNN wins by {diff:.1f} percentage points!")
        print("  -> Shape patterns matter more than sequence for this data")
    else:
        print(f"\n  = TIE - both models perform similarly")
    
    # Store LSTM result
    db = ExperimentDB()
    run_id = f"lstm_vs_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.store_run(
        run_id=run_id,
        strategy="lstm_comparison",
        config={
            'lookback': lookback,
            'hidden_size': 64,
            'num_layers': 2,
            'architecture': 'PriceLSTM',
        },
        metrics={
            'total_trades': len(dataset),
            'wins': int(len(dataset) * lstm_acc),
            'losses': int(len(dataset) * (1 - lstm_acc)),
            'win_rate': lstm_acc,
            'total_pnl': 0,
        },
        model_path=str(lstm_path)
    )
    print(f"\n[+] Saved to ExperimentDB: {run_id}")
    
    return {
        'lstm_acc': lstm_acc,
        'cnn_acc': cnn_acc,
        'lstm_path': str(lstm_path),
        'samples': len(dataset),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LSTM vs CNN Comparison")
    parser.add_argument("--input", type=str, default="results/ict_ifvg/records.jsonl")
    parser.add_argument("--lookback", type=int, default=60, help="Sequence length")
    
    args = parser.parse_args()
    
    results = run_comparison(args.input, args.lookback)

```

### scripts/verify_fixes.py

```python
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

```

### scripts/verify_position_boxes.py

```python
#!/usr/bin/env python
"""
Test Strategy Run with Position Box Verification

This script:
1. Runs a simple EMA crossover strategy for 1 week
2. Verifies position boxes match trade outcomes
3. Checks SL/TP levels are correct for direction

Run: python scripts/verify_position_boxes.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import loader
from src.policy.triggers.indicator_triggers import EMACrossTrigger
from src.policy.brackets import ATRBracket
from src.strategy.scan import run_strategy_scan
from src.config import RESULTS_DIR


def verify_position_boxes(run_dir: Path) -> dict:
    """
    Verify position boxes match trade outcomes.
    
    Checks:
    1. Direction matches between decision.oco and trade
    2. SL is on correct side of entry for direction
    3. TP is on correct side of entry for direction  
    4. Outcome matches what price did (hit SL = LOSS, hit TP = WIN)
    """
    decisions_file = run_dir / "decisions.jsonl"
    trades_file = run_dir / "trades.jsonl"
    
    if not decisions_file.exists() or not trades_file.exists():
        return {"error": "Missing decisions.jsonl or trades.jsonl"}
    
    # Load data
    decisions = []
    with open(decisions_file) as f:
        for line in f:
            if line.strip():
                decisions.append(json.loads(line))
    
    trades = []
    with open(trades_file) as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))
    
    # Build lookup
    decisions_by_id = {d['decision_id']: d for d in decisions}
    
    errors = []
    warnings = []
    checks_passed = 0
    
    for trade in trades:
        trade_dir = trade.get('direction')
        decision_id = trade.get('decision_id')
        entry = trade.get('entry_price')
        exit_price = trade.get('exit_price')
        outcome = trade.get('outcome')
        pnl = trade.get('pnl_dollars')
        
        decision = decisions_by_id.get(decision_id, {})
        oco = decision.get('oco', {})
        oco_dir = oco.get('direction')
        stop = oco.get('stop_price')
        tp = oco.get('tp_price')
        scanner_ctx = decision.get('scanner_context', {})
        ctx_dir = scanner_ctx.get('direction')
        
        # Check 1: Direction consistency
        if oco_dir and trade_dir and oco_dir != trade_dir:
            errors.append(f"Trade {decision_id}: OCO dir={oco_dir} vs Trade dir={trade_dir}")
        elif ctx_dir and trade_dir and ctx_dir != trade_dir:
            errors.append(f"Trade {decision_id}: Context dir={ctx_dir} vs Trade dir={trade_dir}")
        else:
            checks_passed += 1
        
        # Check 2: SL/TP on correct side for direction
        if oco_dir == "LONG":
            if stop and entry and stop >= entry:
                errors.append(f"Trade {decision_id}: LONG but SL({stop}) >= entry({entry})")
            elif stop and entry:
                checks_passed += 1
                
            if tp and entry and tp <= entry:
                errors.append(f"Trade {decision_id}: LONG but TP({tp}) <= entry({entry})")
            elif tp and entry:
                checks_passed += 1
                
        elif oco_dir == "SHORT":
            if stop and entry and stop <= entry:
                errors.append(f"Trade {decision_id}: SHORT but SL({stop}) <= entry({entry})")
            elif stop and entry:
                checks_passed += 1
                
            if tp and entry and tp >= entry:
                errors.append(f"Trade {decision_id}: SHORT but TP({tp}) >= entry({entry})")
            elif tp and entry:
                checks_passed += 1
        
        # Check 3: Outcome vs PnL consistency
        if outcome == "WIN" and pnl and pnl < 0:
            errors.append(f"Trade {decision_id}: WIN but pnl={pnl} (negative)")
        elif outcome == "LOSS" and pnl and pnl > 0:
            errors.append(f"Trade {decision_id}: LOSS but pnl={pnl} (positive)")
        elif pnl is not None:
            checks_passed += 1
    
    return {
        "total_trades": len(trades),
        "checks_passed": checks_passed,
        "errors": errors,
        "warnings": warnings,
        "status": "PASS" if not errors else "FAIL"
    }


def main():
    print("=" * 60)
    print("POSITION BOX VERIFICATION TEST")
    print("=" * 60)
    
    # 1. Create trigger and bracket
    print("\n1. Creating EMA Crossover (9/21) trigger...")
    trigger = EMACrossTrigger(fast=9, slow=21)
    bracket = ATRBracket(stop_atr=2.0, tp_atr=3.0)
    
    # 2. Run strategy for 1 week in April
    print("\n2. Running strategy scan (April 1-7, 2025)...")
    start_date = "2025-04-01"
    weeks = 1
    run_name = "verify_boxes_test"
    
    try:
        result = run_strategy_scan(
            trigger=trigger,
            bracket=bracket,
            start_date=start_date,
            weeks=weeks,
            run_name=run_name,
            timeframe="5m"
        )
        print(f"   Scan complete: {result.total_trades} trades, {result.total_decisions} decisions")
    except Exception as e:
        print(f"   ERROR during scan: {e}")
        return
    
    # 3. Verify position boxes
    print("\n3. Verifying position boxes...")
    run_dir = RESULTS_DIR / "viz" / run_name
    verification = verify_position_boxes(run_dir)
    
    print(f"\n   Total trades: {verification.get('total_trades', 0)}")
    print(f"   Checks passed: {verification.get('checks_passed', 0)}")
    print(f"   Status: {verification.get('status', 'UNKNOWN')}")
    
    if verification.get('errors'):
        print("\n   ERRORS:")
        for err in verification['errors'][:10]:  # First 10
            print(f"     - {err}")
        if len(verification['errors']) > 10:
            print(f"     ... and {len(verification['errors']) - 10} more")
    
    if verification.get('warnings'):
        print("\n   WARNINGS:")
        for warn in verification['warnings'][:5]:
            print(f"     - {warn}")
    
    print("\n" + "=" * 60)
    if verification.get('status') == 'PASS':
        print("✅ ALL CHECKS PASSED")
    else:
        print("❌ VERIFICATION FAILED - SEE ERRORS ABOVE")
    print("=" * 60)


if __name__ == "__main__":
    main()

```

### scripts/verify_replay_inference.py

```python

import sys
import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

# Import the wrapper directly to test logic
from src.models.model_registry_init import PullerXGBoostWrapper

def fetch_yfinance_data(ticker="MES=F", days=7):
    """Fetch 1m data exactly like the backend."""
    print(f"Fetching {days} days of 1m data for {ticker}...")
    end = datetime.now()
    start = end - timedelta(days=days)
    
    # Using the same call structure as src/server/main.py
    yf_ticker = yf.Ticker(ticker)
    df = yf_ticker.history(start=start, end=end, interval="1m")
    
    if df.empty:
        print("No data found!")
        return []
        
    # Standardize columns
    df.columns = [c.lower() for c in df.columns]
    df = df.reset_index()
    
    # Handle time column
    time_col = None
    for col in ['Datetime', 'datetime', 'Date', 'date', 'time']:
        if col in df.columns:
            time_col = col
            break
            
    if not time_col:
        print("No time column found!")
        return []
        
    bars = []
    for _, row in df.iterrows():
        ts = row[time_col]
        # Convert to ISO format string to match UI
        if hasattr(ts, 'isoformat'):
            ts_str = ts.isoformat()
        else:
            ts_str = str(ts)

        bars.append({
            'time': ts_str,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row.get('volume', 0))
        })
        
    print(f"Loaded {len(bars)} bars")
    return bars

def run_simulation(bars, model_path='models/puller_xgb_4class.json'):
    """Run inference bar-by-bar."""
    print(f"Loading model from {model_path}...")
    try:
        model = PullerXGBoostWrapper(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Running simulation...")
    triggers = []
    
    # Need at least 30 bars for features
    for i in range(30, len(bars)):
        # Current window of 30 bars (ending at i)
        window = bars[i-29:i+1] 
        current_bar = bars[i]
        
        # Predict
        # Wrapper expects 'bars' key in features
        try:
            result = model.predict({'bars': window})
        except Exception as e:
            print(f"Prediction error at {i}: {e}")
            continue
        
        if result['triggered']:
            triggers.append({
                'time': current_bar['time'],
                'price': current_bar['close'],
                'direction': result['direction'],
                'probs': result['probs']
            })
            probs_str = ", ".join([f"{p:.2f}" for p in result['probs']])
            print(f"TRIGGER: {current_bar['time']} {result['direction']} @ {current_bar['close']:.2f} (Probs: [{probs_str}])")
            
    print(f"\nTotal Triggers: {len(triggers)}")
    return triggers

if __name__ == "__main__":
    bars = fetch_yfinance_data()
    if bars:
        run_simulation(bars)

```

### src/__init__.py

```python
# MLang2 - Trade Simulation & Research Platform
"""
A deterministic, causal-correct platform for simulating trades,
logging decisions, and training models to predict counterfactual outcomes.
"""

__version__ = "0.1.0"

# Lazy imports to avoid loading heavy dependencies (torch, etc.) on import
def _get_skills_registry():
    """Lazy import of skills registry to avoid torch dependency on module import."""
    from src.skills.registry import registry
    return registry

def _list_available_skills():
    """Lazy import of skills list."""
    from src.skills.registry import list_available_skills
    return list_available_skills()

# For backward compatibility
skills = None  # Will be loaded on first access if needed

```

### src/api/client.ts

```typescript
import { VizDecision, VizTrade, RunManifest, AgentResponse, ChatMessage, ContinuousData } from '../types/viz';

// API base URL - auto-detect port (8000 or 8001)
let API_BASE = import.meta.env.VITE_API_URL || '';

// Flag to track if backend is available - only cache success, always retry on failure
let backendAvailable: boolean | null = null;

// Check backend availability, auto-detecting port if needed
async function checkBackend(): Promise<boolean> {
    // Only cache success - if previously failed, try again
    if (backendAvailable === true) return true;

    // If no explicit URL, try both ports
    if (!API_BASE) {
        for (const port of [8000, 8001]) {
            try {
                const response = await fetch(`http://localhost:${port}/health`, {
                    method: 'GET',
                    signal: AbortSignal.timeout(2000) // 2s timeout
                });
                if (response.ok) {
                    API_BASE = `http://localhost:${port}`;
                    console.log(`Backend detected on port ${port}`);
                    backendAvailable = true;
                    return true;
                }
            } catch {
                // Try next port
            }
        }
        backendAvailable = false;
        return false;
    }

    try {
        const response = await fetch(`${API_BASE}/health`, { method: 'GET' });
        backendAvailable = response.ok;
    } catch {
        backendAvailable = false;
    }
    return backendAvailable;
}

// ============================================================================
// API CLIENT
// ============================================================================
export const api = {
    // Fetch continuous contract data for base chart
    getContinuousContract: async (
        start?: string,
        end?: string,
        timeframe: string = '1m'
    ): Promise<ContinuousData> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const params = new URLSearchParams();
        if (start) params.set('start', start);
        if (end) params.set('end', end);
        params.set('timeframe', timeframe);
        const response = await fetch(`${API_BASE}/market/continuous?${params}`);
        if (!response.ok) throw new Error('Failed to fetch continuous data');
        return response.json();
    },

    getRuns: async (): Promise<string[]> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/runs`);
        if (!response.ok) throw new Error('Failed to fetch runs');
        return response.json();
    },

    clearAllRuns: async (): Promise<void> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/experiments/clear`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Failed to clear runs');
    },

    getRun: async (runId: string): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/runs/${runId}`);
        if (!response.ok) throw new Error(`Failed to fetch run: ${runId}`);
        return response.json();
    },

    getManifest: async (runId: string): Promise<RunManifest> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/runs/${runId}/manifest`);
        if (!response.ok) throw new Error(`Failed to fetch manifest: ${runId}`);
        return response.json();
    },

    getDecisions: async (runId: string): Promise<VizDecision[]> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/runs/${runId}/decisions`);
        if (!response.ok) throw new Error(`Failed to fetch decisions: ${runId}`);
        return response.json();
    },

    getTrades: async (runId: string): Promise<VizTrade[]> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/runs/${runId}/trades`);
        if (!response.ok) throw new Error(`Failed to fetch trades: ${runId}`);
        return response.json();
    },

    postAgent: async (
        messages: ChatMessage[],
        context: { runId: string, currentIndex: number, currentMode: 'DECISION' | 'TRADE' }
    ): Promise<AgentResponse> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            return { reply: 'Backend not connected. Start with: ./start.sh' };
        }
        try {
            const response = await fetch(`${API_BASE}/agent/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages, context }),
            });
            if (!response.ok) return { reply: 'Error contacting agent server.' };
            return response.json();
        } catch {
            return { reply: 'Error contacting agent server.' };
        }
    },

    postLabAgent: async (messages: ChatMessage[]): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            return { reply: 'Backend not connected. Start with: ./start.sh' };
        }
        try {
            const response = await fetch(`${API_BASE}/lab/agent`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages }),
            });
            if (!response.ok) return { reply: 'Error contacting lab agent.' };
            return response.json();
        } catch {
            return { reply: 'Error contacting lab agent.' };
        }
    },

    runStrategy: async (payload: any): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) {
            throw new Error('Backend unavailable. Start with: ./start.sh');
        }
        const response = await fetch(`${API_BASE}/agent/run-strategy`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        return response.json();
    },

    startReplay: async (modelPath: string, startDate?: string, days: number = 1, speed: number = 10.0, threshold: number = 0.6, strategy: string = "ifvg_4class"): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) throw new Error('Backend unavailable');

        const response = await fetch(`${API_BASE}/replay/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_path: modelPath,
                start_date: startDate,
                days: days,
                speed: speed,
                threshold: threshold,
                strategy: strategy
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to start replay');
        }
        return response.json();
    },

    startLiveReplay: async (
        ticker: string = "MES=F",
        strategy: string = "ema_cross",
        days: number = 7,
        speed: number = 10.0,
        entryConfig?: {
            entry_type?: string;     // Now string to support all strategies
            entry_params?: any;      // Dynamic params for strategies
            stop_method?: 'atr' | 'swing' | 'fixed_bars';
            stop_config?: any;       // Future-proofing
            tp_method?: 'atr' | 'r_multiple';
            stop_atr?: number;
            tp_atr?: number;
            tp_r?: number;
        }
    ): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) throw new Error('Backend unavailable');

        const body = {
            ticker,
            strategy,
            days,
            speed,
            // Entry scan config
            entry_type: entryConfig?.entry_type || 'market',
            entry_params: entryConfig?.entry_params || {},
            stop_method: entryConfig?.stop_method || 'atr',
            tp_method: entryConfig?.tp_method || 'atr',
            stop_atr: entryConfig?.stop_atr || 1.0,
            tp_atr: entryConfig?.tp_atr || 2.0,
            tp_r: entryConfig?.tp_r || 2.0
        };

        const response = await fetch(`${API_BASE}/replay/start/live`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to start live replay');
        }
        return response.json();
    },

    stopReplay: async (sessionId: string): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) return;

        await fetch(`${API_BASE}/replay/sessions/${sessionId}`, {
            method: 'DELETE'
        });
    },

    getReplayStreamUrl: (sessionId: string) => {
        // API_BASE is set by checkBackend hopefully? 
        // We might need to ensure checkBackend is called, but startReplay calls it.
        return `${API_BASE}/replay/stream/${sessionId}`;
    },

    getYFinanceData: async (ticker: string, days: number): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) throw new Error('Backend unavailable');
        const response = await fetch(`${API_BASE}/market/yfinance?ticker=${ticker}&days=${days}`);
        if (!response.ok) {
            throw new Error(`YFinance fetch failed: ${response.status}`);
        }
        return response.json();
    },

    getExperiments: async (params: any = {}): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) throw new Error('Backend unavailable');
        const query = new URLSearchParams(params).toString();
        const response = await fetch(`${API_BASE}/experiments?${query}`);
        if (!response.ok) throw new Error('Failed to fetch experiments');
        return response.json();
    },

    deleteExperiment: async (runId: string): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) throw new Error('Backend unavailable');
        const response = await fetch(`${API_BASE}/experiments/${runId}`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Failed to delete experiment');
        return response.json();
    },

    visualizeExperiment: async (runId: string): Promise<any> => {
        const hasBackend = await checkBackend();
        if (!hasBackend) throw new Error('Backend unavailable');
        const response = await fetch(`${API_BASE}/experiments/${runId}/visualize`, { method: 'POST' });
        if (!response.ok) throw new Error('Failed to visualize experiment');
        return response.json();
    }
};
```

### src/App.tsx

```tsx
import React, { useState, useEffect, useMemo, useRef } from 'react';
import { api } from './api/client';
import { VizDecision, VizTrade, UIAction, ContinuousData } from './types/viz';
import { RunPicker } from './components/RunPicker';
import { Navigator } from './components/Navigator';
import { CandleChart } from './components/CandleChart';
import { DetailsPanel } from './components/DetailsPanel';
import { ChatAgent } from './components/ChatAgent';
import { LiveSessionView } from './components/LiveSessionView';
import { StatsPanel } from './components/StatsPanel';
import { LabPage } from './components/LabPage';
import ExperimentsView from './components/ExperimentsView';
import { IndicatorSettingsPanel } from './components/IndicatorSettings';
import { DEFAULT_INDICATOR_SETTINGS, type IndicatorSettings } from './features/chart_indicators';

type PageType = 'trade' | 'lab' | 'experiments';

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<PageType>('trade');
  const [currentRun, setCurrentRun] = useState<string>('');
  const [mode, setMode] = useState<'DECISION' | 'TRADE'>('DECISION');
  const [index, setIndex] = useState<number>(0);
  const [showSimulation, setShowSimulation] = useState<boolean>(false);
  const [simulationMode, setSimulationMode] = useState<'SIMULATION' | 'YFINANCE'>('SIMULATION');

  const [continuousData, setContinuousData] = useState<ContinuousData | null>(null);
  const [continuousLoading, setContinuousLoading] = useState<boolean>(true);

  const [decisions, setDecisions] = useState<VizDecision[]>([]);
  const [trades, setTrades] = useState<VizTrade[]>([]);

  // Layout State
  const [chatHeight, setChatHeight] = useState<number>(300);
  const isResizingRef = useRef(false);

  // Indicator Settings State
  const [indicatorSettings, setIndicatorSettings] = useState<IndicatorSettings>(DEFAULT_INDICATOR_SETTINGS);

  // Load continuous contract data
  useEffect(() => {
    setContinuousLoading(true);

    let startDate: string | undefined;
    let endDate: string | undefined;

    if (decisions.length > 0) {
      const timestamps = decisions
        .map(d => d.timestamp)
        .filter((t): t is string => !!t)
        .sort();

      if (timestamps.length > 0) {
        startDate = timestamps[0];
        endDate = timestamps[timestamps.length - 1];
      }
    }

    api.getContinuousContract(startDate, endDate).then(data => {
      setContinuousData(data);
      setContinuousLoading(false);
    }).catch(err => {
      console.error('Failed to load continuous data:', err);
      setContinuousLoading(false);
    });
  }, [decisions]);

  // Load run-specific data
  useEffect(() => {
    if (currentRun) {
      Promise.all([
        api.getDecisions(currentRun),
        api.getTrades(currentRun)
      ]).then(([d, t]) => {
        setDecisions(d);
        setTrades(t);
        setIndex(0); // Reset index on run change
      });
    }
  }, [currentRun]);

  // Derived State
  const activeDecision = useMemo(() => {
    if (mode === 'DECISION') {
      return decisions.find(d => d.index === index) || decisions[index] || null;
    } else {
      const trade = trades.find(t => t.index === index);
      return trade ? decisions.find(d => d.decision_id === trade.decision_id) || null : null;
    }
  }, [mode, index, decisions, trades]);

  const activeTrade = useMemo(() => {
    if (mode === 'TRADE') {
      return trades.find(t => t.index === index) || null;
    } else {
      return activeDecision ? trades.find(t => t.decision_id === activeDecision.decision_id) || null : null;
    }
  }, [mode, index, trades, activeDecision]);

  const maxIndex = mode === 'DECISION' ? decisions.length - 1 : trades.length - 1;

  // Agent Action Handler
  const handleAgentAction = async (action: UIAction) => {
    switch (action.type) {
      case 'SET_INDEX':
        setIndex(action.payload);
        break;
      case 'SET_MODE':
        setMode(action.payload);
        setIndex(0);
        break;
      case 'LOAD_RUN':
        setCurrentRun(action.payload);
        break;
      case 'RUN_STRATEGY':
        try {
          console.log("Running strategy...", action.payload);
          const result = await api.runStrategy(action.payload);
          if (result.success && result.run_id) {
            setCurrentRun(result.run_id);
            setMode('DECISION');
          } else {
            console.error("Strategy run failed:", result.error);
          }
        } catch (e) {
          console.error('Failed to run strategy:', e);
        }
        break;
      case 'START_REPLAY':
        setSimulationMode('SIMULATION');
        setShowSimulation(true);
        break;
      case 'TRAIN_FROM_SCAN':
        alert("Training started in background (check console)");
        break;
      default:
        console.warn('Unknown action:', action);
    }
  };

  // Resizing Logic
  const startResizing = () => {
    isResizingRef.current = true;
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', stopResizing);
    document.body.style.userSelect = 'none'; // Prevent selection while dragging
  };

  const stopResizing = () => {
    isResizingRef.current = false;
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', stopResizing);
    document.body.style.userSelect = '';
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isResizingRef.current) return;
    const newHeight = window.innerHeight - e.clientY;
    // Constrain height (min 100px, max 80% of screen)
    const constrained = Math.max(100, Math.min(newHeight, window.innerHeight * 0.8));
    setChatHeight(constrained);
  };

  // If Lab page is active, render it instead
  if (currentPage === 'lab') {
    return (
      <div className="flex flex-col h-screen w-full bg-slate-900 overflow-hidden text-slate-100 font-sans">
        <div className="h-14 flex items-center gap-4 px-6 bg-slate-850/80 backdrop-blur border-b border-slate-800 shrink-0">
          <button
            onClick={() => setCurrentPage('trade')}
            className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors px-3 py-1.5 rounded-md hover:bg-slate-800"
          >
            <span>←</span> Back to Trade View
          </button>
        </div>
        <div className="flex-1 overflow-hidden min-h-0 bg-slate-900">
          <LabPage
            onLoadRun={(runId: string) => {
              setCurrentRun(runId);
              setCurrentPage('trade');
            }}
          />
        </div>
      </div>
    );
  }

  // If Experiments page is active
  if (currentPage === 'experiments') {
    return (
      <div className="flex flex-col h-screen w-full bg-slate-900 overflow-hidden text-slate-100 font-sans">
        <div className="h-14 flex items-center gap-4 px-6 bg-slate-850/80 backdrop-blur border-b border-slate-800 shrink-0">
          <button
            onClick={() => setCurrentPage('trade')}
            className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors px-3 py-1.5 rounded-md hover:bg-slate-800"
          >
            <span>←</span> Back to Trade View
          </button>
        </div>
        <div className="flex-1 overflow-hidden min-h-0 bg-slate-900">
          <ExperimentsView
            onLoadRun={(runId: string) => {
              setCurrentRun(runId);
              setCurrentPage('trade');
            }}
          />
        </div>
      </div>
    );
  }

  // Trade View (default)
  return (
    <div className="flex h-screen w-full bg-slate-900 text-slate-100 font-sans overflow-hidden">

      {/* LEFT SIDEBAR - Expanded Width */}
      <div className="w-96 flex flex-col border-r border-slate-800 bg-slate-950 shrink-0 shadow-xl z-20">

        {/* Header */}
        <div className="h-16 flex items-center justify-between px-6 border-b border-slate-800 shrink-0 bg-slate-950">
          <div className="flex items-center gap-4">
            <h1 className="font-bold text-white text-xl tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-teal-400">Trade Viz</h1>
            <div className="flex gap-2">
              <button
                onClick={() => setCurrentPage('lab')}
                className="text-slate-400 hover:text-green-400 hover:bg-green-400/10 transition-all p-2 rounded-md"
                title="Lab"
              >
                🔬
              </button>
              <button
                onClick={() => setCurrentPage('experiments')}
                className="text-slate-400 hover:text-blue-400 hover:bg-blue-400/10 transition-all p-2 rounded-md"
                title="Experiments"
              >
                🧪
              </button>
            </div>
          </div>
          <button
            onClick={() => {
              setSimulationMode('SIMULATION');
              setShowSimulation(true);
            }}
            className="flex items-center justify-center w-8 h-8 rounded-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white shadow-lg shadow-purple-900/20 transition-all transform hover:scale-105"
            title="Replay"
          >
            ▶
          </button>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto overflow-x-hidden p-4 space-y-6 custom-scrollbar">

          <RunPicker onSelect={setCurrentRun} />

          <Navigator
            mode={mode}
            setMode={setMode}
            index={index}
            setIndex={setIndex}
            maxIndex={Math.max(0, maxIndex)}
          />

          {!currentRun ? (
            <div className="p-8 text-sm text-slate-500 text-center border border-dashed border-slate-800 rounded-lg bg-slate-900/50">
              <p>Select a run above to see details.</p>
            </div>
          ) : (
            <>
              {/* Details Panel - Stays in Sidebar */}
              <div className="border border-slate-800 rounded-lg overflow-hidden shadow-sm bg-slate-900/50">
                <div className="bg-slate-800/50 px-4 py-2 text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                  Decision Context
                </div>
                <div className="bg-slate-900/30">
                  <DetailsPanel decision={activeDecision} trade={activeTrade} />
                </div>
              </div>

              <div className="flex justify-between items-center text-[10px] text-slate-600 px-2 font-mono uppercase tracking-wider">
                <span>📊 {continuousData?.count?.toLocaleString() || 0} bars</span>
                <span>📍 {decisions.length} decisions / {trades.length} trades</span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* MAIN CONTENT - Vertical Layout */}
      <div className="flex-1 flex flex-col min-w-0 h-full relative bg-slate-900">

        {/* Stats Panel (Moved Back to Top of Main Content) */}
        {currentRun && (
          <div className="shrink-0 border-b border-slate-800 bg-slate-900 z-10 shadow-sm">
            <StatsPanel decisions={decisions} startingBalance={50000} />
          </div>
        )}

        {/* Chart Top (Flex Grow) */}
        <div className="flex-1 min-h-0 relative bg-slate-900">
          {/* Indicator Settings - Top LEFT of chart */}
          <div className="absolute top-2 left-2 z-30">
            <IndicatorSettingsPanel settings={indicatorSettings} onChange={setIndicatorSettings} />
          </div>

          <CandleChart
            continuousData={continuousData}
            decisions={decisions}
            activeDecision={activeDecision}
            trade={activeTrade}
            trades={trades}
            indicatorSettings={indicatorSettings}
          />
        </div>

        {/* Resizer Handle */}
        <div
          className="h-1 bg-slate-950 hover:bg-blue-500/50 cursor-row-resize shrink-0 flex items-center justify-center transition-all duration-300 border-y border-slate-800 relative group z-30"
          onMouseDown={startResizing}
        >
          <div className="w-16 h-1 bg-slate-700 rounded-full group-hover:bg-blue-400 transition-colors opacity-50 group-hover:opacity-100" />
        </div>

        {/* Chat Bottom (Fixed Height) */}
        <div style={{ height: chatHeight }} className="shrink-0 bg-slate-950 border-t border-slate-800 shadow-[0_-4px_20px_-5px_rgba(0,0,0,0.3)] z-20">
          <ChatAgent
            runId={currentRun || 'none'}
            currentIndex={index}
            currentMode={mode}
            onAction={handleAgentAction}
          />
        </div>

      </div>

      {/* UNIFIED REPLAY OVERLAY */}
      {showSimulation && (
        <LiveSessionView
          onClose={() => setShowSimulation(false)}
          runId={currentRun}
          initialMode={simulationMode}
          lastTradeTimestamp={
            decisions.length > 0
              ? decisions[decisions.length - 1].timestamp || undefined
              : undefined
          }
        />
      )}

    </div>
  );
};

export default App;
```

### src/components/CandleChart.tsx

```tsx
import React, { useEffect, useRef, useState, useMemo } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, Time, SeriesMarker } from 'lightweight-charts';
import { VizDecision, VizTrade, ContinuousData, BarData } from '../types/viz';
import { PositionBox, createTradePositionBoxes } from './PositionBox';
import type { IndicatorSettings, OHLCV } from '../features/chart_indicators';
import { useIndicators } from '../hooks/useIndicators';
import { INDICATOR_COLORS } from './IndicatorSettings';

interface SimulationOco {
    entry: number;
    stop: number;
    tp: number;
    startTime: number;  // Unix timestamp
}

interface CandleChartProps {
    continuousData: ContinuousData | null;  // Full contract data
    decisions: VizDecision[];               // All decisions for markers
    activeDecision: VizDecision | null;     // Currently selected decision
    trade: VizTrade | null;                 // Active trade for position box
    trades?: VizTrade[];                    // All trades for overlay mode
    simulationOco?: SimulationOco | null;   // OCO state for simulation mode
    indicatorSettings?: IndicatorSettings;
    // NOTE: forceShowAllTrades and defaultShowAllTrades were REMOVED
    // The "Show All Trades" feature had broken bars_held parsing
}

type Timeframe = '1m' | '5m' | '15m' | '1h' | '4h';


// Aggregation helper for higher timeframes - aligned to time boundaries
const aggregateData = (bars: BarData[], intervalMinutes: number): BarData[] => {
    if (intervalMinutes === 1) return bars;
    if (bars.length === 0) return [];

    const aggregated: BarData[] = [];
    const intervalMs = intervalMinutes * 60 * 1000;

    // Group bars by aligned time bucket
    const buckets = new Map<number, BarData[]>();

    bars.forEach(bar => {
        const barTime = new Date(bar.time).getTime();
        // Align to interval boundary (floor to nearest interval)
        const bucketTime = Math.floor(barTime / intervalMs) * intervalMs;

        if (!buckets.has(bucketTime)) {
            buckets.set(bucketTime, []);
        }
        buckets.get(bucketTime)!.push(bar);
    });

    // Convert buckets to aggregated bars
    const sortedBuckets = Array.from(buckets.entries()).sort((a, b) => a[0] - b[0]);

    for (const [bucketTime, barsInBucket] of sortedBuckets) {
        if (barsInBucket.length === 0) continue;

        const open = barsInBucket[0].open;
        const close = barsInBucket[barsInBucket.length - 1].close;
        let high = -Infinity;
        let low = Infinity;
        let vol = 0;

        barsInBucket.forEach(c => {
            if (c.high > high) high = c.high;
            if (c.low < low) low = c.low;
            vol += c.volume;
        });

        // Use the first bar's time string (it has the correct local part we want)
        aggregated.push({
            time: barsInBucket[0].time,
            open,
            high,
            low,
            close,
            volume: vol
        });
    }
    return aggregated;
};

// Parse ISO time string to Unix timestamp
const parseTime = (timeStr: string): number => {
    if (!timeStr) return 0;

    // If it contains a timezone offset or Z, let the browser parse it
    if (timeStr.match(/([+-]\d{2}:?\d{2}|Z)$/)) {
        const ts = Date.parse(timeStr);
        if (!isNaN(ts)) return Math.floor(ts / 1000);
    }

    // Naive fallback (treats string as UTC components if no timezone info)
    const m = timeStr.match(/(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})/);
    if (!m) return Math.floor(new Date(timeStr).getTime() / 1000);

    return Math.floor(Date.UTC(
        parseInt(m[1]),
        parseInt(m[2]) - 1,
        parseInt(m[3]),
        parseInt(m[4]),
        parseInt(m[5]),
        parseInt(m[6])
    ) / 1000);
};

// Find bar index by timestamp
const findBarIndex = (bars: BarData[], timestamp: string): number => {
    const targetTime = parseTime(timestamp);
    for (let i = 0; i < bars.length; i++) {
        const barTime = parseTime(bars[i].time);
        if (barTime >= targetTime) return i;
    }
    return Math.max(0, bars.length - 1);
};

export const CandleChart: React.FC<CandleChartProps> = ({
    continuousData,
    decisions,
    activeDecision,
    trade,
    trades = [],
    simulationOco,
    indicatorSettings,
}) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

    // References for position box primitives
    const activeBoxesRef = useRef<PositionBox[]>([]);
    // NOTE: allTradesBoxesRef was REMOVED - "Show All Trades" feature was broken
    const simOcoBoxesRef = useRef<PositionBox[]>([]);
    const simPriceLinesRef = useRef<any[]>([]);

    // Indicator series refs
    const indicatorSeriesRef = useRef<Map<string, ISeriesApi<"Line">>>(new Map());

    const [timeframe, setTimeframe] = useState<Timeframe>('1m');
    const [isLoading, setIsLoading] = useState(true);

    // Process continuous data with current timeframe
    const chartData = useMemo(() => {
        if (!continuousData?.bars?.length) return [];

        const intervalMap: Record<Timeframe, number> = { '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240 };
        const interval = intervalMap[timeframe];
        const aggregated = aggregateData(continuousData.bars, interval);

        return aggregated.map(bar => ({
            time: parseTime(bar.time) as Time,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close
        }));
    }, [continuousData, timeframe]);

    // Get aggregated bars for timestamp lookups
    const aggregatedBars = useMemo(() => {
        if (!continuousData?.bars?.length) return [];
        const intervalMap: Record<Timeframe, number> = { '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240 };
        return aggregateData(continuousData.bars, intervalMap[timeframe]);
    }, [continuousData, timeframe]);

    // Create decision markers
    const decisionMarkers = useMemo((): SeriesMarker<Time>[] => {
        if (!decisions.length || !aggregatedBars.length) return [];

        return decisions
            .filter(d => d.timestamp)
            .map(d => {
                const barIdx = findBarIndex(aggregatedBars, d.timestamp!);
                const bar = aggregatedBars[barIdx];
                if (!bar) return null;

                const isActive = d.decision_id === activeDecision?.decision_id;
                const direction = d.scanner_context?.direction || d.oco?.direction || 'LONG';

                return {
                    time: parseTime(bar.time) as Time,
                    position: direction === 'LONG' ? 'belowBar' : 'aboveBar',
                    color: isActive ? '#f59e0b' : '#6366f1', // amber for active, indigo for others
                    shape: direction === 'LONG' ? 'arrowUp' : 'arrowDown',
                    text: isActive ? 'ACTIVE' : '',
                    size: isActive ? 2 : 1
                } as SeriesMarker<Time>;
            })
            .filter((m): m is SeriesMarker<Time> => m !== null);
    }, [decisions, activeDecision, aggregatedBars]);

    // Initialize chart
    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: '#1e293b' },
                textColor: '#cbd5e1',
            },
            grid: {
                vertLines: { color: '#334155' },
                horzLines: { color: '#334155' },
            },
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight,
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
            },
        });

        const newSeries = chart.addCandlestickSeries({
            upColor: '#22c55e',
            downColor: '#ef4444',
            borderVisible: false,
            wickUpColor: '#22c55e',
            wickDownColor: '#ef4444',
        });

        chartRef.current = chart;
        seriesRef.current = newSeries;

        const handleResize = () => {
            if (chartContainerRef.current) {
                chart.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            // Detach all position boxes
            activeBoxesRef.current.forEach(box => {
                try { seriesRef.current?.detachPrimitive(box); } catch { }
            });
            activeBoxesRef.current = [];
            chart.remove();
        };
    }, []);

    // Update chart data when continuous data or timeframe changes
    useEffect(() => {
        if (!seriesRef.current || !chartData.length) return;

        setIsLoading(true);
        seriesRef.current.setData(chartData);

        // Force resize to pick up container dimensions (fixes simulation view)
        if (chartRef.current && chartContainerRef.current) {
            const width = chartContainerRef.current.clientWidth;
            const height = chartContainerRef.current.clientHeight;
            if (width > 0 && height > 0) {
                chartRef.current.applyOptions({ width, height });
            }
        }

        setIsLoading(false);
    }, [chartData]);

    // Update markers when decisions change
    useEffect(() => {
        if (!seriesRef.current) return;
        seriesRef.current.setMarkers(decisionMarkers);
    }, [decisionMarkers]);

    // Convert AGGREGATED bars to OHLCV format for indicators
    // CRITICAL: Use aggregatedBars (matching timeframe) not raw 1m bars
    const indicatorCandles: OHLCV[] = useMemo(() => {
        if (!aggregatedBars?.length) return [];
        return aggregatedBars.map(bar => ({
            time: parseTime(bar.time),
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume || 0
        }));
    }, [aggregatedBars]);

    // Calculate indicator data
    const indicatorData = useIndicators(
        indicatorCandles,
        indicatorSettings || { ema9: false, ema21: false, ema200: false, vwap: false, atrBands: false, bollingerBands: false, donchianChannels: false, adr: false, customIndicators: [] }
    );

    // Render indicator line series
    useEffect(() => {
        if (!chartRef.current || !indicatorSettings) return;

        const chart = chartRef.current;
        const seriesMap = indicatorSeriesRef.current;

        // Helper to create or update a line series
        const updateLineSeries = (key: string, data: Array<{ time: number | string; value: number }>, color: string, enabled: boolean) => {
            if (enabled && data.length > 0) {
                let series = seriesMap.get(key);
                if (!series) {
                    series = chart.addLineSeries({
                        color,
                        lineWidth: 1,
                        priceLineVisible: false,
                        lastValueVisible: false,
                    });
                    seriesMap.set(key, series);
                }
                series.setData(data.map(p => ({ time: p.time as Time, value: p.value })));
            } else {
                const existing = seriesMap.get(key);
                if (existing) {
                    chart.removeSeries(existing);
                    seriesMap.delete(key);
                }
            }
        };

        // Helper to create band series (upper + lower as separate lines)
        const updateBandSeries = (key: string, data: Array<{ time: number | string; upper: number; lower: number; middle?: number }>, color: string, enabled: boolean) => {
            if (enabled && data.length > 0) {
                // Upper band
                let upperSeries = seriesMap.get(`${key}_upper`);
                if (!upperSeries) {
                    upperSeries = chart.addLineSeries({ color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
                    seriesMap.set(`${key}_upper`, upperSeries);
                }
                upperSeries.setData(data.map(p => ({ time: p.time as Time, value: p.upper })));

                // Lower band
                let lowerSeries = seriesMap.get(`${key}_lower`);
                if (!lowerSeries) {
                    lowerSeries = chart.addLineSeries({ color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
                    seriesMap.set(`${key}_lower`, lowerSeries);
                }
                lowerSeries.setData(data.map(p => ({ time: p.time as Time, value: p.lower })));

                // Middle band (optional)
                if (data[0]?.middle !== undefined) {
                    let middleSeries = seriesMap.get(`${key}_middle`);
                    if (!middleSeries) {
                        middleSeries = chart.addLineSeries({ color, lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });
                        seriesMap.set(`${key}_middle`, middleSeries);
                    }
                    middleSeries.setData(data.map(p => ({ time: p.time as Time, value: p.middle! })));
                }
            } else {
                // Remove all band series
                ['_upper', '_lower', '_middle'].forEach(suffix => {
                    const existing = seriesMap.get(`${key}${suffix}`);
                    if (existing) {
                        chart.removeSeries(existing);
                        seriesMap.delete(`${key}${suffix}`);
                    }
                });
            }
        };

        // EMAs
        updateLineSeries('ema9', indicatorData.ema9, INDICATOR_COLORS.ema9, indicatorSettings.ema9);
        updateLineSeries('ema21', indicatorData.ema21, INDICATOR_COLORS.ema21, indicatorSettings.ema21);
        updateLineSeries('ema200', indicatorData.ema200, INDICATOR_COLORS.ema200, indicatorSettings.ema200);

        // VWAP
        updateLineSeries('vwap', indicatorData.vwap, INDICATOR_COLORS.vwap, indicatorSettings.vwap);

        // Bands
        updateBandSeries('atrBands', indicatorData.atrBands, INDICATOR_COLORS.atrBands, indicatorSettings.atrBands);
        updateBandSeries('bollingerBands', indicatorData.bollingerBands, INDICATOR_COLORS.bollingerBands, indicatorSettings.bollingerBands);
        updateBandSeries('donchianChannels', indicatorData.donchianChannels, INDICATOR_COLORS.donchianChannels, indicatorSettings.donchianChannels);

        // ADR Zones (resistance top/bottom + support top/bottom = 4 lines)
        if (indicatorSettings.adr && indicatorData.adr.length > 0) {
            // Resistance zone
            let resTopSeries = seriesMap.get('adr_resTop');
            if (!resTopSeries) {
                resTopSeries = chart.addLineSeries({ color: '#ef4444', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
                seriesMap.set('adr_resTop', resTopSeries);
            }
            resTopSeries.setData(indicatorData.adr.map(z => ({ time: z.time as Time, value: z.resTop })));

            let resBottomSeries = seriesMap.get('adr_resBottom');
            if (!resBottomSeries) {
                resBottomSeries = chart.addLineSeries({ color: '#f87171', lineWidth: 1, priceLineVisible: false, lastValueVisible: false, lineStyle: 2 });
                seriesMap.set('adr_resBottom', resBottomSeries);
            }
            resBottomSeries.setData(indicatorData.adr.map(z => ({ time: z.time as Time, value: z.resBottom })));

            // Support zone
            let supBottomSeries = seriesMap.get('adr_supBottom');
            if (!supBottomSeries) {
                supBottomSeries = chart.addLineSeries({ color: '#22c55e', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
                seriesMap.set('adr_supBottom', supBottomSeries);
            }
            supBottomSeries.setData(indicatorData.adr.map(z => ({ time: z.time as Time, value: z.supBottom })));

            let supTopSeries = seriesMap.get('adr_supTop');
            if (!supTopSeries) {
                supTopSeries = chart.addLineSeries({ color: '#4ade80', lineWidth: 1, priceLineVisible: false, lastValueVisible: false, lineStyle: 2 });
                seriesMap.set('adr_supTop', supTopSeries);
            }
            supTopSeries.setData(indicatorData.adr.map(z => ({ time: z.time as Time, value: z.supTop })));
        } else {
            // Remove ADR series
            ['adr_resTop', 'adr_resBottom', 'adr_supTop', 'adr_supBottom'].forEach(key => {
                const existing = seriesMap.get(key);
                if (existing) {
                    chart.removeSeries(existing);
                    seriesMap.delete(key);
                }
            });
        }

        // Custom Indicators
        const customIds = new Set(indicatorSettings.customIndicators?.map(i => i.id) || []);

        // Remove deleted custom indicators
        for (const key of seriesMap.keys()) {
            if (key.startsWith('custom_') && !customIds.has(key.replace('custom_', ''))) {
                const series = seriesMap.get(key);
                if (series) {
                    chart.removeSeries(series);
                    seriesMap.delete(key);
                }
            }
        }

        // Add/update custom indicators
        for (const custom of indicatorSettings.customIndicators || []) {
            const data = indicatorData.customIndicators.get(custom.id);
            if (data && data.length > 0) {
                let series = seriesMap.get(`custom_${custom.id}`);
                if (!series) {
                    series = chart.addLineSeries({
                        color: custom.color,
                        lineWidth: 1,
                        priceLineVisible: false,
                        lastValueVisible: false,
                    });
                    seriesMap.set(`custom_${custom.id}`, series);
                }
                series.setData(data.map(p => ({ time: p.time as Time, value: p.value })));
            }
        }

    }, [indicatorSettings, indicatorData]);

    // ========================================
    // SHOW ALL TRADE BOXES ON LOAD - FIXED 2025-12-25
    // ========================================
    // This useEffect renders position boxes (TP/SL zones) for ALL trades on the chart.
    //
    // CRITICAL DEPENDENCIES:
    // - decisions: Re-run when decision data changes
    // - trades: Re-run when trade data changes (for exit_time/bars_held)
    // - aggregatedBars: Re-run when bar data changes (for timestamp lookups)
    // - continuousData: Re-run when raw data loads
    // - chartData: Re-run when chart data is set (ensures chart is ready)
    // - isLoading: Re-run after loading completes
    // - timeframe: CRITICAL - must re-run when timeframe changes so boxes align correctly
    //
    // PREVIOUS BUG: The old "Show All Trades" toggle had broken bars_held parsing
    // (string vs number types) that caused boxes to extend past actual TP/SL hits.
    // This was fixed by using the same endTime calculation as the active decision path.
    // ========================================
    const allTradesBoxesRef = useRef<PositionBox[]>([]);

    useEffect(() => {
        console.log('[ALL_TRADES] useEffect running, timeframe:', timeframe, 'decisions:', decisions.length, 'trades:', trades.length);

        if (!seriesRef.current || !chartRef.current) {
            console.log('[ALL_TRADES] No chart/series ref');
            return;
        }
        if (!aggregatedBars.length || !continuousData?.bars?.length) {
            console.log('[ALL_TRADES] No bars');
            return;
        }

        // Clear old boxes - MUST do this on every timeframe change
        console.log('[ALL_TRADES] Clearing', allTradesBoxesRef.current.length, 'old boxes');
        allTradesBoxesRef.current.forEach(box => {
            try { seriesRef.current?.detachPrimitive(box); } catch { }
        });
        allTradesBoxesRef.current = [];

        // Get all decisions with OCO data
        const decisionsWithOco = decisions.filter(d => d.oco && d.timestamp);
        console.log('[ALL_TRADES] Decisions with OCO:', decisionsWithOco.length);
        if (!decisionsWithOco.length) return;

        const newBoxes: PositionBox[] = [];

        decisionsWithOco.forEach((decision, idx) => {
            const oco = decision.oco;
            if (!oco?.entry_price || !oco?.stop_price || !oco?.tp_price) return;

            // Find matching trade for this decision (for exit_time/bars_held)
            const matchingTrade = trades.find(t => t.decision_id === decision.decision_id);

            const entryPrice = oco.entry_price;
            const stopPrice = oco.stop_price;
            const tpPrice = oco.tp_price;
            const direction = (decision.scanner_context?.direction || oco.direction || 'LONG') as 'LONG' | 'SHORT';

            // ========================================
            // CRITICAL: Snap times to nearest ACTUAL bars on the chart
            // timeToCoordinate returns null if the exact time doesn't exist
            // On 5m bars, 09:39 doesn't exist - only 09:35, 09:40, etc.
            // So we find the nearest bar and use ITS time
            // ========================================
            const startBarIdx = findBarIndex(aggregatedBars, decision.timestamp!);
            const startBar = aggregatedBars[startBarIdx];
            if (!startBar) return;  // Skip if no matching bar found

            const startTime = parseTime(startBar.time) as Time;

            // End time calculation - snap to nearest bar
            let endTime = startTime;

            if (matchingTrade?.exit_time) {
                // Find the bar at or after exit time
                const endBarIdx = findBarIndex(aggregatedBars, matchingTrade.exit_time);
                const endBar = aggregatedBars[endBarIdx];
                if (endBar) {
                    endTime = parseTime(endBar.time) as Time;
                }
            } else if (matchingTrade?.bars_held) {
                // bars_held is in 1-minute bars - need to convert to current timeframe
                const intervalMinutes = { '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240 }[timeframe] || 1;
                const barsOnThisTF = Math.max(1, Math.ceil(matchingTrade.bars_held / intervalMinutes));
                const endBarIdx = Math.min(startBarIdx + barsOnThisTF, aggregatedBars.length - 1);
                const endBar = aggregatedBars[endBarIdx];
                if (endBar) {
                    endTime = parseTime(endBar.time) as Time;
                }
            } else if (oco.max_bars) {
                // Fallback: use max_bars converted to current timeframe
                const intervalMinutes = { '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240 }[timeframe] || 1;
                const barsOnThisTF = Math.max(1, Math.ceil(oco.max_bars / intervalMinutes));
                const endBarIdx = Math.min(startBarIdx + barsOnThisTF, aggregatedBars.length - 1);
                const endBar = aggregatedBars[endBarIdx];
                if (endBar) {
                    endTime = parseTime(endBar.time) as Time;
                }
            } else {
                // Default: 2 hours (120 minutes) converted to bars
                const intervalMinutes = { '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240 }[timeframe] || 1;
                const barsOnThisTF = Math.max(1, Math.ceil(120 / intervalMinutes));
                const endBarIdx = Math.min(startBarIdx + barsOnThisTF, aggregatedBars.length - 1);
                const endBar = aggregatedBars[endBarIdx];
                if (endBar) {
                    endTime = parseTime(endBar.time) as Time;
                }
            }
            const { slBox, tpBox } = createTradePositionBoxes(
                entryPrice,
                stopPrice,
                tpPrice,
                startTime,
                endTime,
                direction,
                decision.decision_id || `all_${idx}`
            );

            seriesRef.current?.attachPrimitive(slBox);
            seriesRef.current?.attachPrimitive(tpBox);
            newBoxes.push(slBox, tpBox);
            console.log('[ALL_TRADES] Attached boxes, total now:', newBoxes.length);
        });

        allTradesBoxesRef.current = newBoxes;
        console.log('[ALL_TRADES] Done! Created', newBoxes.length, 'boxes total');

    }, [decisions, trades, aggregatedBars, continuousData, chartData, isLoading, timeframe]);

    // Handle active decision - scroll to it and show position boxes
    useEffect(() => {
        if (!seriesRef.current || !chartRef.current) return;
        if (!aggregatedBars.length || !continuousData?.bars?.length) return;

        // Remove old active position boxes
        activeBoxesRef.current.forEach(box => {
            try { seriesRef.current?.detachPrimitive(box); } catch { }
        });
        activeBoxesRef.current = [];

        // If no active decision, just clear boxes
        if (!activeDecision?.timestamp) return;

        // Get decision bar
        const decisionIdx = findBarIndex(aggregatedBars, activeDecision.timestamp);
        const decisionBar = aggregatedBars[decisionIdx];
        if (!decisionBar) return;

        // Scroll to decision with context (more bars for better view)
        const fromIdx = Math.max(0, decisionIdx - 50);
        const toIdx = Math.min(aggregatedBars.length - 1, decisionIdx + 50);

        if (aggregatedBars[fromIdx] && aggregatedBars[toIdx]) {
            chartRef.current.timeScale().setVisibleRange({
                from: parseTime(aggregatedBars[fromIdx].time) as Time,
                to: parseTime(aggregatedBars[toIdx].time) as Time
            });
        }

        // Add position boxes if OCO data available
        const oco = activeDecision.oco;
        if (oco) {
            const entryPrice = oco.entry_price;
            const stopPrice = oco.stop_price;
            const tpPrice = oco.tp_price;
            const direction = (activeDecision.scanner_context?.direction || oco.direction || 'LONG') as 'LONG' | 'SHORT';

            // Calculate start time from decision timestamp
            const startTime = parseTime(decisionBar.time) as Time;

            // Calculate end time based on trade data or estimate
            let endTime = startTime;

            if (trade?.exit_time) {
                // Use actual exit timestamp from trade
                endTime = parseTime(trade.exit_time) as Time;
            } else if (trade?.bars_held) {
                // Estimate end time: decision time + bars_held minutes (assuming 1m bars)
                const decisionTimeMs = parseTime(decisionBar.time) * 1000;
                const endTimeMs = decisionTimeMs + (trade.bars_held * 60 * 1000);
                endTime = Math.floor(endTimeMs / 1000) as Time;
            } else if (oco.max_bars) {
                // Fallback: use max_bars from OCO
                const decisionTimeMs = parseTime(decisionBar.time) * 1000;
                const endTimeMs = decisionTimeMs + (oco.max_bars * 60 * 1000);
                endTime = Math.floor(endTimeMs / 1000) as Time;
            } else {
                // Default: 50 bars (50 minutes)
                const decisionTimeMs = parseTime(decisionBar.time) * 1000;
                const endTimeMs = decisionTimeMs + (50 * 60 * 1000);
                endTime = Math.floor(endTimeMs / 1000) as Time;
            }

            // Create position boxes with actual timestamps
            const { slBox, tpBox } = createTradePositionBoxes(
                entryPrice,
                stopPrice,
                tpPrice,
                startTime,
                endTime,
                direction,
                activeDecision.decision_id
            );

            // Attach primitives to series
            seriesRef.current.attachPrimitive(slBox);
            seriesRef.current.attachPrimitive(tpBox);

            activeBoxesRef.current = [slBox, tpBox];
        }

    }, [activeDecision, trade, aggregatedBars, timeframe, continuousData]);

    // Render simulation OCO position boxes
    // Render simulation OCO price lines
    useEffect(() => {
        if (!seriesRef.current) return;

        // Clear existing simulation lines
        simPriceLinesRef.current.forEach(line => {
            try { seriesRef.current?.removePriceLine(line); } catch { }
        });
        simPriceLinesRef.current = [];

        // If no simulation OCO, we're done
        if (!simulationOco) return;

        // Entry Line
        const entryLine = seriesRef.current.createPriceLine({
            price: simulationOco.entry,
            color: '#3b82f6', // Blue
            lineWidth: 2,
            lineStyle: 0, // Solid
            axisLabelVisible: true,
            title: '',
        });

        // TP Line
        const tpLine = seriesRef.current.createPriceLine({
            price: simulationOco.tp,
            color: '#22c55e', // Green
            lineWidth: 2,
            lineStyle: 0, // Solid
            axisLabelVisible: true,
            title: '',
        });

        // SL Line
        const slLine = seriesRef.current.createPriceLine({
            price: simulationOco.stop,
            color: '#ef4444', // Red
            lineWidth: 2,
            lineStyle: 0, // Solid
            axisLabelVisible: true,
            title: '',
        });

        simPriceLinesRef.current = [entryLine, tpLine, slLine];

    }, [simulationOco]);

    return (
        <div className="relative w-full h-full group">
            <div ref={chartContainerRef} className="w-full h-full" />

            {/* Loading indicator */}
            {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-slate-900/50">
                    <div className="text-slate-400">Loading chart...</div>
                </div>
            )}

            {/* No data message */}
            {!continuousData?.bars?.length && !isLoading && (
                <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center text-slate-500">
                        <p className="text-lg mb-2">No continuous data loaded</p>
                        <p className="text-sm">Waiting for market data...</p>
                    </div>
                </div>
            )}

            {/* Controls */}
            <div className="absolute top-3 right-3 flex flex-col gap-2 z-10">
                {/* Timeframe Controls */}
                <div className="flex bg-slate-800 rounded-md border border-slate-700 shadow-lg overflow-hidden">
                    {(['1m', '5m', '15m', '1h', '4h'] as Timeframe[]).map((tf, idx, arr) => (
                        <button
                            key={tf}
                            onClick={() => setTimeframe(tf)}
                            className={`px-3 py-1 text-xs font-bold transition-colors ${timeframe === tf
                                ? 'bg-blue-600 text-white'
                                : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200'
                                } ${idx !== arr.length - 1 ? 'border-r border-slate-700' : ''}`}
                        >
                            {tf}
                        </button>
                    ))}
                </div>
            </div>

            {/* Decision count overlay */}
            {decisions.length > 0 && (
                <div className="absolute bottom-3 right-3 bg-slate-800/80 px-2 py-1 rounded text-xs text-slate-400 z-10">
                    {decisions.length} decisions
                </div>
            )}
        </div>
    );
};

```

### src/components/ChatAgent.tsx

```tsx
import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { api } from '../api/client';
import { ChatMessage, UIAction } from '../types/viz';

interface ChatAgentProps {
  runId: string;
  currentIndex: number;
  currentMode: 'DECISION' | 'TRADE';
  onAction: (action: UIAction) => void;
}

export const ChatAgent: React.FC<ChatAgentProps> = ({ runId, currentIndex, currentMode, onAction }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'assistant', content: 'Hello! I am the **Trade Viz Agent**. How can I help with your analysis today?' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !runId) return;

    const userMsg: ChatMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const response = await api.postAgent([...messages, userMsg], { runId, currentIndex, currentMode });

      setMessages(prev => [...prev, { role: 'assistant', content: response.reply }]);

      if (response.ui_action) {
        onAction(response.ui_action);
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: "Error contacting agent." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-950 font-sans">
      <div className="px-4 py-3 bg-slate-900/50 backdrop-blur-sm border-b border-slate-800 flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
        <h3 className="text-xs font-bold text-slate-300 uppercase tracking-widest">Agent Terminal</h3>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar bg-slate-950/50" ref={scrollRef}>
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}>
            <div className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm shadow-md ${m.role === 'user'
              ? 'bg-blue-600 text-white rounded-br-none'
              : 'bg-slate-800 text-slate-200 border border-slate-700/50 rounded-bl-none'
              }`}>
              {m.role === 'assistant' ? (
                <div className="prose prose-sm prose-invert max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-headings:my-2 prose-code:bg-slate-700 prose-code:px-1 prose-code:rounded prose-pre:bg-slate-900 prose-pre:border prose-pre:border-slate-700">
                  <ReactMarkdown>{m.content}</ReactMarkdown>
                </div>
              ) : (
                m.content
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex items-center gap-2 text-xs text-slate-500 ml-4 animate-pulse">
            <span className="w-1.5 h-1.5 bg-slate-500 rounded-full animate-bounce"></span>
            <span className="w-1.5 h-1.5 bg-slate-500 rounded-full animate-bounce delay-75"></span>
            <span className="w-1.5 h-1.5 bg-slate-500 rounded-full animate-bounce delay-150"></span>
            Agent is thinking...
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="p-3 bg-slate-900 border-t border-slate-800 flex gap-3 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.1)]">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask about market structure, setup validation, or strategy..."
          className="flex-1 bg-slate-800/50 border border-slate-700 rounded-full px-5 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/50 transition-all shadow-inner"
        />
        <button
          type="submit"
          disabled={loading || !runId}
          className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-full px-6 py-2.5 text-sm font-bold shadow-lg shadow-blue-900/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:translate-y-[-1px] active:translate-y-[1px]"
        >
          Send
        </button>
      </form>
    </div>
  );
};

```

### src/components/DetailsPanel.tsx

```tsx
import React from 'react';
import { VizDecision, VizTrade } from '../types/viz';

interface DetailsPanelProps {
  decision: VizDecision | null;
  trade: VizTrade | null;
}

export const DetailsPanel: React.FC<DetailsPanelProps> = ({ decision, trade }) => {
  if (!decision) return <div className="p-8 text-slate-500 text-center italic text-sm">No data point selected</div>;

  return (
    <div className="h-full flex flex-col bg-slate-900/30">
      <div className="bg-slate-950/50 border-b border-slate-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${decision.action === 'ENTER' ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]' : 'bg-slate-500'}`}></span>
            <h3 className="font-bold text-sm text-slate-200">
            {decision.action}
            </h3>
            <span className="font-mono text-slate-600 text-[10px] bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">#{decision.decision_id}</span>
        </div>

        {trade && (
          <span className={`text-[10px] font-bold px-2 py-1 rounded border ${
              trade.outcome === 'WIN'
                ? 'bg-green-500/10 text-green-400 border-green-500/20 shadow-[0_0_10px_-3px_rgba(34,197,94,0.4)]'
                : 'bg-red-500/10 text-red-400 border-red-500/20 shadow-[0_0_10px_-3px_rgba(239,68,68,0.4)]'
            }`}>
            {trade.outcome} <span className="ml-1 opacity-75">(${trade.pnl_dollars})</span>
          </span>
        )}
      </div>

      <div className="flex-1 overflow-auto p-4 flex flex-col gap-5 custom-scrollbar">

        {/* Stats Section */}
        <div className="space-y-4">
          <section>
            <h4 className="text-[10px] font-bold text-slate-500 uppercase mb-2 tracking-widest flex items-center gap-2">
                <span className="h-px w-3 bg-slate-700"></span>
                Decision Specs
                <span className="h-px flex-1 bg-slate-700"></span>
            </h4>
            <div className="grid grid-cols-2 gap-y-2 text-xs">
              <span className="text-slate-500">Scanner</span>
              <span className="text-right font-mono font-semibold text-blue-400">{decision.scanner_id || 'unknown'}</span>

              <span className="text-slate-500">Price</span>
              <span className="text-right font-mono text-slate-300">{decision.current_price?.toFixed?.(2) ?? decision.current_price ?? '-'}</span>

              <span className="text-slate-500">ATR</span>
              <span className="text-right font-mono text-slate-300">{decision.atr?.toFixed?.(2) ?? decision.atr ?? '-'}</span>

              {decision.skip_reason && (
                <>
                    <span className="text-slate-500">Skip Reason</span>
                    <span className="text-right font-mono text-amber-500 font-bold text-[10px] leading-tight">{decision.skip_reason}</span>
                </>
              )}
            </div>
          </section>

          {trade && (
            <section>
              <h4 className="text-[10px] font-bold text-slate-500 uppercase mb-2 tracking-widest flex items-center gap-2">
                <span className="h-px w-3 bg-slate-700"></span>
                Trade Performance
                <span className="h-px flex-1 bg-slate-700"></span>
              </h4>
              <div className="grid grid-cols-2 gap-y-2 text-xs">
                <span className="text-slate-500">Entry</span>
                <span className="text-right font-mono text-slate-300">{trade.entry_price?.toFixed?.(2) ?? trade.entry_price ?? '-'}</span>

                <span className="text-slate-500">Exit</span>
                <span className="text-right font-mono text-slate-300">{trade.exit_price?.toFixed?.(2) ?? trade.exit_price ?? '-'}</span>

                <span className="text-slate-500">R-Multiple</span>
                <span className={`text-right font-mono font-bold ${trade.r_multiple && trade.r_multiple > 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {trade.r_multiple?.toFixed?.(2) ?? '-'}R
                </span>

                <span className="text-slate-500">MAE / MFE</span>
                <span className="text-right font-mono text-slate-400">{trade.mae?.toFixed?.(2) ?? '-'} / {trade.mfe?.toFixed?.(2) ?? '-'}</span>
              </div>
            </section>
          )}
        </div>

        {/* JSON Context */}
        <div className="space-y-2">
             <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Context Data</div>
             <div className="bg-slate-950 rounded-md p-3 overflow-auto text-[10px] font-mono border border-slate-800 shadow-inner max-h-60 custom-scrollbar">
                {decision.scanner_context && (
                    <div className="mb-3">
                        <div className="text-blue-500/70 mb-1 font-bold">// Scanner Context</div>
                        <pre className="text-slate-300 whitespace-pre-wrap leading-relaxed">
                            {JSON.stringify(decision.scanner_context, null, 2)}
                        </pre>
                    </div>
                )}

                {decision.oco && (
                    <div>
                        <div className="text-green-500/70 mb-1 font-bold">// OCO Params</div>
                        <pre className="text-slate-300 whitespace-pre-wrap leading-relaxed">
                            {JSON.stringify(decision.oco, null, 2)}
                        </pre>
                    </div>
                )}
            </div>
        </div>

      </div>
    </div>
  );
};

```

### src/components/ExperimentsView.tsx

```tsx
import React, { useState, useEffect } from 'react';
import { api } from '../api/client';

interface Experiment {
    run_id: string;
    created_at: string;
    strategy: string;
    config: any;
    total_trades: number;
    wins: number;
    losses: number;
    win_rate: number;
    total_pnl: number;
    avg_pnl_per_trade: number;
    has_viz: boolean;
}

interface ExperimentsViewProps {
    onLoadRun: (runId: string) => void;
}

export const ExperimentsView: React.FC<ExperimentsViewProps> = ({ onLoadRun }) => {
    const [experiments, setExperiments] = useState<Experiment[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [sortBy, setSortBy] = useState<string>('created_at');
    const [sortDesc, setSortDesc] = useState<boolean>(true);
    const [processing, setProcessing] = useState<string | null>(null);

    const fetchExperiments = async () => {
        setLoading(true);
        try {
            const data = await api.getExperiments({
                sort_by: sortBy,
                sort_desc: sortDesc,
                limit: 100
            });
            setExperiments(data.items || []);
        } catch (error) {
            console.error("Failed to fetch experiments", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchExperiments();
    }, [sortBy, sortDesc]);

    const handleDelete = async (runId: string) => {
        if (!window.confirm(`Are you sure you want to delete run ${runId}?`)) return;
        try {
            await api.deleteExperiment(runId);
            fetchExperiments();
        } catch (error) {
            console.error("Failed to delete run", error);
            alert("Failed to delete run");
        }
    };

    const handleVisualize = async (runId: string) => {
        setProcessing(runId);
        try {
            await api.visualizeExperiment(runId);
            // Reload experiments to update 'has_viz' status
            await fetchExperiments();
        } catch (error) {
            console.error("Failed to generate viz", error);
            alert("Failed to generate visualization. Check logs.");
        } finally {
            setProcessing(null);
        }
    };

    const formatPnL = (val: number) => {
        return (
            <span className={val >= 0 ? "text-green-500" : "text-red-500"}>
                ${val.toFixed(2)}
            </span>
        );
    };

    return (
        <div className="flex flex-col h-full bg-gray-900 text-white p-6 overflow-hidden">
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-2xl font-bold">Experiments & Backtests</h1>
                <button
                    onClick={fetchExperiments}
                    className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 transition"
                >
                    Refresh
                </button>
            </div>

            <div className="flex-1 overflow-auto bg-gray-800 rounded-lg shadow-lg">
                <table className="w-full text-left border-collapse">
                    <thead className="bg-gray-700 sticky top-0 z-10">
                        <tr>
                            <th className="p-3 font-semibold cursor-pointer hover:bg-gray-600" onClick={() => { setSortBy('created_at'); setSortDesc(!sortDesc); }}>
                                Date {sortBy === 'created_at' && (sortDesc ? '↓' : '↑')}
                            </th>
                            <th className="p-3 font-semibold">Strategy</th>
                            <th className="p-3 font-semibold text-right cursor-pointer hover:bg-gray-600" onClick={() => { setSortBy('total_trades'); setSortDesc(!sortDesc); }}>
                                Trades {sortBy === 'total_trades' && (sortDesc ? '↓' : '↑')}
                            </th>
                            <th className="p-3 font-semibold text-right cursor-pointer hover:bg-gray-600" onClick={() => { setSortBy('win_rate'); setSortDesc(!sortDesc); }}>
                                Win Rate {sortBy === 'win_rate' && (sortDesc ? '↓' : '↑')}
                            </th>
                            <th className="p-3 font-semibold text-right cursor-pointer hover:bg-gray-600" onClick={() => { setSortBy('total_pnl'); setSortDesc(!sortDesc); }}>
                                Total PnL {sortBy === 'total_pnl' && (sortDesc ? '↓' : '↑')}
                            </th>
                            <th className="p-3 font-semibold text-center">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {loading ? (
                            <tr><td colSpan={6} className="p-8 text-center text-gray-400">Loading...</td></tr>
                        ) : experiments.length === 0 ? (
                            <tr><td colSpan={6} className="p-8 text-center text-gray-400">No experiments found.</td></tr>
                        ) : (
                            experiments.map((exp) => (
                                <tr key={exp.run_id} className="border-b border-gray-700 hover:bg-gray-750 transition-colors">
                                    <td className="p-3 text-gray-300">
                                        <div className="font-mono text-sm">{new Date(exp.created_at).toLocaleString()}</div>
                                        <div className="text-xs text-gray-500 mt-1">{exp.run_id}</div>
                                    </td>
                                    <td className="p-3 font-medium text-blue-300">
                                        {exp.strategy}
                                        {exp.config && exp.config.entry_trigger && (
                                            <div className="text-xs text-gray-400 mt-1 font-mono">
                                                {exp.config.entry_trigger.type} ({JSON.stringify(exp.config.entry_trigger).slice(0, 30)}...)
                                            </div>
                                        )}
                                    </td>
                                    <td className="p-3 text-right font-mono">{exp.total_trades}</td>
                                    <td className="p-3 text-right font-mono">
                                        {(exp.win_rate * 100).toFixed(1)}%
                                    </td>
                                    <td className="p-3 text-right font-mono font-bold">
                                        {formatPnL(exp.total_pnl)}
                                    </td>
                                    <td className="p-3">
                                        <div className="flex justify-center gap-2">
                                            {exp.has_viz ? (
                                                <button
                                                    onClick={() => onLoadRun(exp.run_id)}
                                                    className="px-3 py-1 bg-green-600 text-xs rounded hover:bg-green-700"
                                                >
                                                    Load Viz
                                                </button>
                                            ) : (
                                                <button
                                                    onClick={() => handleVisualize(exp.run_id)}
                                                    disabled={processing === exp.run_id}
                                                    className={`px-3 py-1 bg-gray-600 text-xs rounded hover:bg-gray-500 ${processing === exp.run_id ? 'opacity-50 cursor-wait' : ''}`}
                                                >
                                                    {processing === exp.run_id ? 'Generating...' : 'Re-run for Viz'}
                                                </button>
                                            )}

                                            <button
                                                onClick={() => handleDelete(exp.run_id)}
                                                className="px-3 py-1 bg-red-900/50 text-red-300 text-xs rounded hover:bg-red-900"
                                            >
                                                Delete
                                            </button>
                                        </div>
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default ExperimentsView;

```

### src/components/IndicatorSettings.tsx

```tsx
/**
 * Indicator Settings Panel - Vertical Layout
 * 
 * Toggle menu for enabling/disabling chart indicators.
 * Positioned top-left of chart, vertical layout.
 */

import React, { useState } from 'react';
import type { IndicatorSettings, CustomIndicator } from '../features/chart_indicators';

interface IndicatorSettingsProps {
    settings: IndicatorSettings;
    onChange: (settings: IndicatorSettings) => void;
}

interface IndicatorToggle {
    key: keyof Omit<IndicatorSettings, 'customIndicators'>;
    label: string;
    color: string;
}

const PRESET_INDICATORS: IndicatorToggle[] = [
    // EMAs
    { key: 'ema9', label: 'EMA 9', color: '#fbbf24' },   // yellow
    { key: 'ema21', label: 'EMA 21', color: '#3b82f6' }, // blue
    { key: 'ema200', label: 'EMA 200', color: '#ffffff' }, // white

    // VWAP
    { key: 'vwap', label: 'VWAP', color: '#ec4899' }, // pink

    // Bands
    { key: 'atrBands', label: 'ATR Bands', color: '#6b7280' },  // gray
    { key: 'bollingerBands', label: 'Bollinger', color: '#a855f7' }, // purple
    { key: 'donchianChannels', label: 'Donchian', color: '#22d3ee' }, // cyan

    // ADR
    { key: 'adr', label: 'ADR', color: '#f97316' }, // orange
];

export const IndicatorSettingsPanel: React.FC<IndicatorSettingsProps> = ({ settings, onChange }) => {
    const [showCustomForm, setShowCustomForm] = useState(false);
    const [customType, setCustomType] = useState<'ema' | 'sma'>('ema');
    const [customPeriod, setCustomPeriod] = useState<number>(50);
    const [customColor, setCustomColor] = useState<string>('#10b981');

    const toggle = (key: keyof Omit<IndicatorSettings, 'customIndicators'>) => {
        onChange({ ...settings, [key]: !settings[key] });
    };

    const addCustomIndicator = () => {
        const newIndicator: CustomIndicator = {
            id: `${customType}_${customPeriod}_${Date.now()}`,
            type: customType,
            period: customPeriod,
            color: customColor,
        };

        const customIndicators = [...(settings.customIndicators || []), newIndicator];
        onChange({ ...settings, customIndicators });
        setShowCustomForm(false);
    };

    const removeCustomIndicator = (id: string) => {
        const customIndicators = (settings.customIndicators || []).filter(i => i.id !== id);
        onChange({ ...settings, customIndicators });
    };

    return (
        <div className="flex flex-col gap-1 bg-slate-800/90 rounded-lg p-2 min-w-[120px] backdrop-blur-sm">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-slate-700 pb-1 mb-1">
                <span className="text-xs text-slate-400 font-medium">Indicators</span>
                <button
                    onClick={() => setShowCustomForm(!showCustomForm)}
                    className="w-5 h-5 rounded bg-slate-700 hover:bg-slate-600 flex items-center justify-center text-slate-400 hover:text-white transition-colors"
                    title="Add custom indicator"
                >
                    <span className="text-sm">+</span>
                </button>
            </div>

            {/* Custom Indicator Form */}
            {showCustomForm && (
                <div className="bg-slate-700/50 rounded p-2 mb-1 space-y-2">
                    <div className="flex gap-1">
                        <button
                            onClick={() => setCustomType('ema')}
                            className={`flex-1 px-2 py-1 text-[10px] rounded ${customType === 'ema' ? 'bg-blue-600' : 'bg-slate-600'}`}
                        >
                            EMA
                        </button>
                        <button
                            onClick={() => setCustomType('sma')}
                            className={`flex-1 px-2 py-1 text-[10px] rounded ${customType === 'sma' ? 'bg-blue-600' : 'bg-slate-600'}`}
                        >
                            SMA
                        </button>
                    </div>
                    <input
                        type="number"
                        value={customPeriod}
                        onChange={(e) => setCustomPeriod(Number(e.target.value))}
                        className="w-full px-2 py-1 text-xs bg-slate-800 border border-slate-600 rounded"
                        placeholder="Period"
                        min={1}
                        max={500}
                    />
                    <div className="flex items-center gap-2">
                        <input
                            type="color"
                            value={customColor}
                            onChange={(e) => setCustomColor(e.target.value)}
                            className="w-6 h-6 rounded cursor-pointer"
                        />
                        <button
                            onClick={addCustomIndicator}
                            className="flex-1 px-2 py-1 text-[10px] bg-green-600 hover:bg-green-500 rounded"
                        >
                            Add
                        </button>
                    </div>
                </div>
            )}

            {/* Preset Indicators */}
            {PRESET_INDICATORS.map(ind => (
                <button
                    key={ind.key}
                    onClick={() => toggle(ind.key)}
                    className={`
            flex items-center gap-2 px-2 py-1.5 text-xs rounded transition-all w-full text-left
            ${settings[ind.key]
                            ? 'bg-slate-600/80 text-white'
                            : 'text-slate-500 hover:text-slate-300 hover:bg-slate-700/50'
                        }
          `}
                >
                    <span
                        className="w-2.5 h-2.5 rounded-full shrink-0"
                        style={{ backgroundColor: ind.color, opacity: settings[ind.key] ? 1 : 0.4 }}
                    />
                    <span className="truncate">{ind.label}</span>
                </button>
            ))}

            {/* Custom Indicators */}
            {(settings.customIndicators || []).map(ind => (
                <div
                    key={ind.id}
                    className="flex items-center gap-2 px-2 py-1.5 text-xs bg-slate-600/80 rounded"
                >
                    <span
                        className="w-2.5 h-2.5 rounded-full shrink-0"
                        style={{ backgroundColor: ind.color }}
                    />
                    <span className="flex-1 truncate">{ind.type.toUpperCase()} {ind.period}</span>
                    <button
                        onClick={() => removeCustomIndicator(ind.id)}
                        className="text-slate-400 hover:text-red-400 text-[10px]"
                    >
                        ✕
                    </button>
                </div>
            ))}
        </div>
    );
};

export const INDICATOR_COLORS = {
    ema9: '#fbbf24',
    ema21: '#3b82f6',
    ema200: '#ffffff',
    vwap: '#ec4899',
    atrBands: '#6b7280',
    bollingerBands: '#a855f7',
    donchianChannels: '#22d3ee',
    adr: '#f97316',
};

```

### src/components/LabPage.tsx

```tsx
import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { api } from '../api/client';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    type?: 'text' | 'table' | 'chart' | 'code';
    data?: any;
    run_id?: string;
}

interface LabResult {
    strategy: string;
    trades: number;
    wins: number;
    losses: number;
    win_rate: number;
    total_pnl: number;
    equity_curve?: number[];
}

interface LabPageProps {
    onLoadRun?: (runId: string) => void;
}

export const LabPage: React.FC<LabPageProps> = ({ onLoadRun }) => {
    const [messages, setMessages] = useState<Message[]>([
        {
            role: 'assistant',
            content: 'Welcome to the Research Lab! I can help you test strategies, run scans, train models, and analyze results. What would you like to explore?',
            type: 'text'
        }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [currentResult, setCurrentResult] = useState<LabResult | null>(null);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg: Message = { role: 'user', content: input, type: 'text' };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            const response = await api.postLabAgent([...messages, userMsg]);
            const assistantMsg: Message = {
                role: 'assistant',
                content: response.reply || 'Processing...',
                type: response.type || 'text',
                data: response.data,
                run_id: response.run_id
            };
            setMessages(prev => [...prev, assistantMsg]);
            if (response.result) {
                setCurrentResult(response.result);
            }
        } catch (err) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'Error contacting lab agent. Is the backend running?',
                type: 'text'
            }]);
        } finally {
            setLoading(false);
        }
    };

    const quickActions = [
        { label: 'Run EMA Scan', prompt: 'Run an EMA cross scan on the last 7 days' },
        { label: 'Test ORB Strategy', prompt: 'Test the Opening Range Breakout strategy' },
        { label: 'Compare Models', prompt: 'Compare the LSTM vs CNN model accuracy' },
        { label: 'Show Best Config', prompt: 'What is the best configuration from recent experiments?' },
        { label: 'Run Grid Search', prompt: 'Run a grid search on ORB stop and target parameters' },
    ];

    const sendQuickAction = (prompt: string) => {
        setInput(prompt);
    };

    const renderResultTable = (result: LabResult, runId?: string) => (
        <div className="bg-slate-800 rounded-lg p-4 my-3 border border-slate-600">
            <div className="flex items-center justify-between mb-3">
                <div className="text-sm font-bold text-blue-400">{result.strategy}</div>
                {runId && onLoadRun && (
                    <button
                        onClick={() => onLoadRun(runId)}
                        className="bg-blue-600 hover:bg-blue-500 text-white text-xs px-3 py-1.5 rounded transition"
                    >
                        📊 Visualize
                    </button>
                )}
            </div>
            <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                    <div className="text-2xl font-bold text-white">{result.trades}</div>
                    <div className="text-xs text-slate-400">Trades</div>
                </div>
                <div>
                    <div className={`text-2xl font-bold ${result.win_rate >= 0.5 ? 'text-green-400' : 'text-red-400'}`}>
                        {(result.win_rate * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-slate-400">Win Rate</div>
                </div>
                <div>
                    <div className={`text-2xl font-bold ${result.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        ${result.total_pnl.toLocaleString()}
                    </div>
                    <div className="text-xs text-slate-400">P&L</div>
                </div>
            </div>

            <div className="mt-4">
                <div className="flex h-3 rounded overflow-hidden">
                    <div className="bg-green-500" style={{ width: `${result.win_rate * 100}%` }} />
                    <div className="bg-red-500" style={{ width: `${(1 - result.win_rate) * 100}%` }} />
                </div>
                <div className="flex justify-between text-xs text-slate-400 mt-1">
                    <span>{result.wins} Wins</span>
                    <span>{result.losses} Losses</span>
                </div>
            </div>

            {result.equity_curve && result.equity_curve.length > 0 && (
                <div className="mt-4">
                    <div className="text-xs text-slate-400 mb-2">Equity Curve</div>
                    <div className="h-16 flex items-end gap-px">
                        {result.equity_curve.slice(-50).map((val, idx) => {
                            const min = Math.min(...result.equity_curve!.slice(-50));
                            const max = Math.max(...result.equity_curve!.slice(-50));
                            const height = max > min ? ((val - min) / (max - min)) * 100 : 50;
                            return (
                                <div
                                    key={idx}
                                    className={`flex-1 ${val >= result.equity_curve![0] ? 'bg-green-500' : 'bg-red-500'}`}
                                    style={{ height: `${Math.max(5, height)}%` }}
                                />
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );

    const renderMessage = (msg: Message, idx: number) => {
        if (msg.role === 'user') {
            return (
                <div key={idx} className="flex justify-end">
                    <div className="max-w-[80%] bg-blue-600 text-white rounded-xl px-4 py-2">
                        {msg.content}
                    </div>
                </div>
            );
        }

        return (
            <div key={idx} className="flex justify-start">
                <div className="max-w-[90%]">
                    <div className="bg-slate-700 text-slate-100 rounded-xl px-4 py-3">
                        <div className="prose prose-sm prose-invert max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-headings:my-2 prose-headings:text-blue-400 prose-code:bg-slate-600 prose-code:px-1 prose-code:rounded prose-pre:bg-slate-800 prose-pre:border prose-pre:border-slate-600 prose-strong:text-white prose-table:border-collapse prose-th:bg-slate-700 prose-th:border prose-th:border-slate-600 prose-th:px-3 prose-th:py-2 prose-td:border prose-td:border-slate-600 prose-td:px-3 prose-td:py-2 prose-tr:even:bg-slate-800/50">
                            <ReactMarkdown>{msg.content}</ReactMarkdown>
                        </div>
                    </div>
                    {msg.data?.result && renderResultTable(msg.data.result, msg.run_id)}
                </div>
            </div>
        );
    };

    return (
        <div className="flex flex-col h-full bg-slate-900 overflow-hidden">
            {/* Header */}
            <div className="h-14 flex items-center justify-between px-6 border-b border-slate-700 bg-slate-800 shrink-0">
                <div className="flex items-center gap-3">
                    <span className="text-2xl">🔬</span>
                    <h1 className="text-xl font-bold text-white">Research Lab</h1>
                </div>
                <div className="text-sm text-slate-400">
                    AI-Powered Strategy Research
                </div>
            </div>

            {/* Main Content */}
            <div className="flex flex-1 overflow-hidden min-h-0">

                {/* Left Sidebar - Current Result & Commands */}
                <div className="w-80 border-r border-slate-700 bg-slate-800 p-4 overflow-y-auto shrink-0 flex flex-col">
                    <h2 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">
                        Latest Result
                    </h2>

                    {currentResult ? (
                        renderResultTable(currentResult)
                    ) : (
                        <div className="text-slate-500 text-sm text-center py-8 border border-dashed border-slate-700 rounded">
                            Run a strategy to see results here
                        </div>
                    )}

                    <div className="mt-6">
                        <h2 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-3">
                            Quick Commands
                        </h2>
                        <div className="space-y-2 text-xs">
                            <div className="bg-slate-700 p-2 rounded text-slate-300 cursor-pointer hover:bg-slate-600 transition" onClick={() => setInput("Run EMA cross scan")}>
                                <code>"Run EMA cross scan"</code>
                            </div>
                            <div className="bg-slate-700 p-2 rounded text-slate-300 cursor-pointer hover:bg-slate-600 transition" onClick={() => setInput("Test lunch hour fade")}>
                                <code>"Test lunch hour fade"</code>
                            </div>
                            <div className="bg-slate-700 p-2 rounded text-slate-300 cursor-pointer hover:bg-slate-600 transition" onClick={() => setInput("Train LSTM on bounce data")}>
                                <code>"Train LSTM on bounce data"</code>
                            </div>
                            <div className="bg-slate-700 p-2 rounded text-slate-300 cursor-pointer hover:bg-slate-600 transition" onClick={() => setInput("Compare ORB vs MR strategy")}>
                                <code>"Compare ORB vs MR strategy"</code>
                            </div>
                            <div className="bg-slate-700 p-2 rounded text-slate-300 cursor-pointer hover:bg-slate-600 transition" onClick={() => setInput("Show experiment history")}>
                                <code>"Show experiment history"</code>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Chat Area (Right) */}
                <div className="flex-1 flex flex-col min-w-0 bg-slate-900">
                    {/* Messages */}
                    <div className="flex-1 overflow-y-auto p-6 space-y-4" ref={scrollRef}>
                        {messages.map((msg, idx) => renderMessage(msg, idx))}
                        {loading && (
                            <div className="flex justify-start">
                                <div className="bg-slate-700 text-slate-300 rounded-xl px-4 py-3 animate-pulse">
                                    <span className="text-blue-400">Agent is thinking...</span>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Quick Actions */}
                    <div className="px-6 py-3 border-t border-slate-700 bg-slate-800 shrink-0">
                        <div className="flex gap-2 flex-wrap">
                            {quickActions.map((action, idx) => (
                                <button
                                    key={idx}
                                    onClick={() => sendQuickAction(action.prompt)}
                                    className="bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs px-3 py-1.5 rounded-full transition"
                                >
                                    {action.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Input */}
                    <form onSubmit={handleSubmit} className="p-4 border-t border-slate-700 bg-slate-800 shrink-0">
                        <div className="flex gap-3">
                            <input
                                value={input}
                                onChange={e => setInput(e.target.value)}
                                placeholder="Ask me to run a strategy, test a theory, or analyze results..."
                                className="flex-1 bg-slate-900 border border-slate-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                            />
                            <button
                                type="submit"
                                disabled={loading}
                                className="bg-blue-600 hover:bg-blue-500 text-white rounded-lg px-6 py-3 font-bold disabled:opacity-50"
                            >
                                Send
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default LabPage;

```

### src/components/LiveSessionView.tsx

```tsx
import React, { useState, useCallback, useRef, useEffect } from 'react';
import { CandleChart } from './CandleChart';
import { VizTrade, VizDecision } from '../types/viz';
import { api } from '../api/client';

interface LiveSessionViewProps {
    onClose: () => void;
    runId?: string;
    lastTradeTimestamp?: string;
    initialMode?: 'SIMULATION' | 'YFINANCE';
}

interface BarData {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

type DataSourceMode = 'SIMULATION' | 'YFINANCE';
type PlaybackState = 'STOPPED' | 'PLAYING' | 'PAUSED';

const SidebarSection: React.FC<{
    title: string;
    children: React.ReactNode;
    defaultOpen?: boolean;
    colorClass?: string;
}> = ({ title, children, defaultOpen = false, colorClass = "text-blue-400" }) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);
    return (
        <div className="mb-2 border-b border-slate-700 pb-2 last:border-0">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={`flex items-center justify-between w-full text-xs font-bold uppercase py-1 ${colorClass} hover:opacity-80`}
            >
                {title}
                <span className="text-slate-500">{isOpen ? '▼' : '▶'}</span>
            </button>
            {isOpen && <div className="mt-2 text-sm">{children}</div>}
        </div>
    );
};

export const LiveSessionView: React.FC<LiveSessionViewProps> = ({
    onClose,
    runId,
    lastTradeTimestamp,
    initialMode = 'SIMULATION'
}) => {
    // Data Source
    const [dataSourceMode, setDataSourceMode] = useState<DataSourceMode>(initialMode);

    // Playback State
    const [playbackState, setPlaybackState] = useState<PlaybackState>('STOPPED');
    const [speed, setSpeed] = useState(200); // ms per bar
    const [bars, setBars] = useState<BarData[]>([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [startIndex, setStartIndex] = useState(0);
    const [status, setStatus] = useState('Ready');

    // Model/Scanner Selection - Now with enable checkboxes (OFF by default)
    const [useCnnModel, setUseCnnModel] = useState(false);       // OFF by default
    const [usePatternScanner, setUsePatternScanner] = useState(false); // OFF by default
    const [selectedModel, setSelectedModel] = useState('models/ifvg_4class_cnn.pth');
    const [selectedScanner, setSelectedScanner] = useState('ifvg');
    const [availableModels, setAvailableModels] = useState<string[]>([
        'models/ifvg_4class_cnn.pth',
        'models/ifvg_cnn.pth',
        'models/best_model.pth',
        'models/puller_xgb_4class.json'
    ]);

    // Entry Configuration (sent to backend)
    const [entryType, setEntryType] = useState<'market' | 'limit'>('market');
    const [stopMethod, setStopMethod] = useState<'atr' | 'swing' | 'fixed_bars'>('atr');
    const [tpMethod, setTpMethod] = useState<'atr' | 'r_multiple'>('atr');

    // OCO State
    const [ocoState, setOcoState] = useState<{
        entry: number;
        stop: number;
        tp: number;
        startTime: number;
        direction: 'LONG' | 'SHORT';
    } | null>(null);

    // Trade Settings
    const [threshold, setThreshold] = useState(0.35);
    const [stopAtr, setStopAtr] = useState(2.0);
    const [tpAtr, setTpAtr] = useState(4.0);

    // Trade Tracking
    const [triggers, setTriggers] = useState(0);
    const [wins, setWins] = useState(0);
    const [losses, setLosses] = useState(0);
    const [completedTrades, setCompletedTrades] = useState<VizTrade[]>([]);
    const [completedDecisions, setCompletedDecisions] = useState<VizDecision[]>([]);

    // YFinance specific
    const [ticker, setTicker] = useState('MES=F');
    const [yfinanceDays, setYfinanceDays] = useState(7);
    const [isLiveStreaming, setIsLiveStreaming] = useState(false);

    // Refs
    const intervalRef = useRef<NodeJS.Timeout | null>(null);
    const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const allBarsRef = useRef<BarData[]>([]);
    const ocoRef = useRef<typeof ocoState>(null);
    const completedTradesRef = useRef<VizTrade[]>([]);
    const completedDecisionsRef = useRef<VizDecision[]>([]);
    const eventSourceRef = useRef<EventSource | null>(null);
    const dataSourceModeRef = useRef<DataSourceMode>(initialMode);

    // Load data based on selected mode
    useEffect(() => {
        dataSourceModeRef.current = dataSourceMode;
        if (dataSourceMode === 'SIMULATION') {
            loadSimulationData();
        } else {
            // YFinance: auto-load historical data
            resetState();
            setStatus('Loading YFinance data...');
            fetchYFinanceHistory();
        }
    }, [dataSourceMode, lastTradeTimestamp, runId]);

    const resetState = () => {
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        allBarsRef.current = [];
        setBars([]);
        setCurrentIndex(0);
        setStartIndex(0);
        setOcoState(null);
        ocoRef.current = null;
        setTriggers(0);
        setWins(0);
        setLosses(0);
        setCompletedTrades([]);
        setCompletedDecisions([]);
        completedTradesRef.current = [];
        completedDecisionsRef.current = [];
        setPlaybackState('STOPPED');
        setIsLiveStreaming(false);
    };

    const loadSimulationData = async () => {
        // Load from continuous contract JSON
        for (const port of [8000, 8001]) {
            try {
                const params = new URLSearchParams();
                params.set('timeframe', '1m');

                if (lastTradeTimestamp) {
                    const lastTradeDate = new Date(lastTradeTimestamp);
                    const startDate = new Date(lastTradeDate.getTime() - 2 * 24 * 60 * 60 * 1000);
                    const endDate = new Date(lastTradeDate.getTime() + 14 * 24 * 60 * 60 * 1000);
                    params.set('start', startDate.toISOString());
                    params.set('end', endDate.toISOString());
                }

                const res = await fetch(`http://localhost:${port}/market/continuous?${params}`);
                if (res.ok) {
                    const data = await res.json();
                    const loadedBars: BarData[] = data.bars.map((b: any) => ({
                        time: new Date(b.time).getTime() / 1000,
                        open: b.open,
                        high: b.high,
                        low: b.low,
                        close: b.close,
                        volume: b.volume || 0
                    }));

                    allBarsRef.current = loadedBars;

                    let simStartIdx = 0;
                    if (lastTradeTimestamp) {
                        const lastTradeTime = new Date(lastTradeTimestamp).getTime() / 1000;
                        simStartIdx = loadedBars.findIndex(b => b.time >= lastTradeTime);
                        if (simStartIdx === -1) simStartIdx = 0;
                    }
                    setStartIndex(simStartIdx);
                    setCurrentIndex(simStartIdx);
                    setStatus(`Ready (${loadedBars.length} bars from JSON)`);
                    return;
                }
            } catch { }
        }
        setStatus('Failed to load simulation data');
    };


    // Fetch YFinance history using simple JSON endpoint
    const fetchYFinanceHistory = async (): Promise<boolean> => {
        setStatus(`Fetching ${ticker} data from Yahoo...`);
        try {
            const data = await api.getYFinanceData(ticker, yfinanceDays);

            if (!data.bars || data.bars.length === 0) {
                setStatus(data.message || 'No data returned from YFinance');
                return false;
            }

            const historyBars: BarData[] = data.bars.map((b: any) => ({
                time: new Date(b.time).getTime() / 1000,
                open: b.open,
                high: b.high,
                low: b.low,
                close: b.close,
                volume: b.volume || 0
            }));

            console.log('[YFinance] Loaded', historyBars.length, 'bars');
            console.log('[YFinance] First bar:', historyBars[0]);
            console.log('[YFinance] Last bar:', historyBars[historyBars.length - 1]);

            allBarsRef.current = historyBars;
            setStartIndex(0);
            setCurrentIndex(0);
            setStatus(`Ready: ${historyBars.length} bars loaded. Press Play or Go Live.`);
            return true;
        } catch (e: any) {
            setStatus(`YFinance error: ${e.message}`);
            return false;
        }
    };

    // Helper to fetch a single new candle (manual trigger)
    const fetchNextBar = async () => {
        try {
            const data = await api.getYFinanceData(ticker, 1);
            if (data.bars && data.bars.length > 0) {
                const latestBar = data.bars[data.bars.length - 1];
                const latestTime = new Date(latestBar.time).getTime() / 1000;
                const ourLatestTime = allBarsRef.current.length > 0 ? allBarsRef.current[allBarsRef.current.length - 1].time : 0;
                if (latestTime > ourLatestTime) {
                    const newBar: BarData = {
                        time: latestTime,
                        open: latestBar.open,
                        high: latestBar.high,
                        low: latestBar.low,
                        close: latestBar.close,
                        volume: latestBar.volume || 0
                    };
                    console.log('[Manual] New bar fetched:', new Date(latestTime * 1000).toLocaleTimeString());
                    allBarsRef.current.push(newBar);
                    setBars(prev => [...prev, newBar]);
                    setCurrentIndex(prev => prev + 1);
                    processBar(newBar, allBarsRef.current.length - 1);
                    setStatus(`Manual fetch: ${new Date(latestTime * 1000).toLocaleTimeString()}`);
                } else {
                    setStatus('Manual fetch: No newer candle available yet');
                }
            }
        } catch (e) {
            console.error('[Manual] Error fetching next bar:', e);
            setStatus('Error fetching next candle');
        }
    };

    const goLive = async () => {
        // Collect params from UI
        const pctInput = document.getElementById('param_pct') as HTMLInputElement;
        const tfInput = document.getElementById('param_tf') as HTMLSelectElement;

        const entryParams: any = {};
        if (pctInput) entryParams.pct = parseFloat(pctInput.value);
        if (tfInput) entryParams.timeframe = tfInput.value;

        try {
            setStatus('Initializing Backend Session...');
            const config = {
                entry_type: entryType,
                entry_params: entryParams,
                stop_method: stopMethod,
                tp_method: tpMethod,
                stop_atr: stopAtr,
                tp_atr: tpAtr
            };

            const res = await api.startLiveReplay(
                ticker,
                selectedScanner, // Use scanner as strategy
                yfinanceDays,
                speed,
                config
            );

            if (res.session_id) {
                setStatus(`Session Started: ${res.session_id}. Connecting stream...`);
                setupEventSource(res.session_id);
            }
        } catch (e: any) {
            console.error(e);
            setStatus(`Failed to start live session: ${e.message}`);
        }
    };

    const setupEventSource = (sessionId: string) => {
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
        }

        const url = api.getReplayStreamUrl(sessionId);
        const eventSource = new EventSource(url);
        eventSourceRef.current = eventSource;

        setIsLiveStreaming(true);
        setPlaybackState('PLAYING');

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.type === 'BAR') {
                    // Update latest bar
                    const newBar: BarData = {
                        time: new Date(data.timestamp).getTime() / 1000,
                        open: data.data.open,
                        high: data.data.high,
                        low: data.data.low,
                        close: data.data.close,
                        volume: data.data.volume
                    };

                    // Append or update if timestamp matches last
                    setBars(prev => {
                        const last = prev[prev.length - 1];
                        if (last && last.time === newBar.time) {
                            // Update existing (unlikely in this stream but possible)
                            const updated = [...prev];
                            updated[updated.length - 1] = newBar;
                            allBarsRef.current = updated;
                            return updated;
                        } else {
                            // Append
                            const updated = [...prev, newBar];
                            allBarsRef.current = updated;
                            return updated;
                        }
                    });
                    setCurrentIndex(prev => prev + 1); // Auto scroll?
                    setStatus(`Live: ${new Date(newBar.time * 1000).toLocaleTimeString()}`);
                }
                else if (data.type === 'DECISION') {
                    // Backend triggered a pattern
                    // We can visualize this?
                    // For now, key is ORDER_SUBMIT usually follows
                }
                else if (data.type === 'ORDER_SUBMIT') {
                    const ocoData = data.data;
                    const newOco = {
                        entry: ocoData.entry_price,
                        stop: ocoData.stop_price,
                        tp: ocoData.tp_price,
                        startTime: new Date(data.timestamp).getTime() / 1000,
                        direction: ocoData.direction
                    };
                    ocoRef.current = newOco;
                    setOcoState(newOco);
                    setTriggers(prev => prev + 1);
                }
                else if (data.type === 'FILL') {
                    // Trade completed
                    // We construct VizTrade from event data or wait for full update?
                    // Simpler: Backend manages state, we just visualize active OCO closure
                    // OCOEngine (Py) emits outcome.
                    // But here we just clear the OCO box for now, 
                    // ideally we get the trade result from the backend event.

                    // For now, client logic clears OCO if it sees price hit levels? No, backend tells us.
                    ocoRef.current = null;
                    setOcoState(null);

                    // Note: Ideally we parse PnL from FILL event to update stats
                    const pnl = data.data.pnl_dollars || 0;
                    if (pnl > 0) setWins(prev => prev + 1);
                    else setLosses(prev => prev + 1);
                }
                else if (data.type === 'STREAM_END') {
                    eventSource.close();
                    setPlaybackState('STOPPED');
                    setStatus('Session Ends.');
                    setIsLiveStreaming(false);
                }
            } catch (e) {
                console.error('SSE Parse Error', e);
            }
        };

        eventSource.onerror = (e) => {
            console.error('SSE Error', e);
            eventSource.close();
            setIsLiveStreaming(false);
            setPlaybackState('STOPPED');
            setStatus('Connection connection lost.');
        };
    };

    // UI button for manual candle fetch (place near other controls)
    const manualFetchButton = (
        <button
            className="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 text-white rounded"
            onClick={fetchNextBar}
            disabled={dataSourceMode !== 'YFINANCE' || isLiveStreaming === false}
        >
            Fetch Next Candle
        </button>
    );


    const handlePlayPause = useCallback(async () => {
        // In live streaming mode, Play/Pause has no effect - use Stop to exit
        if (isLiveStreaming) {
            return;
        }

        if (playbackState === 'PLAYING') {
            // Pause
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
            setPlaybackState('PAUSED');
            setStatus('Paused');
        } else {
            // For YFinance mode: fetch history first if not loaded
            if (dataSourceMode === 'YFINANCE' && allBarsRef.current.length === 0) {
                const success = await fetchYFinanceHistory();
                if (!success) return;
            }
            // Play or Resume - same behavior for both Simulation and YFinance
            startPlayback();
        }
    }, [playbackState, currentIndex, startIndex, dataSourceMode, isLiveStreaming]);

    const handleStop = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
        }
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }
        setPlaybackState('STOPPED');
        setIsLiveStreaming(false);
        setCurrentIndex(startIndex);
        setBars([]);
        setOcoState(null);
        ocoRef.current = null;
        setTriggers(0);
        setWins(0);
        setLosses(0);
        setCompletedTrades([]);
        setCompletedDecisions([]);
        completedTradesRef.current = [];
        completedDecisionsRef.current = [];
        setStatus('Stopped');
    }, [startIndex]);

    const handleRewind = useCallback(() => {
        // Rewind by 100 bars or to start
        const newIndex = Math.max(startIndex, currentIndex - 100);
        setCurrentIndex(newIndex);

        // If playing, update bars
        if (playbackState === 'PLAYING' || playbackState === 'PAUSED') {
            setBars(allBarsRef.current.slice(startIndex, newIndex + 1));
        }

        setStatus(`Rewound to bar ${newIndex}`);
    }, [currentIndex, startIndex, playbackState]);

    const handleFastForward = useCallback(() => {
        // Fast forward by 100 bars or to end
        const newIndex = Math.min(allBarsRef.current.length - 1, currentIndex + 100);
        setCurrentIndex(newIndex);

        if (playbackState === 'PLAYING' || playbackState === 'PAUSED') {
            setBars(allBarsRef.current.slice(startIndex, newIndex + 1));
        }

        setStatus(`Fast forwarded to bar ${newIndex}`);
    }, [currentIndex, startIndex, playbackState]);

    const handleSeek = useCallback((index: number) => {
        setCurrentIndex(index);
        if (playbackState === 'PLAYING' || playbackState === 'PAUSED') {
            setBars(allBarsRef.current.slice(startIndex, index + 1));
        }
    }, [startIndex, playbackState]);

    const startPlayback = useCallback(() => {
        if (allBarsRef.current.length === 0) {
            setStatus('No data loaded');
            return;
        }

        // IMPORTANT: Check if resuming BEFORE setting state to PLAYING
        const isResuming = playbackState === 'PAUSED';

        // Clear any existing interval first
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }

        setPlaybackState('PLAYING');

        // If resuming, start from current position. Otherwise reset.
        let idx: number;
        if (isResuming) {
            // Resume from where we left off (currentIndex is already the next bar to show)
            idx = currentIndex + 1;
            setStatus('Resuming...');
        } else {
            // Fresh start
            idx = startIndex;
            setCurrentIndex(startIndex);
            setBars([]);
            setOcoState(null);
            ocoRef.current = null;
            setTriggers(0);
            setWins(0);
            setLosses(0);
            setCompletedTrades([]);
            setCompletedDecisions([]);
            completedTradesRef.current = [];
            completedDecisionsRef.current = [];
            setStatus('Playing...');
        }

        intervalRef.current = setInterval(() => {
            if (idx >= allBarsRef.current.length) {
                // In YFinance mode: Wait for new bars instead of stopping
                if (dataSourceModeRef.current === 'YFINANCE') {
                    const lastBar = allBarsRef.current[allBarsRef.current.length - 1];
                    const lastTime = lastBar ? new Date(lastBar.time * 1000).toLocaleTimeString() : 'N/A';
                    setStatus(`Live: Waiting for new candle... (Last: ${lastTime})`);
                    return; // Skip this tick, but keep interval alive!
                }

                // In Simulation mode: Stop as usual
                if (intervalRef.current) {
                    clearInterval(intervalRef.current);
                    intervalRef.current = null;
                }
                setPlaybackState('STOPPED');
                setStatus('Completed');
                return;
            }

            const bar = allBarsRef.current[idx];
            setBars(prev => [...prev, bar]);
            setCurrentIndex(idx);

            // Process OCO exits and model triggers
            processBar(bar, idx);

            idx++;
        }, speed);
    }, [speed, startIndex, currentIndex, playbackState]);

    const processBar = (bar: BarData, idx: number) => {
        // OCO Exit Logic
        if (ocoRef.current) {
            let outcome = '';
            let price = 0;
            const isLong = ocoRef.current.direction === 'LONG';

            if (isLong) {
                if (bar.low <= ocoRef.current.stop) {
                    outcome = 'LOSS';
                    price = ocoRef.current.stop;
                    setLosses(prev => prev + 1);
                } else if (bar.high >= ocoRef.current.tp) {
                    outcome = 'WIN';
                    price = ocoRef.current.tp;
                    setWins(prev => prev + 1);
                }
            } else {
                if (bar.high >= ocoRef.current.stop) {
                    outcome = 'LOSS';
                    price = ocoRef.current.stop;
                    setLosses(prev => prev + 1);
                } else if (bar.low <= ocoRef.current.tp) {
                    outcome = 'WIN';
                    price = ocoRef.current.tp;
                    setWins(prev => prev + 1);
                }
            }

            if (outcome) {
                const tradeId = `sim_${ocoRef.current.startTime}`;
                const barsHeld = Math.max(1, Math.round((bar.time - ocoRef.current.startTime) / 60));

                const decision: VizDecision = {
                    decision_id: tradeId,
                    timestamp: new Date(ocoRef.current.startTime * 1000).toISOString(),
                    bar_idx: 0, index: 0, scanner_id: selectedScanner, scanner_context: {},
                    action: 'OCO', skip_reason: '', current_price: ocoRef.current.entry,
                    atr: 0, cf_outcome: outcome, cf_pnl_dollars: 0,
                    oco: {
                        entry_price: ocoRef.current.entry,
                        stop_price: ocoRef.current.stop,
                        tp_price: ocoRef.current.tp,
                        entry_type: 'MARKET', direction: ocoRef.current.direction, reference_type: '',
                        reference_value: 0, atr_at_creation: 0, max_bars: 100,
                        stop_atr: 0, tp_multiple: 0
                    },
                    oco_results: {
                        simulation: {
                            bars_held: barsHeld,
                            outcome: outcome,
                            pnl_dollars: (isLong ? (price - ocoRef.current.entry) : (ocoRef.current.entry - price)) * 50
                        }
                    }
                };

                const trade: VizTrade = {
                    trade_id: tradeId, decision_id: tradeId, index: completedTradesRef.current.length,
                    direction: ocoRef.current.direction, size: 1,
                    entry_time: new Date(ocoRef.current.startTime * 1000).toISOString(),
                    entry_bar: 0, entry_price: ocoRef.current.entry,
                    exit_time: new Date(bar.time * 1000).toISOString(),
                    exit_bar: 0, exit_price: price,
                    exit_reason: outcome === 'WIN' ? 'TP' : 'SL',
                    outcome: outcome, pnl_points: isLong ? (price - ocoRef.current.entry) : (ocoRef.current.entry - price),
                    pnl_dollars: (isLong ? (price - ocoRef.current.entry) : (ocoRef.current.entry - price)) * 50,
                    r_multiple: outcome === 'WIN' ? 2 : -1, bars_held: 0, mae: 0, mfe: 0, fills: []
                };

                completedDecisionsRef.current.push(decision);
                completedTradesRef.current.push(trade);
                setCompletedDecisions([...completedDecisionsRef.current]);
                setCompletedTrades([...completedTradesRef.current]);

                ocoRef.current = null;
                setOcoState(null);
            }
        }

        // Trigger Logic (Entry) - SIMULATION mode only, either CNN model or pattern scanner
        // In YFinance mode, the backend (session_live.py) handles strategy triggering
        const canTrigger = dataSourceModeRef.current === 'SIMULATION' && !ocoRef.current && idx % 5 === 0 && idx >= 60;
        const shouldUseCnn = canTrigger && useCnnModel;
        const shouldUseScanner = canTrigger && usePatternScanner;

        // Pattern Scanner Trigger (simple local implementation)
        if (shouldUseScanner && !shouldUseCnn) {
            const recentBars = allBarsRef.current.slice(Math.max(0, idx - 13), idx + 1);
            const avgRange = recentBars.reduce((sum, b) => sum + (b.high - b.low), 0) / recentBars.length;
            const atr = avgRange || (bar.close * 0.001);

            // Simple pattern detection based on selected scanner
            let triggered = false;
            let direction: 'LONG' | 'SHORT' | null = null;

            if (selectedScanner === 'ema_cross' && recentBars.length >= 9) {
                // Simple EMA cross check
                const closes = recentBars.map(b => b.close);
                const fast = closes.slice(-3).reduce((a, b) => a + b, 0) / 3;
                const slow = closes.slice(-9).reduce((a, b) => a + b, 0) / 9;
                const prevFast = closes.slice(-4, -1).reduce((a, b) => a + b, 0) / 3;
                const prevSlow = closes.slice(-10, -1).reduce((a, b) => a + b, 0) / 9;

                if (prevFast <= prevSlow && fast > slow) {
                    triggered = true;
                    direction = 'LONG';
                } else if (prevFast >= prevSlow && fast < slow) {
                    triggered = true;
                    direction = 'SHORT';
                }
            } else if (selectedScanner === 'ifvg' && recentBars.length >= 3) {
                // Simple IFVG detection (fair value gap)
                const b1 = recentBars[recentBars.length - 3];
                const b2 = recentBars[recentBars.length - 2];
                const b3 = recentBars[recentBars.length - 1];

                // Bullish FVG: gap between bar1 high and bar3 low
                if (b1.high < b3.low && b2.close > b2.open) {
                    triggered = true;
                    direction = 'LONG';
                }
                // Bearish FVG: gap between bar1 low and bar3 high  
                else if (b1.low > b3.high && b2.close < b2.open) {
                    triggered = true;
                    direction = 'SHORT';
                }
            }

            if (triggered && direction) {
                const entry = bar.close;
                const isLong = direction === 'LONG';
                const stop = isLong ? entry - (stopAtr * atr) : entry + (stopAtr * atr);
                const tp = isLong ? entry + (tpAtr * atr) : entry - (tpAtr * atr);

                const newOco = { entry, stop, tp, startTime: bar.time, direction };
                ocoRef.current = newOco;
                setOcoState(newOco);
                setTriggers(prev => prev + 1);
            }
        }

        // CNN Model Trigger (calls backend /infer endpoint)
        if (shouldUseCnn) {
            const windowBars = allBarsRef.current.slice(Math.max(0, idx - 29), idx + 1);
            const recentBars = allBarsRef.current.slice(Math.max(0, idx - 13), idx + 1);
            const avgRange = recentBars.reduce((sum, b) => sum + (b.high - b.low), 0) / recentBars.length;
            const atr = avgRange || (bar.close * 0.001);

            const tryInfer = async (port: number) => {
                try {
                    const res = await fetch(`http://localhost:${port}/infer`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            bars: windowBars.map(b => ({
                                open: b.open,
                                high: b.high,
                                low: b.low,
                                close: b.close,
                                volume: b.volume || 0
                            })),
                            model_path: selectedModel,
                            threshold: threshold
                        })
                    });
                    return await res.json();
                } catch {
                    return null;
                }
            };

            tryInfer(8000).then(result => {
                if (!result) return tryInfer(8001);
                return result;
            }).then(result => {
                if (result?.triggered && result.direction !== 'NONE' && !ocoRef.current) {
                    const entry = bar.close;
                    const isLong = result.direction === 'LONG';
                    const stop = isLong ? entry - (stopAtr * atr) : entry + (stopAtr * atr);
                    const tp = isLong ? entry + (tpAtr * atr) : entry - (tpAtr * atr);

                    const newOco = {
                        entry,
                        stop,
                        tp,
                        startTime: bar.time,
                        direction: result.direction as 'LONG' | 'SHORT'
                    };
                    ocoRef.current = newOco;
                    setOcoState(newOco);
                    setTriggers(prev => prev + 1);
                }
            }).catch(e => console.error('Infer error:', e));
        }
    };

    // Restart interval when speed changes during playback
    useEffect(() => {
        if (playbackState === 'PLAYING' && intervalRef.current && !isLiveStreaming) {
            // Clear old interval
            clearInterval(intervalRef.current);
            intervalRef.current = null;

            // Create new interval with updated speed, starting from current position + 1
            let idx = currentIndex + 1;
            intervalRef.current = setInterval(() => {
                if (idx >= allBarsRef.current.length) {
                    if (intervalRef.current) {
                        clearInterval(intervalRef.current);
                        intervalRef.current = null;
                    }
                    setPlaybackState('STOPPED');
                    setStatus('Completed');
                    return;
                }

                const bar = allBarsRef.current[idx];
                setBars(prev => [...prev, bar]);
                setCurrentIndex(idx);
                processBar(bar, idx);
                idx++;
            }, speed);

            setStatus(`Speed changed to ${speed}ms`);
        }
    }, [speed]);

    useEffect(() => {
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
                eventSourceRef.current = null;
            }
        };
    }, []);

    return (
        <div className="fixed inset-0 bg-slate-900 z-50 flex flex-col">
            {/* Header */}
            <div className="h-14 bg-slate-800 border-b border-slate-700 flex items-center justify-between px-4">
                <h1 className="text-white font-bold">Live Session Mode</h1>
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <span className="text-xs text-slate-400">Data Source:</span>
                        <select
                            value={dataSourceMode}
                            onChange={e => setDataSourceMode(e.target.value as DataSourceMode)}
                            disabled={playbackState === 'PLAYING'}
                            className="bg-slate-700 border border-slate-600 rounded px-2 py-1 text-xs text-white"
                        >
                            <option value="SIMULATION">Simulation (JSON)</option>
                            <option value="YFINANCE">YFinance (API)</option>
                        </select>
                    </div>
                    <button onClick={onClose} className="text-slate-400 hover:text-white">✕ Close</button>
                </div>
            </div>

            <div className="flex-1 flex overflow-hidden min-h-0">
                {/* Main Chart Area */}
                <div className="flex-1 flex flex-col min-h-0">
                    <div className="flex-1 min-h-[400px]">
                        {/* Debug: Log first few bars to console */}
                        {bars.length > 0 && console.log('[Chart Input] First 3 bars:', bars.slice(0, 3).map(b => ({
                            time: b.time,
                            timeISO: new Date(b.time * 1000).toISOString(),
                            open: b.open, high: b.high, low: b.low, close: b.close,
                            range: b.high - b.low
                        })))}
                        <CandleChart
                            continuousData={bars.length > 0 ? {
                                timeframe: '1m',
                                count: bars.length,
                                bars: bars.map(b => ({
                                    time: new Date(b.time * 1000).toISOString(),
                                    open: b.open, high: b.high, low: b.low, close: b.close, volume: b.volume
                                }))
                            } : null}
                            decisions={completedDecisions}
                            activeDecision={null}
                            trade={null}
                            trades={completedTrades}
                            simulationOco={ocoState}
                        />
                    </div>
                </div>

                {/* Right Sidebar - Controls */}
                <div className="w-80 bg-slate-800 border-l border-slate-700 p-4 overflow-y-auto min-h-0 max-h-full">

                    {/* Playback Controls */}
                    <SidebarSection title="Playback" defaultOpen={true} colorClass="text-green-400">
                        <div className="flex gap-2 mb-3">
                            <button
                                onClick={handlePlayPause}
                                disabled={dataSourceMode === 'SIMULATION' && allBarsRef.current.length === 0}
                                className={`flex-1 font-bold py-2 px-3 rounded text-sm ${(dataSourceMode === 'SIMULATION' && allBarsRef.current.length === 0)
                                    ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                                    : playbackState === 'PLAYING'
                                        ? 'bg-yellow-600 hover:bg-yellow-500 text-white'
                                        : 'bg-green-600 hover:bg-green-500 text-white'
                                    }`}
                            >
                                {playbackState === 'PLAYING' ? '⏸ Pause' : '▶ Play'}
                            </button>
                            <button
                                onClick={handleStop}
                                disabled={playbackState === 'STOPPED'}
                                className="flex-1 bg-red-600 hover:bg-red-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-bold py-2 px-3 rounded text-sm"
                            >
                                ■ Stop
                            </button>
                        </div>

                        {/* Go Live button - YFinance only */}
                        {dataSourceMode === 'YFINANCE' && (
                            <div className="mb-3">
                                <button
                                    onClick={goLive}
                                    disabled={isLiveStreaming}
                                    className={`w-full font-bold py-2 px-3 rounded text-sm ${isLiveStreaming
                                        ? 'bg-orange-700 text-orange-300 cursor-not-allowed'
                                        : 'bg-orange-600 hover:bg-orange-500 text-white'
                                        }`}
                                >
                                    {isLiveStreaming ? '📡 Live Streaming...' : '⏩ Go Live (Realtime)'}
                                </button>
                            </div>
                        )}

                        <div className="flex gap-2 mb-3">
                            <button
                                onClick={handleRewind}
                                disabled={playbackState === 'STOPPED'}
                                className="flex-1 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white py-1 px-2 rounded text-xs"
                            >
                                ⏪ -100
                            </button>
                            <button
                                onClick={handleFastForward}
                                disabled={playbackState === 'STOPPED'}
                                className="flex-1 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white py-1 px-2 rounded text-xs"
                            >
                                +100 ⏩
                            </button>
                        </div>

                        <div className="mb-3">
                            <label className="text-xs text-slate-400 mb-1 block">Speed (ms per bar)</label>
                            <select
                                value={speed}
                                onChange={e => setSpeed(parseInt(e.target.value))}
                                className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                            >
                                <option value={500}>Slow (500ms)</option>
                                <option value={200}>Normal (200ms)</option>
                                <option value={100}>Fast (100ms)</option>
                                <option value={50}>Very Fast (50ms)</option>
                                <option value={10}>Max (10ms)</option>
                            </select>
                        </div>

                        <div className="mb-1">
                            <label className="text-xs text-slate-400 mb-1 block">
                                Position: {currentIndex} / {allBarsRef.current.length}
                            </label>
                            <input
                                type="range"
                                min={startIndex}
                                max={Math.max(startIndex, allBarsRef.current.length - 1)}
                                value={currentIndex}
                                onChange={e => handleSeek(parseInt(e.target.value))}
                                disabled={playbackState === 'STOPPED'}
                                className="w-full"
                            />
                        </div>
                    </SidebarSection>

                    {/* Data Source Specific Settings */}
                    {dataSourceMode === 'YFINANCE' && (
                        <SidebarSection title="YFinance Settings" defaultOpen={true} colorClass="text-purple-400">
                            <div className="mb-1">
                                <label className="text-xs text-slate-400 mb-1 block">Ticker</label>
                                <input
                                    type="text"
                                    value={ticker}
                                    onChange={e => setTicker(e.target.value)}
                                    disabled={playbackState === 'PLAYING' || isLiveStreaming}
                                    className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                                />
                            </div>
                        </SidebarSection>
                    )}

                    {/* Model Selection */}
                    <SidebarSection title="Trigger Sources" colorClass="text-cyan-400">
                        <div className="mb-3">
                            <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer mb-1">
                                <input
                                    type="checkbox"
                                    checked={useCnnModel}
                                    onChange={e => setUseCnnModel(e.target.checked)}
                                    disabled={playbackState === 'PLAYING'}
                                    className="w-4 h-4"
                                />
                                Use CNN Model
                            </label>
                            {useCnnModel && (
                                <select
                                    value={selectedModel}
                                    onChange={e => setSelectedModel(e.target.value)}
                                    disabled={playbackState === 'PLAYING'}
                                    className="w-full mt-1 bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                                >
                                    {availableModels.map(model => (
                                        <option key={model} value={model}>{model.split('/').pop()}</option>
                                    ))}
                                </select>
                            )}
                        </div>

                        <div className="mb-1">
                            <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer mb-1">
                                <input
                                    type="checkbox"
                                    checked={usePatternScanner}
                                    onChange={e => setUsePatternScanner(e.target.checked)}
                                    disabled={playbackState === 'PLAYING'}
                                    className="w-4 h-4"
                                />
                                Use Pattern Scanner
                            </label>
                            {usePatternScanner && (
                                <select
                                    value={selectedScanner}
                                    onChange={e => setSelectedScanner(e.target.value)}
                                    disabled={playbackState === 'PLAYING'}
                                    className="w-full mt-1 bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                                >
                                    <option value="ifvg">IFVG</option>
                                    <option value="ema_cross">EMA Cross</option>
                                    <option value="ema_bounce">EMA Bounce</option>
                                </select>
                            )}
                        </div>

                        {useCnnModel && usePatternScanner && (
                            <p className="text-xs text-yellow-400 mt-1">⚠ Both enabled: requires BOTH to trigger (AND)</p>
                        )}
                    </SidebarSection>

                    {/* Entry Configuration */}
                    <SidebarSection title="Entry Configuration" colorClass="text-purple-400">
                        <div className="mb-3">
                            <label className="text-xs text-slate-400 mb-1 block">Entry Strategy</label>
                            <select
                                value={entryType}
                                onChange={e => setEntryType(e.target.value as any)}
                                disabled={playbackState === 'PLAYING'}
                                className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white mb-2"
                            >
                                <option value="market">Market</option>
                                <option value="limit">Limit (Legacy)</option>
                                <option value="retrace_signal">Retrace (Signal Bar)</option>
                                <option value="retrace_timeframe">Retrace (Timeframe)</option>
                                <option value="breakout">Breakout</option>
                            </select>

                            {/* Dynamic Params based on strategy */}
                            {(entryType === 'retrace_signal' || entryType === 'retrace_timeframe') && (
                                <div className="mb-2 pl-2 border-l-2 border-slate-700">
                                    <label className="text-xs text-slate-500 mb-1 block">Retrace % (0.0 - 1.0)</label>
                                    <input
                                        type="number"
                                        step="0.1"
                                        min="0.1"
                                        max="1.0"
                                        defaultValue="0.5"
                                        id="param_pct"
                                        className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs text-white"
                                    />
                                </div>
                            )}

                            {entryType === 'retrace_timeframe' && (
                                <div className="mb-2 pl-2 border-l-2 border-slate-700">
                                    <label className="text-xs text-slate-500 mb-1 block">Timeframe</label>
                                    <select
                                        id="param_tf"
                                        defaultValue="15m"
                                        className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs text-white"
                                    >
                                        <option value="5m">5m</option>
                                        <option value="15m">15m</option>
                                        <option value="1h">1h</option>
                                    </select>
                                </div>
                            )}
                        </div>

                        <div className="flex gap-2 mb-2">
                            <button
                                className={`flex-1 px-3 py-1 rounded text-sm font-medium ${isLiveStreaming ? 'bg-red-600 hover:bg-red-500' : 'bg-green-600 hover:bg-green-500'} disabled:opacity-50`}
                                onClick={isLiveStreaming ? handleStop : goLive}
                                disabled={playbackState === 'PLAYING' && !isLiveStreaming}
                            >
                                {isLiveStreaming ? 'Stop Live' : 'Go Live'}
                            </button>
                            {isLiveStreaming && (
                                <button
                                    className="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-sm"
                                    onClick={fetchNextBar}
                                    title="Force check for new data"
                                >
                                    Force Fetch
                                </button>
                            )}
                        </div>
                        {dataSourceMode === 'YFINANCE' && (
                            <p className="text-xs text-yellow-500 mb-2">
                                Note: YFinance data is delayed ~15 mins.
                                <br />Last bar: {allBarsRef.current.length > 0 ? new Date(allBarsRef.current[allBarsRef.current.length - 1].time * 1000).toLocaleTimeString() : 'N/A'}
                            </p>
                        )}

                        <div className="mb-3">
                            <label className="text-xs text-slate-400 mb-1 block">Stop Placement</label>
                            <select
                                value={stopMethod}
                                onChange={e => setStopMethod(e.target.value as any)}
                                disabled={playbackState === 'PLAYING'}
                                className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                            >
                                <option value="atr">ATR Multiple</option>
                                <option value="swing">Behind Swing</option>
                                <option value="fixed_bars">Fixed Bars</option>
                            </select>
                        </div>

                        <div className="mb-1">
                            <label className="text-xs text-slate-400 mb-1 block">Take Profit</label>
                            <select
                                value={tpMethod}
                                onChange={e => setTpMethod(e.target.value as any)}
                                disabled={playbackState === 'PLAYING'}
                                className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                            >
                                <option value="atr">ATR Multiple</option>
                                <option value="r_multiple">R-Multiple</option>
                            </select>
                        </div>
                    </SidebarSection>

                    {/* OCO Settings */}
                    <SidebarSection title="OCO Settings" colorClass="text-orange-400">
                        <div className="mb-3">
                            <label className="text-xs text-slate-400 mb-1 block">Threshold</label>
                            <input
                                type="number"
                                step="0.01"
                                min="0.1"
                                max="0.9"
                                value={threshold}
                                onChange={e => setThreshold(parseFloat(e.target.value) || 0.35)}
                                disabled={playbackState === 'PLAYING'}
                                className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                            />
                        </div>
                        <div className="mb-3">
                            <label className="text-xs text-slate-400 mb-1 block">Stop Loss (ATR ×)</label>
                            <input
                                type="number"
                                step="0.5"
                                min="0.5"
                                max="10"
                                value={stopAtr}
                                onChange={e => setStopAtr(parseFloat(e.target.value) || 2)}
                                disabled={playbackState === 'PLAYING'}
                                className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                            />
                        </div>
                        <div className="mb-1">
                            <label className="text-xs text-slate-400 mb-1 block">Take Profit (ATR ×)</label>
                            <input
                                type="number"
                                step="0.5"
                                min="0.5"
                                max="20"
                                value={tpAtr}
                                onChange={e => setTpAtr(parseFloat(e.target.value) || 4)}
                                disabled={playbackState === 'PLAYING'}
                                className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-sm text-white"
                            />
                        </div>
                    </SidebarSection>

                    {/* Status & Stats */}
                    <SidebarSection title="Status & Stats" defaultOpen={true} colorClass="text-white">
                        <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                                <span className="text-slate-400">Status:</span>
                                <span className="text-white bg-slate-900 px-2 py-1 rounded text-xs">{status}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-400">Mode:</span>
                                <span className="text-blue-400">{dataSourceMode}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-400">Triggers:</span>
                                <span className="text-yellow-400">{triggers}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-400">Wins:</span>
                                <span className="text-green-400">{wins}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-400">Losses:</span>
                                <span className="text-red-400">{losses}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-400">Win Rate:</span>
                                <span className="text-cyan-400">
                                    {(wins + losses) > 0 ? ((wins / (wins + losses)) * 100).toFixed(1) : '0.0'}%
                                </span>
                            </div>
                        </div>
                    </SidebarSection>
                </div>
            </div>
        </div>
    );
};

```

### src/components/Navigator.tsx

```tsx
import React from 'react';

interface NavigatorProps {
  mode: 'DECISION' | 'TRADE';
  setMode: (m: 'DECISION' | 'TRADE') => void;
  index: number;
  setIndex: (i: number) => void;
  maxIndex: number;
}

export const Navigator: React.FC<NavigatorProps> = ({ mode, setMode, index, setIndex, maxIndex }) => {
  return (
    <div className="bg-slate-900/40 rounded-lg p-4 border border-slate-800/60 backdrop-blur-sm">
      
      {/* Mode Toggle */}
      <div className="flex bg-slate-950 rounded-lg p-1.5 mb-5 shadow-inner border border-slate-800">
        <button 
          onClick={() => setMode('DECISION')}
          className={`flex-1 text-[11px] uppercase tracking-wider py-2 rounded-md font-bold transition-all duration-200 ${mode === 'DECISION' ? 'bg-blue-600 text-white shadow-md' : 'text-slate-500 hover:text-slate-300'}`}
        >
          Decisions
        </button>
        <button 
          onClick={() => setMode('TRADE')}
          className={`flex-1 text-[11px] uppercase tracking-wider py-2 rounded-md font-bold transition-all duration-200 ${mode === 'TRADE' ? 'bg-blue-600 text-white shadow-md' : 'text-slate-500 hover:text-slate-300'}`}
        >
          Trades
        </button>
      </div>

      {/* Navigation Controls */}
      <div>
        <div className="flex justify-between items-center mb-3">
          <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">
            Index <span className="text-white font-mono bg-slate-800 px-1.5 py-0.5 rounded ml-1">{index}</span> <span className="text-slate-600">/</span> <span className="font-mono text-slate-400">{maxIndex}</span>
          </span>
        </div>
        
        <div className="flex gap-2 mb-4">
          <button 
            onClick={() => setIndex(Math.max(0, index - 1))}
            className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-200 py-2 rounded-md text-xs font-bold border border-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:border-slate-600 active:transform active:scale-95"
            disabled={index <= 0}
          >
            ← Prev
          </button>
          <button 
            onClick={() => setIndex(Math.min(maxIndex, index + 1))}
            className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-200 py-2 rounded-md text-xs font-bold border border-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:border-slate-600 active:transform active:scale-95"
            disabled={index >= maxIndex}
          >
            Next →
          </button>
        </div>

        <div className="relative h-4 flex items-center mb-2">
            <input
              type="range"
              min="0"
              max={maxIndex}
              value={index}
              onChange={(e) => setIndex(parseInt(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500 hover:accent-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-500/30"
            />
        </div>
      </div>

      <div className="relative group">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <svg className="h-3 w-3 text-slate-500 group-focus-within:text-blue-500 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <input 
            type="text" 
            placeholder="Search ID..." 
            className="w-full bg-slate-950 border border-slate-800 rounded-md pl-8 pr-2 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 transition-all"
          />
      </div>

    </div>
  );
};

```

### src/components/PositionBox.ts

```typescript
/**
 * Position Box Primitive for Lightweight Charts
 * 
 * Draws bounded rectangles for SL/TP zones like TradingView's position tool.
 * Only spans from startTime to endTime, not full-width.
 */

import {
    ISeriesPrimitive,
    SeriesAttachedParameter,
    Time,
    ISeriesPrimitivePaneView,
    ISeriesPrimitivePaneRenderer,
    PrimitiveHoveredItem,
    SeriesPrimitivePaneViewZOrder,
} from 'lightweight-charts';

export interface PositionBoxOptions {
    id: string;
    startTime: Time;
    endTime: Time;
    topPrice: number;
    bottomPrice: number;
    fillColor: string;
    borderColor: string;
    borderWidth?: number;
    label?: string;
    labelColor?: string;
}

class PositionBoxRenderer implements ISeriesPrimitivePaneRenderer {
    private _data: PositionBoxOptions;
    private _x1: number = 0;
    private _x2: number = 0;
    private _y1: number = 0;
    private _y2: number = 0;

    constructor(data: PositionBoxOptions) {
        this._data = data;
    }

    update(x1: number, x2: number, y1: number, y2: number) {
        this._x1 = x1;
        this._x2 = x2;
        this._y1 = y1;
        this._y2 = y2;
    }

    // lightweight-charts v4 uses CanvasRenderingTarget2D which wraps the context
    // We use 'any' to avoid type gymnastics - the runtime API is stable
    draw(target: any): void {
        target.useMediaCoordinateSpace(({ context: ctx }: { context: CanvasRenderingContext2D }) => {
            const width = Math.abs(this._x2 - this._x1);
            const height = Math.abs(this._y2 - this._y1);
            const x = Math.min(this._x1, this._x2);
            const y = Math.min(this._y1, this._y2);

            if (width <= 0 || height <= 0) return;

            // Draw filled rectangle
            ctx.fillStyle = this._data.fillColor;
            ctx.fillRect(x, y, width, height);

            // Draw border
            ctx.strokeStyle = this._data.borderColor;
            ctx.lineWidth = this._data.borderWidth || 1;
            ctx.strokeRect(x, y, width, height);

            // Draw label if provided
            if (this._data.label) {
                ctx.font = 'bold 10px sans-serif';
                ctx.fillStyle = this._data.labelColor || this._data.borderColor;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                ctx.fillText(this._data.label, x + 4, y + 2);
            }
        });
    }
}

class PositionBoxPaneView implements ISeriesPrimitivePaneView {
    private _source: PositionBox;
    private _renderer: PositionBoxRenderer;

    constructor(source: PositionBox) {
        this._source = source;
        this._renderer = new PositionBoxRenderer(source.options);
    }

    zOrder(): SeriesPrimitivePaneViewZOrder {
        return 'bottom';
    }

    renderer(): ISeriesPrimitivePaneRenderer {
        const series = this._source.series;
        const timeScale = this._source.chart?.timeScale();

        if (!series || !timeScale) {
            this._renderer.update(0, 0, 0, 0);
            return this._renderer;
        }

        const opts = this._source.options;

        // Convert times to x coordinates
        const x1 = timeScale.timeToCoordinate(opts.startTime);
        const x2 = timeScale.timeToCoordinate(opts.endTime);

        // Convert prices to y coordinates
        const y1 = series.priceToCoordinate(opts.topPrice);
        const y2 = series.priceToCoordinate(opts.bottomPrice);

        if (x1 === null || x2 === null || y1 === null || y2 === null) {
            this._renderer.update(0, 0, 0, 0);
            return this._renderer;
        }

        this._renderer.update(x1, x2, y1, y2);
        return this._renderer;
    }
}

export class PositionBox implements ISeriesPrimitive<Time> {
    private _paneViews: PositionBoxPaneView[];
    private _options: PositionBoxOptions;
    private _series: SeriesAttachedParameter<Time> | null = null;

    constructor(options: PositionBoxOptions) {
        this._options = options;
        this._paneViews = [new PositionBoxPaneView(this)];
    }

    get options(): PositionBoxOptions {
        return this._options;
    }

    get series() {
        return this._series?.series ?? null;
    }

    get chart() {
        return this._series?.chart ?? null;
    }

    attached(param: SeriesAttachedParameter<Time>): void {
        this._series = param;
    }

    detached(): void {
        this._series = null;
    }

    paneViews(): readonly ISeriesPrimitivePaneView[] {
        return this._paneViews;
    }

    updateOptions(options: Partial<PositionBoxOptions>): void {
        this._options = { ...this._options, ...options };
    }

    hitTest(): PrimitiveHoveredItem | null {
        return null;
    }
}

// Helper to create SL and TP boxes for a trade
export function createTradePositionBoxes(
    entryPrice: number,
    stopPrice: number,
    tpPrice: number,
    startTime: Time,
    endTime: Time,
    direction: 'LONG' | 'SHORT',
    tradeId: string = 'default',
    labels?: { sl?: string; tp?: string }
): { slBox: PositionBox; tpBox: PositionBox; } {

    // SL Zone (red)
    const slBox = new PositionBox({
        id: `sl_${tradeId}`,
        startTime,
        endTime,
        topPrice: Math.max(entryPrice, stopPrice),
        bottomPrice: Math.min(entryPrice, stopPrice),
        fillColor: 'rgba(239, 68, 68, 0.15)',  // red-500 @ 15%
        borderColor: '#ef4444',
        borderWidth: 1,
        label: labels?.sl,
        labelColor: '#ef4444'
    });

    // TP Zone (green)
    const tpBox = new PositionBox({
        id: `tp_${tradeId}`,
        startTime,
        endTime,
        topPrice: Math.max(entryPrice, tpPrice),
        bottomPrice: Math.min(entryPrice, tpPrice),
        fillColor: 'rgba(34, 197, 94, 0.15)',  // green-500 @ 15%
        borderColor: '#22c55e',
        borderWidth: 1,
        label: labels?.tp,
        labelColor: '#22c55e'
    });

    return { slBox, tpBox };
}

```

### src/components/ReplayControls.tsx

```tsx
import React, { useState, useEffect, useRef, useCallback } from 'react';

interface ReplayControlsProps {
    maxIndex: number;
    currentIndex: number;
    onIndexChange: (index: number) => void;
    onReplayStart: () => void;
    onReplayEnd: () => void;
}

/**
 * Replay Controls - Auto-step through existing decisions
 * 
 * This doesn't load new data - it animates through the bars/decisions
 * already loaded in the chart.
 */
export const ReplayControls: React.FC<ReplayControlsProps> = ({
    maxIndex,
    currentIndex,
    onIndexChange,
    onReplayStart,
    onReplayEnd
}) => {
    const [isPlaying, setIsPlaying] = useState(false);
    const [speed, setSpeed] = useState<number>(500); // ms per step
    const [playbackIndex, setPlaybackIndex] = useState<number>(0);

    const intervalRef = useRef<NodeJS.Timeout | null>(null);

    // Start playback
    const startReplay = useCallback(() => {
        if (maxIndex <= 0) return;

        setIsPlaying(true);
        setPlaybackIndex(0);
        onIndexChange(0);
        onReplayStart();

        // Start stepping through
        intervalRef.current = setInterval(() => {
            setPlaybackIndex(prev => {
                const next = prev + 1;
                if (next > maxIndex) {
                    // Reached end
                    stopReplay();
                    return prev;
                }
                onIndexChange(next);
                return next;
            });
        }, speed);
    }, [maxIndex, speed, onIndexChange, onReplayStart]);

    // Stop playback
    const stopReplay = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        setIsPlaying(false);
        onReplayEnd();
    }, [onReplayEnd]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, []);

    // Update speed while playing
    useEffect(() => {
        if (isPlaying && intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = setInterval(() => {
                setPlaybackIndex(prev => {
                    const next = prev + 1;
                    if (next > maxIndex) {
                        stopReplay();
                        return prev;
                    }
                    onIndexChange(next);
                    return next;
                });
            }, speed);
        }
    }, [speed, isPlaying, maxIndex, onIndexChange, stopReplay]);

    const progress = maxIndex > 0 ? (playbackIndex / maxIndex) * 100 : 0;

    return (
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-4">
            <h3 className="text-sm font-bold text-blue-400 uppercase mb-3">Replay Mode</h3>

            {/* Speed Control */}
            <div className="mb-3">
                <label className="text-xs text-slate-400">Speed</label>
                <select
                    value={speed}
                    onChange={e => setSpeed(parseInt(e.target.value))}
                    className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-xs text-white"
                >
                    <option value={2000}>0.5x (2s per decision)</option>
                    <option value={1000}>1x (1s per decision)</option>
                    <option value={500}>2x (0.5s per decision)</option>
                    <option value={200}>5x (0.2s per decision)</option>
                    <option value={100}>10x (0.1s per decision)</option>
                </select>
            </div>

            {/* Progress Bar */}
            <div className="mb-3">
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                    <span>Progress</span>
                    <span>{playbackIndex} / {maxIndex}</span>
                </div>
                <div className="w-full bg-slate-900 rounded-full h-2">
                    <div
                        className="bg-blue-500 h-2 rounded-full transition-all duration-200"
                        style={{ width: `${progress}%` }}
                    />
                </div>
            </div>

            {/* Controls */}
            <div className="flex gap-2 mb-3">
                {!isPlaying ? (
                    <button
                        onClick={startReplay}
                        disabled={maxIndex <= 0}
                        className={`flex-1 font-bold py-2 px-4 rounded text-sm ${maxIndex > 0
                                ? 'bg-green-600 hover:bg-green-500 text-white'
                                : 'bg-slate-700 text-slate-500 cursor-not-allowed'
                            }`}
                    >
                        ▶ Play
                    </button>
                ) : (
                    <button
                        onClick={stopReplay}
                        className="flex-1 bg-red-600 hover:bg-red-500 text-white font-bold py-2 px-4 rounded text-sm"
                    >
                        ■ Stop
                    </button>
                )}
            </div>

            {/* Status */}
            <div className="text-xs space-y-1">
                <div className="flex justify-between">
                    <span className="text-slate-400">Status:</span>
                    <span className={isPlaying ? 'text-green-400' : 'text-slate-300'}>
                        {maxIndex <= 0 ? 'No data loaded' : isPlaying ? 'Playing...' : 'Ready'}
                    </span>
                </div>
                {maxIndex <= 0 && (
                    <p className="text-slate-500 text-xs mt-2">
                        Select a run with decisions to enable replay.
                    </p>
                )}
            </div>
        </div>
    );
};

```

### src/components/RunPicker.tsx

```tsx
import React, { useEffect, useState } from 'react';
import { api } from '../api/client';

export const RunPicker: React.FC<{ onSelect: (id: string) => void }> = ({ onSelect }) => {
  const [runs, setRuns] = useState<string[]>([]);
  const [confirmClear, setConfirmClear] = useState(false);

  const refreshRuns = () => {
    api.getRuns().then(setRuns);
  };

  useEffect(() => {
    refreshRuns();
  }, []);

  const handleClearAll = async () => {
    if (!confirmClear) {
      setConfirmClear(true);
      setTimeout(() => setConfirmClear(false), 3000); // Reset after 3s
      return;
    }

    try {
      await api.clearAllRuns(); // We need to add this method to the client first
      setRuns([]);
      onSelect(''); // Clear selection
      setConfirmClear(false);
    } catch (e) {
      console.error('Failed to clear runs:', e);
    }
  };

  return (
    <div className="bg-slate-900/40 rounded-lg p-4 border border-slate-800/60 backdrop-blur-sm shadow-sm">
      <div className="flex justify-between items-center mb-2">
        <label className="text-[10px] text-slate-500 font-bold uppercase tracking-widest flex items-center gap-1">
          <span className="w-1.5 h-1.5 bg-blue-500 rounded-full"></span>
          Select Session
        </label>
        <button
          onClick={refreshRuns}
          className="text-[10px] bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white px-2 py-0.5 rounded transition-colors"
          title="Refresh List"
        >
          REFRESH
        </button>
      </div>
      <div className="flex gap-2">
        <div className="relative flex-1 group">
            <select
            className="w-full appearance-none bg-slate-950 border border-slate-800 rounded-md text-xs p-2.5 text-slate-200 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 transition-all cursor-pointer hover:border-slate-700"
            onChange={(e) => onSelect(e.target.value)}
            >
            <option value="">-- Choose Run --</option>
            {runs.map(r => <option key={r} value={r}>{r}</option>)}
            </select>
            <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none text-slate-500 group-hover:text-slate-300">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
            </div>
        </div>

        <button
          onClick={handleClearAll}
          className={`px-3 rounded-md text-xs font-bold border transition-all duration-200 ${confirmClear
              ? 'bg-red-500 text-white border-red-400 shadow-[0_0_10px_rgba(239,68,68,0.5)]'
              : 'bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700 hover:text-red-400 hover:border-slate-600'
            }`}
          title="Clear all experiments"
        >
          {confirmClear ? '??' : '🗑'}
        </button>
      </div>
    </div>
  );
};

```

### src/components/SimulationChart.tsx

```tsx
import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createChart, IChartApi, ISeriesApi, ColorType } from 'lightweight-charts';

interface SimChartProps {
    /** Called when simulation needs a new bar */
    onRequestBar?: () => void;
    /** Called when simulation ends */
    onSimulationEnd?: () => void;
}

interface BarData {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
}

interface OCOZone {
    entryPrice: number;
    stopPrice: number;
    tpPrice: number;
    entryTime: number;
    active: boolean;
}

/**
 * Simulation Chart - Bar-by-bar forward simulation
 * 
 * Unlike CandleChart which shows a fixed window, this chart:
 * - Starts empty or with minimal history
 * - Adds bars one at a time
 * - Shows OCO zones when model triggers
 * - Animates OCO resolution
 */
export const SimulationChart: React.FC<SimChartProps> = ({
    onRequestBar,
    onSimulationEnd
}) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);

    const [bars, setBars] = useState<BarData[]>([]);
    const [ocoZone, setOcoZone] = useState<OCOZone | null>(null);
    const [isRunning, setIsRunning] = useState(false);
    const [currentBar, setCurrentBar] = useState<number>(0);

    // Initialize chart
    useEffect(() => {
        if (!containerRef.current) return;

        const chart = createChart(containerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: '#0f172a' },
                textColor: '#94a3b8',
            },
            grid: {
                vertLines: { color: '#1e293b' },
                horzLines: { color: '#1e293b' },
            },
            width: containerRef.current.clientWidth,
            height: 400,
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
            },
        });

        const candleSeries = chart.addCandlestickSeries({
            upColor: '#22c55e',
            downColor: '#ef4444',
            borderUpColor: '#22c55e',
            borderDownColor: '#ef4444',
            wickUpColor: '#22c55e',
            wickDownColor: '#ef4444',
        });

        chartRef.current = chart;
        candleSeriesRef.current = candleSeries;

        // Resize handler
        const handleResize = () => {
            if (containerRef.current) {
                chart.resize(containerRef.current.clientWidth, 400);
            }
        };
        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, []);

    // Update chart when bars change
    useEffect(() => {
        if (candleSeriesRef.current && bars.length > 0) {
            candleSeriesRef.current.setData(bars.map(b => ({
                time: b.time as any,
                open: b.open,
                high: b.high,
                low: b.low,
                close: b.close,
            })));

            // Auto-scroll to latest bar
            chartRef.current?.timeScale().scrollToRealTime();
        }
    }, [bars]);

    // Add a new bar
    const addBar = useCallback((bar: BarData) => {
        setBars(prev => [...prev, bar]);
        setCurrentBar(prev => prev + 1);
    }, []);

    // Show OCO zone
    const showOCO = useCallback((entry: number, stop: number, tp: number) => {
        setOcoZone({
            entryPrice: entry,
            stopPrice: stop,
            tpPrice: tp,
            entryTime: bars.length > 0 ? bars[bars.length - 1].time : Date.now() / 1000,
            active: true,
        });

        // Draw horizontal lines for OCO
        if (candleSeriesRef.current) {
            // Note: In full implementation, would use priceLine API
        }
    }, [bars]);

    // Clear OCO zone
    const clearOCO = useCallback(() => {
        setOcoZone(null);
    }, []);

    // Reset simulation
    const reset = useCallback(() => {
        setBars([]);
        setOcoZone(null);
        setCurrentBar(0);
        setIsRunning(false);
        if (candleSeriesRef.current) {
            candleSeriesRef.current.setData([]);
        }
    }, []);

    return (
        <div className="relative">
            <div ref={containerRef} className="w-full h-[400px]" />

            {/* Overlay info */}
            <div className="absolute top-2 left-2 bg-slate-900/80 px-3 py-2 rounded text-xs">
                <div className="text-slate-400">Bars: <span className="text-white">{bars.length}</span></div>
                {ocoZone && (
                    <div className="text-yellow-400 mt-1">
                        OCO Active: Entry ${ocoZone.entryPrice.toFixed(2)}
                    </div>
                )}
            </div>

            {/* Empty state */}
            {bars.length === 0 && (
                <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-slate-500 text-sm">
                        Waiting for simulation to start...
                    </div>
                </div>
            )}
        </div>
    );
};

// Export utility to control the simulation externally
export interface SimulationController {
    addBar: (bar: BarData) => void;
    showOCO: (entry: number, stop: number, tp: number) => void;
    clearOCO: () => void;
    reset: () => void;
}

```

### src/components/StatsPanel.tsx

```tsx
import React, { useMemo } from 'react';
import { VizDecision } from '../types/viz';

interface StatsPanelProps {
    decisions: VizDecision[];
    startingBalance?: number;
}

interface ScanStats {
    totalTrades: number;
    wins: number;
    losses: number;
    winRate: number;
    totalPnL: number;
    avgPnL: number;
    maxDrawdown: number;
    endBalance: number;
    profitFactor: number;
    avgWin: number;
    avgLoss: number;
}

export const StatsPanel: React.FC<StatsPanelProps> = ({
    decisions,
    startingBalance = 50000
}) => {
    const stats = useMemo<ScanStats>(() => {
        if (!decisions || decisions.length === 0) {
            return {
                totalTrades: 0,
                wins: 0,
                losses: 0,
                winRate: 0,
                totalPnL: 0,
                avgPnL: 0,
                maxDrawdown: 0,
                endBalance: startingBalance,
                profitFactor: 0,
                avgWin: 0,
                avgLoss: 0,
            };
        }

        let balance = startingBalance;
        let peakBalance = startingBalance;
        let maxDrawdown = 0;
        let wins = 0;
        let losses = 0;
        let totalWinAmount = 0;
        let totalLossAmount = 0;

        decisions.forEach(d => {
            // Get PnL from oco_results (multiple formats) or cf_pnl_dollars
            const ocoResults = d.oco_results || {};
            let pnl = 0;

            // Format 1: Direct pnl_dollars on oco_results (from IFVG debug scanner)
            if (typeof ocoResults.pnl_dollars === 'number') {
                pnl = ocoResults.pnl_dollars;
            }
            // Format 2: Nested OCO results (from OR scanner)
            else if (typeof Object.values(ocoResults)[0] === 'object') {
                const bestOco = Object.values(ocoResults)[0] as { pnl_dollars?: number } | undefined;
                pnl = bestOco?.pnl_dollars ?? 0;
            }
            // Fallback: cf_pnl_dollars
            else {
                pnl = d.cf_pnl_dollars ?? 0;
            }

            balance += pnl;

            if (pnl > 0) {
                wins++;
                totalWinAmount += pnl;
            } else if (pnl < 0) {
                losses++;
                totalLossAmount += Math.abs(pnl);
            }

            // Track peak and drawdown
            if (balance > peakBalance) {
                peakBalance = balance;
            }
            const currentDrawdown = peakBalance - balance;
            if (currentDrawdown > maxDrawdown) {
                maxDrawdown = currentDrawdown;
            }
        });

        const totalTrades = wins + losses;
        const totalPnL = balance - startingBalance;

        return {
            totalTrades,
            wins,
            losses,
            winRate: totalTrades > 0 ? (wins / totalTrades) * 100 : 0,
            totalPnL,
            avgPnL: totalTrades > 0 ? totalPnL / totalTrades : 0,
            maxDrawdown,
            endBalance: balance,
            profitFactor: totalLossAmount > 0 ? totalWinAmount / totalLossAmount : totalWinAmount > 0 ? Infinity : 0,
            avgWin: wins > 0 ? totalWinAmount / wins : 0,
            avgLoss: losses > 0 ? totalLossAmount / losses : 0,
        };
    }, [decisions, startingBalance]);

    const formatCurrency = (val: number) => {
        const sign = val >= 0 ? '+' : '';
        return `${sign}$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    };

    const StatBox = ({ label, value, color = 'text-white', subValue, trend }: {
        label: string;
        value: string | number;
        color?: string;
        subValue?: string;
        trend?: 'up' | 'down' | 'neutral';
    }) => (
        <div className="bg-slate-800/40 backdrop-blur-sm border border-slate-700/50 rounded-lg p-3 flex flex-col hover:bg-slate-800/60 transition-colors shadow-sm group">
            <span className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider mb-1">{label}</span>
            <span className={`text-base font-bold font-mono ${color} group-hover:scale-105 transition-transform origin-left`}>{value}</span>
            {subValue && (
                <div className="flex items-center mt-1">
                    <span className="text-[10px] text-slate-500 font-mono">{subValue}</span>
                </div>
            )}
        </div>
    );

    if (decisions.length === 0) {
        return (
            <div className="p-4 bg-slate-900 border-b border-slate-800">
                <div className="text-sm text-slate-500 text-center italic">No scan data loaded to analyze.</div>
            </div>
        );
    }

    return (
        <div className="px-4 py-3 bg-slate-900 border-b border-slate-800 shadow-md z-10">
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                {/* Starting Balance */}
                <StatBox
                    label="Initial Capital"
                    value={`$${startingBalance.toLocaleString()}`}
                    color="text-slate-300"
                />

                {/* End Balance */}
                <StatBox
                    label="Current Balance"
                    value={`$${stats.endBalance.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                    color={stats.endBalance >= startingBalance ? 'text-green-400' : 'text-red-400'}
                    trend={stats.endBalance >= startingBalance ? 'up' : 'down'}
                />

                {/* Total P&L */}
                <StatBox
                    label="Net P&L"
                    value={formatCurrency(stats.totalPnL)}
                    color={stats.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}
                    subValue={`${stats.totalTrades} Trades`}
                />

                {/* Win Rate */}
                <StatBox
                    label="Win Rate"
                    value={`${stats.winRate.toFixed(1)}%`}
                    color={stats.winRate >= 50 ? 'text-emerald-400' : 'text-amber-400'}
                    subValue={`${stats.wins}W - ${stats.losses}L`}
                />

                {/* Max Drawdown */}
                <StatBox
                    label="Max Drawdown"
                    value={`-$${stats.maxDrawdown.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                    color="text-rose-400"
                    subValue={`${((stats.maxDrawdown / startingBalance) * 100).toFixed(1)}%`}
                />

                {/* Profit Factor */}
                <StatBox
                    label="Profit Factor"
                    value={stats.profitFactor === Infinity ? '∞' : stats.profitFactor.toFixed(2)}
                    color={stats.profitFactor >= 1.5 ? 'text-purple-400' : stats.profitFactor >= 1 ? 'text-blue-400' : 'text-slate-400'}
                    subValue={`Avg: ${formatCurrency(stats.avgPnL)}`}
                />
            </div>
        </div>
    );
};

```

### src/components/TimeframeBar.tsx

```tsx
import React from 'react';

export type Timeframe = '1m' | '5m' | '15m' | '1h' | '4h';

interface TimeframeBarProps {
    selected: Timeframe;
    onSelect: (tf: Timeframe) => void;
}

const TIMEFRAMES: Timeframe[] = ['1m', '5m', '15m', '1h', '4h'];

export const TimeframeBar: React.FC<TimeframeBarProps> = ({ selected, onSelect }) => {
    return (
        <div className="flex items-center gap-1 bg-slate-800 rounded-lg p-1">
            {TIMEFRAMES.map(tf => (
                <button
                    key={tf}
                    onClick={() => onSelect(tf)}
                    className={`px-3 py-1.5 text-xs font-medium rounded transition-all ${selected === tf
                            ? 'bg-blue-600 text-white shadow-lg'
                            : 'text-slate-400 hover:text-white hover:bg-slate-700'
                        }`}
                >
                    {tf}
                </button>
            ))}
        </div>
    );
};

```

### src/config.py

```python
"""
MLang2 Configuration
Central configuration for paths, constants, and defaults.
"""

from pathlib import Path
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from typing import List

# =============================================================================
# BASE PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = BASE_DIR / "cache"
SHARDS_DIR = BASE_DIR / "shards"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DIR, CACHE_DIR, SHARDS_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TIMEZONE
# =============================================================================

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")
DEFAULT_TZ = NY_TZ

# =============================================================================
# SESSION TIMES (New York)
# =============================================================================

SESSION_RTH_START = "09:30"   # Regular Trading Hours
SESSION_RTH_END = "16:00"
SESSION_GLOBEX_START = "18:00"
SESSION_GLOBEX_END = "09:30"

# =============================================================================
# INSTRUMENT CONSTANTS (MES)
# =============================================================================

TICK_SIZE = 0.25
POINT_VALUE = 5.0
COMMISSION_PER_SIDE = 1.25  # ~$2.50 round trip

# =============================================================================
# INDICATOR DEFAULTS
# =============================================================================

DEFAULT_EMA_PERIOD = 200
DEFAULT_RSI_PERIOD = 14
DEFAULT_ADR_PERIOD = 14
DEFAULT_ATR_PERIOD = 14

# =============================================================================
# FEATURE DEFAULTS
# =============================================================================

DEFAULT_LOOKBACK_MINUTES = 120  # 2 hours
DEFAULT_LOOKBACK_1M = 120       # 2 hours of 1m bars
DEFAULT_LOOKBACK_5M = 24        # 2 hours of 5m bars
DEFAULT_LOOKBACK_15M = 8        # 2 hours of 15m bars

# =============================================================================
# SIMULATION DEFAULTS
# =============================================================================

DEFAULT_MAX_BARS_IN_TRADE = 200
DEFAULT_SLIPPAGE_TICKS = 0.5
DEFAULT_MAX_RISK_DOLLARS = 300.0  # Default max risk per trade for position sizing

# =============================================================================
# DATA FILES
# =============================================================================

CONTINUOUS_CONTRACT_PATH = RAW_DATA_DIR / "continuous_contract.json"

```

### src/core/__init__.py

```python
# Core module
"""Core abstractions and utilities."""

```

### src/core/enums.py

```python
"""
Core Enums
Shared enumerations used across the codebase.
These are in a separate module to avoid circular imports.
"""

from enum import Enum


class RunMode(Enum):
    """
    Execution mode for the system.
    
    Controls what operations are permitted:
    - TRAIN: Can peek at future data for labeling, can learn, cannot trade
    - REPLAY: Cannot peek future, cannot learn, can simulate trades
    - SCAN: Cannot peek future, cannot learn, cannot trade (read-only analysis)
    """
    TRAIN = "TRAIN"
    REPLAY = "REPLAY"
    SCAN = "SCAN"


class ModelRole(Enum):
    """
    Role of a model in the system.
    Determines which RunModes it is allowed to operate in.
    """
    TRAINING_ONLY = "TRAINING_ONLY"  # Only for training/labeling
    FROZEN_EVAL = "FROZEN_EVAL"      # Validated model for metrics
    REPLAY_ONLY = "REPLAY_ONLY"      # Specifically for simulation
    SCAN_ASSIST = "SCAN_ASSIST"      # Low-confidence signal generator

```

### src/core/manifest.py

```python
"""
Run Manifest
Unified contract for all run outputs (SCAN/REPLAY/TRAIN).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json

from src.core.enums import RunMode


@dataclass
class ScannerConfig:
    """Configuration for a scanner used in the run."""
    scanner_id: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scanner_id': self.scanner_id,
            'params': self.params,
        }


@dataclass
class ModelConfig:
    """Configuration for a model used in the run."""
    model_id: str
    model_path: Optional[str] = None
    role: str = "REPLAY_ONLY"
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'model_path': self.model_path,
            'role': self.role,
            'params': self.params,
        }


@dataclass
class ArtifactRefs:
    """References to artifacts produced by the run."""
    decisions: Optional[str] = None  # Path to decisions.jsonl
    trades: Optional[str] = None     # Path to trades.jsonl
    series: Optional[str] = None     # Path to full_series.json
    indicators: Optional[str] = None # Path to indicators.jsonl
    metrics: Optional[str] = None    # Path to metrics.json
    events: Optional[str] = None     # Path to events.jsonl (for replay)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'decisions': self.decisions,
            'trades': self.trades,
            'series': self.series,
            'indicators': self.indicators,
            'metrics': self.metrics,
            'events': self.events,
        }


@dataclass
class Provenance:
    """Provenance tracking for reproducibility."""
    git_hash: Optional[str] = None
    config_hash: Optional[str] = None
    created_by: str = "mlang2"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'git_hash': self.git_hash,
            'config_hash': self.config_hash,
            'created_by': self.created_by,
        }


@dataclass
class RunManifest:
    """
    Unified run manifest.
    
    This is the single source of truth for what a run contains.
    All outputs (SCAN/REPLAY/TRAIN) produce this manifest.
    
    The UI reads this to know:
    - What mode the run was in
    - What scanners/models were used
    - What artifacts are available
    - How to reproduce the run
    """
    # Identity
    run_id: str
    created_at: str  # ISO timestamp
    run_mode: RunMode
    
    # Market context
    symbol: str = "MES"
    timeframe: str = "1m"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Components used
    scanners: List[ScannerConfig] = field(default_factory=list)
    models: List[ModelConfig] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)  # List of indicator IDs used
    
    # Artifacts produced
    artifacts: ArtifactRefs = field(default_factory=ArtifactRefs)
    
    # Provenance
    provenance: Provenance = field(default_factory=Provenance)
    
    # Summary stats (optional)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'created_at': self.created_at,
            'run_mode': self.run_mode.value,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'scanners': [s.to_dict() for s in self.scanners],
            'models': [m.to_dict() for m in self.models],
            'indicators': self.indicators,
            'artifacts': self.artifacts.to_dict(),
            'provenance': self.provenance.to_dict(),
            'stats': self.stats,
        }
    
    def save(self, path: Path):
        """Save manifest to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'RunManifest':
        """Load manifest from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        # Reconstruct nested objects
        scanners = [ScannerConfig(**s) for s in data.get('scanners', [])]
        models = [ModelConfig(**m) for m in data.get('models', [])]
        artifacts = ArtifactRefs(**data.get('artifacts', {}))
        provenance = Provenance(**data.get('provenance', {}))
        
        return cls(
            run_id=data['run_id'],
            created_at=data['created_at'],
            run_mode=RunMode(data['run_mode']),
            symbol=data.get('symbol', 'MES'),
            timeframe=data.get('timeframe', '1m'),
            start_date=data.get('start_date'),
            end_date=data.get('end_date'),
            scanners=scanners,
            models=models,
            indicators=data.get('indicators', []),
            artifacts=artifacts,
            provenance=provenance,
            stats=data.get('stats', {}),
        )
    
    @classmethod
    def create_for_scan(
        cls,
        run_id: str,
        scanner_id: str,
        scanner_params: Dict[str, Any],
        start_date: str,
        end_date: str,
    ) -> 'RunManifest':
        """Factory method for SCAN mode runs."""
        return cls(
            run_id=run_id,
            created_at=datetime.utcnow().isoformat() + 'Z',
            run_mode=RunMode.SCAN,
            start_date=start_date,
            end_date=end_date,
            scanners=[ScannerConfig(scanner_id=scanner_id, params=scanner_params)],
            artifacts=ArtifactRefs(
                decisions=f"{run_id}/decisions.jsonl",
            ),
        )
    
    @classmethod
    def create_for_replay(
        cls,
        run_id: str,
        model_id: str,
        model_path: str,
        start_date: str,
        end_date: str,
    ) -> 'RunManifest':
        """Factory method for REPLAY mode runs."""
        return cls(
            run_id=run_id,
            created_at=datetime.utcnow().isoformat() + 'Z',
            run_mode=RunMode.REPLAY,
            start_date=start_date,
            end_date=end_date,
            models=[ModelConfig(model_id=model_id, model_path=model_path, role="REPLAY_ONLY")],
            artifacts=ArtifactRefs(
                decisions=f"{run_id}/decisions.jsonl",
                trades=f"{run_id}/trades.jsonl",
                events=f"{run_id}/events.jsonl",
            ),
        )
    
    @classmethod
    def create_for_train(
        cls,
        run_id: str,
        start_date: str,
        end_date: str,
    ) -> 'RunManifest':
        """Factory method for TRAIN mode runs."""
        return cls(
            run_id=run_id,
            created_at=datetime.utcnow().isoformat() + 'Z',
            run_mode=RunMode.TRAIN,
            start_date=start_date,
            end_date=end_date,
            artifacts=ArtifactRefs(
                decisions=f"{run_id}/decisions.jsonl",
                trades=f"{run_id}/trades.jsonl",
                metrics=f"{run_id}/metrics.json",
            ),
        )

```

### src/core/registries.py

```python
"""
Plugin Registries
Central registration for scanners, models, and indicators.
"""

from typing import Dict, Callable, Any, List, Protocol
from dataclasses import dataclass


# =============================================================================
# Scanner Registry
# =============================================================================

class Scanner(Protocol):
    """Protocol for scanner implementations."""
    def scan(self, step_result: Any) -> bool:
        """Return True if conditions are met."""
        ...


@dataclass
class ScannerInfo:
    """Metadata about a registered scanner."""
    scanner_id: str
    name: str
    description: str
    params_schema: Dict[str, Any]  # JSON schema for params


class ScannerRegistry:
    """
    Registry for scanner implementations.
    
    Usage:
        @ScannerRegistry.register("ema_cross", "EMA Cross", "Trigger on EMA crossover")
        class EMACrossScanner:
            def __init__(self, fast=12, slow=26):
                self.fast = fast
                self.slow = slow
            
            def scan(self, step_result):
                # Implementation
                pass
    """
    
    _registry: Dict[str, Callable] = {}
    _info: Dict[str, ScannerInfo] = {}
    
    @classmethod
    def register(
        cls,
        scanner_id: str,
        name: str,
        description: str = "",
        params_schema: Dict[str, Any] = None
    ):
        """Decorator to register a scanner."""
        def decorator(scanner_class):
            cls._registry[scanner_id] = scanner_class
            cls._info[scanner_id] = ScannerInfo(
                scanner_id=scanner_id,
                name=name,
                description=description,
                params_schema=params_schema or {},
            )
            return scanner_class
        return decorator
    
    @classmethod
    def create(cls, scanner_id: str, **params) -> Scanner:
        """Create scanner instance by ID."""
        if scanner_id not in cls._registry:
            raise ValueError(f"Unknown scanner: {scanner_id}")
        return cls._registry[scanner_id](**params)
    
    @classmethod
    def list_all(cls) -> List[ScannerInfo]:
        """List all registered scanners."""
        return list(cls._info.values())
    
    @classmethod
    def get_info(cls, scanner_id: str) -> ScannerInfo:
        """Get info for a specific scanner."""
        if scanner_id not in cls._info:
            raise ValueError(f"Unknown scanner: {scanner_id}")
        return cls._info[scanner_id]


# =============================================================================
# Model Registry
# =============================================================================

class PolicyModel(Protocol):
    """Protocol for policy model implementations."""
    def predict(self, features: Any) -> Dict[str, Any]:
        """Return model prediction."""
        ...


@dataclass
class ModelInfo:
    """Metadata about a registered model."""
    model_id: str
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]


class ModelRegistry:
    """
    Registry for model implementations.
    
    Usage:
        @ModelRegistry.register("fusion_cnn", "Fusion CNN Model")
        class FusionModelWrapper:
            def __init__(self, model_path):
                self.model = load_model(model_path)
            
            def predict(self, features):
                return self.model.forward(**features)
    """
    
    _registry: Dict[str, Callable] = {}
    _info: Dict[str, ModelInfo] = {}
    
    @classmethod
    def register(
        cls,
