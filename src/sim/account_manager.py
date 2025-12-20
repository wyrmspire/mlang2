"""
Account Manager
Multi-account simulation tracking.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import pandas as pd

from src.sim.account import Account, TradeRecord
from src.sim.costs import CostModel, DEFAULT_COSTS
from src.sim.execution import Fill


@dataclass
class AccountSnapshot:
    """Snapshot of account state at a point in time."""
    account_id: str
    timestamp: pd.Timestamp
    balance: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    open_positions: int
    total_trades: int
    
    def to_dict(self) -> Dict:
        return {
            'account_id': self.account_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'balance': self.balance,
            'equity': self.equity,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'open_positions': self.open_positions,
            'total_trades': self.total_trades,
        }


class AccountManager:
    """
    Multi-account manager for simulation.
    
    Manages multiple accounts, routes orders, aggregates PnL.
    Useful for:
    - Multiple strategies in one session
    - Different risk profiles
    - Prop firm rule testing (per-account limits)
    """
    
    def __init__(self):
        self.accounts: Dict[str, Account] = {}
        self.snapshots: List[AccountSnapshot] = []
        
    def create_account(
        self,
        account_id: str,
        starting_balance: float = 50000.0,
        costs: CostModel = None
    ) -> Account:
        """Create a new account."""
        if account_id in self.accounts:
            raise ValueError(f"Account {account_id} already exists")
        
        account = Account(
            starting_balance=starting_balance,
            costs=costs or DEFAULT_COSTS
        )
        self.accounts[account_id] = account
        return account
    
    def delete_account(self, account_id: str):
        """Delete an account."""
        if account_id in self.accounts:
            del self.accounts[account_id]
    
    def get_account(self, account_id: str) -> Optional[Account]:
        """Get account by ID."""
        return self.accounts.get(account_id)
    
    def list_accounts(self) -> List[str]:
        """List all account IDs."""
        return list(self.accounts.keys())
    
    def take_snapshot(self, account_id: str, current_price: float, timestamp: pd.Timestamp):
        """Take a snapshot of account state."""
        account = self.accounts.get(account_id)
        if not account:
            return
        
        equity = account.get_equity(current_price)
        unrealized = equity - account.balance
        
        snapshot = AccountSnapshot(
            account_id=account_id,
            timestamp=timestamp,
            balance=account.balance,
            equity=equity,
            realized_pnl=account.realized_pnl,
            unrealized_pnl=unrealized,
            open_positions=len(account.positions),
            total_trades=len(account.trades),
        )
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_total_pnl(self) -> float:
        """Get total PnL across all accounts."""
        return sum(acc.realized_pnl for acc in self.accounts.values())
    
    def get_total_equity(self, current_price: float) -> float:
        """Get total equity across all accounts."""
        return sum(acc.get_equity(current_price) for acc in self.accounts.values())
    
    def get_aggregate_stats(self) -> Dict:
        """Get aggregated stats across all accounts."""
        total_trades = sum(len(acc.trades) for acc in self.accounts.values())
        total_pnl = self.get_total_pnl()
        
        all_trades = []
        for acc in self.accounts.values():
            all_trades.extend(acc.trades)
        
        wins = sum(1 for t in all_trades if t.outcome == 'WIN')
        
        return {
            'total_accounts': len(self.accounts),
            'total_trades': total_trades,
            'wins': wins,
            'losses': total_trades - wins,
            'win_rate': wins / total_trades if total_trades > 0 else 0.0,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0.0,
        }
    
    def reset_all(self):
        """Reset all accounts to starting state."""
        for account in self.accounts.values():
            account.balance = account.starting_balance
            account.positions.clear()
            account.trades.clear()
            account.realized_pnl = 0.0
            account.peak_balance = account.starting_balance
            account.max_drawdown = 0.0
        self.snapshots.clear()
    
    def get_snapshots_for_account(self, account_id: str) -> List[AccountSnapshot]:
        """Get all snapshots for a specific account."""
        return [s for s in self.snapshots if s.account_id == account_id]
