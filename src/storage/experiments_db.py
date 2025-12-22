"""
Experiment Database

SQLite-backed storage for experiment results.
Enables the agent to query past performance and find optimal configurations.

Usage:
    from src.storage.experiments_db import ExperimentDB
    
    db = ExperimentDB()
    db.store_run(run_id, metrics)
    best = db.query_best("win_rate", top_k=5)
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExperimentRecord:
    """Single experiment record."""
    run_id: str
    created_at: str
    strategy: str
    model_path: Optional[str]
    config: Dict[str, Any]
    
    # Key metrics
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    
    # Optional extended metrics
    sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None
    profit_factor: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'run_id': self.run_id,
            'created_at': self.created_at,
            'strategy': self.strategy,
            'model_path': self.model_path,
            'config': self.config,
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'avg_pnl_per_trade': self.avg_pnl_per_trade,
            'sharpe': self.sharpe,
            'max_drawdown': self.max_drawdown,
            'profit_factor': self.profit_factor,
        }


class ExperimentDB:
    """
    SQLite database for experiment results.
    
    Allows the agent to:
    - Store run results
    - Query best configurations by metric
    - Compare strategies
    - Learn from history
    """
    
    def __init__(self, db_path: str = "results/experiments.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                strategy TEXT NOT NULL,
                model_path TEXT,
                config_json TEXT NOT NULL,
                
                total_trades INTEGER NOT NULL,
                wins INTEGER NOT NULL,
                losses INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                total_pnl REAL NOT NULL,
                avg_pnl_per_trade REAL NOT NULL,
                
                sharpe REAL,
                max_drawdown REAL,
                profit_factor REAL
            )
        """)
        
        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy ON experiments(strategy)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_win_rate ON experiments(win_rate DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_total_pnl ON experiments(total_pnl DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created ON experiments(created_at DESC)")
        
        conn.commit()
        conn.close()
    
    def store_run(
        self,
        run_id: str,
        strategy: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        model_path: Optional[str] = None
    ) -> bool:
        """
        Store experiment results.
        
        Args:
            run_id: Unique run identifier
            strategy: Strategy name (e.g., "opening_range", "modular")
            config: Run configuration dict
            metrics: Must include: total_trades, wins, losses, win_rate, total_pnl
            model_path: Path to model file if applicable
        
        Returns:
            True if stored successfully
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO experiments (
                    run_id, created_at, strategy, model_path, config_json,
                    total_trades, wins, losses, win_rate, total_pnl, avg_pnl_per_trade,
                    sharpe, max_drawdown, profit_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                datetime.utcnow().isoformat() + 'Z',
                strategy,
                model_path,
                json.dumps(config),
                metrics.get('total_trades', metrics.get('trades', 0)),
                metrics.get('wins', 0),
                metrics.get('losses', 0),
                metrics.get('win_rate', 0.0),
                metrics.get('total_pnl', 0.0),
                metrics.get('avg_pnl_per_trade', 0.0),
                metrics.get('sharpe'),
                metrics.get('max_drawdown'),
                metrics.get('profit_factor'),
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error storing experiment: {e}")
            return False
        finally:
            conn.close()
    
    def query_best(
        self,
        metric: str = "win_rate",
        strategy: Optional[str] = None,
        min_trades: int = 10,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Query best experiments by metric.
        
        Args:
            metric: Column to sort by (win_rate, total_pnl, sharpe, etc.)
            strategy: Filter by strategy name
            min_trades: Minimum trades required
            top_k: Number of results
        
        Returns:
            List of experiment dicts
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = f"""
            SELECT * FROM experiments
            WHERE total_trades >= ?
            {"AND strategy = ?" if strategy else ""}
            ORDER BY {metric} DESC
            LIMIT ?
        """
        
        params = [min_trades]
        if strategy:
            params.append(strategy)
        params.append(top_k)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            record = dict(row)
            record['config'] = json.loads(record['config_json'])
            del record['config_json']
            results.append(record)
        
        return results
    
    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get a specific run by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM experiments WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            record = dict(row)
            record['config'] = json.loads(record['config_json'])
            del record['config_json']
            return record
        return None
    
    def list_strategies(self) -> List[Dict]:
        """List all strategies with their stats."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                strategy,
                COUNT(*) as run_count,
                AVG(win_rate) as avg_win_rate,
                AVG(total_pnl) as avg_pnl,
                MAX(win_rate) as best_win_rate,
                MAX(total_pnl) as best_pnl
            FROM experiments
            GROUP BY strategy
            ORDER BY run_count DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'strategy': row[0],
                'run_count': row[1],
                'avg_win_rate': row[2],
                'avg_pnl': row[3],
                'best_win_rate': row[4],
                'best_pnl': row[5],
            }
            for row in rows
        ]
    
    def count(self) -> int:
        """Get total number of experiments."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM experiments")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def delete_run(self, run_id: str) -> bool:
        """Delete a run by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM experiments WHERE run_id = ?", (run_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted
