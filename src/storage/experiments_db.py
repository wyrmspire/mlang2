"""
Storage Module - Experiment Results Database

Solves the "Results Management" problem by:
1. Storing experiment metrics in SQLite for queryability
2. Keeping detailed logs in JSONL (existing)
3. Providing query interface for agents to find best configurations

This creates "memory" for agents to optimize iteratively.
"""

import sqlite3
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json
import hashlib


@dataclass
class ExperimentRecord:
    """
    Single experiment record.
    
    Contains high-level metrics and config hash for quick lookup.
    """
    experiment_id: str
    config_hash: str
    strategy_name: str
    model_path: Optional[str]
    
    # Date range
    start_date: str
    end_date: str
    
    # Performance metrics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    sharpe_ratio: Optional[float]
    max_drawdown: float
    total_pnl: float
    
    # Configuration
    config_json: str  # Full config as JSON string
    
    # Metadata
    created_at: str
    duration_seconds: float
    
    # Additional metrics
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_id': self.experiment_id,
            'config_hash': self.config_hash,
            'strategy_name': self.strategy_name,
            'model_path': self.model_path,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'total_pnl': self.total_pnl,
            'config_json': self.config_json,
            'created_at': self.created_at,
            'duration_seconds': self.duration_seconds,
            'metadata': json.dumps(self.metadata),
        }


class ExperimentDatabase:
    """
    SQLite database for experiment results.
    
    Provides fast queries for agents to discover best configurations.
    
    Usage:
        db = ExperimentDatabase("results/experiments.db")
        
        # Store experiment
        db.store_experiment(experiment_record)
        
        # Query for best configurations
        best = db.query(
            "SELECT * FROM experiments WHERE win_rate > 0.6 AND strategy = ?",
            ("opening_range",)
        )
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize experiment database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Main experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                config_hash TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                model_path TEXT,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                total_trades INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                avg_win REAL NOT NULL,
                avg_loss REAL NOT NULL,
                sharpe_ratio REAL,
                max_drawdown REAL NOT NULL,
                total_pnl REAL NOT NULL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                metadata TEXT
            )
        """)
        
        # Create indices for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_strategy_name 
            ON experiments(strategy_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_win_rate 
            ON experiments(win_rate DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_config_hash 
            ON experiments(config_hash)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at 
            ON experiments(created_at DESC)
        """)
        
        self.conn.commit()
    
    def store_experiment(self, record: ExperimentRecord):
        """
        Store an experiment record.
        
        Args:
            record: ExperimentRecord to store
        """
        cursor = self.conn.cursor()
        
        data = record.to_dict()
        
        cursor.execute("""
            INSERT OR REPLACE INTO experiments (
                experiment_id, config_hash, strategy_name, model_path,
                start_date, end_date, total_trades, win_rate,
                avg_win, avg_loss, sharpe_ratio, max_drawdown, total_pnl,
                config_json, created_at, duration_seconds, metadata
            ) VALUES (
                :experiment_id, :config_hash, :strategy_name, :model_path,
                :start_date, :end_date, :total_trades, :win_rate,
                :avg_win, :avg_loss, :sharpe_ratio, :max_drawdown, :total_pnl,
                :config_json, :created_at, :duration_seconds, :metadata
            )
        """, data)
        
        self.conn.commit()
    
    def query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a SQL query.
        
        Args:
            sql: SQL query string
            params: Query parameters
            
        Returns:
            List of result dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute(sql, params)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_best_experiments(
        self,
        metric: str = "win_rate",
        limit: int = 10,
        strategy_name: Optional[str] = None,
        min_trades: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get best experiments by a specific metric.
        
        Args:
            metric: Metric to optimize ("win_rate", "sharpe_ratio", "total_pnl")
            limit: Number of results to return
            strategy_name: Optional filter by strategy
            min_trades: Minimum number of trades required
            
        Returns:
            List of experiment records sorted by metric
        """
        sql = f"""
            SELECT * FROM experiments
            WHERE total_trades >= ?
        """
        params = [min_trades]
        
        if strategy_name:
            sql += " AND strategy_name = ?"
            params.append(strategy_name)
        
        sql += f" ORDER BY {metric} DESC LIMIT ?"
        params.append(limit)
        
        return self.query(sql, tuple(params))
    
    def get_experiment_by_id(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific experiment by ID."""
        results = self.query(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (experiment_id,)
        )
        return results[0] if results else None
    
    def get_experiments_by_config_hash(self, config_hash: str) -> List[Dict[str, Any]]:
        """Get all experiments with a specific configuration."""
        return self.query(
            "SELECT * FROM experiments WHERE config_hash = ?",
            (config_hash,)
        )
    
    def get_recent_experiments(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get most recent experiments."""
        return self.query(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
    
    def compute_config_hash(self, config: Dict[str, Any]) -> str:
        """
        Compute deterministic hash of configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            SHA256 hash as hex string
        """
        # Sort keys for deterministic hashing
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM experiments")
        total = cursor.fetchone()['total']
        
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT strategy_name) as unique_strategies,
                COUNT(DISTINCT config_hash) as unique_configs,
                AVG(win_rate) as avg_win_rate,
                MAX(win_rate) as max_win_rate,
                AVG(total_pnl) as avg_pnl,
                MAX(total_pnl) as max_pnl
            FROM experiments
        """)
        stats = dict(cursor.fetchone())
        stats['total_experiments'] = total
        
        return stats
    
    def close(self):
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_experiment_record(
    experiment_id: str,
    config: Dict[str, Any],
    results: Dict[str, Any],
    model_path: Optional[str] = None,
    duration_seconds: float = 0.0
) -> ExperimentRecord:
    """
    Create an ExperimentRecord from experiment results.
    
    Args:
        experiment_id: Unique experiment identifier
        config: Experiment configuration dict
        results: Results dictionary with metrics
        model_path: Optional path to trained model
        duration_seconds: Experiment duration
        
    Returns:
        ExperimentRecord ready to store
    """
    db = ExperimentDatabase(":memory:")  # Temporary connection for hash
    config_hash = db.compute_config_hash(config)
    db.close()
    
    return ExperimentRecord(
        experiment_id=experiment_id,
        config_hash=config_hash,
        strategy_name=config.get('strategy_name', 'unknown'),
        model_path=model_path,
        start_date=config.get('start_date', ''),
        end_date=config.get('end_date', ''),
        total_trades=results.get('total_trades', 0),
        win_rate=results.get('win_rate', 0.0),
        avg_win=results.get('avg_win', 0.0),
        avg_loss=results.get('avg_loss', 0.0),
        sharpe_ratio=results.get('sharpe_ratio'),
        max_drawdown=results.get('max_drawdown', 0.0),
        total_pnl=results.get('total_pnl', 0.0),
        config_json=json.dumps(config),
        created_at=datetime.now().isoformat(),
        duration_seconds=duration_seconds,
        metadata=results.get('metadata', {}),
    )
