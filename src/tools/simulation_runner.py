"""
Simulation Runner - Run strategies with model inference in simulation mode.

This tool allows agents to:
1. Load a trained model
2. Configure a strategy with triggers
3. Run simulation against playback data
4. Get complete trade results with model inference decisions
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
import json

from src.core.enums import RunMode, ModelRole
from src.experiments.config import ExperimentConfig, ReplayConfig
from src.experiments.strategy_config import StrategyConfig


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    run_id: str
    strategy_config: Dict[str, Any]
    model_path: str
    
    # Performance metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    
    # Trade details
    trades: List[Dict[str, Any]]
    decisions: List[Dict[str, Any]]
    
    # Model info
    model_role: str
    inference_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'strategy_config': self.strategy_config,
            'model_path': self.model_path,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'model_role': self.model_role,
            'inference_count': self.inference_count,
            'trade_count': len(self.trades),
            'decision_count': len(self.decisions),
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        return f"""
Simulation Results for {self.run_id}
{'=' * 60}
Strategy: {self.strategy_config.get('strategy_id', 'Unknown')}
Model: {Path(self.model_path).name}

Performance:
  Total Trades: {self.total_trades}
  Win Rate: {self.win_rate:.1%}
  Total PnL: ${self.total_pnl:.2f}
  Avg Win: ${self.avg_win:.2f}
  Avg Loss: ${self.avg_loss:.2f}
  Max Drawdown: ${self.max_drawdown:.2f}
  Sharpe Ratio: {self.sharpe_ratio:.2f}

Model Inference:
  Role: {self.model_role}
  Inference Calls: {self.inference_count}
  Decisions Made: {len(self.decisions)}
"""


class SimulationRunner:
    """
    Run strategy simulations with model inference.
    
    This is the primary tool for agents to test complete strategies:
    - Triggers detect entry opportunities
    - Model performs inference to decide trade/no-trade
    - Simulation engine executes trades
    - Results include full trade history and metrics
    
    Usage:
        # Simple run
        result = SimulationRunner.run(
            strategy_id="opening_range",
            model_path="runs/exp_001/model.pt",
            start_date="2025-03-01",
            end_date="2025-03-15"
        )
        
        # Advanced configuration
        result = SimulationRunner.run(
            strategy_config=StrategyConfig(
                strategy_id="mean_reversion",
                scanner_id="rsi_extreme",
                scanner_params={"oversold": 25, "overbought": 75},
                oco_tp_multiple=1.5,
                oco_stop_atr=1.0
            ),
            model_path="models/mean_rev_v2.pt",
            start_date="2025-03-01",
            end_date="2025-03-31",
            replay_config=ReplayConfig(
                speed_multiplier=1.0,
                pause_on_decision=False
            )
        )
    """
    
    @staticmethod
    def run(
        strategy_id: Optional[str] = None,
        strategy_config: Optional[StrategyConfig] = None,
        model_path: str = None,
        start_date: str = None,
        end_date: str = None,
        replay_config: Optional[ReplayConfig] = None,
        **kwargs
    ) -> SimulationResult:
        """
        Run a strategy simulation with model inference.
        
        Args:
            strategy_id: Quick strategy selection (or use strategy_config)
            strategy_config: Full strategy configuration object
            model_path: Path to trained model for inference
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            replay_config: Replay configuration (optional)
            **kwargs: Additional parameters to override in strategy_config
            
        Returns:
            SimulationResult with complete metrics and trade history
        """
        # Validate inputs
        if not model_path:
            raise ValueError("model_path is required")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Build strategy config
        if strategy_config is None:
            if strategy_id is None:
                raise ValueError("Either strategy_id or strategy_config required")
            
            strategy_config = StrategyConfig(
                strategy_id=strategy_id,
                start_date=start_date or "",
                end_date=end_date or "",
                **kwargs
            )
        
        # Create experiment config for REPLAY mode
        exp_config = ExperimentConfig(
            name=f"sim_{strategy_config.strategy_id}_{start_date}",
            description=f"Simulation of {strategy_config.strategy_id} with model inference",
            run_mode=RunMode.REPLAY,  # Critical: REPLAY mode for simulation
            replay_config=replay_config or ReplayConfig(),
            start_date=start_date or strategy_config.start_date,
            end_date=end_date or strategy_config.end_date,
            scanner_id=strategy_config.scanner_id,
            scanner_params=strategy_config.scanner_params,
        )
        
        # Run simulation with model
        # This is where the integration happens:
        # 1. Load model with REPLAY_ONLY role
        # 2. Step through data bar-by-bar
        # 3. Scanner triggers decision points
        # 4. Model performs inference
        # 5. Trades executed based on model output
        # 6. Results collected
        
        run_result = SimulationRunner._execute_simulation(
            exp_config=exp_config,
            model_path=model_path,
            strategy_config=strategy_config
        )
        
        return run_result
    
    @staticmethod
    def _execute_simulation(
        exp_config: ExperimentConfig,
        model_path: str,
        strategy_config: StrategyConfig
    ) -> SimulationResult:
        """
        Internal method to execute simulation.
        
        This loads the model, runs the experiment, and collects results.
        """
        # Import here to avoid circular dependencies
        from src.experiments.runner import run_experiment
        from src.models.fusion import FusionModel
        import torch
        
        # Load model with REPLAY_ONLY role
        model = FusionModel.load(model_path, role=ModelRole.REPLAY_ONLY)
        
        # Run experiment (this will use the model for inference)
        # The runner will:
        # - Step through bars
        # - Check scanner for triggers
        # - Call model.predict() for each trigger
        # - Execute trades based on model decision
        # - Track all decisions and trades
        result = run_experiment(exp_config, model=model)
        
        # Extract metrics and trades
        trades = result.get('trades', [])
        decisions = result.get('decisions', [])
        
        # Calculate metrics
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        losing_trades = sum(1 for t in trades if t.get('pnl', 0) < 0)
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        
        wins = [t['pnl'] for t in trades if t.get('pnl', 0) > 0]
        losses = [t['pnl'] for t in trades if t.get('pnl', 0) < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # Calculate drawdown
        equity_curve = []
        running_pnl = 0
        for trade in trades:
            running_pnl += trade.get('pnl', 0)
            equity_curve.append(running_pnl)
        
        max_drawdown = 0
        if equity_curve:
            peak = equity_curve[0]
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                drawdown = peak - equity
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        # Simple Sharpe (using trade PnL)
        import numpy as np
        if trades:
            returns = np.array([t.get('pnl', 0) for t in trades])
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        return SimulationResult(
            run_id=result.get('run_id', 'sim_' + exp_config.name),
            strategy_config=strategy_config.to_dict() if hasattr(strategy_config, 'to_dict') else {},
            model_path=model_path,
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=winning_trades / len(trades) if trades else 0,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            trades=trades,
            decisions=decisions,
            model_role=ModelRole.REPLAY_ONLY.value,
            inference_count=len(decisions),
        )
    
    @staticmethod
    def batch_run(
        strategy_configs: List[Dict[str, Any]],
        model_path: str,
        start_date: str,
        end_date: str
    ) -> List[SimulationResult]:
        """
        Run multiple strategy configurations in batch.
        
        Useful for parameter sweeps or comparing strategies.
        """
        results = []
        for config in strategy_configs:
            result = SimulationRunner.run(
                strategy_config=StrategyConfig(**config) if isinstance(config, dict) else config,
                model_path=model_path,
                start_date=start_date,
                end_date=end_date
            )
            results.append(result)
        return results
    
    @staticmethod
    def compare_models(
        strategy_config: StrategyConfig,
        model_paths: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, SimulationResult]:
        """
        Compare multiple models on the same strategy.
        
        Returns dict mapping model_path -> SimulationResult
        """
        results = {}
        for model_path in model_paths:
            result = SimulationRunner.run(
                strategy_config=strategy_config,
                model_path=model_path,
                start_date=start_date,
                end_date=end_date
            )
            results[model_path] = result
        return results
    
    @staticmethod
    def save_results(result: SimulationResult, output_path: str):
        """Save simulation results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
    
    @staticmethod
    def load_results(input_path: str) -> Dict[str, Any]:
        """Load simulation results from JSON."""
        with open(input_path) as f:
            return json.load(f)


# Convenience function for agents
def run_simulation(**kwargs) -> SimulationResult:
    """
    Quick simulation runner for agents.
    
    Example:
        result = run_simulation(
            strategy_id="opening_range",
            model_path="models/my_model.pt",
            start_date="2025-03-01",
            end_date="2025-03-15"
        )
        print(result.summary())
    """
    return SimulationRunner.run(**kwargs)
