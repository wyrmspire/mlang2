"""
Experiment Runner
Run a single experiment end-to-end.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from src.viz.export import Exporter

from src.experiments.config import ExperimentConfig
from src.experiments.fingerprint import compute_fingerprint
from src.experiments.splits import generate_walk_forward_splits, WalkForwardConfig

from src.data.loader import load_continuous_contract
from src.data.resample import resample_all_timeframes

from src.sim.stepper import MarketStepper
from src.sim.oco_engine import create_oco_bracket
from src.features.pipeline import compute_features, precompute_indicators
from src.policy.scanners import get_scanner
from src.policy.filters import DEFAULT_FILTERS
from src.policy.cooldown import CooldownManager
from src.policy.actions import Action, SkipReason

from src.labels.labeler import Labeler
from src.datasets.decision_record import DecisionRecord
from src.datasets.trade_record import TradeRecord
from src.datasets.writer import ShardWriter
from src.datasets.reader import create_dataloader

from src.models.fusion import FusionModel
from src.models.train import train_model, TrainResult

from src.config import PROCESSED_DIR, SHARDS_DIR, RESULTS_DIR


@dataclass
class ExperimentResult:
    """Result of running an experiment."""
    config: ExperimentConfig
    fingerprint: str
    
    # Dataset stats
    total_records: int
    win_records: int
    loss_records: int
    timeout_records: int
    
    # Training results
    train_result: Optional[TrainResult] = None
    
    # Created at
    created_at: pd.Timestamp = None
    
    def to_dict(self):
        return {
            'fingerprint': self.fingerprint,
            'total_records': self.total_records,
            'win_records': self.win_records,
            'loss_records': self.loss_records,
            'timeout_records': self.timeout_records,
            'best_val_loss': self.train_result.best_val_loss if self.train_result else None,
            'best_epoch': self.train_result.best_epoch if self.train_result else None,
        }


def run_experiment(
    config: ExperimentConfig,
    exporter: Optional['Exporter'] = None
) -> ExperimentResult:
    """
    Run a complete experiment:
    
    1. Load data
    2. Generate decision records at scanner points
    3. Label all records with counterfactual outcomes
    4. Write to shards
    5. Train model
    6. Return results
    
    Args:
        config: Experiment configuration
        exporter: Optional Exporter for viz output
    """
    print(f"Running experiment: {config.name}")
    
    # Compute fingerprint
    fingerprint = compute_fingerprint(config)
    print(f"Fingerprint: {fingerprint}")
    
    # 1. Load and prepare data
    print("Loading data...")
    df = load_continuous_contract()
    
    # Filter by date range
    if config.start_date:
        df = df[df['time'] >= config.start_date]
    if config.end_date:
        df = df[df['time'] <= config.end_date]
    
    df = df.reset_index(drop=True)
    print(f"Data range: {df['time'].min()} to {df['time'].max()}")
    print(f"Total bars: {len(df)}")
    
    # Resample to higher timeframes
    htf_data = resample_all_timeframes(df)
    df_5m = htf_data.get('5m')
    df_15m = htf_data.get('15m')
    
    # Precompute indicators
    print("Precomputing indicators...")
    indicators_map = precompute_indicators(df, df_5m, df_15m)
    
    # 2. Generate decision records
    print("Generating decision records...")
    stepper = MarketStepper(df, start_idx=200, end_idx=len(df) - 200)
    scanner = get_scanner(config.scanner_id, **config.scanner_params)
    labeler = Labeler(config.label_config)
    cooldown = CooldownManager()
    
    records: List[DecisionRecord] = []
    
    while True:
        step = stepper.step()
        if step.is_done:
            break
        
        # Compute features
        features = compute_features(
            stepper,
            config.feature_config,
            df_5m=df_5m,
            df_15m=df_15m,
            precomputed_indicators=indicators_map,
        )
        
        # Check if scanner triggers
        scan_result = scanner.scan(features.market_state, features)
        if not scan_result.triggered:
            continue
        
        # Check filters
        filter_result = DEFAULT_FILTERS.check(features)
        if not filter_result.passed:
            skip_reason = SkipReason.FILTER_BLOCK
        # Check cooldown
        elif cooldown.is_on_cooldown(step.bar_idx, features.timestamp)[0]:
            skip_reason = SkipReason.COOLDOWN
        else:
            skip_reason = SkipReason.NOT_SKIPPED
        
        # Create record
        record = DecisionRecord(
            timestamp=features.timestamp,
            bar_idx=step.bar_idx,
            decision_id=str(uuid.uuid4())[:8],
            scanner_id=config.scanner_id,
            action=Action.NO_TRADE if skip_reason != SkipReason.NOT_SKIPPED else Action.PLACE_ORDER,
            skip_reason=skip_reason,
            x_price_1m=features.x_price_1m,
            x_price_5m=features.x_price_5m,
            x_price_15m=features.x_price_15m,
            x_context=features.x_context,
            current_price=features.current_price,
            atr=features.atr,
        )
        
        # 3. Label with counterfactual outcome
        cf_label = labeler.label_decision_point(df, step.bar_idx, features.atr)
        record.cf_outcome = cf_label.outcome
        record.cf_pnl = cf_label.pnl
        record.cf_pnl_dollars = cf_label.pnl_dollars
        record.cf_mae = cf_label.mae
        record.cf_mfe = cf_label.mfe
        record.cf_mae_atr = cf_label.mae_atr
        record.cf_mfe_atr = cf_label.mfe_atr
        record.cf_bars_held = cf_label.bars_held
        
        records.append(record)
        
        # === VIZ EXPORT HOOK ===
        if exporter:
            curr_idx = step.bar_idx
            
            # Extract RAW OHLCV for chart: 60 bars before + 20 bars after
            start_raw_idx = max(0, curr_idx - 60)
            end_raw_idx = min(len(df), curr_idx + 21)  # +1 for current bar, +20 future
            raw_slice = df.iloc[start_raw_idx : end_raw_idx]
            raw_ohlcv = raw_slice[['open', 'high', 'low', 'close', 'volume']].values.tolist()
            
            # Extract future bars separately (for compatibility)
            future_bars = []
            end_future_idx = min(len(df), curr_idx + 21)
            if end_future_idx > curr_idx + 1:
                future_slice = df.iloc[curr_idx+1 : end_future_idx]
                future_bars = future_slice[['open', 'high', 'low', 'close', 'volume']].values.tolist()
            
            # Extract indicator values for overlay
            ind = features.indicators
            indicators_dict = {}
            if ind:
                indicators_dict = {
                    'ema': ind.ema_5m_20,
                    'atr': ind.atr_5m_14,
                    'rsi': ind.rsi_5m_14,
                }

            exporter.on_decision(
                record, features, 
                future_1m=future_bars,
                raw_ohlcv=raw_ohlcv,
                indicators=indicators_dict
            )
            
            # Export Bracket if trade
            if record.action == Action.PLACE_ORDER:
                # Create bracket to visualize TP/SL
                bracket = create_oco_bracket(
                    config=config.oco_config,
                    base_price=features.current_price,
                    atr=features.atr
                )
                exporter.on_bracket_created(record.decision_id, bracket)
        
        # Update cooldown if trade placed
        if record.action == Action.PLACE_ORDER:
            cooldown.record_trade(step.bar_idx, cf_label.outcome, features.timestamp)
            
            # Export Trade Record for Viz
            if exporter:
                # Approximate exit bar
                exit_bar = step.bar_idx + record.cf_bars_held
                exit_time = features.timestamp + pd.Timedelta(minutes=record.cf_bars_held)
                
                trade = TradeRecord(
                    trade_id=str(uuid.uuid4())[:8],
                    decision_id=record.decision_id,
                    entry_time=features.timestamp,
                    entry_bar=step.bar_idx,
                    entry_price=features.current_price,
                    direction=config.oco_config.direction,
                    exit_time=exit_time,
                    exit_bar=exit_bar,
                    exit_price=features.current_price + (record.cf_pnl / (1 if config.oco_config.direction=="LONG" else -1)),
                    exit_reason=record.cf_outcome,
                    outcome=record.cf_outcome,
                    pnl_points=record.cf_pnl,
                    pnl_dollars=record.cf_pnl_dollars,
                    r_multiple=record.cf_pnl_dollars / (features.atr * config.oco_config.stop_atr * 50) if features.atr > 0 else 0,
                    bars_held=record.cf_bars_held,
                    mae=record.cf_mae,
                    mfe=record.cf_mfe,
                    scanner_id=record.scanner_id,
                    entry_atr=features.atr
                )
                exporter.on_trade_closed(trade)
                
    print(f"Generated {len(records)} decision records")
    
    # Count outcomes
    win_count = sum(1 for r in records if r.cf_outcome == 'WIN')
    loss_count = sum(1 for r in records if r.cf_outcome == 'LOSS')
    timeout_count = sum(1 for r in records if r.cf_outcome == 'TIMEOUT')
    
    print(f"Outcomes: {win_count} WIN, {loss_count} LOSS, {timeout_count} TIMEOUT")
    
    # 4. Write to shards
    shard_dir = SHARDS_DIR / fingerprint
    print(f"Writing shards to {shard_dir}")
    
    with ShardWriter(shard_dir, experiment_id=fingerprint) as writer:
        for record in records:
            writer.write(record)
    
    # 5. Train model (if enough data)
    train_result = None
    if win_count + loss_count >= 100:
        print("Training model...")
        
        # Create dataloaders (simple 80/20 split for now)
        loader = create_dataloader(shard_dir, batch_size=config.train_config.batch_size)
        
        # Split into train/val
        dataset = loader.dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        from torch.utils.data import random_split, DataLoader
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_ds, batch_size=config.train_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.train_config.batch_size)
        
        # Create model
        model = FusionModel(
            context_dim=config.schema.x_context_dim,
            num_classes=2,  # WIN/LOSS
            dropout=config.train_config.dropout,
        )
        
        # Train
        train_result = train_model(model, train_loader, val_loader, config.train_config)
    
    # 6. Return results
    return ExperimentResult(
        config=config,
        fingerprint=fingerprint,
        total_records=len(records),
        win_records=win_count,
        loss_records=loss_count,
        timeout_records=timeout_count,
        train_result=train_result,
        created_at=pd.Timestamp.now(),
    )
