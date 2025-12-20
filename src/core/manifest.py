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
