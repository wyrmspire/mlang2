"""
Scanners
Setup detection - determines when a decision point occurs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from src.features.state import MarketState
from src.features.pipeline import FeatureBundle


@dataclass
class ScannerResult:
    """Result from a scanner check."""
    scanner_id: str
    triggered: bool
    context: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0   # Confidence/strength of signal


class Scanner(ABC):
    """
    Base class for setup scanners.
    
    Scanners define WHEN a decision point occurs.
    They don't decide the action - just whether to evaluate.
    """
    
    @property
    @abstractmethod
    def scanner_id(self) -> str:
        """Unique identifier for this scanner."""
        pass
    
    @abstractmethod
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScannerResult:
        """
        Check if current state triggers this scanner.
        
        Args:
            state: Current market state
            features: Computed features
            
        Returns:
            ScannerResult with triggered flag and context
        """
        pass


class AlwaysScanner(Scanner):
    """
    Scanner that always triggers.
    Useful for testing or fixed-interval strategies.
    """
    
    @property
    def scanner_id(self) -> str:
        return "always"
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScannerResult:
        return ScannerResult(
            scanner_id=self.scanner_id,
            triggered=True,
            score=1.0
        )


class IntervalScanner(Scanner):
    """
    Scanner that triggers every N bars.
    """
    
    def __init__(self, interval: int = 5):
        self.interval = interval
        self._last_triggered = -interval
    
    @property
    def scanner_id(self) -> str:
        return f"interval_{self.interval}"
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScannerResult:
        bar_idx = features.bar_idx
        
        if bar_idx - self._last_triggered >= self.interval:
            self._last_triggered = bar_idx
            return ScannerResult(
                scanner_id=self.scanner_id,
                triggered=True,
                score=1.0
            )
        
        return ScannerResult(
            scanner_id=self.scanner_id,
            triggered=False
        )


class LevelProximityScanner(Scanner):
    """
    Scanner that triggers when price is near key levels.
    """
    
    def __init__(
        self,
        atr_threshold: float = 0.5,
        level_types: List[str] = None
    ):
        self.atr_threshold = atr_threshold
        self.level_types = level_types or ['1h', '4h', 'pdh', 'pdl']
    
    @property
    def scanner_id(self) -> str:
        return f"level_proximity_{self.atr_threshold}"
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScannerResult:
        if features.levels is None or features.atr <= 0:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        levels = features.levels
        atr = features.atr
        price = features.current_price
        
        # Check distances to each level
        min_dist_atr = float('inf')
        nearest_level = None
        
        checks = [
            ('1h_high', levels.dist_1h_high),
            ('1h_low', levels.dist_1h_low),
            ('4h_high', levels.dist_4h_high),
            ('4h_low', levels.dist_4h_low),
            ('pdh', levels.dist_pdh),
            ('pdl', levels.dist_pdl),
        ]
        
        for name, dist in checks:
            dist_atr = abs(dist) / atr if atr > 0 else float('inf')
            if dist_atr < min_dist_atr:
                min_dist_atr = dist_atr
                nearest_level = name
        
        triggered = min_dist_atr <= self.atr_threshold
        
        return ScannerResult(
            scanner_id=self.scanner_id,
            triggered=triggered,
            context={
                'nearest_level': nearest_level,
                'distance_atr': min_dist_atr,
            },
            score=max(0, 1 - min_dist_atr / self.atr_threshold) if triggered else 0
        )


class RSIExtremeScanner(Scanner):
    """
    Scanner that triggers at RSI extremes.
    """
    
    def __init__(
        self,
        oversold: float = 30.0,
        overbought: float = 70.0
    ):
        self.oversold = oversold
        self.overbought = overbought
    
    @property
    def scanner_id(self) -> str:
        return f"rsi_extreme_{int(self.oversold)}_{int(self.overbought)}"
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScannerResult:
        if features.indicators is None:
            return ScannerResult(scanner_id=self.scanner_id, triggered=False)
        
        rsi = features.indicators.rsi_5m_14
        
        oversold = rsi <= self.oversold
        overbought = rsi >= self.overbought
        triggered = oversold or overbought
        
        return ScannerResult(
            scanner_id=self.scanner_id,
            triggered=triggered,
            context={
                'rsi': rsi,
                'condition': 'oversold' if oversold else 'overbought' if overbought else 'neutral',
            },
            score=1.0 if triggered else 0.0
        )


class ScriptScanner(Scanner):
    """
    Scanner that wraps standalone research scripts.
    
    This adapter allows successful research scripts (run_lunch_fade.py, run_rvap_scan.py, etc.)
    to be integrated into the main application pipeline without rewriting them as Scanner classes.
    
    The script must expose either:
    - get_signals(df, **kwargs) -> List[Dict]: Returns list of signal dictionaries
    - scan(df, idx, **kwargs) -> Optional[Dict]: Returns signal dict for specific bar or None
    
    Example:
        scanner = ScriptScanner(
            script_path="scripts/run_lunch_fade.py",
            script_kwargs={'stop_atr_mult': 0.5}
        )
    """
    
    def __init__(
        self,
        script_path: str,
        script_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            script_path: Path to Python script (e.g., 'scripts/run_lunch_fade.py')
            script_kwargs: Optional kwargs to pass to the script's scan/get_signals function
        """
        self.script_path = script_path
        self.script_kwargs = script_kwargs or {}
        self._script_module = None
        self._scan_function = None
        self._load_script()
    
    def _load_script(self):
        """Dynamically import the script and find the scan/get_signals function."""
        import importlib.util
        from pathlib import Path
        
        script_file = Path(self.script_path)
        if not script_file.exists():
            raise FileNotFoundError(f"Script not found: {self.script_path}")
        
        # Load module from file path
        spec = importlib.util.spec_from_file_location("script_module", script_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load script: {self.script_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._script_module = module
        
        # Look for scan or get_signals function
        if hasattr(module, 'get_signals'):
            self._scan_function = module.get_signals
        elif hasattr(module, 'scan'):
            self._scan_function = module.scan
        else:
            raise AttributeError(
                f"Script {self.script_path} must expose either 'get_signals' or 'scan' function"
            )
    
    @property
    def scanner_id(self) -> str:
        script_name = Path(self.script_path).stem
        return f"script_{script_name}"
    
    def scan(
        self,
        state: MarketState,
        features: FeatureBundle
    ) -> ScannerResult:
        """
        Call the script's scan function with current state.
        
        Note: This assumes the script can work with individual bars.
        For batch scripts (get_signals), you may need to buffer state history.
        """
        try:
            # Build a minimal DataFrame from state for compatibility
            # Most scripts expect DataFrame with OHLCV columns
            bar_data = {
                'time': [state.current_time],
                'open': [state.current_price],  # Approximate
                'high': [state.current_price],
                'low': [state.current_price],
                'close': [state.current_price],
                'volume': [0],
            }
            
            import pandas as pd
            df = pd.DataFrame(bar_data)
            
            # Call script function
            # If it's get_signals (batch), call with full df
            # If it's scan (single bar), call with df and idx=0
            if self._scan_function.__name__ == 'get_signals':
                signals = self._scan_function(df, **self.script_kwargs)
                # Check if any signal for current bar
                triggered = len(signals) > 0
                context = signals[0] if signals else {}
            else:
                # scan function for single bar
                signal = self._scan_function(df, 0, **self.script_kwargs)
                triggered = signal is not None
                context = signal or {}
            
            return ScannerResult(
                scanner_id=self.scanner_id,
                triggered=triggered,
                context=context,
                score=context.get('score', 1.0) if triggered else 0.0
            )
        except Exception as e:
            # Log error but don't crash
            return ScannerResult(
                scanner_id=self.scanner_id,
                triggered=False,
                context={'error': str(e)}
            )


def _discover_library_scanners() -> Dict[str, type]:
    """Helper to find all Scanner classes in the library."""
    import importlib
    import pkgutil
    import inspect
    from src.policy import library
    
    found = {}
    
    # Iterate over modules in the library package
    for loader, name, is_pkg in pkgutil.iter_modules(library.__path__):
        full_name = f"src.policy.library.{name}"
        module = importlib.import_module(full_name)
        
        # Find all classes that inherit from Scanner
        for cls_name, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, Scanner) and cls is not Scanner:
                # Use a slugified name or something generic
                # For discovery, we'll try to use a 'base' name or the class name lowercased
                key = name.lower().replace('_', '')
                found[key] = cls
                
    return found


def get_scanner(scanner_id: str, **kwargs) -> Scanner:
    """Factory function to get scanner by ID."""
    scanners = {
        'always': AlwaysScanner,
        'interval': IntervalScanner,
        'level_proximity': LevelProximityScanner,
        'rsi_extreme': RSIExtremeScanner,
        'script': ScriptScanner,  # New: ScriptScanner adapter
    }
    
    # Add discovered library scanners
    scanners.update(_discover_library_scanners())
    
    # Extract base name
    # We support both 'rsi_extreme' and just 'rsi' if we wanted
    base = scanner_id.split('_')[0].lower().replace('_', '')
    
    if base in scanners:
        return scanners[base](**kwargs)
    
    # Also check full name matches in case of library scanners
    # e.g. middayreversal
    clean_id = scanner_id.replace('_', '').lower()
    for name, cls in scanners.items():
        if clean_id.startswith(name):
            return cls(**kwargs)
    
    raise ValueError(f"Unknown scanner: {scanner_id}. Available: {list(scanners.keys())}")
