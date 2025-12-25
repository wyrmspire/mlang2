"""
Scanner Registration
Wire existing scanners into the ScannerRegistry.
"""

from src.core.registries import ScannerRegistry
from src.policy.scanners import AlwaysScanner, IntervalScanner
from src.policy.modular_scanner import ModularScanner


# =============================================================================
# Register built-in scanners
# =============================================================================

@ScannerRegistry.register(
    scanner_id="always",
    name="Always Scanner",
    description="Triggers on every bar - useful for testing or fixed strategies",
    params_schema={}
)
class AlwaysScannerWrapper:
    """Wrapper to adapt AlwaysScanner to registry."""
    def __init__(self):
        self._scanner = AlwaysScanner()
    
    def scan(self, step_result):
        # Adapt to registry interface
        # In real use, would extract state and features from step_result
        from src.policy.scanners import ScanResult
        return ScanResult(
            scanner_id="always",
            triggered=True,
            score=1.0
        )


@ScannerRegistry.register(
    scanner_id="interval",
    name="Interval Scanner",
    description="Triggers every N bars",
    params_schema={
        "interval": {"type": "integer", "default": 5, "min": 1}
    }
)
class IntervalScannerWrapper:
    """Wrapper to adapt IntervalScanner to registry."""
    def __init__(self, interval: int = 5):
        self._scanner = IntervalScanner(interval=interval)
    
    def scan(self, step_result):
        from src.policy.scanners import ScanResult
        # Simplified - real implementation would extract features
        return ScanResult(
            scanner_id=f"interval_{self._scanner.interval}",
            triggered=False,  # Placeholder
            score=0.0
        )


@ScannerRegistry.register(
    scanner_id="modular",
    name="Modular Scanner",
    description="Scanner based on composable triggers (time, candle patterns, indicators)",
    params_schema={
        "trigger_config": {
            "type": "object",
            "description": "Trigger configuration dict",
            "required": True
        },
        "cooldown_bars": {
            "type": "integer",
            "default": 20,
            "min": 0
        }
    }
)
class ModularScannerWrapper:
    """Wrapper to adapt ModularScanner to registry."""
    def __init__(self, trigger_config: dict, cooldown_bars: int = 20):
        self._scanner = ModularScanner(
            trigger_config=trigger_config,
            cooldown_bars=cooldown_bars
        )
    
    def scan(self, step_result):
        from src.policy.scanners import ScanResult
        return ScanResult(
            scanner_id=self._scanner.scanner_id,
            triggered=False,  # Placeholder
            score=0.0
        )


# Auto-register on import
def register_all_scanners():
    """
    Register all available scanners.
    Call this at startup to populate the registry.
    """
    # The decorators above already registered them
    # This function just serves as a hook for explicit initialization
    pass
