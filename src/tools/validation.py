"""
Strategy Validator - Validate strategy configurations before execution.

Ensures strategies are well-formed and catches common errors.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class ValidationSeverity(Enum):
    """Severity of validation issue."""
    ERROR = "ERROR"      # Must fix before running
    WARNING = "WARNING"  # Should review
    INFO = "INFO"        # Informational


@dataclass
class ValidationIssue:
    """Single validation issue."""
    severity: ValidationSeverity
    category: str
    message: str
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        output = f"[{self.severity.value}] {self.category}: {self.message}"
        if self.suggestion:
            output += f"\n  → Suggestion: {self.suggestion}"
        return output


@dataclass
class ValidationResult:
    """Result of strategy validation."""
    valid: bool
    issues: List[ValidationIssue]
    
    def has_errors(self) -> bool:
        return any(i.severity == ValidationSeverity.ERROR for i in self.issues)
    
    def has_warnings(self) -> bool:
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)
    
    def print_report(self):
        """Print validation report."""
        if self.valid and not self.issues:
            print("✓ Strategy validation passed with no issues")
            return
        
        print("\nStrategy Validation Report")
        print("=" * 60)
        
        errors = [i for i in self.issues if i.severity == ValidationSeverity.ERROR]
        warnings = [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
        infos = [i for i in self.issues if i.severity == ValidationSeverity.INFO]
        
        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            for issue in errors:
                print(f"  {issue}")
        
        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for issue in warnings:
                print(f"  {issue}")
        
        if infos:
            print(f"\n💡 INFO ({len(infos)}):")
            for issue in infos:
                print(f"  {issue}")
        
        print(f"\nOverall: {'❌ FAILED' if not self.valid else '✓ PASSED'}")


class StrategyValidator:
    """
    Validate strategy configurations.
    
    Checks:
    - Trigger configurations
    - Scanner parameters
    - Bracket/risk management
    - Date ranges
    - Model compatibility
    - Common pitfalls
    
    Usage:
        result = StrategyValidator.validate(strategy_config)
        if not result.valid:
            result.print_report()
            raise ValueError("Strategy validation failed")
    """
    
    @staticmethod
    def validate(strategy_config: Any) -> ValidationResult:
        """
        Validate a strategy configuration.
        
        Args:
            strategy_config: StrategyConfig object or dict
            
        Returns:
            ValidationResult
        """
        issues = []
        
        # Convert to dict if needed
        if hasattr(strategy_config, 'to_dict'):
            config = strategy_config.to_dict()
        else:
            config = strategy_config
        
        # Check scanner
        issues.extend(StrategyValidator._validate_scanner(config))
        
        # Check brackets/risk
        issues.extend(StrategyValidator._validate_risk(config))
        
        # Check dates
        issues.extend(StrategyValidator._validate_dates(config))
        
        # Check trigger if modular scanner
        if config.get('scanner_id') == 'modular':
            issues.extend(StrategyValidator._validate_trigger(config))
        
        # Check for common issues
        issues.extend(StrategyValidator._check_common_issues(config))
        
        # Valid if no errors
        valid = not any(i.severity == ValidationSeverity.ERROR for i in issues)
        
        return ValidationResult(valid=valid, issues=issues)
    
    @staticmethod
    def _validate_scanner(config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate scanner configuration."""
        issues = []
        
        scanner_id = config.get('scanner_id')
        if not scanner_id:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="Scanner",
                message="No scanner_id specified",
                suggestion="Set scanner_id to a valid scanner (e.g., 'rsi_extreme', 'level_proximity')"
            ))
            return issues
        
        # Known scanners
        known_scanners = [
            'always', 'interval', 'level_proximity', 'rsi_extreme', 
            'modular', 'openingrange', 'simpletime', 'middayreversal',
            'meanreversion'
        ]
        
        if scanner_id not in known_scanners:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="Scanner",
                message=f"Unknown scanner_id: {scanner_id}",
                suggestion=f"Known scanners: {', '.join(known_scanners)}"
            ))
        
        return issues
    
    @staticmethod
    def _validate_risk(config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate risk/bracket configuration."""
        issues = []
        
        # Check stop ATR
        stop_atr = config.get('oco_stop_atr', config.get('stop_atr'))
        if stop_atr is not None:
            if stop_atr <= 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="Risk",
                    message=f"Invalid stop_atr: {stop_atr} (must be > 0)",
                ))
            elif stop_atr > 3.0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="Risk",
                    message=f"Large stop_atr: {stop_atr} (typically 0.5-2.0)",
                    suggestion="Very wide stops may have poor risk/reward"
                ))
        
        # Check TP multiple
        tp_multiple = config.get('oco_tp_multiple', config.get('tp_multiple'))
        if tp_multiple is not None:
            if tp_multiple <= 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="Risk",
                    message=f"Invalid tp_multiple: {tp_multiple} (must be > 0)",
                ))
            elif tp_multiple < 1.0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="Risk",
                    message=f"Low tp_multiple: {tp_multiple} (risk > reward)",
                    suggestion="Consider tp_multiple >= 1.0 for positive risk/reward"
                ))
            elif tp_multiple > 5.0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="Risk",
                    message=f"High tp_multiple: {tp_multiple} (may not reach target)",
                    suggestion="Very high targets may have low hit rate"
                ))
        
        # Check if both are set
        if stop_atr and tp_multiple:
            r_multiple = tp_multiple
            if r_multiple < 1.0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="Risk",
                    message=f"Risk/Reward ratio: 1:{r_multiple:.1f}",
                ))
        
        return issues
    
    @staticmethod
    def _validate_dates(config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate date ranges."""
        issues = []
        
        start_date = config.get('start_date')
        end_date = config.get('end_date')
        
        if not start_date:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="Dates",
                message="No start_date specified",
                suggestion="Set start_date for reproducible backtests"
            ))
        
        if not end_date:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="Dates",
                message="No end_date specified",
                suggestion="Set end_date for reproducible backtests"
            ))
        
        if start_date and end_date:
            try:
                from datetime import datetime
                start = datetime.fromisoformat(start_date)
                end = datetime.fromisoformat(end_date)
                
                if start >= end:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="Dates",
                        message="start_date must be before end_date",
                    ))
                
                days = (end - start).days
                if days < 5:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="Dates",
                        message=f"Short date range: {days} days",
                        suggestion="Consider longer range for statistically significant results"
                    ))
                elif days > 365:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        category="Dates",
                        message=f"Long date range: {days} days",
                    ))
            except:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="Dates",
                    message="Could not parse dates",
                    suggestion="Use YYYY-MM-DD format"
                ))
        
        return issues
    
    @staticmethod
    def _validate_trigger(config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate trigger configuration for modular scanner."""
        issues = []
        
        scanner_params = config.get('scanner_params', {})
        trigger_config = scanner_params.get('trigger_config')
        
        if not trigger_config:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="Trigger",
                message="Modular scanner requires trigger_config in scanner_params",
                suggestion="Add trigger_config: {'type': 'time', 'hour': 10, ...}"
            ))
            return issues
        
        # Validate trigger can be created
        try:
            from src.policy.triggers import trigger_from_dict
            trigger_from_dict(trigger_config)
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="Trigger",
                message=f"Invalid trigger_config: {e}",
            ))
        
        return issues
    
    @staticmethod
    def _check_common_issues(config: Dict[str, Any]) -> List[ValidationIssue]:
        """Check for common strategy issues."""
        issues = []
        
        # Check for overfitting indicators
        if config.get('use_1h_features') and config.get('use_4h_features'):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="Features",
                message="Using both 1h and 4h features",
            ))
        
        # Check cooldown
        scanner_params = config.get('scanner_params', {})
        cooldown = scanner_params.get('cooldown_bars', scanner_params.get('cooldown', 20))
        if isinstance(cooldown, (int, float)) and cooldown < 5:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="Scanner",
                message=f"Very short cooldown: {cooldown} bars",
                suggestion="Short cooldown may lead to overtrading"
            ))
        
        return issues
    
    @staticmethod
    def quick_check(strategy_config: Any) -> bool:
        """Quick validation - returns True if valid, raises on errors."""
        result = StrategyValidator.validate(strategy_config)
        if not result.valid:
            result.print_report()
            raise ValueError("Strategy validation failed")
        return True
