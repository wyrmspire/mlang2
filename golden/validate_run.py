"""
Golden Run Validator

Validates that a run directory conforms to the architectural requirements
defined in ARCHITECTURE_AGREEMENT.md.

Usage:
    python golden/validator.py results/viz/my_run
    python -m golden.validator results/viz/my_run
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: str  # 'ERROR' or 'WARNING'
    category: str  # 'file_inventory', 'manifest', 'decision', 'trade', 'oco'
    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    
    def __str__(self):
        loc = f" at {self.file}" if self.file else ""
        line_info = f":{self.line}" if self.line else ""
        return f"[{self.severity}] {self.category}{loc}{line_info}: {self.message}"


class RunValidator:
    """Validates run directory structure and content."""
    
    def __init__(self, run_path: Path):
        self.run_path = Path(run_path)
        self.issues: List[ValidationIssue] = []
    
    def add_error(self, category: str, message: str, file: str = None, line: int = None):
        """Add an error to the issues list."""
        self.issues.append(ValidationIssue(
            severity='ERROR',
            category=category,
            message=message,
            file=file,
            line=line
        ))
    
    def add_warning(self, category: str, message: str, file: str = None, line: int = None):
        """Add a warning to the issues list."""
        self.issues.append(ValidationIssue(
            severity='WARNING',
            category=category,
            message=message,
            file=file,
            line=line
        ))
    
    def validate(self) -> List[ValidationIssue]:
        """Run all validations and return list of issues."""
        self.issues = []
        
        if not self.run_path.exists():
            self.add_error('file_inventory', f"Run directory does not exist: {self.run_path}")
            return self.issues
        
        if not self.run_path.is_dir():
            self.add_error('file_inventory', f"Path is not a directory: {self.run_path}")
            return self.issues
        
        # Validate file inventory
        self._validate_file_inventory()
        
        # Validate manifest
        manifest_path = self.run_path / "manifest.json"
        if manifest_path.exists():
            self._validate_manifest(manifest_path)
        
        # Validate decisions
        decisions_path = self.run_path / "decisions.jsonl"
        if decisions_path.exists():
            self._validate_decisions(decisions_path)
        
        # Validate trades
        trades_path = self.run_path / "trades.jsonl"
        if trades_path.exists():
            self._validate_trades(trades_path)
        
        return self.issues
    
    def _validate_file_inventory(self):
        """Validate required files exist."""
        required_files = ['manifest.json', 'decisions.jsonl', 'trades.jsonl']
        
        for file in required_files:
            file_path = self.run_path / file
            if not file_path.exists():
                self.add_error('file_inventory', f"Required file missing: {file}")
    
    def _validate_manifest(self, manifest_path: Path):
        """Validate manifest.json structure."""
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            self.add_error('manifest', f"Invalid JSON: {e}", file='manifest.json')
            return
        except Exception as e:
            self.add_error('manifest', f"Failed to read manifest: {e}", file='manifest.json')
            return
        
        # Check required fields
        required_fields = ['run_id', 'fingerprint', 'created_at']
        for field in required_fields:
            if field not in manifest:
                self.add_error('manifest', f"Missing required field: {field}", file='manifest.json')
        
        # Validate created_at is ISO 8601
        if 'created_at' in manifest:
            if not self._is_valid_iso8601(manifest['created_at']):
                self.add_error('manifest', 
                    f"created_at is not valid ISO 8601 with timezone: {manifest['created_at']}", 
                    file='manifest.json')
        
        # Validate config is present
        if 'config' not in manifest:
            self.add_warning('manifest', "Missing 'config' field", file='manifest.json')
        elif not isinstance(manifest['config'], dict):
            self.add_error('manifest', "'config' must be a dictionary", file='manifest.json')
        
        # Validate file_inventory
        if 'file_inventory' in manifest:
            if not isinstance(manifest['file_inventory'], list):
                self.add_error('manifest', "'file_inventory' must be a list", file='manifest.json')
            else:
                # Check that listed files actually exist
                for file in manifest['file_inventory']:
                    file_path = self.run_path / file
                    if not file_path.exists():
                        self.add_warning('manifest', 
                            f"File listed in inventory but not found: {file}", 
                            file='manifest.json')
    
    def _validate_decisions(self, decisions_path: Path):
        """Validate decisions.jsonl structure."""
        try:
            with open(decisions_path) as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        decision = json.loads(line)
                    except json.JSONDecodeError as e:
                        self.add_error('decision', f"Invalid JSON: {e}", 
                            file='decisions.jsonl', line=line_num)
                        continue
                    
                    self._validate_decision_structure(decision, line_num)
        except Exception as e:
            self.add_error('decision', f"Failed to read decisions: {e}", file='decisions.jsonl')
    
    def _validate_decision_structure(self, decision: Dict[str, Any], line_num: int):
        """Validate a single decision structure."""
        # Required fields
        required = ['decision_id', 'timestamp', 'bar_idx', 'action']
        for field in required:
            if field not in decision:
                self.add_error('decision', f"Missing required field: {field}", 
                    file='decisions.jsonl', line=line_num)
        
        # Validate timestamp
        if 'timestamp' in decision:
            if not self._is_valid_iso8601(decision['timestamp']):
                self.add_error('decision', 
                    f"timestamp is not valid ISO 8601 with timezone: {decision['timestamp']}", 
                    file='decisions.jsonl', line=line_num)
        
        # If action is PLACE_ORDER, validate OCO
        if decision.get('action') == 'PLACE_ORDER':
            if 'oco' not in decision:
                self.add_error('decision', 
                    "action is PLACE_ORDER but 'oco' is missing", 
                    file='decisions.jsonl', line=line_num)
            else:
                self._validate_oco_structure(decision['oco'], line_num)
            
            # Validate oco_results if present
            if 'oco_results' in decision:
                self._validate_oco_results(decision, line_num)
    
    def _validate_oco_structure(self, oco: Dict[str, Any], line_num: int):
        """Validate OCO bracket structure."""
        # Check for contracts field
        if 'contracts' not in oco:
            self.add_error('oco', 
                "OCO bracket missing required 'contracts' field", 
                file='decisions.jsonl', line=line_num)
        
        # Check that oco doesn't have nested results (common error)
        if 'results' in oco:
            self.add_error('oco', 
                "oco_results must be FLAT at decision level, not nested in 'oco.results'", 
                file='decisions.jsonl', line=line_num)
        
        # Check for basic price fields
        price_fields = ['entry_price', 'stop_price', 'tp_price']
        for field in price_fields:
            if field not in oco:
                self.add_warning('oco', f"OCO missing recommended field: {field}", 
                    file='decisions.jsonl', line=line_num)
    
    def _validate_oco_results(self, decision: Dict[str, Any], line_num: int):
        """Validate oco_results structure - must be FLAT, not nested."""
        oco_results = decision.get('oco_results')
        
        # oco_results must be a dict at the decision level, not nested in oco
        if not isinstance(oco_results, dict):
            self.add_error('oco', 
                "oco_results must be a dictionary", 
                file='decisions.jsonl', line=line_num)
            return
        
        # Check that oco doesn't have nested results
        oco = decision.get('oco', {})
        if isinstance(oco, dict) and 'results' in oco:
            self.add_error('oco', 
                "oco_results must be FLAT at decision level, not nested in 'oco.results'", 
                file='decisions.jsonl', line=line_num)
        
        # Validate oco_results contains expected fields
        expected = ['filled', 'bars_held', 'pnl_dollars', 'outcome']
        missing = [f for f in expected if f not in oco_results]
        if missing:
            self.add_warning('oco', 
                f"oco_results missing recommended fields: {', '.join(missing)}", 
                file='decisions.jsonl', line=line_num)
    
    def _validate_trades(self, trades_path: Path):
        """Validate trades.jsonl structure."""
        try:
            with open(trades_path) as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        trade = json.loads(line)
                    except json.JSONDecodeError as e:
                        self.add_error('trade', f"Invalid JSON: {e}", 
                            file='trades.jsonl', line=line_num)
                        continue
                    
                    self._validate_trade_structure(trade, line_num)
        except Exception as e:
            self.add_error('trade', f"Failed to read trades: {e}", file='trades.jsonl')
    
    def _validate_trade_structure(self, trade: Dict[str, Any], line_num: int):
        """Validate a single trade structure."""
        # Required fields
        required = ['trade_id', 'decision_id', 'entry_price', 'exit_price', 
                   'pnl_dollars', 'outcome']
        for field in required:
            if field not in trade:
                self.add_error('trade', f"Missing required field: {field}", 
                    file='trades.jsonl', line=line_num)
        
        # Recommended fields
        recommended = ['bars_held', 'entry_time', 'exit_time', 'exit_reason']
        for field in recommended:
            if field not in trade:
                self.add_warning('trade', f"Missing recommended field: {field}", 
                    file='trades.jsonl', line=line_num)
        
        # Validate timestamps if present
        for time_field in ['entry_time', 'exit_time']:
            if time_field in trade and trade[time_field]:
                if not self._is_valid_iso8601(trade[time_field]):
                    self.add_error('trade', 
                        f"{time_field} is not valid ISO 8601 with timezone: {trade[time_field]}", 
                        file='trades.jsonl', line=line_num)
    
    def _is_valid_iso8601(self, timestamp_str: str) -> bool:
        """Check if timestamp is valid ISO 8601 format with timezone."""
        if not timestamp_str:
            return False
        
        # Must contain timezone info (either Z or +/-HH:MM)
        has_tz = ('Z' in timestamp_str or '+' in timestamp_str or 
                  timestamp_str.count('-') > 2)  # More than just date dashes
        
        if not has_tz:
            return False
        
        # Try to parse
        try:
            datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return True
        except (ValueError, AttributeError):
            return False


def validate_run_structure(run_path: Path, golden_path: Path = None) -> List[ValidationIssue]:
    """
    Validate a run against architectural requirements.
    
    Args:
        run_path: Path to the run directory to validate
        golden_path: Optional path to golden reference (for future comparison)
    
    Returns:
        List of ValidationIssue objects
    """
    validator = RunValidator(run_path)
    issues = validator.validate()
    
    # TODO: If golden_path provided, do structural comparison
    # (Phase 2 extension - compare decision count, trade count, etc.)
    
    return issues


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python golden/validator.py <run_directory>")
        print("Example: python golden/validator.py results/viz/my_run")
        sys.exit(1)
    
    run_path = Path(sys.argv[1])
    
    print(f"Validating run: {run_path}")
    print("=" * 60)
    
    issues = validate_run_structure(run_path)
    
    if not issues:
        print("[OK] No issues found! Run structure is valid.")
        sys.exit(0)
    
    # Group by severity
    errors = [i for i in issues if i.severity == 'ERROR']
    warnings = [i for i in issues if i.severity == 'WARNING']
    
    if errors:
        print(f"\n[ERROR] {len(errors)} ERROR(S) found:\n")
        for issue in errors:
            print(f"  {issue}")
    
    if warnings:
        print(f"\n[WARN] {len(warnings)} WARNING(S) found:\n")
        for issue in warnings:
            print(f"  {issue}")
    
    print("\n" + "=" * 60)
    print(f"Total: {len(errors)} errors, {len(warnings)} warnings")
    
    # Exit with error if any errors found
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
