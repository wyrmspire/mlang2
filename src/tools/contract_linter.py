"""
Contract Linter - Validates Run Artifacts

This linter enforces the "Single Run Artifact Contract" defined in ARCHITECTURE_AGREEMENT.md.

Checks:
1. Required files exist (manifest.json, decisions.jsonl, trades.jsonl if applicable)
2. Manifest schema is valid
3. Timestamps are ISO 8601 with timezone
4. oco_results is flat (not nested)
5. Required fields are present
6. Contracts field exists for decisions
7. Strategy spec is present (if using StrategySpec)

Usage:
    python -m src.tools.contract_linter <run_dir>
    
Integration:
    - Called by build/CI to validate artifacts before merge
    - Can be used in exporter to validate during write
    - Part of golden run validation
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime


class ContractLinterError:
    """Represents a linting error."""
    
    def __init__(self, severity: str, category: str, message: str, file: str = "", line: int = 0):
        self.severity = severity  # ERROR, WARNING, INFO
        self.category = category  # MISSING_FILE, SCHEMA, TIMESTAMP, etc.
        self.message = message
        self.file = file
        self.line = line
    
    def __str__(self):
        location = f"{self.file}:{self.line}" if self.file else ""
        return f"[{self.severity}] {self.category}: {self.message} ({location})"


class ContractLinter:
    """
    Linter for run artifact contracts.
    
    Validates that run outputs conform to the canonical schema.
    """
    
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.errors: List[ContractLinterError] = []
    
    def lint(self) -> Tuple[bool, List[ContractLinterError]]:
        """
        Run all linting checks.
        
        Returns:
            (is_valid, errors)
        """
        self.errors = []
        
        # Check directory exists
        if not self.run_dir.exists():
            self.errors.append(ContractLinterError(
                "ERROR", "MISSING_DIR",
                f"Run directory does not exist: {self.run_dir}"
            ))
            return False, self.errors
        
        # Run checks
        self._check_required_files()
        self._check_manifest()
        self._check_decisions()
        self._check_trades()
        self._check_windows()
        
        # Determine if valid (no ERRORs)
        has_errors = any(e.severity == "ERROR" for e in self.errors)
        return not has_errors, self.errors
    
    def _check_required_files(self):
        """Check that required files exist."""
        required = ["manifest.json"]
        for filename in required:
            if not (self.run_dir / filename).exists():
                self.errors.append(ContractLinterError(
                    "ERROR", "MISSING_FILE",
                    f"Required file missing: {filename}"
                ))
        
        # decisions.jsonl should exist if there are any decisions
        decisions_file = self.run_dir / "decisions.jsonl"
        if not decisions_file.exists():
            self.errors.append(ContractLinterError(
                "WARNING", "MISSING_FILE",
                "decisions.jsonl not found - run may have no decisions"
            ))
    
    def _check_manifest(self):
        """Validate manifest.json structure."""
        manifest_path = self.run_dir / "manifest.json"
        if not manifest_path.exists():
            return  # Already reported in required files
        
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(ContractLinterError(
                "ERROR", "SCHEMA",
                f"manifest.json is not valid JSON: {e}",
                file="manifest.json"
            ))
            return
        
        # Check required fields
        required_fields = ["run_id", "created_at", "strategy"]
        for field in required_fields:
            if field not in manifest:
                self.errors.append(ContractLinterError(
                    "ERROR", "SCHEMA",
                    f"manifest.json missing required field: {field}",
                    file="manifest.json"
                ))
        
        # Check timestamp format
        if "created_at" in manifest:
            self._check_timestamp(manifest["created_at"], "manifest.json", "created_at")
        
        # Check if strategy_spec exists (recommended)
        if "strategy_spec" not in manifest:
            self.errors.append(ContractLinterError(
                "WARNING", "SCHEMA",
                "manifest.json missing strategy_spec field - recommended for reproducibility",
                file="manifest.json"
            ))
    
    def _check_decisions(self):
        """Validate decisions.jsonl structure."""
        decisions_path = self.run_dir / "decisions.jsonl"
        if not decisions_path.exists():
            return  # Already reported
        
        try:
            with open(decisions_path) as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        decision = json.loads(line)
                    except json.JSONDecodeError as e:
                        self.errors.append(ContractLinterError(
                            "ERROR", "SCHEMA",
                            f"Invalid JSON: {e}",
                            file="decisions.jsonl",
                            line=line_num
                        ))
                        continue
                    
                    # Check required fields
                    required = ["decision_id", "timestamp", "action"]
                    for field in required:
                        if field not in decision:
                            self.errors.append(ContractLinterError(
                                "ERROR", "SCHEMA",
                                f"Decision missing required field: {field}",
                                file="decisions.jsonl",
                                line=line_num
                            ))
                    
                    # Check timestamp
                    if "timestamp" in decision:
                        self._check_timestamp(
                            decision["timestamp"],
                            "decisions.jsonl",
                            f"decision.timestamp (line {line_num})"
                        )
                    
                    # Check oco_results is flat if present
                    if "oco_results" in decision:
                        self._check_oco_results_flat(
                            decision["oco_results"],
                            "decisions.jsonl",
                            line_num
                        )
                    
                    # Check contracts field if action is PLACE_ORDER
                    if decision.get("action") == "PLACE_ORDER":
                        if "contracts" not in decision:
                            self.errors.append(ContractLinterError(
                                "WARNING", "SCHEMA",
                                "PLACE_ORDER decision missing 'contracts' field",
                                file="decisions.jsonl",
                                line=line_num
                            ))
        
        except Exception as e:
            self.errors.append(ContractLinterError(
                "ERROR", "FILE_READ",
                f"Failed to read decisions.jsonl: {e}",
                file="decisions.jsonl"
            ))
    
    def _check_trades(self):
        """Validate trades.jsonl if it exists."""
        trades_path = self.run_dir / "trades.jsonl"
        if not trades_path.exists():
            return  # Optional file
        
        try:
            with open(trades_path) as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        trade = json.loads(line)
                    except json.JSONDecodeError as e:
                        self.errors.append(ContractLinterError(
                            "ERROR", "SCHEMA",
                            f"Invalid JSON: {e}",
                            file="trades.jsonl",
                            line=line_num
                        ))
                        continue
                    
                    # Check required fields
                    required = ["trade_id", "entry_time", "exit_time", "outcome"]
                    for field in required:
                        if field not in trade:
                            self.errors.append(ContractLinterError(
                                "ERROR", "SCHEMA",
                                f"Trade missing required field: {field}",
                                file="trades.jsonl",
                                line=line_num
                            ))
                    
                    # Check timestamps
                    for ts_field in ["entry_time", "exit_time"]:
                        if ts_field in trade:
                            self._check_timestamp(
                                trade[ts_field],
                                "trades.jsonl",
                                f"trade.{ts_field} (line {line_num})"
                            )
        
        except Exception as e:
            self.errors.append(ContractLinterError(
                "ERROR", "FILE_READ",
                f"Failed to read trades.jsonl: {e}",
                file="trades.jsonl"
            ))
    
    def _check_windows(self):
        """Validate windows.json if it exists."""
        windows_path = self.run_dir / "windows.json"
        if not windows_path.exists():
            return  # Optional file
        
        try:
            with open(windows_path) as f:
                windows = json.load(f)
            
            # Check structure
            if not isinstance(windows, dict):
                self.errors.append(ContractLinterError(
                    "ERROR", "SCHEMA",
                    "windows.json must be a dictionary",
                    file="windows.json"
                ))
                return
            
            if "windows" not in windows:
                self.errors.append(ContractLinterError(
                    "ERROR", "SCHEMA",
                    "windows.json missing 'windows' field",
                    file="windows.json"
                ))
        
        except json.JSONDecodeError as e:
            self.errors.append(ContractLinterError(
                "ERROR", "SCHEMA",
                f"windows.json is not valid JSON: {e}",
                file="windows.json"
            ))
        except Exception as e:
            self.errors.append(ContractLinterError(
                "ERROR", "FILE_READ",
                f"Failed to read windows.json: {e}",
                file="windows.json"
            ))
    
    def _check_timestamp(self, ts: str, file: str, field: str):
        """Check timestamp is ISO 8601 with timezone."""
        if not isinstance(ts, str):
            self.errors.append(ContractLinterError(
                "ERROR", "TIMESTAMP",
                f"{field} must be a string, got {type(ts)}",
                file=file
            ))
            return
        
        # Check for timezone indicator
        if not (ts.endswith('Z') or '+' in ts or ts.endswith('00:00')):
            self.errors.append(ContractLinterError(
                "ERROR", "TIMESTAMP",
                f"{field} missing timezone (must end with Z or have +offset): {ts}",
                file=file
            ))
        
        # Try to parse as ISO 8601
        try:
            datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except ValueError as e:
            self.errors.append(ContractLinterError(
                "ERROR", "TIMESTAMP",
                f"{field} is not valid ISO 8601: {ts} ({e})",
                file=file
            ))
    
    def _check_oco_results_flat(self, oco_results: Any, file: str, line: int):
        """
        Check that oco_results is a flat dict, not nested.
        
        According to ARCHITECTURE_AGREEMENT.md:
        oco_results must be a flat dictionary with keys like:
        - filled, outcome, exit_price, bars_held, etc.
        
        NOT nested like: {"LONG_1R": {...}, "SHORT_2R": {...}}
        """
        if not isinstance(oco_results, dict):
            self.errors.append(ContractLinterError(
                "ERROR", "OCO_RESULTS",
                f"oco_results must be a dict, got {type(oco_results)}",
                file=file,
                line=line
            ))
            return
        
        # Check for nested structure (values are dicts)
        for key, value in oco_results.items():
            if isinstance(value, dict):
                self.errors.append(ContractLinterError(
                    "ERROR", "OCO_RESULTS",
                    f"oco_results must be flat, found nested dict at key '{key}'",
                    file=file,
                    line=line
                ))
                return
        
        # Check for required fields
        required = ["filled", "outcome"]
        for field in required:
            if field not in oco_results:
                self.errors.append(ContractLinterError(
                    "WARNING", "OCO_RESULTS",
                    f"oco_results missing recommended field: {field}",
                    file=file,
                    line=line
                ))


def lint_run_artifact(run_dir: Path, verbose: bool = False) -> bool:
    """
    Lint a run artifact directory.
    
    Args:
        run_dir: Path to run directory
        verbose: Print all errors/warnings
    
    Returns:
        True if valid (no errors)
    """
    linter = ContractLinter(run_dir)
    is_valid, errors = linter.lint()
    
    if verbose or not is_valid:
        print(f"\n{'='*60}")
        print(f"Contract Linter: {run_dir}")
        print(f"{'='*60}\n")
        
        if errors:
            for error in errors:
                print(error)
            
            error_count = sum(1 for e in errors if e.severity == "ERROR")
            warning_count = sum(1 for e in errors if e.severity == "WARNING")
            
            print(f"\n{'='*60}")
            print(f"Total: {error_count} errors, {warning_count} warnings")
            print(f"{'='*60}\n")
        else:
            print("✓ No issues found\n")
    
    return is_valid


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.tools.contract_linter <run_dir>")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    is_valid = lint_run_artifact(run_dir, verbose=True)
    
    sys.exit(0 if is_valid else 1)
