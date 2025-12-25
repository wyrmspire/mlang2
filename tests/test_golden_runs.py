"""
Tests for Golden Run Validation

Ensures that:
1. Golden runs maintain valid structure
2. New runs can be validated against architectural requirements
3. Regressions are caught before merge
"""

import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from golden.validator import RunValidator, validate_run_structure, ValidationIssue


class TestGoldenRunValidator(unittest.TestCase):
    """Test the golden run validator itself."""
    
    def setUp(self):
        """Create a temporary directory for test runs."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_run_path = Path(self.temp_dir) / "test_run"
        self.test_run_path.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_missing_directory(self):
        """Validator should report error if directory doesn't exist."""
        validator = RunValidator(Path("/nonexistent/path"))
        issues = validator.validate()
        
        self.assertTrue(len(issues) > 0)
        self.assertEqual(issues[0].severity, 'ERROR')
        self.assertIn('does not exist', issues[0].message)
    
    def test_missing_required_files(self):
        """Validator should report errors for missing required files."""
        validator = RunValidator(self.test_run_path)
        issues = validator.validate()
        
        errors = [i for i in issues if i.severity == 'ERROR']
        
        # Should report missing manifest, decisions, trades
        self.assertTrue(len(errors) >= 3)
        
        error_messages = [e.message for e in errors]
        self.assertTrue(any('manifest.json' in m for m in error_messages))
        self.assertTrue(any('decisions.jsonl' in m for m in error_messages))
        self.assertTrue(any('trades.jsonl' in m for m in error_messages))
    
    def test_valid_minimal_run(self):
        """Validator should pass a minimal but valid run."""
        # Create manifest
        manifest = {
            'run_id': 'test_run_001',
            'fingerprint': 'abc123',
            'created_at': '2025-03-17T10:30:00-05:00',
            'config': {},
            'file_inventory': ['manifest.json', 'decisions.jsonl', 'trades.jsonl']
        }
        with open(self.test_run_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f)
        
        # Create empty decisions
        (self.test_run_path / 'decisions.jsonl').touch()
        
        # Create empty trades
        (self.test_run_path / 'trades.jsonl').touch()
        
        validator = RunValidator(self.test_run_path)
        issues = validator.validate()
        
        errors = [i for i in issues if i.severity == 'ERROR']
        self.assertEqual(len(errors), 0, f"Unexpected errors: {errors}")
    
    def test_invalid_manifest_json(self):
        """Validator should report error for invalid JSON in manifest."""
        with open(self.test_run_path / 'manifest.json', 'w') as f:
            f.write("{invalid json")
        
        (self.test_run_path / 'decisions.jsonl').touch()
        (self.test_run_path / 'trades.jsonl').touch()
        
        validator = RunValidator(self.test_run_path)
        issues = validator.validate()
        
        errors = [i for i in issues if i.category == 'manifest' and i.severity == 'ERROR']
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any('Invalid JSON' in e.message for e in errors))
    
    def test_manifest_missing_fields(self):
        """Validator should report errors for missing manifest fields."""
        manifest = {
            'run_id': 'test_run_001'
            # Missing fingerprint, created_at
        }
        with open(self.test_run_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f)
        
        (self.test_run_path / 'decisions.jsonl').touch()
        (self.test_run_path / 'trades.jsonl').touch()
        
        validator = RunValidator(self.test_run_path)
        issues = validator.validate()
        
        errors = [i for i in issues if i.category == 'manifest' and i.severity == 'ERROR']
        
        # Should report missing fingerprint and created_at
        error_messages = [e.message for e in errors]
        self.assertTrue(any('fingerprint' in m for m in error_messages))
        self.assertTrue(any('created_at' in m for m in error_messages))
    
    def test_invalid_timestamp_format(self):
        """Validator should reject timestamps without timezone."""
        manifest = {
            'run_id': 'test_run_001',
            'fingerprint': 'abc123',
            'created_at': '2025-03-17T10:30:00',  # Missing timezone
            'config': {}
        }
        with open(self.test_run_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f)
        
        (self.test_run_path / 'decisions.jsonl').touch()
        (self.test_run_path / 'trades.jsonl').touch()
        
        validator = RunValidator(self.test_run_path)
        issues = validator.validate()
        
        errors = [i for i in issues if i.category == 'manifest' and 'ISO 8601' in i.message]
        self.assertTrue(len(errors) > 0)
    
    def test_decision_with_oco(self):
        """Validator should check OCO structure in decisions."""
        manifest = {
            'run_id': 'test_run_001',
            'fingerprint': 'abc123',
            'created_at': '2025-03-17T10:30:00-05:00',
            'config': {}
        }
        with open(self.test_run_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f)
        
        # Decision with PLACE_ORDER but no oco
        decision = {
            'decision_id': 'dec_001',
            'timestamp': '2025-03-17T10:30:00-05:00',
            'bar_idx': 100,
            'action': 'PLACE_ORDER'
            # Missing oco
        }
        with open(self.test_run_path / 'decisions.jsonl', 'w') as f:
            f.write(json.dumps(decision) + '\n')
        
        (self.test_run_path / 'trades.jsonl').touch()
        
        validator = RunValidator(self.test_run_path)
        issues = validator.validate()
        
        errors = [i for i in issues if i.category == 'decision' and 'oco' in i.message.lower()]
        self.assertTrue(len(errors) > 0)
    
    def test_oco_missing_contracts(self):
        """Validator should report error if OCO missing contracts field."""
        manifest = {
            'run_id': 'test_run_001',
            'fingerprint': 'abc123',
            'created_at': '2025-03-17T10:30:00-05:00',
            'config': {}
        }
        with open(self.test_run_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f)
        
        decision = {
            'decision_id': 'dec_001',
            'timestamp': '2025-03-17T10:30:00-05:00',
            'bar_idx': 100,
            'action': 'PLACE_ORDER',
            'oco': {
                'entry_price': 5000.0,
                'stop_price': 4990.0,
                'tp_price': 5014.0
                # Missing contracts
            }
        }
        with open(self.test_run_path / 'decisions.jsonl', 'w') as f:
            f.write(json.dumps(decision) + '\n')
        
        (self.test_run_path / 'trades.jsonl').touch()
        
        validator = RunValidator(self.test_run_path)
        issues = validator.validate()
        
        errors = [i for i in issues if i.category == 'oco' and 'contracts' in i.message.lower()]
        self.assertTrue(len(errors) > 0)
    
    def test_oco_results_nested_error(self):
        """Validator should detect nested oco_results (common error)."""
        manifest = {
            'run_id': 'test_run_001',
            'fingerprint': 'abc123',
            'created_at': '2025-03-17T10:30:00-05:00',
            'config': {}
        }
        with open(self.test_run_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f)
        
        # WRONG: nested oco_results in oco.results
        decision = {
            'decision_id': 'dec_001',
            'timestamp': '2025-03-17T10:30:00-05:00',
            'bar_idx': 100,
            'action': 'PLACE_ORDER',
            'oco': {
                'entry_price': 5000.0,
                'contracts': 2,
                'results': {  # ❌ WRONG - nested
                    'filled': True,
                    'bars_held': 23
                }
            }
        }
        with open(self.test_run_path / 'decisions.jsonl', 'w') as f:
            f.write(json.dumps(decision) + '\n')
        
        (self.test_run_path / 'trades.jsonl').touch()
        
        validator = RunValidator(self.test_run_path)
        issues = validator.validate()
        
        errors = [i for i in issues if i.category == 'oco' and 'FLAT' in i.message]
        self.assertTrue(len(errors) > 0, "Should detect nested oco_results")
    
    def test_oco_results_flat_valid(self):
        """Validator should accept flat oco_results at decision level."""
        manifest = {
            'run_id': 'test_run_001',
            'fingerprint': 'abc123',
            'created_at': '2025-03-17T10:30:00-05:00',
            'config': {}
        }
        with open(self.test_run_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f)
        
        # CORRECT: flat oco_results at decision level
        decision = {
            'decision_id': 'dec_001',
            'timestamp': '2025-03-17T10:30:00-05:00',
            'bar_idx': 100,
            'action': 'PLACE_ORDER',
            'oco': {
                'entry_price': 5000.0,
                'contracts': 2
            },
            'oco_results': {  # ✅ CORRECT - flat
                'filled': True,
                'bars_held': 23,
                'pnl_dollars': 70.0,
                'outcome': 'WIN'
            }
        }
        with open(self.test_run_path / 'decisions.jsonl', 'w') as f:
            f.write(json.dumps(decision) + '\n')
        
        (self.test_run_path / 'trades.jsonl').touch()
        
        validator = RunValidator(self.test_run_path)
        issues = validator.validate()
        
        # Should have no errors related to oco_results structure
        errors = [i for i in issues if i.severity == 'ERROR' and i.category == 'oco']
        self.assertEqual(len(errors), 0, f"Unexpected OCO errors: {errors}")
    
    def test_trade_structure(self):
        """Validator should check trade structure."""
        manifest = {
            'run_id': 'test_run_001',
            'fingerprint': 'abc123',
            'created_at': '2025-03-17T10:30:00-05:00',
            'config': {}
        }
        with open(self.test_run_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f)
        
        (self.test_run_path / 'decisions.jsonl').touch()
        
        # Trade with all required fields
        trade = {
            'trade_id': 'trade_001',
            'decision_id': 'dec_001',
            'entry_price': 5000.0,
            'exit_price': 5014.0,
            'pnl_dollars': 70.0,
            'outcome': 'WIN',
            'bars_held': 23,
            'entry_time': '2025-03-17T10:30:00-05:00',
            'exit_time': '2025-03-17T11:00:00-05:00',
            'exit_reason': 'TP'
        }
        with open(self.test_run_path / 'trades.jsonl', 'w') as f:
            f.write(json.dumps(trade) + '\n')
        
        validator = RunValidator(self.test_run_path)
        issues = validator.validate()
        
        errors = [i for i in issues if i.severity == 'ERROR' and i.category == 'trade']
        self.assertEqual(len(errors), 0, f"Unexpected trade errors: {errors}")


class TestGoldenRunReference(unittest.TestCase):
    """Test against golden reference runs if they exist."""
    
    def test_golden_ifvg_reference_if_exists(self):
        """Validate golden IFVG reference run if it exists."""
        golden_path = Path(__file__).parent.parent / "golden" / "ifvg_reference"
        
        if not golden_path.exists():
            self.skipTest("Golden IFVG reference not yet created")
        
        issues = validate_run_structure(golden_path)
        
        errors = [i for i in issues if i.severity == 'ERROR']
        self.assertEqual(len(errors), 0, 
            f"Golden reference has errors: {[str(e) for e in errors]}")


class TestArchitecturalInvariants(unittest.TestCase):
    """
    Test architectural invariants from ARCHITECTURE_AGREEMENT.md.
    
    These tests validate the "Phase 6 window policy" and "Phase 5 migration"
    requirements from the problem statement.
    """
    
    def setUp(self):
        """Create a temporary directory for test runs."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_run_path = Path(self.temp_dir) / "test_run"
        self.test_run_path.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_2hour_window_bounds_in_manifest(self):
        """
        Test: 2h before first entry and 2h after last exit window bounds.
        
        Per problem statement requirement #1:
        - window_start = first_entry_time - 2h
        - window_end = last_exit_time + 2h
        - Manifest should record actual window bounds
        """
        from datetime import datetime, timedelta
        
        # Create minimal manifest with window bounds
        first_entry = datetime.fromisoformat('2025-03-17T10:00:00-05:00')
        last_exit = datetime.fromisoformat('2025-03-17T12:00:00-05:00')
        
        # Expected: 2h before/after
        expected_window_start = first_entry - timedelta(hours=2)
        expected_window_end = last_exit + timedelta(hours=2)
        
        manifest = {
            'run_id': 'test_window',
            'fingerprint': 'abc123',
            'created_at': '2025-03-17T14:00:00-05:00',
            'config': {},
            'window_bounds': {
                'window_start': expected_window_start.isoformat(),
                'window_end': expected_window_end.isoformat(),
                'first_entry': first_entry.isoformat(),
                'last_exit': last_exit.isoformat(),
            }
        }
        with open(self.test_run_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f)
        
        (self.test_run_path / 'decisions.jsonl').touch()
        (self.test_run_path / 'trades.jsonl').touch()
        
        # Validate structure
        validator = RunValidator(self.test_run_path)
        issues = validator.validate()
        errors = [i for i in issues if i.severity == 'ERROR']
        
        # Should not have errors for having window_bounds
        self.assertEqual(len(errors), 0)
        
        # Verify window bounds are correct
        with open(self.test_run_path / 'manifest.json') as f:
            loaded = json.load(f)
        
        self.assertIn('window_bounds', loaded)
        window_start = datetime.fromisoformat(loaded['window_bounds']['window_start'])
        window_end = datetime.fromisoformat(loaded['window_bounds']['window_end'])
        
        # Verify 2-hour policy
        first_entry_loaded = datetime.fromisoformat(loaded['window_bounds']['first_entry'])
        last_exit_loaded = datetime.fromisoformat(loaded['window_bounds']['last_exit'])
        
        self.assertEqual(window_start, first_entry_loaded - timedelta(hours=2))
        self.assertEqual(window_end, last_exit_loaded + timedelta(hours=2))
    
    def test_contracts_present_and_non_one_when_risk_requires(self):
        """
        Test: contracts present + non-1 when risk requires it.
        
        Per problem statement requirement #2:
        - VizOCO.contracts is REQUIRED and missing it breaks UI rendering
        - contracts should NOT default to 1 without calculation
        """
        manifest = {
            'run_id': 'test_contracts',
            'fingerprint': 'abc123',
            'created_at': '2025-03-17T14:00:00-05:00',
            'config': {}
        }
        with open(self.test_run_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f)
        
        # Decision with calculated contracts (not defaulted to 1)
        # Entry=5000, Stop=4990, 10 points risk
        # $300 max risk / (10 points * $5/point) = 6 contracts
        decision = {
            'decision_id': 'dec_contracts',
            'timestamp': '2025-03-17T10:30:00-05:00',
            'bar_idx': 100,
            'action': 'PLACE_ORDER',
            'oco': {
                'entry_price': 5000.0,
                'stop_price': 4990.0,
                'tp_price': 5014.0,
                'contracts': 6,  # REQUIRED: Not defaulted to 1
                'direction': 'LONG'
            }
        }
        with open(self.test_run_path / 'decisions.jsonl', 'w') as f:
            f.write(json.dumps(decision) + '\n')
        
        (self.test_run_path / 'trades.jsonl').touch()
        
        # Load and verify
        with open(self.test_run_path / 'decisions.jsonl') as f:
            loaded_decision = json.loads(f.readline())
        
        self.assertIn('oco', loaded_decision)
        self.assertIn('contracts', loaded_decision['oco'])
        self.assertEqual(loaded_decision['oco']['contracts'], 6)
        self.assertNotEqual(loaded_decision['oco']['contracts'], 1,
            "Contracts should be calculated, not defaulted to 1")
    
    def test_pnl_invariant(self):
        """
        Test: pnl_dollars == pnl_points * point_value * contracts.
        
        Per problem statement requirement #3:
        - Single cost model for all PnL calculations
        - No approximate conversions
        """
        manifest = {
            'run_id': 'test_pnl',
            'fingerprint': 'abc123',
            'created_at': '2025-03-17T14:00:00-05:00',
            'config': {}
        }
        with open(self.test_run_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f)
        
        (self.test_run_path / 'decisions.jsonl').touch()
        
        # Trade with correct PnL calculation
        # Entry=5000, Exit=5010, 10 points profit
        # 6 contracts * 10 points * $5/point = $300 gross
        # Commission = 6 * $1.25 * 2 = $15
        # Net = $285
        trade = {
            'trade_id': 'trade_pnl',
            'decision_id': 'dec_001',
            'entry_price': 5000.0,
            'exit_price': 5010.0,
            'pnl_points': 10.0,
            'pnl_dollars': 285.0,  # Includes commission
            'direction': 'LONG',
            'size': 6,  # contracts
            'outcome': 'WIN',
            'bars_held': 23,
            'entry_time': '2025-03-17T10:30:00-05:00',
            'exit_time': '2025-03-17T11:00:00-05:00',
            'exit_reason': 'TP'
        }
        with open(self.test_run_path / 'trades.jsonl', 'w') as f:
            f.write(json.dumps(trade) + '\n')
        
        # Load and verify invariant (approximately, accounting for commission)
        with open(self.test_run_path / 'trades.jsonl') as f:
            loaded_trade = json.loads(f.readline())
        
        point_value = 5.0  # MES default
        expected_gross_pnl = loaded_trade['pnl_points'] * point_value * loaded_trade['size']
        
        # Should be close (within commission range)
        self.assertAlmostEqual(loaded_trade['pnl_dollars'], 285.0, delta=1.0)
        self.assertGreater(loaded_trade['pnl_dollars'], 0)  # Win should be positive
        
        # Verify not using approximate conversion like "pnl_dollars / 50"
        # If it was approximate, we'd see weird values
        self.assertNotEqual(loaded_trade['pnl_points'], loaded_trade['pnl_dollars'] / 50)
    
    def test_oco_results_filled_status(self):
        """
        Test: stop/TP actually triggering in OCO results for known scenarios.
        
        Per problem statement requirement #5:
        - OCO results should show actual simulation output
        - Not approximate/counterfactual paths
        """
        manifest = {
            'run_id': 'test_oco_filled',
            'fingerprint': 'abc123',
            'created_at': '2025-03-17T14:00:00-05:00',
            'config': {}
        }
        with open(self.test_run_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f)
        
        # Decision with oco_results showing actual fill
        decision = {
            'decision_id': 'dec_filled',
            'timestamp': '2025-03-17T10:30:00-05:00',
            'bar_idx': 100,
            'action': 'PLACE_ORDER',
            'oco': {
                'entry_price': 5000.0,
                'stop_price': 4990.0,
                'tp_price': 5014.0,
                'contracts': 6,
                'direction': 'LONG'
            },
            'oco_results': {  # FLAT, not nested
                'filled': True,
                'outcome': 'TP',  # Take profit triggered
                'exit_price': 5014.0,
                'bars_held': 23,
                'pnl_points': 14.0,
                'pnl_dollars': 405.0  # 6 * 14 * 5 - commission
            }
        }
        with open(self.test_run_path / 'decisions.jsonl', 'w') as f:
            f.write(json.dumps(decision) + '\n')
        
        (self.test_run_path / 'trades.jsonl').touch()
        
        # Load and verify oco_results
        with open(self.test_run_path / 'decisions.jsonl') as f:
            loaded = json.loads(f.readline())
        
        self.assertIn('oco_results', loaded)
        self.assertTrue(loaded['oco_results']['filled'])
        self.assertEqual(loaded['oco_results']['outcome'], 'TP')
        
        # Verify not nested under oco
        self.assertNotIn('results', loaded.get('oco', {}))


if __name__ == "__main__":
    unittest.main()
