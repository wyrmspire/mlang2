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


if __name__ == "__main__":
    unittest.main()
