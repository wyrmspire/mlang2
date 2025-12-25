"""
Tests for Declarative Strategy Specification

Ensures:
1. StrategySpec can be created and validated
2. Serialization/deserialization works
3. Fingerprinting is deterministic
4. Convenience functions work correctly
"""

import sys
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.strategy.spec import (
    StrategySpec, TriggerConfig, BracketConfig, SizingConfig, FilterConfig,
    TriggerType, BracketType, SizingMethod,
    create_ema_cross_strategy, create_ifvg_strategy, create_model_strategy
)


class TestTriggerConfig(unittest.TestCase):
    """Test TriggerConfig."""
    
    def test_create_trigger(self):
        """Should create trigger config."""
        trigger = TriggerConfig(
            type=TriggerType.EMA_CROSS,
            params={'fast': 9, 'slow': 21}
        )
        
        self.assertEqual(trigger.type, TriggerType.EMA_CROSS)
        self.assertEqual(trigger.params['fast'], 9)
    
    def test_trigger_to_dict(self):
        """Should serialize to dict."""
        trigger = TriggerConfig(
            type=TriggerType.RSI_THRESHOLD,
            params={'overbought': 70, 'oversold': 30},
            filters=['session:rth']
        )
        
        d = trigger.to_dict()
        self.assertEqual(d['type'], 'rsi_threshold')
        self.assertEqual(d['params']['overbought'], 70)
        self.assertIn('session:rth', d['filters'])
    
    def test_trigger_from_dict(self):
        """Should deserialize from dict."""
        data = {
            'type': 'ema_cross',
            'params': {'fast': 12, 'slow': 26},
            'model_id': None,
            'filters': []
        }
        
        trigger = TriggerConfig.from_dict(data)
        self.assertEqual(trigger.type, TriggerType.EMA_CROSS)
        self.assertEqual(trigger.params['fast'], 12)


class TestBracketConfig(unittest.TestCase):
    """Test BracketConfig."""
    
    def test_create_atr_bracket(self):
        """Should create ATR bracket."""
        bracket = BracketConfig(
            type=BracketType.ATR,
            stop_atr=2.0,
            tp_atr=3.0
        )
        
        self.assertEqual(bracket.type, BracketType.ATR)
        self.assertEqual(bracket.stop_atr, 2.0)
    
    def test_bracket_to_dict(self):
        """Should serialize bracket."""
        bracket = BracketConfig(
            type=BracketType.PERCENT,
            stop_percent=0.01,
            tp_percent=0.015
        )
        
        d = bracket.to_dict()
        self.assertEqual(d['type'], 'percent')
        self.assertEqual(d['stop_percent'], 0.01)
    
    def test_bracket_from_dict(self):
        """Should deserialize bracket."""
        data = {
            'type': 'atr',
            'stop_atr': 1.5,
            'tp_atr': 2.5,
            'entry_type': 'LIMIT',
            'max_bars': 150
        }
        
        bracket = BracketConfig.from_dict(data)
        self.assertEqual(bracket.type, BracketType.ATR)
        self.assertEqual(bracket.stop_atr, 1.5)
        self.assertEqual(bracket.max_bars, 150)


class TestSizingConfig(unittest.TestCase):
    """Test SizingConfig."""
    
    def test_create_fixed_contracts(self):
        """Should create fixed contracts sizing."""
        sizing = SizingConfig(
            method=SizingMethod.FIXED_CONTRACTS,
            contracts=2
        )
        
        self.assertEqual(sizing.method, SizingMethod.FIXED_CONTRACTS)
        self.assertEqual(sizing.contracts, 2)
    
    def test_create_fixed_risk(self):
        """Should create fixed risk sizing."""
        sizing = SizingConfig(
            method=SizingMethod.FIXED_RISK,
            risk_percent=0.02,
            max_contracts=5
        )
        
        self.assertEqual(sizing.method, SizingMethod.FIXED_RISK)
        self.assertEqual(sizing.risk_percent, 0.02)
    
    def test_sizing_to_dict(self):
        """Should serialize sizing."""
        sizing = SizingConfig(
            method=SizingMethod.KELLY,
            max_contracts=10
        )
        
        d = sizing.to_dict()
        self.assertEqual(d['method'], 'kelly')
    
    def test_sizing_from_dict(self):
        """Should deserialize sizing."""
        data = {
            'method': 'fixed_risk',
            'risk_percent': 0.01,
            'max_contracts': 3
        }
        
        sizing = SizingConfig.from_dict(data)
        self.assertEqual(sizing.method, SizingMethod.FIXED_RISK)
        self.assertEqual(sizing.risk_percent, 0.01)


class TestStrategySpec(unittest.TestCase):
    """Test StrategySpec."""
    
    def test_create_strategy_spec(self):
        """Should create complete strategy spec."""
        spec = StrategySpec(
            strategy_id="test_strategy",
            trigger=TriggerConfig(
                type=TriggerType.EMA_CROSS,
                params={'fast': 9, 'slow': 21}
            ),
            bracket=BracketConfig(
                type=BracketType.ATR,
                stop_atr=2.0,
                tp_atr=3.0
            ),
            sizing=SizingConfig(
                method=SizingMethod.FIXED_RISK,
                risk_percent=0.02
            )
        )
        
        self.assertEqual(spec.strategy_id, "test_strategy")
        self.assertEqual(spec.trigger.type, TriggerType.EMA_CROSS)
        self.assertEqual(spec.bracket.type, BracketType.ATR)
    
    def test_strategy_to_dict(self):
        """Should serialize to dict."""
        spec = StrategySpec(
            strategy_id="ema_9_21",
            trigger=TriggerConfig(type=TriggerType.EMA_CROSS, params={'fast': 9, 'slow': 21}),
            bracket=BracketConfig(type=BracketType.ATR, stop_atr=2.0, tp_atr=3.0),
            sizing=SizingConfig(method=SizingMethod.FIXED_RISK, risk_percent=0.02),
            name="EMA Cross 9/21",
            tags=['trend', 'ema']
        )
        
        d = spec.to_dict()
        self.assertEqual(d['strategy_id'], 'ema_9_21')
        self.assertEqual(d['trigger']['type'], 'ema_cross')
        self.assertIn('trend', d['tags'])
    
    def test_strategy_from_dict(self):
        """Should deserialize from dict."""
        data = {
            'strategy_id': 'ifvg_test',
            'trigger': {
                'type': 'ifvg',
                'params': {},
                'model_id': None,
                'filters': []
            },
            'bracket': {
                'type': 'atr',
                'stop_atr': 1.5,
                'tp_atr': 2.5,
                'entry_type': 'LIMIT',
                'max_bars': 200
            },
            'sizing': {
                'method': 'fixed_contracts',
                'contracts': 1
            },
            'filters': [],
            'indicators': ['ema', 'atr'],
            'version': '1.0'
        }
        
        spec = StrategySpec.from_dict(data)
        self.assertEqual(spec.strategy_id, 'ifvg_test')
        self.assertEqual(spec.trigger.type, TriggerType.IFVG)
        self.assertIn('ema', spec.indicators)
    
    def test_strategy_to_json(self):
        """Should serialize to JSON string."""
        spec = StrategySpec(
            strategy_id="test",
            trigger=TriggerConfig(type=TriggerType.TIME, params={'hour': 9, 'minute': 30}),
            bracket=BracketConfig(type=BracketType.FIXED, stop_points=10, tp_points=15),
            sizing=SizingConfig(method=SizingMethod.FIXED_CONTRACTS, contracts=1)
        )
        
        json_str = spec.to_json()
        self.assertIsInstance(json_str, str)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        self.assertEqual(parsed['strategy_id'], 'test')
    
    def test_strategy_from_json(self):
        """Should deserialize from JSON string."""
        json_str = '''
        {
            "strategy_id": "json_test",
            "trigger": {
                "type": "ema_cross",
                "params": {"fast": 12, "slow": 26},
                "model_id": null,
                "filters": []
            },
            "bracket": {
                "type": "atr",
                "stop_atr": 2.0,
                "tp_atr": 3.0
            },
            "sizing": {
                "method": "fixed_risk",
                "risk_percent": 0.02
            },
            "filters": [],
            "indicators": []
        }
        '''
        
        spec = StrategySpec.from_json(json_str)
        self.assertEqual(spec.strategy_id, 'json_test')
        self.assertEqual(spec.trigger.params['fast'], 12)
    
    def test_strategy_fingerprint(self):
        """Should generate deterministic fingerprint."""
        spec1 = StrategySpec(
            strategy_id="fp_test",
            trigger=TriggerConfig(type=TriggerType.EMA_CROSS, params={'fast': 9, 'slow': 21}),
            bracket=BracketConfig(type=BracketType.ATR, stop_atr=2.0, tp_atr=3.0),
            sizing=SizingConfig(method=SizingMethod.FIXED_RISK, risk_percent=0.02)
        )
        
        spec2 = StrategySpec(
            strategy_id="fp_test",
            trigger=TriggerConfig(type=TriggerType.EMA_CROSS, params={'fast': 9, 'slow': 21}),
            bracket=BracketConfig(type=BracketType.ATR, stop_atr=2.0, tp_atr=3.0),
            sizing=SizingConfig(method=SizingMethod.FIXED_RISK, risk_percent=0.02)
        )
        
        # Same spec should produce same fingerprint
        fp1 = spec1.fingerprint()
        fp2 = spec2.fingerprint()
        self.assertEqual(fp1, fp2)
        
        # Fingerprint should be hex string
        self.assertEqual(len(fp1), 16)
        self.assertTrue(all(c in '0123456789abcdef' for c in fp1))
    
    def test_strategy_fingerprint_different(self):
        """Different specs should have different fingerprints."""
        spec1 = StrategySpec(
            strategy_id="test1",
            trigger=TriggerConfig(type=TriggerType.EMA_CROSS, params={'fast': 9, 'slow': 21}),
            bracket=BracketConfig(type=BracketType.ATR, stop_atr=2.0, tp_atr=3.0),
            sizing=SizingConfig(method=SizingMethod.FIXED_RISK, risk_percent=0.02)
        )
        
        spec2 = StrategySpec(
            strategy_id="test2",  # Different ID
            trigger=TriggerConfig(type=TriggerType.EMA_CROSS, params={'fast': 9, 'slow': 21}),
            bracket=BracketConfig(type=BracketType.ATR, stop_atr=2.0, tp_atr=3.0),
            sizing=SizingConfig(method=SizingMethod.FIXED_RISK, risk_percent=0.02)
        )
        
        self.assertNotEqual(spec1.fingerprint(), spec2.fingerprint())
    
    def test_strategy_validation(self):
        """Should validate strategy spec."""
        # Valid spec
        spec = StrategySpec(
            strategy_id="valid",
            trigger=TriggerConfig(type=TriggerType.EMA_CROSS, params={'fast': 9, 'slow': 21}),
            bracket=BracketConfig(type=BracketType.ATR, stop_atr=2.0, tp_atr=3.0),
            sizing=SizingConfig(method=SizingMethod.FIXED_RISK, risk_percent=0.02)
        )
        
        errors = spec.validate()
        self.assertEqual(len(errors), 0)
    
    def test_strategy_validation_errors(self):
        """Should detect validation errors."""
        # Missing stop_atr for ATR bracket
        spec = StrategySpec(
            strategy_id="invalid",
            trigger=TriggerConfig(type=TriggerType.EMA_CROSS, params={}),
            bracket=BracketConfig(type=BracketType.ATR),  # Missing stop/tp
            sizing=SizingConfig(method=SizingMethod.FIXED_CONTRACTS)  # Missing contracts
        )
        
        errors = spec.validate()
        self.assertTrue(len(errors) > 0)
        # Should have errors about missing ATR values and missing contracts
        self.assertTrue(any('atr' in e.lower() for e in errors))
        self.assertTrue(any('contracts' in e.lower() for e in errors))


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def test_create_ema_cross_strategy(self):
        """Should create EMA cross strategy."""
        spec = create_ema_cross_strategy(fast=9, slow=21)
        
        self.assertEqual(spec.strategy_id, 'ema_cross_9_21')
        self.assertEqual(spec.trigger.type, TriggerType.EMA_CROSS)
        self.assertEqual(spec.trigger.params['fast'], 9)
        self.assertEqual(spec.bracket.type, BracketType.ATR)
        
        # Should be valid
        errors = spec.validate()
        self.assertEqual(len(errors), 0)
    
    def test_create_ifvg_strategy(self):
        """Should create IFVG strategy."""
        spec = create_ifvg_strategy(stop_atr=1.5, tp_atr=2.5)
        
        self.assertEqual(spec.strategy_id, 'ifvg')
        self.assertEqual(spec.trigger.type, TriggerType.IFVG)
        self.assertEqual(spec.bracket.stop_atr, 1.5)
        
        # Should be valid
        errors = spec.validate()
        self.assertEqual(len(errors), 0)
    
    def test_create_model_strategy(self):
        """Should create ML model strategy."""
        spec = create_model_strategy(model_id='fusion_cnn')
        
        self.assertEqual(spec.strategy_id, 'model_fusion_cnn')
        self.assertEqual(spec.trigger.type, TriggerType.MODEL)
        self.assertEqual(spec.trigger.model_id, 'fusion_cnn')
        
        # Should be valid
        errors = spec.validate()
        self.assertEqual(len(errors), 0)


class TestIndicatorDeclaration(unittest.TestCase):
    """Test indicator_ids declaration in StrategySpec."""
    
    def test_strategy_with_indicators(self):
        """StrategySpec should declare indicator_ids."""
        spec = StrategySpec(
            strategy_id="test_with_indicators",
            trigger=TriggerConfig(type=TriggerType.EMA_CROSS, params={"fast": 9, "slow": 21}),
            bracket=BracketConfig(type=BracketType.ATR, stop_atr=2.0, tp_atr=3.0),
            sizing=SizingConfig(method=SizingMethod.FIXED_CONTRACTS, contracts=1),
            indicators=['ema_9', 'ema_21', 'atr_14', 'rsi_14']
        )
        
        # Verify indicators are declared
        self.assertEqual(len(spec.indicators), 4)
        self.assertIn('ema_9', spec.indicators)
        self.assertIn('ema_21', spec.indicators)
        self.assertIn('atr_14', spec.indicators)
        self.assertIn('rsi_14', spec.indicators)
        
        # Verify they serialize
        spec_dict = spec.to_dict()
        self.assertIn('indicators', spec_dict)
        self.assertEqual(spec_dict['indicators'], ['ema_9', 'ema_21', 'atr_14', 'rsi_14'])
        
        # Verify they deserialize
        restored = StrategySpec.from_dict(spec_dict)
        self.assertEqual(restored.indicators, spec.indicators)
    
    def test_strategy_without_indicators(self):
        """StrategySpec without indicators should have empty list."""
        spec = StrategySpec(
            strategy_id="test_no_indicators",
            trigger=TriggerConfig(type=TriggerType.IFVG),
            bracket=BracketConfig(type=BracketType.ATR, stop_atr=2.0, tp_atr=3.0),
            sizing=SizingConfig(method=SizingMethod.FIXED_CONTRACTS, contracts=1)
        )
        
        # Should have empty indicators list
        self.assertEqual(spec.indicators, [])
        
        # Should serialize with empty list
        spec_dict = spec.to_dict()
        self.assertEqual(spec_dict['indicators'], [])


if __name__ == "__main__":
    unittest.main()
