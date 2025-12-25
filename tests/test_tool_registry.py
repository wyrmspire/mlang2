"""
Tests for Unified Tool Registry

Ensures:
1. ToolRegistry works correctly
2. Backward compatibility adapters function
3. Gemini function declarations are generated
4. Tool catalog export works
"""

import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.tool_registry import (
    ToolRegistry, ToolCategory, ToolInfo,
    ScannerRegistryAdapter, ModelRegistryAdapter,
    IndicatorRegistryAdapter, SkillRegistryAdapter
)


class TestToolRegistry(unittest.TestCase):
    """Test core ToolRegistry functionality."""
    
    def setUp(self):
        """Clear registry before each test."""
        ToolRegistry._registry = {}
        ToolRegistry._info = {}
    
    def test_register_tool(self):
        """Should register a tool with metadata."""
        @ToolRegistry.register(
            tool_id="test_scanner",
            category=ToolCategory.SCANNER,
            name="Test Scanner",
            description="A test scanner"
        )
        class TestScanner:
            def execute(self, **inputs):
                return {'result': True}
        
        # Check registration
        self.assertIn('test_scanner', ToolRegistry._info)
        info = ToolRegistry.get_info('test_scanner')
        self.assertEqual(info.tool_id, 'test_scanner')
        self.assertEqual(info.category, ToolCategory.SCANNER)
        self.assertEqual(info.name, 'Test Scanner')
    
    def test_create_tool_instance(self):
        """Should create tool instances."""
        @ToolRegistry.register(
            tool_id="test_tool",
            category=ToolCategory.UTILITY,
            name="Test Tool"
        )
        class TestTool:
            def __init__(self, param=10):
                self.param = param
            
            def execute(self, **inputs):
                return {'param': self.param}
        
        # Create instance
        tool = ToolRegistry.create('test_tool', param=20)
        result = tool.execute()
        self.assertEqual(result['param'], 20)
    
    def test_list_all_tools(self):
        """Should list all registered tools."""
        @ToolRegistry.register(
            tool_id="scanner1",
            category=ToolCategory.SCANNER,
            name="Scanner 1"
        )
        class Scanner1:
            def execute(self, **inputs):
                return {}
        
        @ToolRegistry.register(
            tool_id="model1",
            category=ToolCategory.MODEL,
            name="Model 1"
        )
        class Model1:
            def execute(self, **inputs):
                return {}
        
        all_tools = ToolRegistry.list_all()
        self.assertEqual(len(all_tools), 2)
        
        scanners = ToolRegistry.list_all(ToolCategory.SCANNER)
        self.assertEqual(len(scanners), 1)
        self.assertEqual(scanners[0].tool_id, 'scanner1')
    
    def test_list_by_tag(self):
        """Should filter tools by tag."""
        @ToolRegistry.register(
            tool_id="tagged_tool",
            category=ToolCategory.INDICATOR,
            name="Tagged Tool",
            tags=['experimental', 'beta']
        )
        class TaggedTool:
            def execute(self, **inputs):
                return {}
        
        @ToolRegistry.register(
            tool_id="stable_tool",
            category=ToolCategory.INDICATOR,
            name="Stable Tool",
            tags=['stable']
        )
        class StableTool:
            def execute(self, **inputs):
                return {}
        
        experimental = ToolRegistry.list_by_tag('experimental')
        self.assertEqual(len(experimental), 1)
        self.assertEqual(experimental[0].tool_id, 'tagged_tool')
    
    def test_gemini_function_declaration(self):
        """Should generate Gemini function declarations."""
        @ToolRegistry.register(
            tool_id="test_func",
            category=ToolCategory.SKILL,
            name="Test Function",
            description="Does something useful",
            input_schema={
                'type': 'object',
                'properties': {
                    'param1': {'type': 'string'},
                    'param2': {'type': 'number'}
                },
                'required': ['param1']
            }
        )
        class TestFunc:
            def execute(self, **inputs):
                return {}
        
        declarations = ToolRegistry.get_gemini_function_declarations()
        self.assertEqual(len(declarations), 1)
        
        decl = declarations[0]
        self.assertEqual(decl['name'], 'test_func')
        self.assertEqual(decl['description'], 'Does something useful')
        self.assertIn('param1', decl['parameters']['properties'])
    
    def test_gemini_declaration_category_filter(self):
        """Should filter Gemini declarations by category."""
        @ToolRegistry.register(
            tool_id="scanner1",
            category=ToolCategory.SCANNER,
            name="Scanner 1"
        )
        class Scanner1:
            def execute(self, **inputs):
                return {}
        
        @ToolRegistry.register(
            tool_id="skill1",
            category=ToolCategory.SKILL,
            name="Skill 1"
        )
        class Skill1:
            def execute(self, **inputs):
                return {}
        
        # Get only skills
        skill_decls = ToolRegistry.get_gemini_function_declarations(
            categories=[ToolCategory.SKILL]
        )
        self.assertEqual(len(skill_decls), 1)
        self.assertEqual(skill_decls[0]['name'], 'skill1')
    
    def test_export_catalog(self):
        """Should export tool catalog as JSON."""
        @ToolRegistry.register(
            tool_id="tool1",
            category=ToolCategory.SCANNER,
            name="Tool 1",
            version="1.2.3"
        )
        class Tool1:
            def execute(self, **inputs):
                return {}
        
        catalog = ToolRegistry.export_catalog()
        
        self.assertEqual(catalog['version'], '1.0')
        self.assertEqual(catalog['total_tools'], 1)
        self.assertIn('scanner', catalog['categories'])
        self.assertEqual(len(catalog['tools']), 1)
        self.assertEqual(catalog['tools'][0]['tool_id'], 'tool1')
        self.assertEqual(catalog['tools'][0]['version'], '1.2.3')
    
    def test_export_catalog_to_file(self):
        """Should write catalog to JSON file."""
        @ToolRegistry.register(
            tool_id="tool1",
            category=ToolCategory.UTILITY,
            name="Tool 1"
        )
        class Tool1:
            def execute(self, **inputs):
                return {}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            catalog = ToolRegistry.export_catalog(temp_path)
            
            # Verify file was written
            with open(temp_path) as f:
                loaded = json.load(f)
            
            self.assertEqual(loaded['total_tools'], 1)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestBackwardCompatibilityAdapters(unittest.TestCase):
    """Test backward compatibility adapters."""
    
    def setUp(self):
        """Clear registry before each test."""
        ToolRegistry._registry = {}
        ToolRegistry._info = {}
    
    def test_scanner_registry_adapter(self):
        """ScannerRegistryAdapter should work like old ScannerRegistry."""
        @ScannerRegistryAdapter.register(
            scanner_id="ema_cross",
            name="EMA Cross",
            description="EMA crossover scanner",
            params_schema={
                'type': 'object',
                'properties': {
                    'fast': {'type': 'number'},
                    'slow': {'type': 'number'}
                }
            }
        )
        class EMACrossScanner:
            def __init__(self, fast=12, slow=26):
                self.fast = fast
                self.slow = slow
            
            def execute(self, **inputs):
                return {'triggered': True}
        
        # Should be registered in ToolRegistry
        info = ToolRegistry.get_info('ema_cross')
        self.assertEqual(info.category, ToolCategory.SCANNER)
        
        # Create via adapter
        scanner = ScannerRegistryAdapter.create('ema_cross', fast=9, slow=21)
        self.assertEqual(scanner.fast, 9)
        
        # List via adapter
        scanners = ScannerRegistryAdapter.list_all()
        self.assertEqual(len(scanners), 1)
    
    def test_model_registry_adapter(self):
        """ModelRegistryAdapter should work like old ModelRegistry."""
        @ModelRegistryAdapter.register(
            model_id="fusion_cnn",
            name="Fusion CNN",
            description="Multi-timeframe CNN"
        )
        class FusionCNN:
            def execute(self, **inputs):
                return {'prediction': [0.1, 0.7, 0.1, 0.1]}
        
        info = ToolRegistry.get_info('fusion_cnn')
        self.assertEqual(info.category, ToolCategory.MODEL)
        
        models = ModelRegistryAdapter.list_all()
        self.assertEqual(len(models), 1)
    
    def test_indicator_registry_adapter(self):
        """IndicatorRegistryAdapter should work like old IndicatorRegistry."""
        @IndicatorRegistryAdapter.register(
            indicator_id="ema",
            name="EMA",
            output_type="line",
            description="Exponential moving average"
        )
        class EMAIndicator:
            def execute(self, **inputs):
                return {'values': [100, 101, 102]}
        
        info = ToolRegistry.get_info('ema')
        self.assertEqual(info.category, ToolCategory.INDICATOR)
        self.assertIn('output_type:line', info.tags)
        
        indicators = IndicatorRegistryAdapter.list_all()
        self.assertEqual(len(indicators), 1)
    
    def test_skill_registry_adapter(self):
        """SkillRegistryAdapter should work like old SkillRegistry."""
        def my_skill(param1, param2=10):
            return {'result': param1 + param2}
        
        skill_registry = SkillRegistryAdapter()
        skill_registry.register('my_skill', my_skill, 'Does something')
        
        # Should be registered as a tool
        info = ToolRegistry.get_info('my_skill')
        self.assertEqual(info.category, ToolCategory.SKILL)
        
        # List skills
        skills = skill_registry.list_skills()
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0]['name'], 'my_skill')
        
        # Get skill function
        func = skill_registry.get_skill('my_skill')
        result = func(param1=5, param2=15)
        self.assertEqual(result['result'], 20)


class TestToolInfo(unittest.TestCase):
    """Test ToolInfo dataclass."""
    
    def test_to_dict(self):
        """ToolInfo should serialize to dict."""
        info = ToolInfo(
            tool_id='test_tool',
            category=ToolCategory.SCANNER,
            name='Test Tool',
            description='A test',
            version='2.0',
            tags=['beta'],
            produces_artifacts=True,
            artifact_spec={'files': ['manifest.json']}
        )
        
        d = info.to_dict()
        self.assertEqual(d['tool_id'], 'test_tool')
        self.assertEqual(d['category'], 'scanner')
        self.assertEqual(d['version'], '2.0')
        self.assertTrue(d['produces_artifacts'])
    
    def test_to_gemini_function_declaration(self):
        """ToolInfo should convert to Gemini format."""
        info = ToolInfo(
            tool_id='run_scan',
            category=ToolCategory.SKILL,
            name='Run Scan',
            description='Execute a strategy scan',
            input_schema={
                'type': 'object',
                'properties': {
                    'strategy': {'type': 'string'},
                    'weeks': {'type': 'integer'}
                },
                'required': ['strategy']
            }
        )
        
        decl = info.to_gemini_function_declaration()
        self.assertEqual(decl['name'], 'run_scan')
        self.assertEqual(decl['description'], 'Execute a strategy scan')
        self.assertIn('strategy', decl['parameters']['properties'])
        self.assertIn('required', decl['parameters'])


if __name__ == "__main__":
    unittest.main()
