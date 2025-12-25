"""
Unified Tool Registry

A single, consistent registry system for all executable components in MLang2:
scanners, models, indicators, skills, and agent tools.

This replaces the fragmented system of:
- ScannerRegistry, ModelRegistry, IndicatorRegistry (src/core/registries.py)
- SkillRegistry (src/skills/registry.py)
- Hardcoded AGENT_TOOLS (src/server/main.py)

Usage:
    from src/core.tool_registry import ToolRegistry, ToolCategory
    
    @ToolRegistry.register(
        tool_id="ema_cross_scanner",
        category=ToolCategory.SCANNER,
        name="EMA Cross Scanner",
        description="Detects EMA crossovers",
        input_schema={...},
        output_schema={...}
    )
    class EMACrossScanner:
        def __init__(self, fast=12, slow=26):
            ...
        
        def execute(self, **inputs):
            # Tool execution
            ...
"""

from typing import Dict, Any, List, Callable, Optional, Protocol
from dataclasses import dataclass, field
from enum import Enum
import inspect
import json


class ToolCategory(Enum):
    """Categories for tools - replaces separate registry systems."""
    SCANNER = "scanner"        # Pattern/signal detection (was ScannerRegistry)
    MODEL = "model"            # ML model inference (was ModelRegistry)
    INDICATOR = "indicator"    # Technical indicators (was IndicatorRegistry)
    SKILL = "skill"            # Agent capabilities (was SkillRegistry)
    STRATEGY = "strategy"      # Strategy executors
    EXPORTER = "exporter"      # Viz/data exporters
    UTILITY = "utility"        # Helper tools


class ToolProtocol(Protocol):
    """Protocol that all tools must implement."""
    
    def execute(self, **inputs) -> Dict[str, Any]:
        """Execute the tool with given inputs and return outputs."""
        ...


@dataclass
class ToolInfo:
    """
    Metadata about a registered tool.
    
    This is the unified info structure that replaces:
    - ScannerInfo
    - ModelInfo
    - IndicatorInfo
    - SkillRegistry's dict structure
    """
    tool_id: str
    category: ToolCategory
    name: str
    description: str
    
    # JSON schemas for validation and agent tool generation
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    
    # Optional metadata
    version: str = "1.0"
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # For deterministic artifact outputs
    produces_artifacts: bool = False
    artifact_spec: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'tool_id': self.tool_id,
            'category': self.category.value,
            'name': self.name,
            'description': self.description,
            'input_schema': self.input_schema,
            'output_schema': self.output_schema,
            'version': self.version,
            'author': self.author,
            'tags': self.tags,
            'produces_artifacts': self.produces_artifacts,
            'artifact_spec': self.artifact_spec,
        }
    
    def to_gemini_function_declaration(self) -> Dict[str, Any]:
        """
        Convert to Gemini function calling format.
        Replaces hardcoded AGENT_TOOLS definitions.
        """
        return {
            'name': self.tool_id,
            'description': self.description,
            'parameters': self.input_schema if self.input_schema else {
                'type': 'object',
                'properties': {}
            }
        }


class ToolRegistry:
    """
    Unified registry for all tools in the system.
    
    Replaces:
    - ScannerRegistry
    - ModelRegistry  
    - IndicatorRegistry
    - SkillRegistry
    - Hardcoded AGENT_TOOLS
    
    Benefits:
    - Single contract for all executable components
    - Auto-generate agent tool schemas
    - Consistent validation
    - Deterministic artifact outputs
    - Category labels instead of separate mechanisms
    """
    
    _registry: Dict[str, Callable] = {}
    _info: Dict[str, ToolInfo] = {}
    
    @classmethod
    def register(
        cls,
        tool_id: str,
        category: ToolCategory,
        name: str,
        description: str = "",
        input_schema: Dict[str, Any] = None,
        output_schema: Dict[str, Any] = None,
        version: str = "1.0",
        author: str = None,
        tags: List[str] = None,
        produces_artifacts: bool = False,
        artifact_spec: Dict[str, Any] = None,
    ):
        """
        Decorator to register a tool.
        
        Args:
            tool_id: Unique identifier (e.g., "ema_cross_scanner")
            category: ToolCategory enum value
            name: Human-readable name
            description: What the tool does
            input_schema: JSON schema for inputs
            output_schema: JSON schema for outputs
            version: Semantic version
            author: Optional author
            tags: Optional tags for filtering
            produces_artifacts: Whether tool creates files
            artifact_spec: Specification of produced artifacts
        
        Returns:
            Decorator function
        """
        def decorator(tool_class):
            cls._registry[tool_id] = tool_class
            cls._info[tool_id] = ToolInfo(
                tool_id=tool_id,
                category=category,
                name=name,
                description=description,
                input_schema=input_schema or {},
                output_schema=output_schema or {},
                version=version,
                author=author,
                tags=tags or [],
                produces_artifacts=produces_artifacts,
                artifact_spec=artifact_spec,
            )
            return tool_class
        return decorator
    
    @classmethod
    def create(cls, tool_id: str, **params) -> ToolProtocol:
        """
        Create a tool instance by ID.
        
        Args:
            tool_id: Tool identifier
            **params: Parameters to pass to tool constructor
        
        Returns:
            Tool instance
        """
        if tool_id not in cls._registry:
            raise ValueError(f"Unknown tool: {tool_id}")
        return cls._registry[tool_id](**params)
    
    @classmethod
    def get_info(cls, tool_id: str) -> ToolInfo:
        """Get metadata for a specific tool."""
        if tool_id not in cls._info:
            raise ValueError(f"Unknown tool: {tool_id}")
        return cls._info[tool_id]
    
    @classmethod
    def list_all(cls, category: ToolCategory = None) -> List[ToolInfo]:
        """
        List all registered tools, optionally filtered by category.
        
        Args:
            category: Optional category filter
        
        Returns:
            List of ToolInfo objects
        """
        if category is None:
            return list(cls._info.values())
        return [info for info in cls._info.values() if info.category == category]
    
    @classmethod
    def list_by_tag(cls, tag: str) -> List[ToolInfo]:
        """List tools with a specific tag."""
        return [info for info in cls._info.values() if tag in info.tags]
    
    @classmethod
    def get_gemini_function_declarations(
        cls, 
        categories: List[ToolCategory] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate Gemini function declarations from registered tools.
        
        This replaces hardcoded AGENT_TOOLS and LAB_TOOLS.
        
        Args:
            categories: Optional list of categories to include
        
        Returns:
            List of Gemini function declarations
        """
        tools_to_export = cls._info.values()
        
        if categories:
            tools_to_export = [
                info for info in tools_to_export 
                if info.category in categories
            ]
        
        return [info.to_gemini_function_declaration() for info in tools_to_export]
    
    @classmethod
    def validate_tool_output(cls, tool_id: str, output: Dict[str, Any]) -> List[str]:
        """
        Validate tool output against its schema.
        
        Args:
            tool_id: Tool identifier
            output: Output to validate
        
        Returns:
            List of validation errors (empty if valid)
        """
        info = cls.get_info(tool_id)
        
        # TODO: Implement JSON schema validation
        # For now, just check if output is a dict
        if not isinstance(output, dict):
            return [f"Tool output must be a dictionary, got {type(output)}"]
        
        return []
    
    @classmethod
    def export_catalog(cls, output_path: str = None) -> Dict[str, Any]:
        """
        Export complete tool catalog as JSON.
        
        Used for:
        - API endpoint /tools/catalog
        - Documentation generation
        - Agent discovery
        
        Args:
            output_path: Optional path to write JSON file
        
        Returns:
            Catalog dictionary
        """
        catalog = {
            'version': '1.0',
            'total_tools': len(cls._info),
            'categories': {},
            'tools': []
        }
        
        # Count by category
        for category in ToolCategory:
            tools_in_category = cls.list_all(category)
            catalog['categories'][category.value] = len(tools_in_category)
        
        # Add all tools
        for info in cls._info.values():
            catalog['tools'].append(info.to_dict())
        
        # Write to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(catalog, f, indent=2)
        
        return catalog


# =============================================================================
# Backward Compatibility Adapters
# =============================================================================

class ScannerRegistryAdapter:
    """
    Adapter to make ToolRegistry work like old ScannerRegistry.
    Allows gradual migration.
    """
    
    @classmethod
    def register(cls, scanner_id: str, name: str, description: str = "", 
                 params_schema: Dict[str, Any] = None):
        """Register a scanner using the new ToolRegistry."""
        return ToolRegistry.register(
            tool_id=scanner_id,
            category=ToolCategory.SCANNER,
            name=name,
            description=description,
            input_schema=params_schema,
        )
    
    @classmethod
    def create(cls, scanner_id: str, **params):
        """Create scanner instance."""
        return ToolRegistry.create(scanner_id, **params)
    
    @classmethod
    def list_all(cls):
        """List all scanners."""
        return ToolRegistry.list_all(ToolCategory.SCANNER)
    
    @classmethod
    def get_info(cls, scanner_id: str):
        """Get scanner info."""
        return ToolRegistry.get_info(scanner_id)


class ModelRegistryAdapter:
    """Adapter for ModelRegistry."""
    
    @classmethod
    def register(cls, model_id: str, name: str, description: str = "",
                 input_schema: Dict[str, Any] = None,
                 output_schema: Dict[str, Any] = None):
        return ToolRegistry.register(
            tool_id=model_id,
            category=ToolCategory.MODEL,
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
        )
    
    @classmethod
    def create(cls, model_id: str, **params):
        return ToolRegistry.create(model_id, **params)
    
    @classmethod
    def list_all(cls):
        return ToolRegistry.list_all(ToolCategory.MODEL)
    
    @classmethod
    def get_info(cls, model_id: str):
        return ToolRegistry.get_info(model_id)


class IndicatorRegistryAdapter:
    """Adapter for IndicatorRegistry."""
    
    @classmethod
    def register(cls, indicator_id: str, name: str, output_type: str = "line",
                 description: str = "", params_schema: Dict[str, Any] = None):
        return ToolRegistry.register(
            tool_id=indicator_id,
            category=ToolCategory.INDICATOR,
            name=name,
            description=description,
            input_schema=params_schema,
            tags=[f"output_type:{output_type}"],
        )
    
    @classmethod
    def create(cls, indicator_id: str, **params):
        return ToolRegistry.create(indicator_id, **params)
    
    @classmethod
    def list_all(cls):
        return ToolRegistry.list_all(ToolCategory.INDICATOR)
    
    @classmethod
    def get_info(cls, indicator_id: str):
        return ToolRegistry.get_info(indicator_id)


class SkillRegistryAdapter:
    """Adapter for SkillRegistry."""
    
    def __init__(self):
        # Compatibility with old SkillRegistry instance creation
        pass
    
    def register(self, name: str, func: Callable, description: str):
        """Register a skill as a tool."""
        # Wrap the function in a simple tool class
        class SkillToolWrapper:
            def __init__(self, skill_func=func):
                self.skill_func = skill_func
            
            def execute(self, **inputs):
                return self.skill_func(**inputs)
        
        # Get function signature for input schema
        sig = inspect.signature(func)
        input_schema = {
            'type': 'object',
            'properties': {
                param: {'type': 'string'}  # Simple default
                for param in sig.parameters.keys()
            }
        }
        
        ToolRegistry.register(
            tool_id=name,
            category=ToolCategory.SKILL,
            name=name,
            description=description,
            input_schema=input_schema,
        )(SkillToolWrapper)
    
    def list_skills(self) -> List[Dict]:
        """List skills in old format."""
        skills = ToolRegistry.list_all(ToolCategory.SKILL)
        return [
            {
                'name': info.tool_id,
                'description': info.description,
                'signature': str(info.input_schema)
            }
            for info in skills
        ]
    
    def get_skill(self, name: str) -> Callable:
        """Get skill function (compatibility)."""
        tool = ToolRegistry.create(name)
        return lambda **kwargs: tool.execute(**kwargs)


# =============================================================================
# Convenience Functions
# =============================================================================

def register_scanner(scanner_id: str, name: str, **kwargs):
    """Convenience function to register a scanner."""
    return ToolRegistry.register(
        tool_id=scanner_id,
        category=ToolCategory.SCANNER,
        name=name,
        **kwargs
    )


def register_model(model_id: str, name: str, **kwargs):
    """Convenience function to register a model."""
    return ToolRegistry.register(
        tool_id=model_id,
        category=ToolCategory.MODEL,
        name=name,
        **kwargs
    )


def register_indicator(indicator_id: str, name: str, **kwargs):
    """Convenience function to register an indicator."""
    return ToolRegistry.register(
        tool_id=indicator_id,
        category=ToolCategory.INDICATOR,
        name=name,
        **kwargs
    )


def register_skill(skill_id: str, name: str, **kwargs):
    """Convenience function to register a skill."""
    return ToolRegistry.register(
        tool_id=skill_id,
        category=ToolCategory.SKILL,
        name=name,
        **kwargs
    )
