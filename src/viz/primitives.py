"""
Visualization Primitives - Generic Drawing Interface

Solves the hardcoded visualization problem by:
1. Creating a generic DrawingPrimitive interface
2. Supporting Lines, Boxes, Text, Icons as JSON events
3. Allowing backend to emit any visualization without frontend changes

This enables agents to "draw" on charts programmatically.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Literal
from enum import Enum


class PrimitiveType(Enum):
    """Types of drawing primitives."""
    LINE = "line"
    BOX = "box"
    TEXT = "text"
    ICON = "icon"
    MARKER = "marker"
    POLYGON = "polygon"


@dataclass
class Color:
    """RGB color specification."""
    r: int  # 0-255
    g: int  # 0-255
    b: int  # 0-255
    a: float = 1.0  # Alpha: 0.0-1.0
    
    def to_hex(self) -> str:
        """Convert to hex color string."""
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"
    
    def to_rgba(self) -> str:
        """Convert to rgba() string."""
        return f"rgba({self.r}, {self.g}, {self.b}, {self.a})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {'r': self.r, 'g': self.g, 'b': self.b, 'a': self.a}
    
    @classmethod
    def from_hex(cls, hex_color: str) -> 'Color':
        """Create color from hex string (e.g., '#FF0000')."""
        hex_color = hex_color.lstrip('#')
        return cls(
            r=int(hex_color[0:2], 16),
            g=int(hex_color[2:4], 16),
            b=int(hex_color[4:6], 16),
        )


# Predefined colors
COLORS = {
    'red': Color(255, 0, 0),
    'green': Color(0, 255, 0),
    'blue': Color(0, 0, 255),
    'yellow': Color(255, 255, 0),
    'orange': Color(255, 165, 0),
    'purple': Color(128, 0, 128),
    'cyan': Color(0, 255, 255),
    'magenta': Color(255, 0, 255),
    'white': Color(255, 255, 255),
    'black': Color(0, 0, 0),
    'gray': Color(128, 128, 128),
}


@dataclass
class Point:
    """Point in time-price space."""
    time: str  # ISO timestamp or bar index
    price: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {'time': self.time, 'price': self.price}


@dataclass
class DrawingPrimitive:
    """
    Base drawing primitive.
    
    All visualizations inherit from this.
    """
    primitive_type: PrimitiveType
    id: str  # Unique identifier
    layer: int = 0  # Drawing layer (higher = on top)
    visible: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            'type': self.primitive_type.value,
            'id': self.id,
            'layer': self.layer,
            'visible': self.visible,
        }
        return data


@dataclass
class LinePrimitive(DrawingPrimitive):
    """
    Line between two points.
    
    Example: Trend line, support/resistance level.
    """
    start: Point = field(default_factory=lambda: Point("", 0.0))
    end: Point = field(default_factory=lambda: Point("", 0.0))
    color: Color = field(default_factory=lambda: COLORS['blue'])
    width: int = 2
    style: Literal['solid', 'dashed', 'dotted'] = 'solid'
    
    def __post_init__(self):
        self.primitive_type = PrimitiveType.LINE
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'start': self.start.to_dict(),
            'end': self.end.to_dict(),
            'color': self.color.to_rgba(),
            'width': self.width,
            'style': self.style,
        })
        return data


@dataclass
class BoxPrimitive(DrawingPrimitive):
    """
    Rectangle box.
    
    Example: OCO bracket zones, FVG boxes, time ranges.
    """
    top_left: Point = field(default_factory=lambda: Point("", 0.0))
    bottom_right: Point = field(default_factory=lambda: Point("", 0.0))
    fill_color: Optional[Color] = None
    border_color: Optional[Color] = field(default_factory=lambda: COLORS['blue'])
    border_width: int = 1
    opacity: float = 0.3
    
    def __post_init__(self):
        self.primitive_type = PrimitiveType.BOX
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'top_left': self.top_left.to_dict(),
            'bottom_right': self.bottom_right.to_dict(),
            'fill_color': self.fill_color.to_rgba() if self.fill_color else None,
            'border_color': self.border_color.to_rgba() if self.border_color else None,
            'border_width': self.border_width,
            'opacity': self.opacity,
        })
        return data


@dataclass
class TextPrimitive(DrawingPrimitive):
    """
    Text label.
    
    Example: Trade annotations, indicator values, alerts.
    """
    position: Point = field(default_factory=lambda: Point("", 0.0))
    text: str = ""
    color: Color = field(default_factory=lambda: COLORS['white'])
    font_size: int = 12
    background_color: Optional[Color] = None
    align: Literal['left', 'center', 'right'] = 'left'
    
    def __post_init__(self):
        self.primitive_type = PrimitiveType.TEXT
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'position': self.position.to_dict(),
            'text': self.text,
            'color': self.color.to_rgba(),
            'font_size': self.font_size,
            'background_color': self.background_color.to_rgba() if self.background_color else None,
            'align': self.align,
        })
        return data


@dataclass
class IconPrimitive(DrawingPrimitive):
    """
    Icon marker.
    
    Example: Entry/exit arrows, alerts, warnings.
    """
    position: Point = field(default_factory=lambda: Point("", 0.0))
    icon_type: str = "arrow"  # 'arrow', 'dot', 'triangle', 'star', 'warning', etc.
    color: Color = field(default_factory=lambda: COLORS['green'])
    size: int = 16
    rotation: int = 0  # Degrees (0 = up, 90 = right, 180 = down, 270 = left)
    
    def __post_init__(self):
        self.primitive_type = PrimitiveType.ICON
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'position': self.position.to_dict(),
            'icon_type': self.icon_type,
            'color': self.color.to_rgba(),
            'size': self.size,
            'rotation': self.rotation,
        })
        return data


@dataclass
class MarkerPrimitive(DrawingPrimitive):
    """
    Price marker (horizontal line at price level).
    
    Example: Stop loss, take profit, entry price.
    """
    price: float = 0.0
    time_start: Optional[str] = None  # If None, extends across entire chart
    time_end: Optional[str] = None
    label: str = ""
    color: Color = field(default_factory=lambda: COLORS['red'])
    width: int = 1
    style: Literal['solid', 'dashed', 'dotted'] = 'dashed'
    
    def __post_init__(self):
        self.primitive_type = PrimitiveType.MARKER
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'price': self.price,
            'time_start': self.time_start,
            'time_end': self.time_end,
            'label': self.label,
            'color': self.color.to_rgba(),
            'width': self.width,
            'style': self.style,
        })
        return data


@dataclass
class PolygonPrimitive(DrawingPrimitive):
    """
    Arbitrary polygon shape.
    
    Example: Complex zones, pattern highlighting.
    """
    points: List[Point] = field(default_factory=list)
    fill_color: Optional[Color] = None
    border_color: Optional[Color] = field(default_factory=lambda: COLORS['blue'])
    border_width: int = 1
    opacity: float = 0.3
    
    def __post_init__(self):
        self.primitive_type = PrimitiveType.POLYGON
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'points': [p.to_dict() for p in self.points],
            'fill_color': self.fill_color.to_rgba() if self.fill_color else None,
            'border_color': self.border_color.to_rgba() if self.border_color else None,
            'border_width': self.border_width,
            'opacity': self.opacity,
        })
        return data


# =============================================================================
# Visualization Manager
# =============================================================================

class VisualizationManager:
    """
    Manages a collection of drawing primitives.
    
    Usage:
        viz = VisualizationManager()
        
        # Add primitives
        viz.add_line("trend", Point("2025-01-01", 100), Point("2025-01-10", 110))
        viz.add_box("fvg_zone", Point("2025-01-05", 105), Point("2025-01-06", 108))
        viz.add_text("entry", Point("2025-01-05", 105), "Entry Signal")
        
        # Export to JSON
        json_data = viz.to_json()
        # Send to frontend...
    """
    
    def __init__(self):
        """Initialize visualization manager."""
        self.primitives: Dict[str, DrawingPrimitive] = {}
    
    def add_primitive(self, primitive: DrawingPrimitive):
        """Add a drawing primitive."""
        self.primitives[primitive.id] = primitive
    
    def add_line(
        self,
        id: str,
        start: Point,
        end: Point,
        color: Color = COLORS['blue'],
        width: int = 2,
        style: str = 'solid',
        layer: int = 0
    ):
        """Convenience method to add a line."""
        line = LinePrimitive(
            primitive_type=PrimitiveType.LINE,
            id=id,
            start=start,
            end=end,
            color=color,
            width=width,
            style=style,
            layer=layer,
        )
        self.add_primitive(line)
    
    def add_box(
        self,
        id: str,
        top_left: Point,
        bottom_right: Point,
        fill_color: Optional[Color] = None,
        border_color: Color = COLORS['blue'],
        opacity: float = 0.3,
        layer: int = 0
    ):
        """Convenience method to add a box."""
        box = BoxPrimitive(
            primitive_type=PrimitiveType.BOX,
            id=id,
            top_left=top_left,
            bottom_right=bottom_right,
            fill_color=fill_color,
            border_color=border_color,
            opacity=opacity,
            layer=layer,
        )
        self.add_primitive(box)
    
    def add_text(
        self,
        id: str,
        position: Point,
        text: str,
        color: Color = COLORS['white'],
        font_size: int = 12,
        layer: int = 1
    ):
        """Convenience method to add text."""
        text_prim = TextPrimitive(
            primitive_type=PrimitiveType.TEXT,
            id=id,
            position=position,
            text=text,
            color=color,
            font_size=font_size,
            layer=layer,
        )
        self.add_primitive(text_prim)
    
    def add_marker(
        self,
        id: str,
        price: float,
        label: str = "",
        color: Color = COLORS['red'],
        style: str = 'dashed',
        layer: int = 0
    ):
        """Convenience method to add a price marker."""
        marker = MarkerPrimitive(
            primitive_type=PrimitiveType.MARKER,
            id=id,
            price=price,
            label=label,
            color=color,
            style=style,
            layer=layer,
        )
        self.add_primitive(marker)
    
    def remove_primitive(self, id: str) -> bool:
        """Remove a primitive by ID."""
        if id in self.primitives:
            del self.primitives[id]
            return True
        return False
    
    def clear(self):
        """Remove all primitives."""
        self.primitives.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export all primitives to dictionary."""
        return {
            'primitives': [p.to_dict() for p in self.primitives.values()]
        }
    
    def to_json(self) -> str:
        """Export all primitives to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)
