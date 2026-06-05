"""
Parsed-document intermediate representation (IR).

Small, dependency-free dataclasses describing a document as a list of pages,
each holding typed elements (text, section_header, table, formula, figure, ...)
with normalized bounding boxes. This is the single data contract the chunker
consumes; the ``adapters`` module builds it from Markdown, NoPaddle output, or
Docling-style JSON.

Coordinates are normalized to a 0-1000 integer scale so they are resolution-
and DPI-independent. ``from_dict`` is permissive: it accepts both ``type`` and
``element_type`` keys, and both ``elements`` and ``regions`` page lists, so
NoPaddle ``ParsedDocument`` JSON drops in unchanged.
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BBox:
    """Normalized bounding box (0-1000 scale)."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __post_init__(self):
        """Validate and fix inverted/out-of-range coordinates (PP9 fix)."""
        # Swap inverted coordinates
        if self.x_min > self.x_max:
            self.x_min, self.x_max = self.x_max, self.x_min
        if self.y_min > self.y_max:
            self.y_min, self.y_max = self.y_max, self.y_min
        # Clamp to valid range
        self.x_min = max(0, min(self.x_min, 1000))
        self.y_min = max(0, min(self.y_min, 1000))
        self.x_max = max(0, min(self.x_max, 1000))
        self.y_max = max(0, min(self.y_max, 1000))

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> 'BBox':
        """Create from a dict, ignoring any extra keys (e.g. a coord-system label)."""
        return cls(
            x_min=data.get('x_min', 0),
            y_min=data.get('y_min', 0),
            x_max=data.get('x_max', 1000),
            y_max=data.get('y_max', 1000),
        )


@dataclass
class Element:
    """Single document element (text, figure, table, etc.)."""
    element_type: str  # text, section_header, formula, table, etc.
    content: str
    bbox: BBox
    page: int
    confidence: float = 1.0
    metadata: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'type': self.element_type,
            'content': self.content,
            'bbox': self.bbox.to_dict(),
            'page': self.page,
            'confidence': self.confidence,
        }
        if self.metadata:
            result['metadata'] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Element':
        """Create from a dict.

        Accepts both ``type`` and ``element_type`` keys. Transcript-segment
        top-level keys (``start_time``, ``end_time``, ``words``) — which some
        producers place at the top level of the element dict — are migrated
        into ``metadata`` so the chunker can read them through the canonical
        ``metadata`` channel and propagate audio timecodes onto chunks.
        Explicit ``metadata`` wins on conflict.
        """
        # Support both 'type' and 'element_type' field names
        element_type = data.get('type') or data.get('element_type', 'text')

        # Support both 'bbox' as dict or missing bbox (use default)
        bbox_data = data.get('bbox')
        if bbox_data:
            bbox = BBox.from_dict(bbox_data)
        else:
            # Default bbox if missing
            bbox = BBox(x_min=0, y_min=0, x_max=1000, y_max=1000)

        # Start with any explicit metadata; then fold in transcript-segment
        # timecodes (and word-level timings) that producers historically
        # placed at the top level. Explicit metadata wins on conflict so
        # callers can override.
        metadata = dict(data.get('metadata') or {})
        for key in ('start_time', 'end_time', 'words'):
            if key in data and key not in metadata:
                metadata[key] = data[key]
        # Don't store an empty dict — keep metadata=None for the
        # "no metadata" case (matches pre-existing serialization output).
        metadata_arg = metadata if metadata else None

        return cls(
            element_type=element_type,
            content=data.get('content', ''),
            bbox=bbox,
            page=data.get('page', 0),
            confidence=data.get('confidence', 1.0),
            metadata=metadata_arg,
        )


@dataclass
class Page:
    """Single document page with elements."""
    page_number: int
    width: int = 1000  # Default normalized width
    height: int = 1000  # Default normalized height
    elements: list[Element] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'page_number': self.page_number,
            'width': self.width,
            'height': self.height,
            'elements': [elem.to_dict() for elem in self.elements]
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Page':
        """Create from a dict. Accepts both ``elements`` and ``regions`` lists."""
        width = data.get('width') or 1000
        height = data.get('height') or 1000
        # NoPaddle pages use 'regions'; Docling-style pages use 'elements'.
        raw = data.get('elements')
        if raw is None:
            raw = data.get('regions', [])
        return cls(
            page_number=data['page_number'],
            width=width,
            height=height,
            elements=[Element.from_dict(elem) for elem in raw],
        )


@dataclass
class Document:
    """A complete parsed document: ordered pages plus document-level metadata."""
    pages: list[Page]
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pages': [page.to_dict() for page in self.pages],
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Document':
        """Create from dictionary."""
        return cls(
            pages=[
                Page.from_dict(page)
                for page in data.get('pages', [])
            ],
            metadata=data.get('metadata') or {},
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Document':
        """Create from JSON string."""
        import json
        return cls.from_dict(json.loads(json_str))