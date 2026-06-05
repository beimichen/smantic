"""
Input adapters.

The chunker consumes a :class:`smantic.ir.Document`. These helpers build one
from the formats people actually have on hand:

  * ``from_markdown(text)``       — raw Markdown / VLM page text (headings, code
                                    fences, pipe tables, ``$$``/``\\[`` math,
                                    lists, images, prose).
  * ``from_nopaddle(doc)``        — a NoPaddle ``ParsedDocument`` (object, dict,
                                    or JSON string). NoPaddle pages use a
                                    ``regions`` key; the IR accepts it directly.
  * ``from_docling_json(text)``   — Docling-style ``{"pages": [...]}`` JSON.
  * ``from_dict`` / ``from_json`` — pass-throughs for the IR's own shape.

Everything funnels into :class:`smantic.ir.Document`, so the chunker never has
to know where the document came from.
"""

import json
from typing import Any

from .ir import Document


def from_markdown(text: str, *, page: int = 1) -> Document:
    """Parse Markdown text into a single-page :class:`Document`."""
    from .ir import Page
    from .markdown import parse_markdown

    elements = parse_markdown(text, page=page)
    return Document(
        pages=[Page(page_number=page, elements=elements)],
        metadata={"source": "markdown"},
    )


def from_dict(data: dict) -> Document:
    """Build a :class:`Document` from its dict form (accepts the NoPaddle shape too)."""
    return Document.from_dict(data)


def from_json(text: str) -> Document:
    """Build a :class:`Document` from a JSON string (IR / Docling / NoPaddle shape)."""
    return Document.from_dict(json.loads(text))


def from_docling_json(text: str) -> Document:
    """Alias of :func:`from_json` for Docling-style ``{"pages": [...]}`` JSON."""
    return from_json(text)


def from_nopaddle(doc: Any | dict | str) -> Document:
    """Build a :class:`Document` from a NoPaddle ``ParsedDocument``.

    Accepts a ``ParsedDocument`` object (anything with ``.to_dict()``), its dict
    form, or a JSON string. NoPaddle pages carry a ``regions`` list, which the
    IR's ``Page.from_dict`` reads natively, so no key surgery is needed.
    """
    if isinstance(doc, str):
        data = json.loads(doc)
    elif hasattr(doc, "to_dict"):
        data = doc.to_dict()
    else:
        data = doc
    return Document.from_dict(data)


__all__ = [
    "from_markdown",
    "from_dict",
    "from_json",
    "from_docling_json",
    "from_nopaddle",
]
