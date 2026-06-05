"""
smantic — structure-aware semantic chunking, minus the heavyweight stack.

Turns a parsed document (or raw Markdown) into retrieval-ready chunks: atomic
blocks (code / tables / formulas / figures) stay intact or split parent->child,
prose runs get genuine semantic boundary detection via ONNX all-MiniLM-L6-v2
embeddings, headings become a searchable trail, and tiny chunks get merged. No
torch, no transformers, no MLX.

Quick start:

    import smantic

    chunks = smantic.chunk_markdown(open("notes.md").read())
    for c in chunks:
        print(c.dominant_type, c.token_count, c.content[:80])

Pairs with NoPaddle:

    import nopaddle, smantic
    doc = nopaddle.parse_pdf("paper.pdf")          # PDF -> typed regions
    chunks = smantic.chunk_document(smantic.from_nopaddle(doc))
"""

from pathlib import Path

from . import config
from .adapters import (
    from_dict,
    from_docling_json,
    from_json,
    from_markdown,
    from_nopaddle,
)
from .chunker import Chunk, StructureAwareChunker
from .ir import BBox, Document, Element, Page

__version__ = "0.1.0"

__all__ = [
    "chunk_document",
    "chunk_markdown",
    "make_chunker",
    "StructureAwareChunker",
    "Chunk",
    "Document",
    "Page",
    "Element",
    "BBox",
    "from_markdown",
    "from_nopaddle",
    "from_docling_json",
    "from_json",
    "from_dict",
    "__version__",
]


def make_chunker(
    *,
    max_tokens: int = config.DEFAULT_MAX_TOKENS,
    overlap_tokens: int = config.DEFAULT_OVERLAP_TOKENS,
    model_dir: str | Path | None = None,
) -> StructureAwareChunker:
    """Build a :class:`StructureAwareChunker`.

    ``max_tokens`` is the soft per-chunk ceiling, ``overlap_tokens`` the token
    overlap between consecutive prose chunks. ``model_dir`` points at a local
    all-MiniLM-L6-v2 ONNX dir; when omitted the model is fetched from the Hugging
    Face Hub on first use and cached (set ``MODEL_CACHE_DIR`` to relocate). If
    the ONNX extra/model is unavailable the chunker still runs on structural
    boundaries alone.
    """
    return StructureAwareChunker(
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        model_dir=Path(model_dir) if model_dir else None,
    )


def chunk_document(
    doc: Document,
    *,
    max_tokens: int = config.DEFAULT_MAX_TOKENS,
    overlap_tokens: int = config.DEFAULT_OVERLAP_TOKENS,
    model_dir: str | Path | None = None,
) -> list[Chunk]:
    """One-shot chunking of a :class:`Document`. Builds, runs, then frees the model."""
    chunker = make_chunker(
        max_tokens=max_tokens, overlap_tokens=overlap_tokens, model_dir=model_dir
    )
    try:
        return chunker.chunk_document(doc)
    finally:
        chunker.release()


def chunk_markdown(
    text: str,
    *,
    max_tokens: int = config.DEFAULT_MAX_TOKENS,
    overlap_tokens: int = config.DEFAULT_OVERLAP_TOKENS,
    model_dir: str | Path | None = None,
) -> list[Chunk]:
    """One-shot chunking of raw Markdown text. Convenience over ``from_markdown``."""
    return chunk_document(
        from_markdown(text),
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        model_dir=model_dir,
    )
