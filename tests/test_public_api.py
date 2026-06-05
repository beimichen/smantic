"""Tests for the top-level public API + Chunk serialization."""

import smantic
from smantic import Chunk, Document, chunk_document, chunk_markdown, make_chunker


def test_exports_present():
    for name in [
        "chunk_document", "chunk_markdown", "make_chunker",
        "StructureAwareChunker", "Chunk", "Document", "Page", "Element", "BBox",
        "from_markdown", "from_nopaddle", "from_docling_json", "from_json", "from_dict",
        "__version__",
    ]:
        assert hasattr(smantic, name), f"missing export: {name}"


def test_version_is_string():
    assert isinstance(smantic.__version__, str)
    assert smantic.__version__.count(".") >= 1


def test_chunk_markdown_smoke():
    md = (
        "# Heading\n\n"
        "A reasonably sized paragraph that should survive as prose without being "
        "merged away, with enough words to clear the minimum and read naturally.\n\n"
        "```python\nprint('hi')\n```\n"
    )
    chunks = chunk_markdown(md)
    assert chunks, "expected at least one chunk"
    assert all(isinstance(c, Chunk) for c in chunks)
    # Reading order: prose before the code block.
    types = [c.dominant_type for c in chunks]
    assert "prose" in types and "code_block" in types
    assert types.index("prose") < types.index("code_block")
    # Sequence numbers are contiguous from 0.
    assert [c.sequence for c in chunks] == list(range(len(chunks)))


def test_chunk_document_accepts_ir_document():
    doc = Document.from_dict({
        "pages": [{"page_number": 1, "elements": [
            {"type": "text", "content": "Just a short line of prose here.", "page": 1}]}],
        "metadata": {},
    })
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].dominant_type == "prose"


def test_make_chunker_respects_max_tokens():
    c = make_chunker(max_tokens=123, overlap_tokens=7)
    assert c.max_tokens == 123
    assert c.overlap_tokens == 7


def test_boundary_threshold_is_wired_from_config():
    # The SMANTIC_BOUNDARY_THRESHOLD knob must actually reach the chunker.
    assert make_chunker().boundary_threshold == smantic.config.BOUNDARY_THRESHOLD


def test_chunk_to_dict_is_json_ready():
    chunks = chunk_markdown("# H\n\nSome prose content here for the chunk.\n")
    import json

    d = chunks[0].to_dict()
    json.dumps(d)  # must not raise
    assert set(d) >= {"content", "token_count", "dominant_type", "page_numbers", "metadata"}


def test_backmatter_references_are_dropped():
    md = (
        "# Body\n\nReal content paragraph that should be kept in the output.\n\n"
        "## References\n\nDoe, J. 2021. A citation that must be skipped.\n"
    )
    chunks = chunk_markdown(md)
    assert all("A citation that must be skipped" not in c.content for c in chunks)
