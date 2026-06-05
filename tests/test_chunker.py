#!/usr/bin/env python3
"""
Tests for Structure-Aware Hierarchical Chunker.

Tests cover:
- Atomic block detection (code, table, formula, visual)
- Prose semantic chunking with ONNX embeddings
- Three-tier boundary detection (hard/soft/emergency)
- Overlap between consecutive chunks
- Dominant type classification
- Content flags (has_code, has_math, has_table)
- Visual block caption handling
- Abbreviation handling (CK9)
- Whitespace joining (CK10)
- Integration with existing chunker logic
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from smantic.chunker import (
    Chunk,
    HeadingStack,
    SentenceSegmenter,
    StructureAwareChunker,
)
from smantic.ir import BBox, Document, Element, Page


@pytest.fixture
def chunker():
    """Provide structure-aware chunker (ONNX model loaded if available)."""
    return StructureAwareChunker(max_tokens=500, overlap_tokens=50)


@pytest.fixture
def chunker_no_overlap():
    """Provide structure-aware chunker with no overlap."""
    return StructureAwareChunker(max_tokens=500, overlap_tokens=0)


@pytest.fixture
def sample_prose_document():
    """Create a document with only prose elements."""
    pages = [
        Page(
            page_number=1,
            width=1000,
            height=1000,
            elements=[
                Element(
                    element_type='section_header',
                    content='Introduction',
                    bbox=BBox(100, 100, 900, 150),
                    page=1,
                    confidence=0.95
                ),
                Element(
                    element_type='text',
                    content='This is the first paragraph. It contains multiple sentences. '
                           'Each sentence should be properly segmented.',
                    bbox=BBox(100, 200, 900, 300),
                    page=1,
                    confidence=0.92
                ),
                Element(
                    element_type='text',
                    content='This is a second paragraph on a different topic. '
                           'It should be detected as a potential boundary.',
                    bbox=BBox(100, 350, 900, 450),
                    page=1,
                    confidence=0.90
                )
            ]
        )
    ]

    return Document(
        pages=pages,
        metadata={
            'source_file': 'test.pdf',
            'processor': 'test',
            'extraction_method': 'standard'
        }
    )


@pytest.fixture
def sample_code_document():
    """Create a document with code elements."""
    code_content = '''def hello_world():
    """Simple greeting function."""
    print("Hello, World!")
    return True

def calculate_sum(a, b):
    """Add two numbers."""
    return a + b'''

    pages = [
        Page(
            page_number=1,
            width=1000,
            height=1000,
            elements=[
                Element(
                    element_type='text',
                    content='Here is some example code:',
                    bbox=BBox(100, 100, 900, 150),
                    page=1
                ),
                Element(
                    element_type='code',
                    content=code_content,
                    bbox=BBox(100, 200, 900, 500),
                    page=1,
                    metadata={'language': 'python'}
                )
            ]
        )
    ]

    return Document(
        pages=pages,
        metadata={'extraction_method': 'standard'}
    )


@pytest.fixture
def sample_table_document():
    """Create a document with table elements."""
    table_content = '''| Name | Age | City |
|------|-----|------|
| Alice | 30 | NYC |
| Bob | 25 | LA |
| Charlie | 35 | Chicago |'''

    pages = [
        Page(
            page_number=1,
            width=1000,
            height=1000,
            elements=[
                Element(
                    element_type='text',
                    content='Here is a summary table:',
                    bbox=BBox(100, 100, 900, 150),
                    page=1
                ),
                Element(
                    element_type='table',
                    content=table_content,
                    bbox=BBox(100, 200, 900, 400),
                    page=1,
                    metadata={'caption': 'User Data'}
                )
            ]
        )
    ]

    return Document(
        pages=pages,
        metadata={'extraction_method': 'standard'}
    )


@pytest.fixture
def sample_formula_document():
    """Create a document with formula elements."""
    formula_content = r'\begin{equation} E = mc^2 \end{equation}'

    pages = [
        Page(
            page_number=1,
            width=1000,
            height=1000,
            elements=[
                Element(
                    element_type='text',
                    content='Einstein\'s famous equation:',
                    bbox=BBox(100, 100, 900, 150),
                    page=1
                ),
                Element(
                    element_type='formula',
                    content=formula_content,
                    bbox=BBox(100, 200, 900, 300),
                    page=1
                )
            ]
        )
    ]

    return Document(
        pages=pages,
        metadata={'extraction_method': 'standard'}
    )


@pytest.fixture
def sample_visual_document():
    """Create a document with visual elements."""
    pages = [
        Page(
            page_number=1,
            width=1000,
            height=1000,
            elements=[
                Element(
                    element_type='text',
                    content='The system architecture is shown below:',
                    bbox=BBox(100, 100, 900, 150),
                    page=1
                ),
                Element(
                    element_type='picture',
                    content='',
                    bbox=BBox(100, 200, 900, 600),
                    page=1,
                    metadata={
                        'caption': 'Figure 1: System Architecture Diagram',
                        'alt_text': 'A diagram showing the three-tier architecture'
                    }
                ),
                Element(
                    element_type='text',
                    content='The diagram illustrates the layered approach.',
                    bbox=BBox(100, 650, 900, 700),
                    page=1
                )
            ]
        )
    ]

    return Document(
        pages=pages,
        metadata={'extraction_method': 'standard'}
    )


@pytest.fixture
def sample_mixed_document():
    """Create a document with mixed content types."""
    pages = [
        Page(
            page_number=1,
            width=1000,
            height=1000,
            elements=[
                Element(
                    element_type='section_header',
                    content='Data Processing Example',
                    bbox=BBox(100, 100, 900, 150),
                    page=1
                ),
                Element(
                    element_type='text',
                    content='This section demonstrates data processing.',
                    bbox=BBox(100, 200, 900, 250),
                    page=1
                ),
                Element(
                    element_type='code',
                    content='def process(data):\n    return data.strip()',
                    bbox=BBox(100, 300, 900, 400),
                    page=1
                ),
                Element(
                    element_type='table',
                    content='| Input | Output |\n|-------|--------|\n| " x " | "x" |',
                    bbox=BBox(100, 450, 900, 550),
                    page=1
                ),
                Element(
                    element_type='text',
                    content='The table shows example transformations.',
                    bbox=BBox(100, 600, 900, 650),
                    page=1
                )
            ]
        )
    ]

    return Document(
        pages=pages,
        metadata={'extraction_method': 'standard'}
    )


# ============================================================================
# Sentence Segmenter Tests (including CK9 abbreviation fixes)
# ============================================================================

class TestSentenceSegmenter:
    """Tests for sentence segmentation, including abbreviation handling."""

    def test_basic_segmentation(self):
        segmenter = SentenceSegmenter()
        boundaries = segmenter.segment("First sentence. Second sentence. Third.")
        assert len(boundaries) == 3

    def test_abbreviation_dr(self):
        segmenter = SentenceSegmenter()
        boundaries = segmenter.segment("Dr. Smith went home. He was tired.")
        assert len(boundaries) == 2

    def test_abbreviation_multi_period(self):
        """CK9: Multi-period abbreviations like U.S. should not split."""
        segmenter = SentenceSegmenter()
        boundaries = segmenter.segment("The U.S. government passed a law. It was significant.")
        assert len(boundaries) == 2

    def test_abbreviation_phd(self):
        """CK9: Ph.D. should not cause a split."""
        segmenter = SentenceSegmenter()
        boundaries = segmenter.segment("She earned her Ph.D. in physics. Then she left.")
        assert len(boundaries) == 2

    def test_empty_text(self):
        segmenter = SentenceSegmenter()
        assert segmenter.segment("") == []
        assert segmenter.segment("   ") == []


# ============================================================================
# Atomic Block Detection Tests
# ============================================================================

class TestAtomicBlockDetection:
    """Tests for atomic block classification."""

    def test_is_atomic_block_code(self, chunker):
        element = Element(element_type='code', content='print("hello")',
                                 bbox=BBox(0, 0, 1000, 1000), page=1)
        assert chunker._is_atomic_block(element) is True

    def test_is_atomic_block_table(self, chunker):
        element = Element(element_type='table', content='| A | B |',
                                 bbox=BBox(0, 0, 1000, 1000), page=1)
        assert chunker._is_atomic_block(element) is True

    def test_is_atomic_block_formula(self, chunker):
        element = Element(element_type='formula', content='E = mc^2',
                                 bbox=BBox(0, 0, 1000, 1000), page=1)
        assert chunker._is_atomic_block(element) is True

    def test_is_atomic_block_picture(self, chunker):
        element = Element(element_type='picture', content='',
                                 bbox=BBox(0, 0, 1000, 1000), page=1)
        assert chunker._is_atomic_block(element) is True

    def test_is_atomic_block_chart(self, chunker):
        element = Element(element_type='chart', content='',
                                 bbox=BBox(0, 0, 1000, 1000), page=1)
        assert chunker._is_atomic_block(element) is True

    def test_is_atomic_block_figure(self, chunker):
        element = Element(element_type='figure', content='',
                                 bbox=BBox(0, 0, 1000, 1000), page=1)
        assert chunker._is_atomic_block(element) is True

    def test_is_not_atomic_block_text(self, chunker):
        element = Element(element_type='text', content='This is prose.',
                                 bbox=BBox(0, 0, 1000, 1000), page=1)
        assert chunker._is_atomic_block(element) is False

    def test_is_not_atomic_block_paragraph(self, chunker):
        element = Element(element_type='paragraph', content='This is a paragraph.',
                                 bbox=BBox(0, 0, 1000, 1000), page=1)
        assert chunker._is_atomic_block(element) is False

    def test_is_not_atomic_block_section_header(self, chunker):
        element = Element(element_type='section_header', content='Introduction',
                                 bbox=BBox(0, 0, 1000, 1000), page=1)
        assert chunker._is_atomic_block(element) is False


# ============================================================================
# Prose Semantic Chunking Tests
# ============================================================================

class TestProseSemanticChunking:
    """Tests for prose chunking logic."""

    def test_prose_elements_semantic_chunked(self, chunker, sample_prose_document):
        chunks = chunker.chunk_document(sample_prose_document)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.dominant_type == 'prose' for c in chunks)

    def test_prose_respects_token_limits(self, chunker, sample_prose_document):
        chunks = chunker.chunk_document(sample_prose_document)
        assert all(c.token_count <= chunker.max_tokens for c in chunks)

    def test_mixed_prose_types_grouped(self, chunker):
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(element_type='section_header', content='Chapter 1',
                                   bbox=BBox(100, 100, 900, 150), page=1),
                    Element(element_type='text', content='First paragraph of chapter.',
                                   bbox=BBox(100, 200, 900, 250), page=1),
                    Element(element_type='list_item', content='Point one',
                                   bbox=BBox(100, 300, 900, 330), page=1),
                    Element(element_type='list_item', content='Point two',
                                   bbox=BBox(100, 340, 900, 370), page=1),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)
        assert all(c.dominant_type == 'prose' for c in chunks)


# ============================================================================
# Overlap Tests (CK2 fix)
# ============================================================================

class TestOverlap:
    """Tests for chunk overlap implementation."""

    def test_overlap_produces_shared_content(self):
        """When overlap > 0, consecutive chunks should share trailing/leading text."""
        # Create a document with enough content to produce multiple chunks
        long_text = ". ".join([f"Sentence number {i} about topic {i % 5}" for i in range(100)])

        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(element_type='text', content=long_text,
                                   bbox=BBox(100, 100, 900, 900), page=1)
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})

        chunker_with_overlap = StructureAwareChunker(max_tokens=200, overlap_tokens=50)
        chunks = chunker_with_overlap.chunk_document(doc)

        if len(chunks) >= 2:
            # Verify that consecutive chunks share some text
            for i in range(len(chunks) - 1):
                # The end of chunk i should overlap with the start of chunk i+1
                # Check that some words from the end of chunk i appear at
                # the start of chunk i+1
                end_words = set(chunks[i].content.split()[-10:])
                start_words = set(chunks[i + 1].content.split()[:10])
                overlap = end_words & start_words
                assert len(overlap) > 0, (
                    f"No overlap between chunk {i} and {i+1}"
                )

    def test_no_overlap_when_zero(self):
        """When overlap=0, consecutive chunks should not share content."""
        long_text = ". ".join([f"Sentence number {i}" for i in range(50)])
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(element_type='text', content=long_text,
                                   bbox=BBox(100, 100, 900, 900), page=1)
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})

        chunker = StructureAwareChunker(max_tokens=200, overlap_tokens=0)
        chunks = chunker.chunk_document(doc)

        # With no overlap, the full concatenation should roughly equal the original
        if len(chunks) >= 2:
            full_text = " ".join(c.content for c in chunks)
            # No duplicate sentences expected
            assert full_text.count("Sentence number 0") == 1


# ============================================================================
# Whitespace Joining Tests (CK10 fix)
# ============================================================================

class TestWhitespaceJoining:
    """Tests for consistent whitespace joining."""

    def test_paragraph_breaks_preserved(self, chunker):
        """Paragraph breaks between elements are preserved as \\n\\n."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(element_type='text', content='First paragraph text.',
                                   bbox=BBox(100, 100, 900, 200), page=1),
                    Element(element_type='text', content='Second paragraph text.',
                                   bbox=BBox(100, 250, 900, 350), page=1),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        if len(chunks) == 1:
            # Both paragraphs in one chunk — should have \n\n between them
            assert "\n\n" in chunks[0].content


# ============================================================================
# Atomic Block Preservation Tests
# ============================================================================

class TestAtomicBlockPreservation:
    """Tests for atomic block integrity."""

    def test_small_code_block_stays_intact(self, chunker, sample_code_document):
        chunks = chunker.chunk_document(sample_code_document)
        code_chunks = [c for c in chunks if c.dominant_type == 'code_block']
        assert len(code_chunks) == 1
        code_chunk = code_chunks[0]
        assert 'hello_world' in code_chunk.content
        assert 'calculate_sum' in code_chunk.content

    def test_small_table_stays_intact(self, chunker, sample_table_document):
        chunks = chunker.chunk_document(sample_table_document)
        table_chunks = [c for c in chunks if c.dominant_type == 'table_block']
        assert len(table_chunks) == 1
        table_chunk = table_chunks[0]
        assert 'Alice' in table_chunk.content
        assert 'Bob' in table_chunk.content
        assert 'Charlie' in table_chunk.content

    def test_small_formula_stays_intact(self, chunker, sample_formula_document):
        chunks = chunker.chunk_document(sample_formula_document)
        formula_chunks = [c for c in chunks if c.dominant_type == 'formula_block']
        assert len(formula_chunks) == 1
        formula_chunk = formula_chunks[0]
        assert 'E = mc^2' in formula_chunk.content


# ============================================================================
# Visual Block Tests
# ============================================================================

class TestVisualBlockHandling:
    """Tests for visual element handling."""

    def test_visual_block_with_caption(self, chunker, sample_visual_document):
        chunks = chunker.chunk_document(sample_visual_document)
        visual_chunks = [c for c in chunks if c.dominant_type == 'visual_block']
        assert len(visual_chunks) == 1
        visual_chunk = visual_chunks[0]
        assert 'Figure 1' in visual_chunk.content
        assert 'Architecture Diagram' in visual_chunk.content
        assert 'three-tier' in visual_chunk.content

    def test_visual_block_without_caption_is_skipped(self, chunker):
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(element_type='picture', content='',
                                   bbox=BBox(100, 100, 900, 500), page=1, metadata={})
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)
        visual_chunks = [c for c in chunks if c.dominant_type == 'visual_block']
        assert len(visual_chunks) == 0

    def test_visual_block_with_alt_text_only(self, chunker):
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(element_type='figure', content='',
                                   bbox=BBox(100, 100, 900, 500), page=1,
                                   metadata={'alt_text': 'Diagram showing data flow'})
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)
        visual_chunks = [c for c in chunks if c.dominant_type == 'visual_block']
        assert len(visual_chunks) == 1
        assert 'data flow' in visual_chunks[0].content

    def test_visual_block_sequence_preserved(self, chunker, sample_visual_document):
        chunks = chunker.chunk_document(sample_visual_document)
        # Intro text "shown below:" is now attached to the visual block (Change 3),
        # so we get 2 chunks: visual_block (with intro prepended) + trailing prose.
        assert len(chunks) == 2
        assert chunks[0].dominant_type == 'visual_block'
        assert 'shown below' in chunks[0].content
        assert chunks[1].dominant_type == 'prose'
        assert chunks[0].sequence == 0
        assert chunks[1].sequence == 1


# ============================================================================
# Dominant Type & Content Flags Tests
# ============================================================================

class TestDominantTypeClassification:
    def test_dominant_type_prose(self, chunker, sample_prose_document):
        chunks = chunker.chunk_document(sample_prose_document)
        assert all(c.dominant_type == 'prose' for c in chunks)

    def test_dominant_type_code_block(self, chunker, sample_code_document):
        chunks = chunker.chunk_document(sample_code_document)
        code_chunks = [c for c in chunks if c.has_code]
        assert all(c.dominant_type == 'code_block' for c in code_chunks)

    def test_dominant_type_table_block(self, chunker, sample_table_document):
        chunks = chunker.chunk_document(sample_table_document)
        table_chunks = [c for c in chunks if c.has_table]
        assert all(c.dominant_type == 'table_block' for c in table_chunks)

    def test_dominant_type_formula_block(self, chunker, sample_formula_document):
        chunks = chunker.chunk_document(sample_formula_document)
        formula_chunks = [c for c in chunks if c.has_math]
        assert all(c.dominant_type == 'formula_block' for c in formula_chunks)

    def test_dominant_type_visual_block(self, chunker, sample_visual_document):
        chunks = chunker.chunk_document(sample_visual_document)
        visual_chunks = [c for c in chunks if c.dominant_type == 'visual_block']
        assert len(visual_chunks) == 1


class TestContentFlags:
    def test_has_code_flag(self, chunker, sample_code_document):
        chunks = chunker.chunk_document(sample_code_document)
        code_chunks = [c for c in chunks if c.dominant_type == 'code_block']
        assert all(c.has_code is True for c in code_chunks)
        prose_chunks = [c for c in chunks if c.dominant_type == 'prose']
        assert all(c.has_code is False for c in prose_chunks)

    def test_has_math_flag(self, chunker, sample_formula_document):
        chunks = chunker.chunk_document(sample_formula_document)
        formula_chunks = [c for c in chunks if c.dominant_type == 'formula_block']
        assert all(c.has_math is True for c in formula_chunks)

    def test_has_table_flag(self, chunker, sample_table_document):
        chunks = chunker.chunk_document(sample_table_document)
        table_chunks = [c for c in chunks if c.dominant_type == 'table_block']
        assert all(c.has_table is True for c in table_chunks)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegrationWithExistingChunker:
    def test_sequence_numbering(self, chunker, sample_mixed_document):
        chunks = chunker.chunk_document(sample_mixed_document)
        for i, chunk in enumerate(chunks):
            assert chunk.sequence == i

    def test_page_numbers_preserved(self, chunker, sample_mixed_document):
        chunks = chunker.chunk_document(sample_mixed_document)
        assert all(hasattr(c, 'page_numbers') for c in chunks)
        assert all(len(c.page_numbers) > 0 for c in chunks)
        assert all(1 in c.page_numbers for c in chunks)

    def test_empty_document(self, chunker):
        doc = Document(pages=[], metadata={})
        chunks = chunker.chunk_document(doc)
        assert chunks == []

    def test_fallback_chunking_still_works(self, chunker):
        """Fitz fallback documents are still chunked (as prose elements)."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(element_type='text',
                                   content='HEADER:\\n\\nParagraph one.\\n\\nParagraph two.',
                                   bbox=BBox(100, 100, 900, 900), page=1)
                ]
            )
        ]
        doc = Document(
            pages=pages,
            metadata={'extraction_method': 'fitz_fallback'}
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 0
        assert all(c.chunking_method == 'semantic' for c in chunks)


class TestMixedDocumentChunking:
    def test_mixed_document_chunks_by_type(self, chunker, sample_mixed_document):
        chunks = chunker.chunk_document(sample_mixed_document)
        dominant_types = [c.dominant_type for c in chunks]
        assert 'prose' in dominant_types
        assert 'code_block' in dominant_types
        assert 'table_block' in dominant_types

    def test_mixed_document_order_preserved(self, chunker, sample_mixed_document):
        chunks = chunker.chunk_document(sample_mixed_document)
        sequences = [c.sequence for c in chunks]
        assert sequences == sorted(sequences)

    def test_prose_not_interrupted_by_blocks(self, chunker):
        """Short prose flanking a code block merges into one prose chunk
        (both are below _MERGE_MIN_TOKENS). The code block stays separate."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(element_type='text', content='Text before code.',
                                   bbox=BBox(100, 100, 900, 150), page=1),
                    Element(element_type='code', content='print("hello")',
                                   bbox=BBox(100, 200, 900, 250), page=1),
                    Element(element_type='text', content='Text after code.',
                                   bbox=BBox(100, 300, 900, 350), page=1),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        # Short prose chunks merge across the code block
        assert len(chunks) == 2
        assert chunks[0].dominant_type == 'code_block'
        # Merged prose contains both before and after text
        assert chunks[1].dominant_type == 'prose'
        assert 'before' in chunks[1].content.lower()
        assert 'after' in chunks[1].content.lower()

    def test_long_prose_flanking_block_stays_separate(self, chunker):
        """Prose chunks above _MERGE_MIN_TOKENS stay separate from each other."""
        # Each paragraph needs >100 tokens to avoid merging
        long_before = (
            "Electronegativity is a measure of an atom's ability to attract the valence "
            "electrons of another atom. We generally use Pauling's scale for electronegativity. "
            "Electronegativity increases as you go across rows due to increasing effective nuclear "
            "charge and smaller atomic radius. It decreases as you go down columns due to constant "
            "effective nuclear charge and larger radius. Polarity depends on molecular shape and "
            "the vectors of dipoles. If a molecule's shape is not symmetrical it is likely polar. "
            "The concept of electronegativity was first proposed by Linus Pauling in 1932 as part "
            "of his theory of chemical bonding. The Pauling scale assigns values ranging from about "
            "0.7 for francium to 4.0 for fluorine. Elements with high electronegativity tend to "
            "gain electrons in chemical reactions while elements with low electronegativity tend to "
            "lose electrons. The difference in electronegativity between two bonded atoms determines "
            "whether the bond is ionic, polar covalent, or nonpolar covalent."
        )
        long_after = (
            "Lewis structures represent the arrangement of electrons in a molecule. To draw them "
            "first count all valence electrons for every atom. Then draw single bonds and subtract "
            "two electrons per bond from the total. If you need more electrons than are available "
            "form double or triple bonds. Expanded octets use empty d orbitals. Resonance structures "
            "show different valid arrangements of double bonds. Move electrons not atoms when "
            "drawing resonance structures. The stronger the bond the closer the atoms. "
            "Formal charge is calculated by taking the number of valence electrons minus the number "
            "of lone pair electrons minus half the number of bonding electrons. The structure with "
            "formal charges closest to zero is usually the most stable. When drawing Lewis structures "
            "for polyatomic ions remember that the total number of electrons includes the charge. "
            "Exceptions to the octet rule include molecules with odd numbers of electrons, incomplete "
            "octets such as boron trifluoride, and expanded octets using d orbitals."
        )
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(element_type='text', content=long_before,
                                   bbox=BBox(100, 100, 900, 150), page=1),
                    Element(element_type='code', content='print("hello")',
                                   bbox=BBox(100, 200, 900, 250), page=1),
                    Element(element_type='text', content=long_after,
                                   bbox=BBox(100, 300, 900, 350), page=1),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        types = [c.dominant_type for c in chunks]
        assert 'code_block' in types
        # Both prose chunks are above merge threshold, so they stay separate
        assert types.count('prose') == 2


# ============================================================================
# Parent-Child Hierarchy Tests
# ============================================================================

class TestParentChunkIDHierarchy:
    def test_large_code_block_has_parent_children(self, chunker):
        functions = []
        for i in range(40):
            functions.append(
                f"def function_{i}(x, y, z):\n"
                f"    \"\"\"Process input values and return computed result for case {i}.\"\"\"\n"
                f"    result = x * {i + 1} + y * {i + 2} + z * {i + 3}\n"
                f"    intermediate = result ** 2 + {i * 10}\n"
                f"    if intermediate > {i * 100}:\n"
                f"        return intermediate / {i + 1}\n"
                f"    return result + intermediate\n"
            )
        large_code = "\n\n".join(functions)

        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(element_type='code', content=large_code,
                                   bbox=BBox(100, 100, 900, 900), page=1,
                                   metadata={'language': 'python'})
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        code_chunks = [c for c in chunks if c.dominant_type == 'code_block']
        assert len(code_chunks) >= 2

        parent = code_chunks[0]
        assert parent.metadata.get('has_children') is True
        assert parent.chunking_method == 'atomic_block'

        children = code_chunks[1:]
        for child in children:
            assert child.block_sequence is not None
            assert child.chunking_method == 'ast_split'
        # block_sequence values should be monotonically increasing
        seqs = [child.block_sequence for child in children]
        assert seqs == sorted(seqs)

    def test_regular_prose_has_no_parent_link(self, chunker, sample_prose_document):
        chunks = chunker.chunk_document(sample_prose_document)
        for chunk in chunks:
            assert chunk.parent_chunk_id is None
            assert chunk.block_sequence is None


# ============================================================================
# Heading Context Tests (Chunk Heading Traceability)
# ============================================================================

class TestHeadingStack:
    """Unit tests for the HeadingStack helper."""

    def test_push_and_trail(self):
        hs = HeadingStack()
        hs.push(1, "Chapter 1")
        assert hs.get_trail() == ["Chapter 1"]

        hs.push(2, "Introduction")
        assert hs.get_trail() == ["Chapter 1", "Introduction"]

    def test_same_level_replaces(self):
        hs = HeadingStack()
        hs.push(1, "Chapter 1")
        hs.push(2, "Intro")
        hs.push(2, "Background")
        assert hs.get_trail() == ["Chapter 1", "Background"]

    def test_deeper_level_cleared_on_shallower(self):
        hs = HeadingStack()
        hs.push(1, "Ch1")
        hs.push(2, "Sec")
        hs.push(3, "Sub")
        assert hs.get_trail() == ["Ch1", "Sec", "Sub"]
        hs.push(1, "Ch2")
        assert hs.get_trail() == ["Ch2"]

    def test_to_metadata_empty(self):
        hs = HeadingStack()
        assert hs.to_metadata() == {}

    def test_to_metadata_format(self):
        hs = HeadingStack()
        hs.push(1, "Ch1")
        hs.push(2, "Background")
        meta = hs.to_metadata()
        assert meta == {
            "heading_trail": ["Ch1", "Background"],
            "heading_level": 2,
            "nearest_heading": "Background",
        }


class TestHeadingLevelExtraction:
    """Tests for _extract_heading_level static method."""

    def test_section_header_level_n(self):
        elem = Element(
            element_type='section_header_level_3',
            content='Subsection',
            bbox=BBox(0, 0, 1000, 1000), page=1
        )
        assert StructureAwareChunker._extract_heading_level(elem) == 3

    def test_metadata_level(self):
        elem = Element(
            element_type='section_header',
            content='Chapter',
            bbox=BBox(0, 0, 1000, 1000), page=1,
            metadata={'level': 2}
        )
        assert StructureAwareChunker._extract_heading_level(elem) == 2

    def test_docx_style(self):
        elem = Element(
            element_type='section_header',
            content='Title',
            bbox=BBox(0, 0, 1000, 1000), page=1,
            metadata={'style': 'Heading 2'}
        )
        assert StructureAwareChunker._extract_heading_level(elem) == 2

    def test_bare_section_header(self):
        elem = Element(
            element_type='section_header',
            content='Slide Title',
            bbox=BBox(0, 0, 1000, 1000), page=1
        )
        assert StructureAwareChunker._extract_heading_level(elem) == 1

    def test_non_header_returns_none(self):
        elem = Element(
            element_type='text',
            content='Hello',
            bbox=BBox(0, 0, 1000, 1000), page=1
        )
        assert StructureAwareChunker._extract_heading_level(elem) is None


class TestHeadingContextInChunks:
    """Tests that heading context flows into chunk metadata."""

    def test_heading_context_in_prose_chunks(self, chunker):
        """H1 → text → H2 → text. Verify heading trail and prepended content."""
        # Text must be long enough (>100 tokens each) to avoid the small-chunk merge pass.
        ch1_body = (
            'This is chapter one introduction text that provides a detailed overview '
            'of the fundamental concepts and principles discussed throughout this section. '
            'It covers key terminology and sets the stage for the material that follows in '
            'the remaining subsections. We begin by establishing the core definitions and '
            'notation that will be used consistently throughout the rest of this document. '
            'The chapter is organized into several subsections each of which builds upon the '
            'ideas presented in the previous one. By the end of this chapter readers should '
            'have a solid understanding of the theoretical framework and practical techniques '
            'that form the basis for the advanced topics covered in subsequent chapters.'
        )
        bg_body = (
            'Background information goes here with additional context about the prior work '
            'and relevant literature that informed the development of the methods described. '
            'This section summarizes findings from multiple previous studies in the field and '
            'identifies the key gaps in existing approaches that motivated our research. '
            'We also review the theoretical foundations upon which our methodology is built. '
            'Previous attempts to solve this problem have relied on simplified assumptions '
            'that do not hold in practice. Our approach addresses these limitations by '
            'introducing a novel formulation that accounts for the full complexity of the '
            'underlying system dynamics and their interactions with the environment.'
        )
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='section_header_level_1',
                        content='Chapter 1',
                        bbox=BBox(0, 0, 1000, 50), page=1,
                        metadata={'level': 1}
                    ),
                    Element(
                        element_type='text',
                        content=ch1_body,
                        bbox=BBox(0, 50, 1000, 200), page=1
                    ),
                    Element(
                        element_type='section_header_level_2',
                        content='Background',
                        bbox=BBox(0, 200, 1000, 250), page=1,
                        metadata={'level': 2}
                    ),
                    Element(
                        element_type='text',
                        content=bg_body,
                        bbox=BBox(0, 250, 1000, 400), page=1
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        ch1_text = [c for c in chunks if 'chapter one introduction' in c.content.lower()]
        bg_text = [c for c in chunks if 'background information' in c.content.lower()]
        assert len(ch1_text) >= 1
        assert len(bg_text) >= 1

        # Heading text prepended into the first content chunk
        assert ch1_text[0].content.startswith('Chapter 1\n\n')

        # Metadata trail
        assert ch1_text[0].metadata.get('heading_trail') == ['Chapter 1']
        assert ch1_text[0].metadata.get('nearest_heading') == 'Chapter 1'

        assert bg_text[0].content.startswith('Background\n\n')
        assert bg_text[0].metadata.get('heading_trail') == ['Chapter 1', 'Background']
        assert bg_text[0].metadata.get('nearest_heading') == 'Background'

        # No standalone heading-only chunks
        for c in chunks:
            assert c.content.strip() not in ('Chapter 1', 'Background'), \
                f"Heading should not be a standalone chunk: {c.content!r}"

    def test_heading_context_on_atomic_blocks(self, chunker):
        """H1 → code block. Code chunk should inherit heading trail."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='section_header_level_1',
                        content='Examples',
                        bbox=BBox(0, 0, 1000, 50), page=1,
                        metadata={'level': 1}
                    ),
                    Element(
                        element_type='code',
                        content='print("hello world")',
                        bbox=BBox(0, 50, 1000, 200), page=1
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        code_chunks = [c for c in chunks if c.dominant_type == 'code_block']
        assert len(code_chunks) >= 1
        assert code_chunks[0].metadata.get('heading_trail') == ['Examples']

    def test_heading_stack_level_replacement(self, chunker):
        """H1 → H3 → H2. Verify H2 clears H3 from trail."""
        # Text must be long enough (>100 tokens each) to avoid the small-chunk merge pass.
        deep_body = (
            'Text under deep section providing detailed analysis of the lower level '
            'subsection concepts and their implications for the broader theoretical framework. '
            'This includes several important observations about system behavior under load '
            'and identifies performance bottlenecks that emerge when scaling to production '
            'workloads with thousands of concurrent users accessing the distributed system. '
            'Furthermore the analysis reveals that memory allocation patterns play a critical '
            'role in determining overall throughput and that careful optimization of buffer '
            'management strategies can yield significant performance improvements across a wide '
            'range of workload characteristics and deployment configurations.'
        )
        mid_body = (
            'Text under mid section covering the intermediate level concepts that bridge '
            'the gap between the high level overview and the detailed implementation specifics. '
            'Several key design decisions are discussed and their trade-offs are evaluated '
            'in the context of real-world deployment constraints including latency requirements '
            'and resource limitations encountered during the initial prototype development phase. '
            'The section concludes with a comprehensive comparison of alternative approaches '
            'that were considered during the design process and explains the rationale behind '
            'the final architecture choices that were made to balance competing requirements '
            'of scalability reliability and maintainability in the production environment.'
        )
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='section_header_level_1',
                        content='Top Level',
                        bbox=BBox(0, 0, 1000, 50), page=1,
                        metadata={'level': 1}
                    ),
                    Element(
                        element_type='section_header_level_3',
                        content='Deep Section',
                        bbox=BBox(0, 50, 1000, 100), page=1,
                        metadata={'level': 3}
                    ),
                    Element(
                        element_type='text',
                        content=deep_body,
                        bbox=BBox(0, 100, 1000, 200), page=1
                    ),
                    Element(
                        element_type='section_header_level_2',
                        content='Mid Section',
                        bbox=BBox(0, 200, 1000, 250), page=1,
                        metadata={'level': 2}
                    ),
                    Element(
                        element_type='text',
                        content=mid_body,
                        bbox=BBox(0, 250, 1000, 400), page=1
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        deep_text = [c for c in chunks if 'text under deep' in c.content.lower()]
        mid_text = [c for c in chunks if 'text under mid' in c.content.lower()]
        assert len(deep_text) >= 1
        assert len(mid_text) >= 1

        # Consecutive headings (H1→H3) both prepended to first content chunk
        assert 'Top Level' in deep_text[0].content
        assert 'Deep Section' in deep_text[0].content

        # Metadata trail
        assert deep_text[0].metadata.get('heading_trail') == ['Top Level', 'Deep Section']

        # H2 clears H3 from trail
        assert mid_text[0].content.startswith('Mid Section\n\n')
        assert mid_text[0].metadata.get('heading_trail') == ['Top Level', 'Mid Section']

    def test_no_heading_context_for_headingless_doc(self, chunker):
        """Document with only text elements has no heading keys in metadata."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='text',
                        content='Just some text.',
                        bbox=BBox(0, 0, 1000, 200), page=1
                    ),
                    Element(
                        element_type='text',
                        content='More text here.',
                        bbox=BBox(0, 200, 1000, 400), page=1
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        for c in chunks:
            assert 'heading_trail' not in c.metadata
            assert 'nearest_heading' not in c.metadata

    def test_heading_level_from_docx_style(self):
        """Element with metadata={'style': 'Heading 2'} → level 2."""
        elem = Element(
            element_type='section_header',
            content='Methods',
            bbox=BBox(0, 0, 1000, 1000), page=1,
            metadata={'style': 'Heading 2'}
        )
        assert StructureAwareChunker._extract_heading_level(elem) == 2

    def test_heading_text_prepended_not_standalone(self, chunker):
        """Heading text is prepended to following chunk, never standalone."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='section_header_level_1',
                        content='Introduction',
                        bbox=BBox(0, 0, 1000, 50), page=1,
                        metadata={'level': 1}
                    ),
                    Element(
                        element_type='section_header_level_2',
                        content='Overview',
                        bbox=BBox(0, 50, 1000, 100), page=1,
                        metadata={'level': 2}
                    ),
                    Element(
                        element_type='text',
                        content='This section provides an overview.',
                        bbox=BBox(0, 100, 1000, 200), page=1
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        # Consecutive headings prepended to the first content chunk
        assert len(chunks) == 1
        assert chunks[0].content.startswith('Introduction\n\nOverview\n\n')
        assert 'overview.' in chunks[0].content.lower()

        # No standalone heading chunks
        for c in chunks:
            assert c.content.strip() not in ('Introduction', 'Overview')


    def test_trailing_heading_absorbed(self, chunker):
        """Heading at end of document is absorbed into the last chunk."""
        body = (
            'Some initial content providing enough context for the document introduction. '
            'This text covers the basic concepts and sets up the framework for the discussion '
            'that would normally follow in subsequent sections of the document.'
        )
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='text',
                        content=body,
                        bbox=BBox(0, 0, 1000, 100), page=1
                    ),
                    Element(
                        element_type='section_header_level_1',
                        content='Appendix',
                        bbox=BBox(0, 100, 1000, 150), page=1,
                        metadata={'level': 1}
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        # Trailing heading absorbed into the last chunk, not a standalone chunk
        all_content = ' '.join(c.content for c in chunks)
        assert 'Appendix' in all_content, "Trailing heading should be absorbed into last chunk"
        # No standalone heading-only chunk
        for c in chunks:
            assert c.content.strip() != 'Appendix', \
                "Trailing heading should not be a standalone chunk"


    def test_page_header_footer_excluded(self, chunker):
        """Page headers/footers (extracted page numbers) must not appear in chunks."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='page_header',
                        content='Chapter 1 — Draft',
                        bbox=BBox(0, 0, 1000, 30), page=1
                    ),
                    Element(
                        element_type='section_header_level_1',
                        content='Introduction',
                        bbox=BBox(0, 30, 1000, 80), page=1,
                        metadata={'level': 1}
                    ),
                    Element(
                        element_type='text',
                        content='The actual content of the page.',
                        bbox=BBox(0, 80, 1000, 500), page=1
                    ),
                    Element(
                        element_type='page_footer',
                        content='5',
                        bbox=BBox(0, 950, 1000, 1000), page=1
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        all_content = '\n'.join(c.content for c in chunks)
        # Page footer "5" must not appear as chunk content
        assert '5' not in all_content.split(), \
            f"Page number leaked into chunk content: {all_content!r}"
        # Page header must not appear
        assert 'Draft' not in all_content, \
            f"Page header leaked into chunk content: {all_content!r}"
        # Actual content must be present
        assert 'actual content' in all_content.lower()


class TestProseCompatibleTypes:
    """Tests for Change 1: Prose-compatible type equivalence."""

    def test_list_items_merged_with_intro_text(self, chunker):
        """text → list_item should NOT create hard boundary; intro + list merge."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='text',
                        content='The ingredients are:',
                        bbox=BBox(0, 0, 1000, 50), page=1
                    ),
                    Element(
                        element_type='list_item',
                        content='Two cups of flour.',
                        bbox=BBox(0, 50, 1000, 80), page=1
                    ),
                    Element(
                        element_type='list_item',
                        content='One cup of sugar.',
                        bbox=BBox(0, 80, 1000, 110), page=1
                    ),
                    Element(
                        element_type='list_item',
                        content='Three eggs.',
                        bbox=BBox(0, 110, 1000, 140), page=1
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        # All should be in a single chunk (total tokens << min_tokens)
        assert len(chunks) == 1
        assert 'ingredients' in chunks[0].content
        assert 'flour' in chunks[0].content
        assert 'sugar' in chunks[0].content
        assert 'eggs' in chunks[0].content

    def test_paragraph_no_hard_boundary_with_text(self, chunker):
        """text → paragraph (VLM) should not create a hard break."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='text',
                        content='First part of the discussion.',
                        bbox=BBox(0, 0, 1000, 50), page=1
                    ),
                    Element(
                        element_type='paragraph',
                        content='Continuation of the same topic.',
                        bbox=BBox(0, 50, 1000, 100), page=1
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert 'First part' in chunks[0].content
        assert 'Continuation' in chunks[0].content

    def test_footnote_flows_with_prose(self, chunker):
        """text → footnote should not create a hard break."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='text',
                        content='The study found significant results.',
                        bbox=BBox(0, 0, 1000, 50), page=1
                    ),
                    Element(
                        element_type='footnote',
                        content='Statistical significance was p < 0.05.',
                        bbox=BBox(0, 50, 1000, 100), page=1
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert 'significant results' in chunks[0].content
        assert 'p < 0.05' in chunks[0].content


class TestVLMElementTypes:
    """Tests for Change 2: VLM element type classification."""

    def test_title_treated_as_heading(self, chunker):
        """VLM 'title' element should behave like a heading (level 1)."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='title',
                        content='Document Title',
                        bbox=BBox(0, 0, 1000, 50), page=1
                    ),
                    Element(
                        element_type='text',
                        content='Content under the title.',
                        bbox=BBox(0, 50, 1000, 100), page=1
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        # Title text prepended to following content chunk
        assert len(chunks) == 1
        assert chunks[0].content.startswith('Document Title\n\n')
        assert 'Content under the title' in chunks[0].content
        # Heading metadata set
        assert chunks[0].metadata.get('nearest_heading') == 'Document Title'

    def test_subtitle_treated_as_heading(self, chunker):
        """VLM 'subtitle' element should behave like a heading (level 2)."""
        elem = Element(
            element_type='subtitle',
            content='A Subtitle',
            bbox=BBox(0, 0, 1000, 50), page=1
        )
        level = StructureAwareChunker._extract_heading_level(elem)
        assert level == 2

    def test_footnote_reference_skipped(self, chunker):
        """footnote_reference elements should be skipped entirely."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='text',
                        content='Main content here.',
                        bbox=BBox(0, 0, 1000, 50), page=1
                    ),
                    Element(
                        element_type='footnote_reference',
                        content='1',
                        bbox=BBox(200, 30, 210, 40), page=1
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        all_content = '\n'.join(c.content for c in chunks)
        # The bare "1" from footnote_reference must not be a separate chunk
        assert 'Main content' in all_content
        # Ensure footnote_reference content is not in any chunk
        for c in chunks:
            assert c.content.strip() != '1'


class TestCaptionIntroAttachment:
    """Tests for Change 3: Caption and intro text attachment to atomic blocks."""

    def test_caption_element_attached_to_table(self, chunker):
        """VLM caption element should be prepended to the following table chunk."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='caption',
                        content='Table 1: Student demographics',
                        bbox=BBox(0, 0, 1000, 30), page=1
                    ),
                    Element(
                        element_type='table',
                        content='| Name | Age |\n|------|-----|\n| Alice | 20 |',
                        bbox=BBox(0, 30, 1000, 200), page=1
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        # Caption should be prepended to the table chunk
        table_chunks = [c for c in chunks if c.dominant_type == 'table_block']
        assert len(table_chunks) == 1
        assert 'Student demographics' in table_chunks[0].content
        assert 'Alice' in table_chunks[0].content

    def test_caption_pattern_attached_to_table(self, chunker):
        """Short 'Table N:' text element should be attached to following table."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='text',
                        content='Table 2: Demographics overview',
                        bbox=BBox(0, 0, 1000, 30), page=1
                    ),
                    Element(
                        element_type='table',
                        content='| City | Pop |\n|------|-----|\n| NYC | 8M |',
                        bbox=BBox(0, 30, 1000, 200), page=1
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        table_chunks = [c for c in chunks if c.dominant_type == 'table_block']
        assert len(table_chunks) == 1
        assert 'Demographics overview' in table_chunks[0].content
        assert 'NYC' in table_chunks[0].content

    def test_intro_text_attached_to_code_block(self, chunker):
        """Short intro ending with ':' should attach to following code block."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='text',
                        content='The following code shows the solution:',
                        bbox=BBox(0, 0, 1000, 30), page=1
                    ),
                    Element(
                        element_type='code',
                        content='def hello():\n    print("hello")',
                        bbox=BBox(0, 30, 1000, 200), page=1
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        code_chunks = [c for c in chunks if c.dominant_type == 'code_block']
        assert len(code_chunks) == 1
        assert 'following code' in code_chunks[0].content
        assert 'def hello' in code_chunks[0].content

    def test_long_text_not_extracted_as_context(self, chunker):
        """Text longer than 60 tokens before an atomic block should NOT be popped."""
        long_text = ' '.join(['word'] * 80)  # well over 60 tokens
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='text',
                        content=long_text,
                        bbox=BBox(0, 0, 1000, 200), page=1
                    ),
                    Element(
                        element_type='table',
                        content='| A | B |\n|---|---|\n| 1 | 2 |',
                        bbox=BBox(0, 200, 1000, 400), page=1
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        # Long text should be in its own prose chunk, not attached to table
        prose_chunks = [c for c in chunks if c.dominant_type == 'prose']
        table_chunks = [c for c in chunks if c.dominant_type == 'table_block']
        assert len(prose_chunks) >= 1
        assert len(table_chunks) >= 1
        # Table chunk should NOT contain the long text
        assert 'word word word' not in table_chunks[0].content

    def test_caption_creates_visual_chunk_when_no_metadata(self, chunker):
        """Caption + figure (no metadata caption) should create visual chunk from context."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='caption',
                        content='Figure 3: Neural network architecture',
                        bbox=BBox(0, 0, 1000, 30), page=1
                    ),
                    Element(
                        element_type='figure',
                        content='',
                        bbox=BBox(0, 30, 1000, 400), page=1
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        # Should have a visual chunk with the caption content
        visual_chunks = [c for c in chunks if c.dominant_type == 'visual_block']
        assert len(visual_chunks) == 1
        assert 'Neural network architecture' in visual_chunks[0].content

    def test_fallback_visual_chunk_gets_heading_metadata(self, chunker):
        """Fallback visual chunk (from context_text) inherits heading trail."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='section_header_level_1',
                        content='Results',
                        bbox=BBox(0, 0, 1000, 30), page=1,
                        metadata={'level': 1}
                    ),
                    Element(
                        element_type='caption',
                        content='Figure 5: Performance comparison',
                        bbox=BBox(0, 30, 1000, 60), page=1
                    ),
                    Element(
                        element_type='figure',
                        content='',
                        bbox=BBox(0, 60, 1000, 400), page=1
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        visual_chunks = [c for c in chunks if c.dominant_type == 'visual_block']
        assert len(visual_chunks) == 1
        assert visual_chunks[0].metadata.get('heading_trail') == ['Results']


class TestSpeakerNotesMerging:
    """Tests for Change 4: Speaker notes merging (PPTX)."""

    def test_speaker_notes_merged_with_slide(self, chunker):
        """Speaker notes should merge with preceding slide content."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='text',
                        content='Key Takeaways for Q3.',
                        bbox=BBox(0, 0, 1000, 100), page=1
                    ),
                    Element(
                        element_type='text',
                        content='[Speaker Notes]\nThe main point here is that revenue grew 15% year over year.',
                        bbox=BBox(0, 100, 1000, 300), page=1,
                        metadata={'is_speaker_notes': True}
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        # Slide text + speaker notes should be in the same chunk
        assert len(chunks) == 1
        assert 'Key Takeaways' in chunks[0].content
        assert 'revenue grew 15%' in chunks[0].content


class TestSectionTypeMetadata:
    """Tests for Change 5: Section type metadata propagation."""

    def test_abstract_section_type_in_metadata(self, chunker):
        """Elements with metadata type='abstract' should propagate to chunk."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='text',
                        content='This paper presents a novel approach to chunking.',
                        bbox=BBox(0, 0, 1000, 100), page=1,
                        metadata={'type': 'abstract'}
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].metadata.get('section_type') == 'abstract'

    def test_bibliography_section_type_in_metadata(self, chunker):
        """Elements with metadata type='bibliography_entry' should propagate."""
        pages = [
            Page(
                page_number=1, width=1000, height=1000,
                elements=[
                    Element(
                        element_type='text',
                        content='Smith, J. (2024). A Study of Chunking. Journal of NLP, 42(1).',
                        bbox=BBox(0, 0, 1000, 100), page=1,
                        metadata={'type': 'bibliography_entry'}
                    ),
                ]
            )
        ]
        doc = Document(pages=pages, metadata={'extraction_method': 'standard'})
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].metadata.get('section_type') == 'bibliography_entry'


class TestSentenceDedup:
    """Parser-artifact deduplication.

    The parser (PaddleOCR-VL for PDFs, and sometimes others) occasionally
    emits the same content twice — cut-off then full version, duplicated
    headers/footers, figure captions repeated alongside body. The chunker
    deduplicates this before embedding + boundary detection so downstream
    LLM extraction doesn't get confused by apparent repetition.
    """

    def _make_sentence(self, text, idx=0):
        from smantic.chunker import Sentence
        return Sentence(
            text=text, start_idx=idx, end_idx=idx + len(text),
            page_num=1, element_type='prose', starts_paragraph=False,
            metadata={},
        )

    def test_exact_duplicate_dropped(self):
        from smantic.chunker import _dedup_sentences
        s1 = self._make_sentence("Assuming the rental price of the H800 GPU is $2 per GPU hour.")
        s2 = self._make_sentence("Then we describe the training setup.")
        s3 = self._make_sentence("Assuming the rental price of the H800 GPU is $2 per GPU hour.")
        result = _dedup_sentences([s1, s2, s3])
        assert len(result) == 2
        assert result[0].text.startswith("Assuming")
        assert result[1].text.startswith("Then we")

    def test_prefix_cut_off_dropped(self):
        from smantic.chunker import _dedup_sentences
        # Short cut-off sentence followed by the full version
        s1 = self._make_sentence("and generation length.")
        s2 = self._make_sentence("and generation length. We evaluate DeepSeek-V3 on benchmarks.")
        result = _dedup_sentences([s1, s2])
        assert len(result) == 1
        assert result[0].text.startswith("and generation length. We evaluate")

    def test_unrelated_sentences_preserved(self):
        from smantic.chunker import _dedup_sentences
        sentences = [
            self._make_sentence("The attention mechanism uses softmax."),
            self._make_sentence("Positional embeddings encode token order."),
            self._make_sentence("Training uses gradient descent."),
        ]
        result = _dedup_sentences(sentences)
        assert len(result) == 3

    def test_very_short_sentence_never_acts_as_prefix(self):
        from smantic.chunker import _dedup_sentences
        # "We." is a prefix of "We evaluate the model on..." but it's too
        # short to assume it's an artifact.
        s1 = self._make_sentence("We.")
        s2 = self._make_sentence("We evaluate the model on benchmarks.")
        result = _dedup_sentences([s1, s2])
        assert len(result) == 2

    def test_short_prefix_with_small_addition_preserved(self):
        from smantic.chunker import _dedup_sentences
        # Prefix rule requires the next sentence to add substantially
        # more content (> 20 chars). A near-clone shouldn't drop the first.
        s1 = self._make_sentence("The model is trained on tokens.")
        s2 = self._make_sentence("The model is trained on tokens!")  # trivially different
        result = _dedup_sentences([s1, s2])
        # s1 is not a prefix of s2 (differs at end), and they're not
        # exact match. Both keep.
        assert len(result) == 2

    def test_deepseek_v3_chunk_602_scenario(self):
        """The exact pattern observed in chunk 602 of the DeepSeek-V3 paper."""
        from smantic.chunker import _dedup_sentences
        sentences = [
            self._make_sentence("Table 1 | Training costs of DeepSeek-V3."),
            self._make_sentence("and generation length."),  # cut-off
            self._make_sentence("and generation length. We evaluate DeepSeek-V3 on a comprehensive array of benchmarks."),
            self._make_sentence("Lastly, we emphasize again the economical training costs."),
            self._make_sentence("Assuming the rental price of the H800 GPU is $2 per GPU hour."),
            self._make_sentence("Assuming the rental price of the H800 GPU is $2 per GPU hour."),  # exact dup
        ]
        result = _dedup_sentences(sentences)
        texts = [s.text for s in result]
        # Cut-off version dropped
        assert "and generation length." not in texts
        # Full version kept
        assert any("We evaluate DeepSeek-V3" in t for t in texts)
        # Duplicated rental-price sentence kept exactly once
        assert texts.count("Assuming the rental price of the H800 GPU is $2 per GPU hour.") == 1

    def test_empty_input(self):
        from smantic.chunker import _dedup_sentences
        assert _dedup_sentences([]) == []

    def test_single_sentence(self):
        from smantic.chunker import _dedup_sentences
        s = self._make_sentence("Just one sentence.")
        result = _dedup_sentences([s])
        assert len(result) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# ─── Heading-body bleed (PaddleOCR-VL bbox overlap artifact) ────────────────


class TestHeadingBodyBleedStripping:
    """Strip body-text bleed from section_header elements.

    PaddleOCR-VL's layout detector occasionally creates a section_header
    bbox that overlaps the first line of the next paragraph. The VLM
    transcribes both, the chunker concatenates, the duplication shows up
    in chunk content. Production sample (page 21 of the DeepSeek-V3
    paper)::

        [section_header] 3.5.1. Communication Hardware
                         In DeenSeek-V3, we implement the
        [text]           In DeepSeek-V3, we implement the overlap...
    """

    def _make_elem(self, et, content, page=1):
        return Element(
            element_type=et,
            content=content,
            bbox=BBox(0, 0, 1000, 100),
            page=page,
            confidence=0.9,
        )

    def test_strips_body_bleed_with_ocr_drift(self):
        """The exact production failure: heading bleeds in a near-prefix
        of the next text element with one-character OCR drift."""
        elements = [
            self._make_elem(
                'section_header',
                '3.5.1. Communication Hardware\nIn DeenSeek-V3, we implement the',
            ),
            self._make_elem(
                'text',
                'In DeepSeek-V3, we implement the overlap between computation '
                'and communication to hide the communication latency...',
            ),
        ]
        out = StructureAwareChunker._strip_heading_body_bleed(elements)
        assert out[0].content == '3.5.1. Communication Hardware'
        assert out[1].content.startswith('In DeepSeek-V3')

    def test_keeps_legitimate_multiline_heading(self):
        """A multi-line heading whose continuation is NOT the start of the
        next paragraph must survive untouched."""
        elements = [
            self._make_elem(
                'section_header',
                'Chapter 5\nThe Hidden Chamber',
            ),
            self._make_elem(
                'text',
                'It was a dark and stormy night when the protagonist...',
            ),
        ]
        out = StructureAwareChunker._strip_heading_body_bleed(elements)
        assert out[0].content == 'Chapter 5\nThe Hidden Chamber'

    def test_strips_when_exact_bleed(self):
        """No OCR drift — exact prefix match should also be stripped."""
        elements = [
            self._make_elem(
                'section_header',
                '4. Pre-Training\nIn this section we describe',
            ),
            self._make_elem(
                'text',
                'In this section we describe the data construction process '
                'and hyper-parameter choices.',
            ),
        ]
        out = StructureAwareChunker._strip_heading_body_bleed(elements)
        assert out[0].content == '4. Pre-Training'

    def test_singleline_heading_unchanged(self):
        """A heading with no newline is never modified."""
        elements = [
            self._make_elem('section_header', 'Introduction'),
            self._make_elem('text', 'Some unrelated body content here.'),
        ]
        out = StructureAwareChunker._strip_heading_body_bleed(elements)
        assert out[0].content == 'Introduction'

    def test_short_input_returns_unchanged(self):
        """A list of < 2 elements has nothing to compare against."""
        elements = [
            self._make_elem(
                'section_header',
                'Heading\nWith something appended',
            ),
        ]
        out = StructureAwareChunker._strip_heading_body_bleed(elements)
        assert out[0].content == 'Heading\nWith something appended'

    def test_skips_intermediate_empty_elements(self):
        """If the next non-empty element matches, strip — even if blank
        elements sit between heading and real body."""
        elements = [
            self._make_elem(
                'section_header',
                '3.5. Suggestions on Hardware Design\nBased on our implementation',
            ),
            self._make_elem('text', '   '),  # whitespace-only
            self._make_elem(
                'text',
                'Based on our implementation of the all-to-all communication '
                'and FP8 training scheme, we propose...',
            ),
        ]
        out = StructureAwareChunker._strip_heading_body_bleed(elements)
        assert out[0].content == '3.5. Suggestions on Hardware Design'


class TestTranscriptTimecodes:
    """PM41 — chunker propagates transcript_segment timecodes into
    chunk metadata so concepts can trace back to audio offsets.
    """

    def test_chunk_carries_aggregated_time_range(self, chunker):
        # Three transcript segments that should land in one chunk (small).
        elements = [
            Element(
                element_type='transcript_segment',
                content='First segment of the transcript discussion. ',
                bbox=BBox(0, 0, 1000, 1000),
                page=1,
                metadata={'start_time': 0.0, 'end_time': 30.0},
            ),
            Element(
                element_type='transcript_segment',
                content='Second segment continuing the same topic. ',
                bbox=BBox(0, 0, 1000, 1000),
                page=1,
                metadata={'start_time': 30.0, 'end_time': 60.0},
            ),
            Element(
                element_type='transcript_segment',
                content='Third segment wrapping up. ',
                bbox=BBox(0, 0, 1000, 1000),
                page=1,
                metadata={'start_time': 60.0, 'end_time': 90.0},
            ),
        ]
        doc = Document(
            pages=[Page(page_number=1, width=1000, height=1000, elements=elements)],
            metadata={'processor': 'asr_whisper'},
        )
        chunks = chunker.chunk_document(doc)
        assert chunks, "chunker produced no chunks"
        chunk_with_times = [c for c in chunks if c.metadata.get('start_time') is not None]
        assert chunk_with_times, "no chunk carries start_time — propagation broken"
        for c in chunk_with_times:
            assert c.metadata['start_time'] >= 0.0
            assert c.metadata['start_time'] <= 60.0
            assert c.metadata['end_time'] >= c.metadata['start_time']
            assert c.metadata['segment_count'] >= 1

    def test_non_transcript_chunks_have_no_time_metadata(self, chunker, sample_prose_document):
        # PDF-style docs (no transcript_segments) must NOT have spurious
        # start_time / end_time in chunk metadata.
        chunks = chunker.chunk_document(sample_prose_document)
        for c in chunks:
            assert 'start_time' not in c.metadata
            assert 'end_time' not in c.metadata
            assert 'segment_count' not in c.metadata


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def test_chunk_document_collapses_cross_element_span_dup(chunker):
    """The final chunk-content pass collapses an adjacent verbatim span
    repeat (the cross-element OCR-duplicate shape) that the per-sentence
    _dedup_sentences misses because the span crosses no sentence boundary."""
    span = "the rate depends on both nucleophile and substrate"  # 8 words, no period
    content = f"In an SN2 reaction {span} {span} concentration also matters for kinetics."
    doc = Document(
        pages=[Page(page_number=1, width=1000, height=1000, elements=[
            Element(element_type='text', content=content,
                           bbox=BBox(100, 200, 900, 320), page=1, confidence=0.9),
        ])],
        metadata={'source_file': 'dup.pdf', 'processor': 'test'},
    )
    chunks = chunker.chunk_document(doc)
    joined = " ".join(c.content for c in chunks if c.dominant_type == 'prose')
    assert joined.count(span) == 1, f"span not collapsed: {joined!r}"
