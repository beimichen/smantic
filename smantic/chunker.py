#!/usr/bin/env python3
"""
Structure-aware semantic chunker.

Performs structure-aware semantic chunking that respects element_type boundaries.
Uses ONNX sentence embeddings (all-MiniLM-L6-v2) for genuine semantic boundary
detection within prose runs.

Key features:
- Atomic block detection: code, table, formula, visual elements stay intact
- Parent-child splitting: large blocks split with hierarchical relationships
- Semantic prose chunking: real ONNX sentence embeddings for boundary detection
- Three-tier boundary detection: hard (structural) > soft (semantic) > emergency
- Overlap: configurable token overlap between consecutive chunks
- Block type tracking: dominant_type and has_* flags for retrieval
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .block_splitters import (
    code_splitter,
    formula_splitter,
    table_splitter,
)
from .embedder import OnnxSentenceEmbedder
from .ir import Document, Element

logger = logging.getLogger(__name__)


# Configuration constants
CHUNK_CONFIG = {
    # Prose chunks (target 200-500 tokens)
    "prose_target_tokens": 350,      # Ideal size (middle of 200-500 range)
    "prose_min_tokens": 200,         # Soft minimum for semantic boundaries
    "prose_max_tokens": 500,         # Soft maximum (can extend for atomic blocks)
    "prose_overlap_tokens": 50,      # ~15% overlap

    # Atomic block thresholds
    "block_large_threshold": 800,    # Split blocks larger than this

    # Code block children
    "code_child_target": 300,        # AST-split target
    "code_child_max": 500,           # Hard max for code children

    # Table block children
    "table_child_target": 350,       # Row-group target
    "table_row_threshold": 40,       # Split if >40 rows

    # Formula block children
    "formula_child_target": 200,     # Step-based split target

    # Semantic boundary threshold (cosine similarity)
    # Lower than the old 0.7 because real sentence embeddings are more
    # discriminative than token-frequency histograms.
    "boundary_threshold": 0.5,
}


class HeadingStack:
    """
    Maintains a hierarchical heading trail.

    When a level-N heading is pushed, all levels >= N are cleared before
    setting the new heading at level N.
    """

    def __init__(self):
        self._headings: dict[int, str] = {}  # level -> heading text

    def push(self, level: int, text: str) -> None:
        """Push a heading, clearing all deeper levels."""
        # Clear levels >= N
        for lvl in list(self._headings):
            if lvl >= level:
                del self._headings[lvl]
        self._headings[level] = text

    def get_trail(self) -> list[str]:
        """Return ordered list of headings from shallowest to deepest."""
        if not self._headings:
            return []
        return [self._headings[k] for k in sorted(self._headings)]

    def to_metadata(self) -> dict[str, Any]:
        """
        Return heading metadata dict, or ``{}`` when empty.

        Format::

            {
                "heading_trail": ["Chapter 1", "Background"],
                "heading_level": 2,
                "nearest_heading": "Background"
            }
        """
        trail = self.get_trail()
        if not trail:
            return {}
        deepest_level = max(self._headings)
        return {
            "heading_trail": list(trail),
            "heading_level": deepest_level,
            "nearest_heading": self._headings[deepest_level],
        }


def _normalize_for_dedup(text: str) -> str:
    """Collapse whitespace + lowercase for duplicate comparison."""
    return " ".join(text.lower().split())


def _dedup_sentences(sentences):
    """Drop parser-artifact duplicate sentences.

    Two drop rules:
      1. Strict-prefix: if sentence `s` is a prefix of the NEXT sentence and
         the next sentence adds substantially more content, drop `s`. This
         handles the "cut-off → full version" pattern from multi-column PDF
         parsing.
      2. Recent-identical: if normalized sentence text exactly matches a
         sentence within the last 8 sentences, drop it. Handles repeated
         content from figure captions, headers/footers, etc.

    Very short sentences (< 10 normalized chars) are never treated as
    prefixes (too easy to coincide with real sentence beginnings).
    """
    if len(sentences) < 2:
        return sentences

    WINDOW = 8
    PREFIX_MIN_CHARS = 10
    PREFIX_MIN_ADDED_CHARS = 20

    keep = [True] * len(sentences)
    recent = []  # list of (normalized_text, kept-index)

    for i, sent in enumerate(sentences):
        norm = _normalize_for_dedup(sent.text)

        # Rule 1: strict prefix of NEXT sentence (that also survives so far)
        if i + 1 < len(sentences) and keep[i + 1]:
            next_norm = _normalize_for_dedup(sentences[i + 1].text)
            if (len(norm) >= PREFIX_MIN_CHARS
                    and next_norm.startswith(norm)
                    and len(next_norm) >= len(norm) + PREFIX_MIN_ADDED_CHARS):
                keep[i] = False
                continue

        # Rule 2: exact match within the recent window
        dup = False
        for prev_norm, _ in recent[-WINDOW:]:
            if prev_norm == norm:
                dup = True
                break
        if dup:
            keep[i] = False
            continue

        recent.append((norm, i))

    return [s for i, s in enumerate(sentences) if keep[i]]


@dataclass
class Sentence:
    """Represents a single sentence with metadata."""
    text: str
    start_idx: int  # Position within the source element
    end_idx: int
    page_num: int
    element_type: str
    embedding: np.ndarray | None = None
    starts_paragraph: bool = False  # First sentence of a new element/paragraph
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Represents a semantic chunk with hierarchical metadata."""
    content: str
    token_count: int
    page_numbers: list[int]
    span_start: int
    span_end: int
    chunking_method: str = "semantic"

    # Hierarchical fields
    parent_chunk_id: int | None = None
    dominant_type: str = "prose"
    block_sequence: int | None = None

    # Content type flags
    has_code: bool = False
    has_math: bool = False
    has_table: bool = False

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Sequence assigned during chunking
    sequence: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the chunk to a plain JSON-ready dict."""
        return {
            "content": self.content,
            "token_count": self.token_count,
            "page_numbers": list(self.page_numbers),
            "span_start": self.span_start,
            "span_end": self.span_end,
            "chunking_method": self.chunking_method,
            "dominant_type": self.dominant_type,
            "parent_chunk_id": self.parent_chunk_id,
            "block_sequence": self.block_sequence,
            "has_code": self.has_code,
            "has_math": self.has_math,
            "has_table": self.has_table,
            "metadata": self.metadata,
            "sequence": self.sequence,
        }


class SentenceSegmenter:
    """Segments text into sentences using rule-based approach."""

    SENTENCE_ENDINGS = re.compile(r'([.!?]+)\s+')
    ABBREVIATIONS = {
        'dr', 'mr', 'mrs', 'ms', 'prof', 'sr', 'jr',
        'etc', 'vs', 'i.e', 'e.g', 'fig', 'vol',
        'no', 'nos', 'p', 'pp', 'ch', 'sec',
        # Multi-period abbreviations
        'u.s', 'u.k', 'e.u', 'ph.d', 'b.s', 'b.a', 'm.s', 'm.a',
        'inc', 'ltd', 'corp', 'dept', 'approx', 'govt',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug',
        'sep', 'oct', 'nov', 'dec',
    }
    # Match X.Y. or X.Y.Z. patterns (initials, acronyms like U.S.A.)
    MULTI_PERIOD_PATTERN = re.compile(r'\b(?:[A-Za-z]\.){2,}$')

    def segment(self, text: str) -> list[tuple[int, int]]:
        """
        Segment text into sentence boundaries.

        Returns:
            List of (start_idx, end_idx) tuples for each sentence
        """
        if not text.strip():
            return []

        sentences = []
        current_start = 0

        for match in self.SENTENCE_ENDINGS.finditer(text):
            end_idx = match.end()

            # Check if it's a known abbreviation
            before = text[max(0, match.start() - 10):match.start()].strip().lower()
            if any(before.endswith(abbr) for abbr in self.ABBREVIATIONS):
                continue

            # Check for multi-period abbreviations (e.g., U.S.A.)
            before_word = text[max(0, match.start() - 20):match.start() + 1]
            if self.MULTI_PERIOD_PATTERN.search(before_word):
                continue

            # Valid sentence boundary
            sentence_text = text[current_start:end_idx].strip()
            if sentence_text:
                sentences.append((current_start, end_idx))
                current_start = end_idx

        # Add remaining text as final sentence
        if current_start < len(text):
            remaining = text[current_start:].strip()
            if remaining:
                sentences.append((current_start, len(text)))

        # If no sentences found, treat entire text as one sentence
        if not sentences:
            sentences.append((0, len(text)))

        return sentences


class StructureAwareChunker:
    """
    Structure-aware semantic chunker that respects element_type boundaries.

    Algorithm:
    1. Classify elements into PROSE vs ATOMIC_BLOCK
    2. Group consecutive PROSE elements -> semantic chunking with ONNX embeddings
    3. Keep ATOMIC_BLOCK elements intact OR split with parent-child
    4. Produce chunks with proper dominant_type and parent links
    5. Three-tier boundary detection: hard (structural) > soft (semantic) > emergency
    6. Overlap between consecutive prose chunks for context continuity
    """

    def __init__(self, max_tokens: int = 500, overlap_tokens: int = 50,
                 model_dir: Path | None = None):
        """
        Initialize chunker.

        Args:
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap between chunks
            model_dir: Optional path to ONNX model directory
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.boundary_threshold = config.BOUNDARY_THRESHOLD

        # Initialize the ONNX sentence embedder (platform-agnostic)
        self.embedder = self._create_embedder(model_dir)
        self.segmenter = SentenceSegmenter()

        logger.info(f"StructureAwareChunker initialized (max_tokens={max_tokens}, "
                     f"overlap={overlap_tokens}, embeddings={'enabled' if self.embedder.available else 'disabled'})")

    @staticmethod
    def _create_embedder(model_dir: Path | None = None):
        """Create the ONNX sentence embedder (platform-agnostic)."""
        return OnnxSentenceEmbedder(model_dir)

    def release(self) -> None:
        """Release the embedder and its ONNX session to free memory."""
        if self.embedder is not None:
            self.embedder.release()

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.embedder.count_tokens(text)

    @staticmethod
    def _extract_heading_level(element: Element) -> int | None:
        """
        Extract heading level from an element, or None if not a heading.

        Handles conventions from all parsers:
        1. ``section_header_level_N`` → parse suffix (PDF, Markdown)
        2. ``metadata['level']`` → int (EPUB, LaTeX, ODF)
        3. ``metadata['style']`` → regex ``heading\\s*(\\d)`` (DOCX)
        4. Bare ``section_header`` → default to 1 (PPTX, RTF, source code)
        """
        etype = element.element_type

        # Parser title/subtitle elements -> heading levels 1/2
        if etype == 'title':
            return 1
        if etype == 'subtitle':
            return 2

        if not etype.startswith('section_header'):
            return None

        # 1. section_header_level_N
        m = re.match(r'^section_header_level_(\d+)$', etype)
        if m:
            return int(m.group(1))

        # 2. metadata['level']
        meta = element.metadata or {}
        if 'level' in meta:
            try:
                return int(meta['level'])
            except (ValueError, TypeError):
                pass

        # 3. metadata['style'] (DOCX: "Heading 2", "heading2", etc.)
        style = meta.get('style', '')
        if style:
            sm = re.search(r'heading\s*(\d)', style, re.IGNORECASE)
            if sm:
                return int(sm.group(1))

        # 4. Bare section_header
        return 1

    # Numbered section heading pattern (e.g. "3.2.1 Mixed Precision", "A.1 Results")
    _NUMBERED_HEADING_RE = re.compile(r'^\s*(?:\d+(\.\d+)*\.?|[A-Z](\.\d+)*\.?)\s+\S')

    @staticmethod
    def _validate_heading(text: str) -> bool:
        """
        Return True if text looks like a real section heading.

        Rejects sentence fragments the VLM parser misclassifies as
        section_header (e.g. "be efficiently implemented.").
        """
        text = text.strip()

        # Length bounds: too short or too long for a heading
        if len(text) < 3 or len(text) > 120:
            return False

        # Numbered section pattern: always valid
        if StructureAwareChunker._NUMBERED_HEADING_RE.match(text):
            return True

        # Reject sentence fragments: ends with lowercase letter + period
        if re.search(r'[a-z]\.\s*$', text):
            return False

        # Reject if contains mid-text sentence punctuation
        # (period + space + lowercase = prose, not heading)
        if re.search(r'\.\s+[a-z]', text):
            return False

        return True

    # Headings that mark backmatter sections with no study value.
    # All elements after these headings are skipped until a non-backmatter
    # heading of equal or shallower depth resets the flag.
    _BACKMATTER_HEADING_RE = re.compile(
        r'^(?:References|Bibliography|Works\s+Cited|Literature|Cited\s+Works'
        r'|Acknowledge?ments?'
        r'|Author\s+Contributions?'
        r'|Conflicts?\s+of\s+Interest'
        r'|Funding|Data\s+Availability'
        r'|Supplementary\s+Materials?'
        r'|Competing\s+Interests?'
        r'|Disclosure)\s*$',
        re.IGNORECASE,
    )

    # Element types that carry no semantic value as chunk content.
    # Parsers often extract these as literal text (e.g. "5", "Page 5")
    # which would otherwise pollute chunk content.
    _SKIP_ELEMENT_TYPES = {'page_header', 'page_footer', 'footnote_reference'}

    # Types that are semantically compatible and should NOT trigger hard
    # boundaries when transitioning between each other (e.g. text → list_item).
    _PROSE_COMPATIBLE_TYPES = {'text', 'list_item', 'paragraph', 'caption', 'footnote'}

    def _is_atomic_block(self, element: Element) -> bool:
        """Check if element is an atomic block that shouldn't be split."""
        return element.element_type in {'code', 'table', 'formula', 'picture', 'chart', 'figure'}

    # Patterns that indicate introductory text for a following block
    _INTRO_PATTERN = re.compile(
        r'(?::\s*$|as follows|below|following\s+(?:code|table|figure|chart|listing|example))',
        re.IGNORECASE,
    )
    _CAPTION_PATTERN = re.compile(
        r'^(?:Table|Figure|Fig\.|Chart|Listing|Eq\.)\s*\d',
        re.IGNORECASE,
    )

    def _extract_context_for_block(
        self,
        prose_buffer: list,
        block_element: Element,
    ) -> str:
        """
        Pop and return the last prose element if it is a caption or short
        intro text for the upcoming atomic block.

        Mutates *prose_buffer* (pops the last entry) when a match is found.
        Returns ``""`` when nothing qualifies.
        """
        if not prose_buffer:
            return ""

        last_elem, _heading_ctx = prose_buffer[-1]
        text = last_elem.content.strip()

        # Explicit VLM caption element
        if last_elem.element_type == 'caption':
            prose_buffer.pop()
            return text

        # Short text (≤ 60 tokens) matching intro or caption patterns
        if self.count_tokens(text) <= 60:
            if self._INTRO_PATTERN.search(text) or self._CAPTION_PATTERN.match(text):
                prose_buffer.pop()
                return text

        return ""

    @staticmethod
    def _strip_heading_body_bleed(
        elements: list[Element],
    ) -> list[Element]:
        """Strip body-text bleed from ``section_header`` elements.

        Some VLM/OCR layout parsers occasionally create a
        ``section_header`` bbox that overlaps with the first line of the
        paragraph immediately below. The parser transcribes both (the
        heading proper *and* the bled-in line) into the section_header
        element's ``content`` (separated by a newline). The same body
        line is also emitted as the start of the next ``text`` element,
        producing duplication like::

            [section_header] 3.5.1. Communication Hardware
                             In this section we describe the
            [text]           In this section we describe the overlap...

        The two transcripts often differ by a few characters because
        they're independent passes on overlapping crops (e.g. a single
        dropped or transposed letter, ``Documemt`` vs ``Document``),
        defeating exact-match dedup.

        Detection rule: when a ``section_header`` content has a newline
        and the post-newline portion is a near-prefix (>= 0.85
        character-similarity ratio over the same-length prefixes) of the
        next text-bearing element's content, strip the bled portion and
        keep only the heading proper. The threshold tolerates 1-2
        character OCR drift while being strict enough that legitimately
        multi-line headings (rare) survive.
        """
        if len(elements) < 2:
            return elements

        try:
            from difflib import SequenceMatcher
        except ImportError:
            return elements

        # Below this length the bled portion is too short to attribute
        # confidently, better to leave the heading alone than to mis-strip
        # a legitimate two-line heading like "Chapter 5\nThe Hidden Chamber".
        MIN_BLEED_LEN = 10
        # Character-level similarity ratio over the same-length prefixes.
        # 0.85 tolerates 1-2 char OCR drift (e.g. a single dropped or
        # transposed letter) over a ~30 char bleed; rejects unrelated
        # continuations.
        BLEED_RATIO_THRESHOLD = 0.85

        stripped = 0
        for i, elem in enumerate(elements):
            if elem.element_type != 'section_header':
                continue
            if '\n' not in elem.content:
                continue
            heading_part, _, body_bleed = elem.content.partition('\n')
            heading_part = heading_part.strip()
            body_bleed = body_bleed.strip()
            if not heading_part or len(body_bleed) < MIN_BLEED_LEN:
                continue

            # Find the next element with non-empty content.
            next_text = None
            for j in range(i + 1, len(elements)):
                cand = elements[j].content.strip()
                if cand:
                    next_text = cand
                    break
            if not next_text:
                continue

            # Compare bleed against the SAME-LENGTH prefix of next_text.
            # Fixed-window comparison (e.g. always 60 chars) systematically
            # underrates ratios when bleed is short (~30 chars), because
            # the unmatched tail of next_text dominates.
            probe_len = min(len(body_bleed), len(next_text))
            if probe_len < MIN_BLEED_LEN:
                continue
            ratio = SequenceMatcher(
                None,
                body_bleed[:probe_len].lower(),
                next_text[:probe_len].lower(),
            ).ratio()
            if ratio >= BLEED_RATIO_THRESHOLD:
                elem.content = heading_part
                stripped += 1

        if stripped:
            logger.info(
                "Stripped heading-body bleed from %d section_header element(s)",
                stripped,
            )

        return elements

    @staticmethod
    def _deduplicate_adjacent_elements(
        elements: list[Element],
    ) -> list[Element]:
        """
        Remove elements whose content is duplicated on an adjacent page.

        The VLM parser sometimes emits identical title/abstract text for
        both page 1 and page 2 (overlapping bounding boxes). This pass
        keeps the first occurrence and drops later duplicates.
        """
        if len(elements) < 2:
            return elements

        seen: dict[tuple, int] = {}   # (norm_content, element_type) -> page
        result = []
        removed = 0

        for elem in elements:
            key = (elem.content.strip().lower(), elem.element_type)
            prev_page = seen.get(key)

            if prev_page is not None and abs(elem.page - prev_page) <= 1:
                removed += 1
                continue

            seen[key] = elem.page
            result.append(elem)

        if removed:
            logger.info(f"Dedup removed {removed} duplicate elements across adjacent pages")

        return result

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """
        Chunk document using structure-aware semantic chunking.

        Args:
            doc: Docling document

        Returns:
            List of chunks with hierarchical relationships
        """
        # Extract all elements from document
        elements = []
        for page in doc.pages:
            elements.extend(page.elements)

        # Remove duplicate elements across adjacent pages (parser artifact:
        # VLM sometimes emits identical content for overlapping page regions)
        elements = self._deduplicate_adjacent_elements(elements)

        # Strip body-text bleed from section_header elements (parser
        # artifact: a VLM/OCR layout bbox can overlap the first line of
        # the next paragraph; the bled line then also appears as the next
        # text element, producing duplication in chunk content).
        elements = self._strip_heading_body_bleed(elements)

        logger.info(f"Chunking document with {len(elements)} elements")

        chunks = []
        prose_buffer = []  # List of (element, heading_meta_dict)
        heading_stack = HeadingStack()
        # (heading_text, page_number) tuples awaiting prepend
        pending_headings: list[tuple[str, int]] = []

        # Backmatter detection: skip references, acknowledgments, etc.
        in_backmatter = False
        backmatter_depth: int | None = None
        backmatter_element_count = 0

        def _flush_prose():
            """Flush prose buffer, prepending any pending heading texts."""
            nonlocal prose_buffer, pending_headings
            if not prose_buffer:
                return
            prose_chunks = self._semantic_chunk_prose(prose_buffer)
            self._prepend_heading_texts(prose_chunks,
                                        [text for text, _ in pending_headings])
            chunks.extend(prose_chunks)
            prose_buffer = []
            pending_headings = []

        for element in elements:
            # Skip elements with no semantic value (e.g. extracted page numbers)
            if element.element_type in self._SKIP_ELEMENT_TYPES:
                continue

            # Update heading stack before any branching
            level = self._extract_heading_level(element)
            if level is not None:
                if self._validate_heading(element.content.strip()):
                    heading_stack.push(level, element.content.strip())
                else:
                    # Demote misclassified sentence fragment to prose
                    logger.debug(f"Demoting invalid heading to prose: {element.content.strip()!r}")
                    level = None

            # --- Backmatter detection ---
            if level is not None:
                heading_text = element.content.strip()
                if self._BACKMATTER_HEADING_RE.match(heading_text):
                    if not in_backmatter:
                        _flush_prose()
                        in_backmatter = True
                        backmatter_depth = level
                        backmatter_element_count = 0
                        logger.info(f"Entering backmatter section: {heading_text!r}")
                    continue
                elif in_backmatter and level <= backmatter_depth:
                    # A non-backmatter heading at same or shallower depth
                    # exits backmatter (e.g., Appendix with real content
                    # after References)
                    logger.info(
                        f"Exiting backmatter after {backmatter_element_count} "
                        f"skipped elements, resuming at: {heading_text!r}"
                    )
                    in_backmatter = False
                    backmatter_depth = None
                    # Fall through to normal heading processing below

            if in_backmatter:
                backmatter_element_count += 1
                continue

            if self._is_atomic_block(element):
                # Extract caption/intro text BEFORE flushing prose
                context_text = self._extract_context_for_block(prose_buffer, element)
                _flush_prose()

                # Handle atomic block: inject heading context
                block_chunks = self._chunk_atomic_block(element)
                heading_meta = heading_stack.to_metadata()
                for c in block_chunks:
                    c.metadata.update(heading_meta)

                # Prepend caption/intro context to first block chunk
                if context_text and block_chunks:
                    first = block_chunks[0]
                    first.content = context_text + '\n\n' + first.content
                    first.token_count = self.count_tokens(first.content)
                elif context_text and not block_chunks and element.element_type in {'picture', 'chart', 'figure'}:
                    # Visual block returned [] but we have caption context
                    fallback_meta = {'element_type': element.element_type}
                    fallback_meta.update(heading_meta)
                    block_chunks = [Chunk(
                        content=context_text,
                        token_count=self.count_tokens(context_text),
                        page_numbers=[element.page],
                        span_start=0, span_end=len(context_text),
                        dominant_type='visual_block',
                        chunking_method='visual_reference',
                        metadata=fallback_meta,
                    )]

                chunks.extend(block_chunks)
                # Heading text consumed via heading_trail metadata
                pending_headings = []
            elif level is not None:
                # Section header: consumed by HeadingStack for metadata.
                # Flush preceding prose so it keeps pre-heading context,
                # then accumulate heading text to prepend into the first
                # content chunk that follows.
                _flush_prose()
                pending_headings.append((element.content.strip(), element.page))
            else:
                # Accumulate prose elements with current heading snapshot
                prose_buffer.append((element, heading_stack.to_metadata()))

        # Flush remaining prose
        _flush_prose()

        # Trailing headings with no following content: absorb into the
        # last chunk rather than creating a tiny standalone chunk.
        if pending_headings:
            trailing_text = '\n\n'.join(t for t, _ in pending_headings)
            if chunks:
                last = chunks[-1]
                last.content = last.content + '\n\n' + trailing_text
                last.token_count = self.count_tokens(last.content)
                logger.debug(
                    f"Absorbed {len(pending_headings)} trailing heading(s) "
                    f"into last chunk: {[t for t, _ in pending_headings]}"
                )
            else:
                logger.debug(
                    f"Dropping {len(pending_headings)} trailing heading(s) "
                    f"(no chunks to absorb into): {[t for t, _ in pending_headings]}"
                )

        if in_backmatter and backmatter_element_count > 0:
            logger.info(
                f"Backmatter filtering skipped {backmatter_element_count} elements"
            )

        # Post-chunking merge pass: absorb small chunks into neighbours
        chunks = self._merge_small_chunks(chunks)

        # Collapse adjacent verbatim span repeats in the ASSEMBLED chunk
        # content. VLM/OCR parsers emit duplicate layout-detection elements
        # (overlapping bboxes / multi-column reading order) whose text the
        # element-assembly above concatenates (joined by newlines), so the
        # same span lands twice inside one chunk. The per-region collapse in
        # the parser can't see across elements; _dedup_sentences only catches
        # exact single-sentence repeats within an 8-sentence window. This
        # final pass catches the cross-element case (~98% of observed dups).
        # Prose only; tables/formulas/code legitimately repeat tokens.
        from .text_dedup import collapse_repeated_spans
        collapsed_chunks = 0
        for chunk in chunks:
            if chunk.dominant_type != "prose":
                continue
            new_content, removed = collapse_repeated_spans(chunk.content)
            if removed:
                chunk.content = new_content
                collapsed_chunks += 1
        if collapsed_chunks:
            logger.info(
                "Collapsed duplicated spans in %d/%d prose chunks "
                "(cross-element OCR repeats)", collapsed_chunks, len(chunks),
            )

        # Assign global sequence numbers
        for i, chunk in enumerate(chunks):
            chunk.sequence = i

        if chunks:
            avg_tokens = sum(c.token_count for c in chunks) / len(chunks)
            logger.info(f"Created {len(chunks)} chunks (avg {avg_tokens:.0f} tokens)")
        else:
            logger.info("Created 0 chunks (empty document)")
        return chunks

    def _semantic_chunk_prose(self, elements) -> list[Chunk]:
        """
        Apply semantic chunking to prose elements.
        Uses ONNX sentence embeddings for genuine boundary detection.

        Args:
            elements: Either List[Element] (legacy) or
                      List[Tuple[Element, Dict]] with heading context.
        """
        # Normalise input: accept both legacy and new tuple format
        normalised: list[tuple[Element, dict[str, Any]]] = []
        for item in elements:
            if isinstance(item, tuple):
                normalised.append(item)
            else:
                normalised.append((item, {}))

        # Extract sentences from prose elements, tracking paragraph boundaries
        sentences = []
        doc_offset = 0

        for element, heading_ctx in normalised:
            text = element.content.strip()
            if not text:
                continue

            boundaries = self.segmenter.segment(text)
            is_first_sentence = True

            # Speaker notes (PPTX) should flow with slide content,
            # not trigger a paragraph break.
            is_speaker_notes = bool(
                element.metadata and element.metadata.get('is_speaker_notes')
            )
            elem_meta = element.metadata or {}

            for start, end in boundaries:
                sentence_text = text[start:end].strip()
                if not sentence_text:
                    continue

                # Transcript timecode propagation. When a sentence
                # originates from a transcript_segment element, capture
                # the segment's start_time / end_time / words so the
                # chunk that absorbs this sentence can later aggregate
                # the time range. ASR writes these as top-level dict
                # keys; Element.from_dict migrates them into
                # element.metadata. Field is None on non-transcript
                # paths (PDF, EPUB, etc.); chunk-level aggregation skips
                # when no sentence carries a start_time.
                sent_meta: dict[str, Any] = {
                    'bbox': element.bbox.to_dict() if element.bbox else None,
                    'confidence': element.confidence,
                    'heading_context': heading_ctx,
                    'section_type': elem_meta.get('type'),
                }
                if 'start_time' in elem_meta:
                    sent_meta['start_time'] = elem_meta.get('start_time')
                if 'end_time' in elem_meta:
                    sent_meta['end_time'] = elem_meta.get('end_time')
                sentence = Sentence(
                    text=sentence_text,
                    start_idx=doc_offset + start,
                    end_idx=doc_offset + end,
                    page_num=element.page,
                    element_type=element.element_type,
                    starts_paragraph=is_first_sentence and not is_speaker_notes,
                    metadata=sent_meta,
                )
                sentences.append(sentence)
                is_first_sentence = False

            doc_offset += len(text) + 1  # +1 for element separator

        if not sentences:
            return []

        # Parser-artifact dedup: VLM/OCR parsers sometimes emit the same
        # text twice (a cut-off version followed by the full version,
        # multi-column text picked up by both columns, or figure captions
        # duplicated alongside body prose). This shows up as visible
        # duplication inside chunks (e.g. a cut-off span immediately
        # followed by its full version). Drop these before embedding /
        # boundary detection so they don't inflate chunk size or pollute
        # chunk content.
        sentences = _dedup_sentences(sentences)

        # Batch-embed all sentences at once (much faster than one-by-one)
        sentence_texts = [s.text for s in sentences]
        embeddings = self.embedder.embed_sentences(sentence_texts)
        for i, sentence in enumerate(sentences):
            sentence.embedding = embeddings[i]

        # Detect three-tier boundaries
        boundaries = self._detect_boundaries(sentences)

        # Form chunks with overlap
        return self._form_prose_chunks(sentences, boundaries)

    def _detect_boundaries(self, sentences: list[Sentence]) -> list[tuple[int, str]]:
        """
        Detect boundaries using three-tier hierarchy.

        Returns:
            List of (sentence_index, boundary_type) where boundary_type is:
            - "hard": section headers, element-type transitions (always break)
            - "soft": semantic similarity drop (prefer to break)
        """
        if len(sentences) <= 1:
            return [(0, "hard")]

        boundaries = [(0, "hard")]  # Document start is always a boundary

        for i in range(1, len(sentences)):
            prev_sent = sentences[i - 1]
            curr_sent = sentences[i]

            # Hard boundary: section header (also matches section_header_level_N)
            if curr_sent.element_type.startswith('section_header'):
                boundaries.append((i, "hard"))
                continue

            # Hard boundary: element type change, unless both types are
            # prose-compatible (e.g. text → list_item, text → paragraph).
            if prev_sent.element_type != curr_sent.element_type:
                both_compatible = (
                    prev_sent.element_type in self._PROSE_COMPATIBLE_TYPES
                    and curr_sent.element_type in self._PROSE_COMPATIBLE_TYPES
                )
                if not both_compatible:
                    boundaries.append((i, "hard"))
                    continue
                # Prose-compatible: fall through to paragraph/semantic checks

            # Hard/soft boundary: new paragraph from a different element
            if curr_sent.starts_paragraph:
                # Downgrade to soft when both types are prose-compatible,
                # whether same type (list_item → list_item) or different
                # (text → list_item), so short intro text + lists merge.
                if (curr_sent.element_type in self._PROSE_COMPATIBLE_TYPES
                        and prev_sent.element_type in self._PROSE_COMPATIBLE_TYPES):
                    boundaries.append((i, "soft"))
                else:
                    boundaries.append((i, "hard"))
                continue

            # Soft boundary: semantic similarity drop
            similarity = self._compute_similarity(prev_sent.embedding, curr_sent.embedding)
            if similarity < self.boundary_threshold:
                boundaries.append((i, "soft"))

        return boundaries

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        if emb1 is None or emb2 is None:
            return 0.5

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def _form_prose_chunks(
        self,
        sentences: list[Sentence],
        boundaries: list[tuple[int, str]],
    ) -> list[Chunk]:
        """
        Form chunks respecting token limits, boundary hierarchy, and overlap.

        Boundary hierarchy:
        - Hard boundaries: always start a new chunk (even if current is small)
        - Soft boundaries: start new chunk if current has >= min_tokens
        - Emergency: split at any sentence when chunk exceeds max_tokens
        """
        chunks = []
        target_tokens = CHUNK_CONFIG["prose_target_tokens"]
        min_tokens = CHUNK_CONFIG["prose_min_tokens"]
        max_tokens = self.max_tokens

        # Build a set for fast boundary lookup
        boundary_map: dict[int, str] = {idx: btype for idx, btype in boundaries}

        i = 0
        while i < len(sentences):
            current_sentences: list[Sentence] = []
            current_tokens = 0

            # Add sentences until we hit a reason to stop
            while i < len(sentences):
                sentence = sentences[i]
                sentence_tokens = self.count_tokens(sentence.text)

                # Check if this sentence starts a boundary (and we already have content)
                if current_sentences and i in boundary_map:
                    btype = boundary_map[i]
                    if btype == "hard":
                        # Always break at hard boundaries
                        break
                    elif btype == "soft" and current_tokens >= min_tokens:
                        # Break at soft boundaries if we have enough content
                        break

                # Emergency: would exceed max_tokens
                if current_tokens + sentence_tokens > max_tokens and current_sentences:
                    break

                current_sentences.append(sentence)
                current_tokens += sentence_tokens
                i += 1

                # If we've reached target and next is a boundary, stop
                if current_tokens >= target_tokens and i in boundary_map:
                    break

            # Create chunk from accumulated sentences
            if current_sentences:
                # Join text preserving paragraph structure
                content = self._join_sentences(current_sentences)
                page_nums = sorted(set(s.page_num for s in current_sentences))
                # Count tokens on joined content (includes join whitespace)
                actual_token_count = self.count_tokens(content)

                # Extract heading context from the first sentence.
                # Section headers always cause hard boundaries, so a chunk
                # never spans across a heading change.
                heading_ctx = current_sentences[0].metadata.get('heading_context', {})

                chunk = Chunk(
                    content=content,
                    token_count=actual_token_count,
                    page_numbers=page_nums,
                    span_start=current_sentences[0].start_idx,
                    span_end=current_sentences[-1].end_idx,
                    chunking_method="semantic",
                    dominant_type="prose",
                    has_code=False,
                    has_math=False,
                    has_table=False,
                    metadata=dict(heading_ctx),
                )

                # Propagate section_type when all sentences share the same one
                section_types = {
                    s.metadata.get('section_type')
                    for s in current_sentences
                    if s.metadata.get('section_type')
                }
                if len(section_types) == 1:
                    chunk.metadata['section_type'] = section_types.pop()

                # Transcript timecode aggregation. If any
                # sentence in this chunk originated from a
                # transcript_segment (carries start_time in its
                # metadata), aggregate the chunk's time range:
                #   start_time = earliest sentence start
                #   end_time   = latest sentence end (falls back to
                #                last start_time when end_time is None)
                #   segment_count = number of distinct
                #                   transcript_segments contributing
                # Lets a consumer map each chunk back to the audio
                # offset it came from.
                transcript_starts: list[float] = []
                transcript_ends: list[float] = []
                for s in current_sentences:
                    st = s.metadata.get('start_time')
                    if isinstance(st, (int, float)):
                        transcript_starts.append(float(st))
                    en = s.metadata.get('end_time')
                    if isinstance(en, (int, float)):
                        transcript_ends.append(float(en))
                if transcript_starts:
                    chunk.metadata['start_time'] = min(transcript_starts)
                    if transcript_ends:
                        chunk.metadata['end_time'] = max(transcript_ends)
                    # Count distinct transcript_segments via unique
                    # (start_time, end_time) pairs across the chunk's
                    # sentences. Sentences split from the same segment
                    # share the same pair.
                    seg_keys = {
                        (s.metadata.get('start_time'), s.metadata.get('end_time'))
                        for s in current_sentences
                        if isinstance(s.metadata.get('start_time'), (int, float))
                    }
                    chunk.metadata['segment_count'] = len(seg_keys)

                chunks.append(chunk)

                # Overlap: rewind so the next chunk shares trailing sentences.
                # Skip overlap across hard boundaries to prevent degenerate
                # micro-chunks (rewind would re-hit the same boundary).
                if self.overlap_tokens > 0 and i < len(sentences) and boundary_map.get(i) != "hard":
                    overlap_tokens_sum = 0
                    rewind = 0
                    for j in range(len(current_sentences) - 1, 0, -1):
                        stokens = self.count_tokens(current_sentences[j].text)
                        if overlap_tokens_sum + stokens > self.overlap_tokens:
                            break
                        overlap_tokens_sum += stokens
                        rewind += 1
                    if rewind > 0:
                        i -= rewind

        return chunks

    @staticmethod
    def _join_sentences(sentences: list[Sentence]) -> str:
        """
        Join sentences preserving paragraph structure.

        Uses "\\n\\n" between paragraphs (starts_paragraph=True) and
        " " within the same paragraph.
        """
        if not sentences:
            return ""

        parts = [sentences[0].text]
        for sentence in sentences[1:]:
            if sentence.starts_paragraph:
                parts.append("\n\n" + sentence.text)
            else:
                parts.append(" " + sentence.text)
        return "".join(parts)

    def _prepend_heading_texts(self, chunks: list[Chunk], heading_texts: list[str]) -> None:
        """
        Prepend accumulated heading texts to the first chunk's content.

        This integrates section header text (e.g. "Chapter 1") into the
        first content chunk that follows, so the heading is searchable
        without creating a wasteful standalone heading chunk.

        Some parsers (notably docling for academic papers) include the
        heading text as the leading paragraph of the next element's body
        even after we've captured it in heading_trail metadata. Without
        deduplication, prepending then produces ``"Heading\\n\\nHeading\\n
        body"`` (the heading appearing twice in the chunk content). That
        duplicated signal wastes tokens and skews any downstream signal
        computed over the chunk text. Strip any duplicate occurrences from
        the leading body before prepending.
        """
        if not heading_texts or not chunks:
            return
        first = chunks[0]
        body = first.content

        # Peel off the heading text(s) only when they appear at the very
        # start of the body followed by a newline, that's the signature of
        # the parser duplicating the heading as a leading paragraph. We do
        # NOT strip when the heading is followed by a space (e.g. body
        # "Background information goes here..." with heading "Background"
        # is real prose, not duplication; the heading is a substring of a
        # longer word).
        leading_ws_len = len(body) - len(body.lstrip())
        after_ws = body[leading_ws_len:]
        for ht in heading_texts:
            if not ht:
                continue
            # Strip only on newline boundary or exact equality.
            if after_ws.startswith(ht + '\n') or after_ws == ht:
                remainder = after_ws[len(ht):].lstrip('\n').lstrip(' ').lstrip('\n')
                after_ws = remainder

        prefix = '\n\n'.join(heading_texts) + '\n\n'
        first.content = prefix + after_ws
        first.token_count = self.count_tokens(first.content)

    # Hard minimum tokens for a chunk to be useful. Chunks below this are
    # merged into neighbours. 100 tokens ~= 75 words, below this a chunk
    # usually lacks enough context to stand alone for retrieval or
    # downstream processing.
    _MERGE_MIN_TOKENS = 100

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Merge chunks below _MERGE_MIN_TOKENS into their nearest neighbour.

        Rules:
        - Never merge across dominant_type boundaries (prose ≠ table_block)
        - Never merge parent/child chunks (structural relationships)
        - Prefer merging backward (into predecessor); fall back to forward
        - For prose runts separated from prose by an atomic block, skip past
          the atomic block to find a prose neighbour
        - Cap merged result at max_tokens to prevent oversized chunks
        - Visual blocks are exempt (captions can be legitimately short)
        """
        if not chunks:
            return chunks

        merged: list[Chunk] = []

        for i, chunk in enumerate(chunks):
            # Visual blocks and child chunks are exempt from merging
            if (chunk.token_count >= self._MERGE_MIN_TOKENS
                    or chunk.dominant_type == 'visual_block'
                    or chunk.parent_chunk_id is not None
                    or chunk.metadata.get('has_children')):
                merged.append(chunk)
                continue

            # Try merging backward into nearest same-type predecessor.
            # Skip past intervening atomic blocks (e.g., prose-formula-prose)
            # so a tiny prose chunk after a formula merges with the prose before it.
            merge_target = None
            for j in range(len(merged) - 1, -1, -1):
                candidate = merged[j]
                if candidate.dominant_type == chunk.dominant_type:
                    if (candidate.token_count + chunk.token_count <= self.max_tokens
                            and not candidate.metadata.get('has_children')):
                        merge_target = candidate
                    break
                # Only skip past atomic blocks, not other prose chunks
                if candidate.dominant_type not in ('code_block', 'table_block', 'formula_block', 'visual_block'):
                    break

            if merge_target is not None:
                merge_target.content = merge_target.content + '\n\n' + chunk.content
                merge_target.token_count = self.count_tokens(merge_target.content)
                merge_target.page_numbers = sorted(
                    set(merge_target.page_numbers + chunk.page_numbers)
                )
                merge_target.span_end = chunk.span_end
                continue

            # Try merging forward into nearest same-type successor
            for j in range(i + 1, len(chunks)):
                succ = chunks[j]
                if succ.dominant_type == chunk.dominant_type:
                    if (chunk.token_count + succ.token_count <= self.max_tokens
                            and not succ.metadata.get('has_children')
                            and succ.parent_chunk_id is None):
                        succ.content = chunk.content + '\n\n' + succ.content
                        succ.token_count = self.count_tokens(succ.content)
                        succ.page_numbers = sorted(
                            set(chunk.page_numbers + succ.page_numbers)
                        )
                        succ.span_start = chunk.span_start
                        merge_target = succ
                    break
                if succ.dominant_type not in ('code_block', 'table_block', 'formula_block', 'visual_block'):
                    break

            if merge_target is not None:
                continue

            # Can't merge, keep as-is
            merged.append(chunk)

        removed = len(chunks) - len(merged)
        if removed:
            logger.info(f"Merge pass absorbed {removed} small chunks (<{self._MERGE_MIN_TOKENS} tokens)")

        return merged

    def _chunk_atomic_block(self, element: Element) -> list[Chunk]:
        """
        Handle an atomic block: keep intact or split with parent-child.

        Visual blocks (picture/chart/figure) create chunks with searchable
        captions/alt_text. They act as document separators AND provide
        text retrieval via their descriptive metadata.
        """
        # Visual blocks: create chunk only if there is searchable content.
        if element.element_type in {'picture', 'chart', 'figure'}:
            caption = element.metadata.get('caption', '') if element.metadata else ''
            alt_text = element.metadata.get('alt_text', '') if element.metadata else ''
            if caption.strip() or alt_text.strip():
                return [self._create_visual_block_chunk(element)]
            else:
                logger.debug(
                    "Skipping visual block with no caption/alt_text (page %d, type=%s)",
                    element.page, element.element_type
                )
                return []

        token_count = self.count_tokens(element.content)

        if token_count <= CHUNK_CONFIG["block_large_threshold"]:
            # Small block: single chunk
            return [self._create_block_chunk(element)]

        # Large block: create parent + children (Phase 2)
        logger.info(f"Large {element.element_type} block ({token_count} tokens) - splitting with parent-child")
        if element.element_type == 'code':
            return self._split_code_block(element)
        elif element.element_type == 'table':
            return self._split_table_block(element)
        elif element.element_type == 'formula':
            return self._split_formula_block(element)
        else:
            # Fallback: single chunk even if large
            return [self._create_block_chunk(element)]

    def _create_block_chunk(self, element: Element) -> Chunk:
        """Create a single chunk for an intact atomic block."""
        type_map = {
            'code': 'code_block',
            'table': 'table_block',
            'formula': 'formula_block',
        }

        return Chunk(
            content=element.content,
            token_count=self.count_tokens(element.content),
            page_numbers=[element.page],
            span_start=0,
            span_end=len(element.content),
            dominant_type=type_map.get(element.element_type, 'mixed'),
            chunking_method='atomic_block',
            has_code=element.element_type == 'code',
            has_math=element.element_type == 'formula',
            has_table=element.element_type == 'table',
            metadata={'element_type': element.element_type}
        )

    def _create_visual_block_chunk(self, element: Element) -> Chunk:
        """Create a chunk for visual elements with searchable caption."""
        caption = element.metadata.get('caption', '') if element.metadata else ''
        alt_text = element.metadata.get('alt_text', '') if element.metadata else ''

        # Combine caption + alt_text for searchable content
        searchable_content = '\n'.join(filter(None, [caption, alt_text])).strip()

        return Chunk(
            content=searchable_content,
            token_count=self.count_tokens(searchable_content) if searchable_content else 0,
            page_numbers=[element.page],
            span_start=0,
            span_end=len(searchable_content),
            dominant_type='visual_block',
            chunking_method='visual_reference',
            has_code=False,
            has_math=False,
            has_table=False,
            metadata={
                'element_type': element.element_type,
                'caption': caption,
                'alt_text': alt_text,
                'skip_text_indexing': not bool(searchable_content),
            }
        )

    def _split_code_block(self, element: Element) -> list[Chunk]:
        """Split large code block by AST boundaries."""
        code = element.content
        language = self._detect_language(code, element.metadata)

        if language == 'python':
            children_content = code_splitter.split_python_code(code)
        elif language in {'javascript', 'typescript'}:
            children_content = code_splitter.split_javascript_code(code)
        else:
            children_content = code_splitter._split_code_heuristic(code)

        if len(children_content) == 1 and children_content[0][1].get('type') == 'module':
            return [self._create_block_chunk(element)]

        parent = self._create_block_chunk(element)
        parent.metadata['has_children'] = True
        parent.metadata['child_count'] = len(children_content)

        children = []
        for i, (content, context) in enumerate(children_content):
            child_tokens = self.count_tokens(content)
            child = Chunk(
                content=content,
                token_count=child_tokens,
                page_numbers=[element.page],
                span_start=context.get('lineno', 0),
                span_end=context.get('end_lineno', len(content)),
                dominant_type='code_block',
                chunking_method='ast_split',
                parent_chunk_id=None,  # Set after parent DB insert
                block_sequence=i,
                has_code=True,
                has_math=False,
                has_table=False,
                metadata={
                    'ast_node': context.get('type'),
                    'name': context.get('name'),
                    'start_line': context.get('lineno'),
                    'end_line': context.get('end_lineno'),
                    'language': language
                }
            )
            children.append(child)

        return [parent] + children

    def _split_table_block(self, element: Element) -> list[Chunk]:
        """Split large table by row groups with headers repeated."""
        content = element.content
        metadata = element.metadata or {}

        children_content = table_splitter.split_table(
            content=content,
            target_tokens=CHUNK_CONFIG["table_child_target"],
            token_counter=self.count_tokens,
            metadata=metadata
        )

        if len(children_content) == 1:
            return [self._create_block_chunk(element)]

        parent = self._create_block_chunk(element)
        parent.metadata['has_children'] = True
        parent.metadata['child_count'] = len(children_content)
        parent.metadata['row_count'] = len(content.split('\n')) - 2

        children = []
        for i, (child_content, context) in enumerate(children_content):
            child_tokens = self.count_tokens(child_content)
            child = Chunk(
                content=child_content,
                token_count=child_tokens,
                page_numbers=[element.page],
                span_start=context.get('row_start', 0),
                span_end=context.get('row_end', 0),
                dominant_type='table_block',
                chunking_method='row_group',
                parent_chunk_id=None,
                block_sequence=i,
                has_code=False,
                has_math=False,
                has_table=True,
                metadata={
                    'row_start': context.get('row_start'),
                    'row_end': context.get('row_end'),
                    'column_headers': context.get('column_headers', ''),
                    'table_caption': metadata.get('caption', '')
                }
            )
            children.append(child)

        return [parent] + children

    def _split_formula_block(self, element: Element) -> list[Chunk]:
        """Split large derivation/equation sequence by environment or step."""
        content = element.content

        children_content = formula_splitter.split_formula(content)

        if len(children_content) == 1 and children_content[0][1].get('split_method') == 'none':
            return [self._create_block_chunk(element)]

        parent = self._create_block_chunk(element)
        parent.metadata['has_children'] = True
        parent.metadata['child_count'] = len(children_content)

        children = []
        for i, (child_content, context) in enumerate(children_content):
            child_tokens = self.count_tokens(child_content)
            child = Chunk(
                content=child_content,
                token_count=child_tokens,
                page_numbers=[element.page],
                span_start=context.get('span_start', 0),
                span_end=context.get('span_end', len(child_content)),
                dominant_type='formula_block',
                chunking_method='env_split',
                parent_chunk_id=None,
                block_sequence=i,
                has_code=False,
                has_math=True,
                has_table=False,
                metadata={
                    'environment': context.get('env'),
                    'step': context.get('step'),
                    'split_method': context.get('split_method'),
                    'char_start': context.get('span_start'),
                    'char_end': context.get('span_end')
                }
            )
            children.append(child)

        return [parent] + children

    def _detect_language(self, code: str, metadata: dict | None = None) -> str:
        """Detect programming language from code content or metadata."""
        if metadata and 'language' in metadata:
            lang = metadata['language'].lower()
            if lang in {'python', 'py'}:
                return 'python'
            elif lang in {'javascript', 'js', 'jsx'}:
                return 'javascript'
            elif lang in {'typescript', 'ts', 'tsx'}:
                return 'typescript'

        code_lower = code.lower()
        # Check Python first: 'def ', 'class ...:', ':\n' are unambiguous Python
        # syntax that won't appear in JS/TS. Checking JS first would
        # false-positive on shared keywords like 'import'/'from', so
        # Python-first is safer and avoids false positives from English words
        # like "function" in docstrings.
        if any(pattern in code for pattern in ['def ', 'class ', ':\n']):
            return 'python'
        if any(pattern in code_lower for pattern in ['function ', 'const ', 'let ', 'var ', '=>', 'export ']):
            if any(pattern in code for pattern in [': string', ': number', 'interface ', 'type ']):
                return 'typescript'
            return 'javascript'

        return 'unknown'
