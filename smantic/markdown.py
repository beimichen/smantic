"""
Markdown -> Element parser.

Turns a Markdown (or Markdown-ish) document into a flat list of typed
``Element`` objects in reading order, ready for the chunker. It handles the
block constructs that matter for chunking:

  * ATX (``# ``) and setext (``===`` / ``---``) headings -> section_header
  * fenced code (``` ``` ``` / ``~~~``)                  -> code (+ language)
  * pipe tables (header + ``|---|`` separator)           -> table
  * display math (``$$ ... $$`` / ``\\[ ... \\]``)         -> formula
  * bullet / ordered lists                               -> list_item
  * standalone images (``![alt](url)``)                  -> picture
  * block quotes + prose paragraphs                      -> text

Inline formatting is left as-is; the chunker works on text, not a rendered AST.
This is deliberately small, dependency-free (stdlib ``re`` only), and not a
CommonMark implementation. Its one job is to hand the chunker correctly typed,
correctly ordered blocks.
"""

import re

from .ir import BBox, Element

# Every block gets the full-page bbox: Markdown has no spatial layout, and the
# chunker only uses bboxes opportunistically (for metadata).
_FULL_BBOX = BBox(0, 0, 1000, 1000)

_ATX_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_ORDERED_RE = re.compile(r"^(\s*)\d+[.)]\s+(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$")
_PIPE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")  # a GFM-style row: starts and ends with |
_IMAGE_ONLY_RE = re.compile(r"^\s*!\[([^\]]*)\]\([^)]*\)\s*$")
_HR_RE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_SETEXT_RE = re.compile(r"^\s*(=+|-+)\s*$")


def parse_markdown(text: str, *, page: int = 1) -> list[Element]:
    """Parse Markdown ``text`` into ``Element`` objects in reading order."""
    if not text or not text.strip():
        return []

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    elements: list[Element] = []
    n = len(lines)
    i = 0

    def emit(etype: str, content: str, metadata: dict | None = None) -> None:
        content = content.strip()
        if content or etype == "picture":
            elements.append(Element(etype, content, _FULL_BBOX, page, 1.0, metadata))

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Blank line: nothing to emit.
        if not stripped:
            i += 1
            continue

        # Fenced code block.
        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(2)
            fence_char = marker[0]
            info = fence.group(3).strip()
            lang = info.split()[0] if info else ""
            body: list[str] = []
            i += 1
            while i < n and not _is_closing_fence(lines[i], fence_char, len(marker)):
                body.append(lines[i])
                i += 1
            i += 1  # consume the closing fence (or run off the end)
            emit("code", "\n".join(body), {"language": lang} if lang else None)
            continue

        # ATX heading.
        atx = _ATX_RE.match(line)
        if atx:
            emit("section_header", atx.group(2), {"level": len(atx.group(1))})
            i += 1
            continue

        # Display math: $$ ... $$  or  \[ ... \]
        if stripped.startswith("$$"):
            content, i = _consume_delimited(lines, i, "$$", "$$")
            emit("formula", content)
            continue
        if stripped.startswith("\\["):
            content, i = _consume_delimited(lines, i, "\\[", "\\]")
            emit("formula", content)
            continue

        # Standalone image.
        img = _IMAGE_ONLY_RE.match(line)
        if img:
            caption = img.group(1).strip()
            emit("picture", caption, {"caption": caption} if caption else None)
            i += 1
            continue

        # Pipe table: a GFM header + |---| separator, OR a run of two or more
        # pipe-delimited rows. The second form keeps a separator-less table (as a
        # VLM can emit) grouped as one atomic block instead of shredding it into
        # prose. A lone pipe in a sentence stays prose (needs >=2 |...| rows).
        if _is_table_start(lines, i):
            table_lines = [lines[i]]
            i += 1
            while i < n and lines[i].strip() and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            emit("table", "\n".join(table_lines))
            continue

        # Thematic break (and a bare setext underline with no paragraph above):
        # carries no chunkable content.
        if _HR_RE.match(line) or _SETEXT_RE.match(line):
            i += 1
            continue

        # List block (bullet or ordered): one list_item per marker, continuation
        # lines folded into the current item.
        if _BULLET_RE.match(line) or _ORDERED_RE.match(line):
            i = _consume_list(lines, i, emit)
            continue

        # Block quote: strip the markers, treat the run as a prose paragraph.
        if stripped.startswith(">"):
            quote: list[str] = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            emit("text", "\n".join(quote))
            continue

        # Paragraph: accumulate until a blank line or a new block starts. A
        # setext underline directly below turns the paragraph into a heading.
        para = [line]
        i += 1
        became_heading = False
        while i < n:
            nxt = lines[i]
            s = nxt.strip()
            if not s:
                break
            # A setext underline directly under paragraph text turns the whole
            # paragraph into a heading. In this context '---' is setext H2, not a
            # thematic break (which only applies after a blank line), so we do
            # NOT exclude it here even though it also matches the HR pattern.
            if _SETEXT_RE.match(nxt):
                level = 1 if s[0] == "=" else 2
                emit("section_header", "\n".join(para), {"level": level})
                i += 1
                became_heading = True
                break
            if _starts_block(lines, i):
                break
            para.append(nxt)
            i += 1
        if not became_heading:
            emit("text", "\n".join(para))

    return elements


# Backwards-compatible alias.
def parse_full_page_output(text: str, page: int) -> list[Element]:
    """Alias of :func:`parse_markdown` (positional ``page`` for older callers)."""
    return parse_markdown(text, page=page)


# ── helpers ──────────────────────────────────────────────────────────────────
def _is_closing_fence(line: str, fence_char: str, min_len: int) -> bool:
    """True if ``line`` is a code-fence close (only fence chars, >= opening len)."""
    s = line.strip()
    return len(s) >= min_len and set(s) == {fence_char}


def _consume_delimited(lines, i, open_tok, close_tok):
    """Consume a display-math block delimited by ``open_tok`` / ``close_tok``.

    Handles both the single-line form (``$$ x $$``) and the multi-line form.
    Returns ``(inner_content, next_index)``.
    """
    first = lines[i].strip()
    inner = first[len(open_tok):]
    # Single-line: closing delimiter on the same line.
    end = inner.rfind(close_tok)
    if end != -1:
        return inner[:end].strip(), i + 1

    body = [inner]
    i += 1
    n = len(lines)
    while i < n:
        s = lines[i].strip()
        if s.endswith(close_tok) or s == close_tok:
            body.append(s[: -len(close_tok)] if s != close_tok else "")
            i += 1
            break
        body.append(lines[i])
        i += 1
    return "\n".join(body).strip(), i


def _consume_list(lines, i, emit):
    """Consume a contiguous list block, emitting one ``list_item`` per marker."""
    n = len(lines)
    current: list[str] | None = None

    def flush():
        if current:
            emit("list_item", "\n".join(current))

    while i < n:
        line = lines[i]
        s = line.strip()
        if not s:
            # A blank line ends the list unless the next non-blank line is still
            # a list item (loose list). Peek ahead.
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j < n and (_BULLET_RE.match(lines[j]) or _ORDERED_RE.match(lines[j])):
                i = j
                continue
            break
        # A thematic break ('* * *', '- - -') would otherwise match _BULLET_RE;
        # treat it as the end of the list (the top level drops it as an HR).
        if _HR_RE.match(line):
            break
        m = _BULLET_RE.match(line) or _ORDERED_RE.match(line)
        if m:
            flush()
            current = [m.group(2)]
            i += 1
            continue
        if _starts_block(lines, i, in_list=True):
            break
        # Continuation line for the current item.
        if current is not None:
            current.append(s)
            i += 1
        else:
            break
    flush()
    return i


def _starts_block(lines, i, in_list: bool = False) -> bool:
    """True if ``lines[i]`` begins a block that should end a paragraph/list item."""
    line = lines[i]
    if _ATX_RE.match(line) or _FENCE_RE.match(line) or _IMAGE_ONLY_RE.match(line):
        return True
    if line.strip().startswith("$$") or line.strip().startswith("\\["):
        return True
    if line.strip().startswith(">"):  # block quote ends the current paragraph
        return True
    if _HR_RE.match(line):
        return True
    if _is_table_start(lines, i):
        return True
    if not in_list and (_BULLET_RE.match(line) or _ORDERED_RE.match(line)):
        return True
    return False


def _is_table_start(lines, i: int) -> bool:
    """True if a pipe table starts at ``lines[i]``.

    Two accepted shapes: a GFM header followed by a ``|---|`` separator row, or a
    run of two or more pipe-delimited rows (a separator-less table). Requiring two
    rows for the second shape keeps a stray ``|`` in a sentence as prose.
    """
    line = lines[i]
    if i + 1 >= len(lines):
        return False
    if "|" in line and _TABLE_SEP_RE.match(lines[i + 1]):
        return True
    return bool(_PIPE_ROW_RE.match(line) and _PIPE_ROW_RE.match(lines[i + 1]))


__all__ = ["parse_markdown", "parse_full_page_output"]
