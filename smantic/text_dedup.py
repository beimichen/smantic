"""Shared text-dedup utility: collapse immediately-adjacent verbatim
word-span repeats (``S S -> S``).

OCR/VLM parsers can re-emit spans verbatim, and concatenating overlapping
layout-detection elements produces a span immediately repeated within the
text. This collapses them at
whichever stage it's applied. The chunker applies it on ASSEMBLED chunk
content (where cross-element duplicates land, joined by newlines); the
parser applies a per-OCR-region variant for intra-region repeats.

Word-level (whitespace-delimited) so the inter-copy separator (space /
newline / period-space) doesn't defeat the match. Loops until stable, so
``S S S -> S`` and several independent repeats all resolve. Preserves the
surviving copy's spacing and the surrounding text.
"""
from __future__ import annotations

import re

# A 4-token verbatim span immediately repeated is effectively never
# legitimate prose; observed OCR/chunker duplicates are full sentences /
# paragraphs (far longer).
DEFAULT_MIN_WORDS = 4
# Token cap on the span length searched per scan (a performance bound);
# real duplicated spans observed were <= ~31 tokens.
DEFAULT_MAX_SPAN = 80


def collapse_repeated_spans(
    text: str,
    min_words: int = DEFAULT_MIN_WORDS,
    max_span: int = DEFAULT_MAX_SPAN,
) -> tuple[str, int]:
    """Return ``(collapsed_text, removed_token_count)``.

    Removes the second copy of every immediately-adjacent verbatim
    word-span of length ``>= min_words``, longest-span-first, looping
    until stable. ``removed_token_count`` is 0 when nothing changed.
    """
    if not text or min_words is None:
        return text, 0
    removed = 0
    while True:
        toks = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        n = len(toks)
        found = None
        for i in range(n):
            kmax = min((n - i) // 2, max_span)
            # Largest span first: stronger evidence, avoids collapsing a
            # sub-span of a larger duplicate.
            for k in range(kmax, min_words - 1, -1):
                if all(toks[i + a][0] == toks[i + k + a][0] for a in range(k)):
                    found = (i, k)
                    break
            if found:
                break
        if not found:
            break
        i, k = found
        gap_start = toks[i + k - 1][2]        # end char of copy-A's last token
        copyb_end = toks[i + 2 * k - 1][2]    # end char of copy-B's last token
        text = text[:gap_start] + text[copyb_end:]
        removed += k
    return text, removed
