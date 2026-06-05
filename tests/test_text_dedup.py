"""Tests for tasks.text_dedup.collapse_repeated_spans — the shared
adjacent-duplicate-span collapser used by the chunker (assembled chunk
content) and the PaddleOCR-VL parser (per-OCR-region)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smantic.text_dedup import collapse_repeated_spans


def _double_free(text, mink=4):
    w = text.split()
    n = len(w)
    for i in range(n):
        for k in range((n - i) // 2, mink - 1, -1):
            if w[i:i + k] == w[i + k:i + 2 * k]:
                return False
    return True


def test_collapses_duplicated_sentence_midcontent():
    s = ("The offline results matched the baseline. "
         "Relative to generating only one answer per question, both variants "
         "provide clear average improvements of approximately six percent. "
         "Relative to generating only one answer per question, both variants "
         "provide clear average improvements of approximately six percent. "
         "Adaptive policies did better.")
    out, removed = collapse_repeated_spans(s)
    assert removed > 0
    assert out.count("Relative to generating only one answer") == 1
    assert "Adaptive policies did better." in out


def test_returns_removed_count_zero_when_clean():
    s = "Transistors have three terminals: a collector, an emitter, and a base."
    out, removed = collapse_repeated_spans(s)
    assert (out, removed) == (s, 0)


def test_collapses_triple_repeat():
    out, _ = collapse_repeated_spans("alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta end")
    assert out == "alpha beta gamma delta end"


def test_collapses_cross_sentence_span_dedup_sentences_would_miss():
    # span crosses no sentence boundary punctuation -> the chunker's
    # per-sentence _dedup_sentences can't see it; the span collapse must.
    span = "the rate depends on both nucleophile and substrate"
    s = f"In an SN2 reaction {span} {span} concentration matters."
    out, removed = collapse_repeated_spans(s)
    assert removed == 8
    assert out.count(span) == 1


def test_below_threshold_preserved():
    # 3-word immediate repeat stays (below min_words=4 default)
    assert collapse_repeated_spans("New York New York office tower")[0] == "New York New York office tower"


def test_preserves_newlines_of_surviving_copy():
    s = "alpha beta\ngamma delta after alpha beta gamma delta after tail end"
    out, removed = collapse_repeated_spans(s)
    assert removed > 0
    assert out.count("alpha beta") == 1
    assert "\n" in out and out.endswith("tail end")


def test_empty_and_tiny_noop():
    assert collapse_repeated_spans("") == ("", 0)
    assert collapse_repeated_spans("short text only") == ("short text only", 0)


def test_multiple_independent_repeats():
    out, _ = collapse_repeated_spans(
        "one two three four one two three four mid here five six seven eight five six seven eight tail")
    assert out == "one two three four mid here five six seven eight tail"
    assert _double_free(out)
