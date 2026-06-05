"""Tests for the Markdown -> Element parser (reading order + block typing)."""

from smantic.markdown import parse_markdown


def _types(elements):
    return [e.element_type for e in elements]


def test_empty_input():
    assert parse_markdown("") == []
    assert parse_markdown("   \n  \n") == []


def test_reading_order_is_preserved():
    md = (
        "# Title\n\n"
        "Intro paragraph.\n\n"
        "```python\nx = 1\n```\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "Closing paragraph."
    )
    els = parse_markdown(md)
    assert _types(els) == ["section_header", "text", "code", "table", "text"]
    # First block is the heading, last is the closing prose - order intact.
    assert els[0].content == "Title"
    assert els[-1].content == "Closing paragraph."


def test_atx_heading_levels():
    els = parse_markdown("# H1\n\n## H2\n\n### H3\n")
    assert _types(els) == ["section_header"] * 3
    assert [e.metadata["level"] for e in els] == [1, 2, 3]


def test_setext_headings():
    els = parse_markdown("Big Title\n=========\n\nSub Title\n---------\n")
    assert _types(els) == ["section_header", "section_header"]
    assert els[0].content == "Big Title" and els[0].metadata["level"] == 1
    assert els[1].content == "Sub Title" and els[1].metadata["level"] == 2


def test_fenced_code_keeps_body_and_language():
    md = "```python\ndef f():\n    return 1\n```\n"
    els = parse_markdown(md)
    assert len(els) == 1
    assert els[0].element_type == "code"
    assert els[0].content == "def f():\n    return 1"
    assert els[0].metadata["language"] == "python"
    # The fence markers themselves never leak into content.
    assert "```" not in els[0].content


def test_tilde_fence_and_no_language():
    els = parse_markdown("~~~\nplain code\n~~~\n")
    assert len(els) == 1 and els[0].element_type == "code"
    assert els[0].content == "plain code"
    assert els[0].metadata is None


def test_pipe_table_grouped_as_one_block():
    md = "| Name | Value |\n|------|-------|\n| a | 1 |\n| b | 2 |\n"
    els = parse_markdown(md)
    assert len(els) == 1 and els[0].element_type == "table"
    assert els[0].content.count("\n") == 3  # header + sep + 2 rows


def test_pipe_without_separator_is_prose_not_table():
    els = parse_markdown("this | has a pipe but no separator row\n")
    assert _types(els) == ["text"]


def test_display_math_single_and_multiline():
    single = parse_markdown("$$ E = mc^2 $$\n")
    assert _types(single) == ["formula"] and single[0].content == "E = mc^2"

    multi = parse_markdown("$$\n\\int_0^1 x\\,dx\n$$\n")
    assert _types(multi) == ["formula"]
    assert "\\int_0^1" in multi[0].content

    bracket = parse_markdown("\\[ a + b \\]\n")
    assert _types(bracket) == ["formula"] and bracket[0].content == "a + b"


def test_lists_emit_one_item_each():
    md = "- first\n- second\n- third\n"
    els = parse_markdown(md)
    assert _types(els) == ["list_item"] * 3
    assert [e.content for e in els] == ["first", "second", "third"]

    ordered = parse_markdown("1. one\n2. two\n")
    assert _types(ordered) == ["list_item", "list_item"]


def test_standalone_image_becomes_picture():
    els = parse_markdown("![a diagram](img.png)\n")
    assert len(els) == 1 and els[0].element_type == "picture"
    assert els[0].content == "a diagram"
    assert els[0].metadata["caption"] == "a diagram"


def test_blockquote_is_prose():
    els = parse_markdown("> quoted line one\n> quoted line two\n")
    assert _types(els) == ["text"]
    assert "quoted line one" in els[0].content
    assert ">" not in els[0].content


def test_thematic_break_is_dropped():
    els = parse_markdown("Para one.\n\n---\n\nPara two.\n")
    assert _types(els) == ["text", "text"]
    assert els[0].content == "Para one." and els[1].content == "Para two."


def test_page_number_propagates():
    els = parse_markdown("# H\n\ntext\n", page=7)
    assert all(e.page == 7 for e in els)


def test_blockquote_right_after_prose_no_blank_line():
    # Regression: a block quote with no blank line above must not be swallowed
    # into the preceding paragraph with literal '>' markers.
    els = parse_markdown("some prose\n> a quote\n> more quote\n")
    assert _types(els) == ["text", "text"]
    assert els[0].content == "some prose"
    assert ">" not in els[1].content
    assert "a quote" in els[1].content and "more quote" in els[1].content


def test_thematic_break_inside_list_is_not_an_item():
    # Regression: '* * *' / '- - -' match the bullet regex; they must be treated
    # as a thematic break (dropped), not emitted as a garbage list item.
    for hr in ("* * *", "- - -"):
        els = parse_markdown(f"- a\n{hr}\n- b\n")
        assert _types(els) == ["list_item", "list_item"]
        assert [e.content for e in els] == ["a", "b"]
