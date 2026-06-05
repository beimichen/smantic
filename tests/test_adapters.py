"""Tests for the input adapters (markdown / IR JSON / NoPaddle shape)."""

import json

from smantic import from_dict, from_docling_json, from_json, from_markdown, from_nopaddle
from smantic.ir import Document


def test_from_markdown_builds_single_page_document():
    doc = from_markdown("# H\n\nSome prose.\n")
    assert isinstance(doc, Document)
    assert len(doc.pages) == 1
    assert doc.metadata["source"] == "markdown"
    assert [e.element_type for e in doc.pages[0].elements] == ["section_header", "text"]


def test_from_docling_json_roundtrip():
    src = {
        "pages": [
            {
                "page_number": 1,
                "elements": [
                    {"type": "text", "content": "hello", "page": 1},
                ],
            }
        ],
        "metadata": {"source": "x"},
    }
    doc = from_docling_json(json.dumps(src))
    assert doc.pages[0].elements[0].content == "hello"
    assert doc.metadata["source"] == "x"


def test_from_nopaddle_accepts_regions_key():
    # NoPaddle pages use 'regions'; the adapter must read them as elements.
    nopaddle_doc = {
        "pages": [
            {
                "page_number": 1,
                "regions": [
                    {"type": "section_header", "content": "Methods", "page": 1},
                    {"type": "text", "content": "We did things.", "page": 1},
                ],
            }
        ],
        "metadata": {"filename": "paper.pdf"},
    }
    doc = from_nopaddle(nopaddle_doc)
    assert len(doc.pages[0].elements) == 2
    assert doc.pages[0].elements[0].element_type == "section_header"
    assert doc.pages[0].elements[1].content == "We did things."


def test_from_nopaddle_accepts_object_with_to_dict():
    class FakeParsedDocument:
        def to_dict(self):
            return {
                "pages": [{"page_number": 1, "regions": [
                    {"type": "text", "content": "obj path", "page": 1}]}],
                "metadata": {},
            }

    doc = from_nopaddle(FakeParsedDocument())
    assert doc.pages[0].elements[0].content == "obj path"


def test_from_nopaddle_accepts_json_string():
    s = json.dumps({"pages": [{"page_number": 1, "regions": [
        {"type": "text", "content": "json path", "page": 1}]}], "metadata": {}})
    doc = from_nopaddle(s)
    assert doc.pages[0].elements[0].content == "json path"


def test_from_dict_and_from_json_equivalent():
    data = {"pages": [{"page_number": 1, "elements": [
        {"type": "text", "content": "z", "page": 1}]}], "metadata": {}}
    a = from_dict(data)
    b = from_json(json.dumps(data))
    assert a.to_dict() == b.to_dict()


def test_type_and_element_type_both_accepted():
    doc = from_dict({"pages": [{"page_number": 1, "elements": [
        {"element_type": "code", "content": "x=1", "page": 1}]}], "metadata": {}})
    assert doc.pages[0].elements[0].element_type == "code"


def test_bbox_tolerates_extra_keys():
    # A region whose bbox carries an extra key (coord label, flag, ...) must not
    # crash ingest: real parser output is not guaranteed to be exactly 4 keys.
    doc = from_dict({"pages": [{"page_number": 1, "elements": [
        {"type": "text", "content": "x",
         "bbox": {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100, "label": "text"},
         "page": 1}]}], "metadata": {}})
    el = doc.pages[0].elements[0]
    assert (el.bbox.x_min, el.bbox.x_max) == (0, 100)


def test_metadata_null_collapses_to_empty_dict():
    # An explicit metadata: null must not leave Document.metadata as None.
    doc = from_dict({"pages": [], "metadata": None})
    assert doc.metadata == {}
