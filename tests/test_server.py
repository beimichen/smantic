"""Tests for the FastAPI service (offline: chunker runs on the fallback embedder)."""

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from smantic.server.main import create_app  # noqa: E402


@pytest.fixture
def client():
    return TestClient(create_app())


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_info(client):
    r = client.get("/info")
    assert r.status_code == 200
    body = r.json()
    assert "version" in body and "embeddings_available" in body


def test_chunk_markdown_text(client):
    md = "# Title\n\nA paragraph of prose with enough words to be a real chunk here.\n"
    r = client.post("/chunk", json={"text": md})
    assert r.status_code == 200
    body = r.json()
    assert body["num_chunks"] >= 1
    assert body["chunks"][0]["dominant_type"] == "prose"
    assert "Title" in body["chunks"][0]["content"]


def test_chunk_document_payload(client):
    doc = {
        "pages": [{"page_number": 1, "regions": [
            {"type": "text", "content": "Short prose from a parsed document.", "page": 1}]}],
        "metadata": {},
    }
    r = client.post("/chunk", json={"document": doc})
    assert r.status_code == 200
    assert r.json()["num_chunks"] == 1


def test_chunk_requires_exactly_one_source(client):
    # Neither text nor document -> 422 from the validator.
    assert client.post("/chunk", json={}).status_code == 422
    # Both -> also 422.
    both = {"text": "x", "document": {"pages": [], "metadata": {}}}
    assert client.post("/chunk", json=both).status_code == 422


def test_chunk_respects_max_tokens_override(client):
    # Distinct sentences (identical ones would be collapsed by dedup).
    md = " ".join(
        f"This is distinct sentence number {i} carrying a little filler content."
        for i in range(60)
    )
    r = client.post("/chunk", json={"text": md, "max_tokens": 60, "overlap_tokens": 0})
    assert r.status_code == 200
    # A tight max_tokens should yield more than one chunk via emergency splits.
    assert r.json()["num_chunks"] >= 2
