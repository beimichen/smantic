"""Tests for the ONNX sentence embedder + its graceful fallback."""

import numpy as np
import pytest

from smantic.embedder import OnnxSentenceEmbedder


def test_fallback_is_deterministic_when_unavailable():
    """With no model loaded, embeddings are uniform unit vectors (sim 1.0)."""
    e = OnnxSentenceEmbedder()
    e.release()  # force the unavailable state regardless of installed extras
    assert e.available is False

    vecs = e.embed_sentences(["alpha", "beta", "gamma"])
    assert vecs.shape == (3, 384)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0)
    # Uniform vectors => every pair is maximally similar => no false boundaries.
    assert np.isclose(vecs[0] @ vecs[1], 1.0)


def test_count_tokens_fallback_heuristic():
    e = OnnxSentenceEmbedder()
    e.release()
    assert e.tokenizer is None
    assert e.count_tokens("") == 1  # max(1, 0 // 4)
    assert e.count_tokens("a" * 40) == 10  # ~4 chars per token


def test_embed_empty_list_returns_single_unit_vector():
    e = OnnxSentenceEmbedder()
    e.release()
    out = e.embed_sentences([])
    assert out.shape == (1, 384)


@pytest.mark.slow
def test_real_embeddings_are_semantically_sane():
    """With the [onnx] extra + model, related sentences score higher."""
    e = OnnxSentenceEmbedder()
    if not e.available:
        pytest.skip("ONNX embedder/model not available")

    assert e.count_tokens("hello world") == 2  # not padded to 128
    v = e.embed_sentences([
        "The cat sat on the mat.",
        "A feline rested upon the rug.",
        "Quantum chromodynamics governs the strong force.",
    ])
    assert v.shape == (3, 384)
    related = float(v[0] @ v[1])
    unrelated = float(v[0] @ v[2])
    assert related > unrelated
