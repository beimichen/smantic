"""
Tiny, env-driven settings.

No heavy deps here, just stdlib. Every constant can be overridden with the
matching ``SMANTIC_*`` env var so a fork or a deployment can nudge behaviour
without touching code.
"""

import os

# Hugging Face repo for the sentence embedder used in semantic boundary
# detection. all-MiniLM-L6-v2 ships an ONNX graph + a tokenizer.json, so we
# run it on raw onnxruntime + the Rust ``tokenizers`` lib (no torch, no
# transformers). Override to point at a mirror or a compatible 384-dim model.
EMBED_REPO: str = os.getenv("SMANTIC_EMBED_REPO", "sentence-transformers/all-MiniLM-L6-v2")


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


#: Soft maximum tokens per chunk (chunks may exceed this only for atomic blocks).
DEFAULT_MAX_TOKENS: int = _env_int("SMANTIC_MAX_TOKENS", 500)

#: Token overlap between consecutive prose chunks (context continuity).
DEFAULT_OVERLAP_TOKENS: int = _env_int("SMANTIC_OVERLAP_TOKENS", 50)

#: Cosine-similarity threshold below which a soft (semantic) boundary is placed.
BOUNDARY_THRESHOLD: float = _env_float("SMANTIC_BOUNDARY_THRESHOLD", 0.5)


__all__ = [
    "EMBED_REPO",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_OVERLAP_TOKENS",
    "BOUNDARY_THRESHOLD",
]
