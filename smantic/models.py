"""
Model resolution + a tiny on-disk cache.

The sentence embedder is pulled from the Hugging Face Hub on first use and
cached on disk. There is no bundled-resource / app-private path; everything
resolves to a single cache root you can point anywhere with
``MODEL_CACHE_DIR``.

  * ``resolve_model_dir(repo)`` - download (if needed) + return a local dir

A ``.download_complete`` marker makes subsequent calls offline and instant.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def sanitize_model_id(model_id: str) -> str:
    """``org/name`` -> ``org--name`` (a filesystem-safe cache dir name)."""
    sanitized = model_id.replace("/", "--").replace("\\", "--")
    if ".." in sanitized or sanitized.startswith("."):
        raise ValueError(f"Invalid model id after sanitization: {sanitized!r}")
    return sanitized


def unsanitize_model_id(sanitized: str) -> str:
    """``org--name`` -> ``org/name``."""
    return sanitized.replace("--", "/")


def _cache_root() -> Path:
    """Where downloaded models live. ``MODEL_CACHE_DIR`` wins; else XDG cache."""
    env = os.getenv("MODEL_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    base = os.getenv("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "smantic" / "models"


def resolve_model_dir(repo_id: str) -> Path:
    """Return a local dir for ``repo_id``, downloading from HF Hub on first use.

    ``repo_id`` may be a real HF id (``org/name``) or the sanitized form
    (``org--name``); both are accepted. The download is a full
    ``snapshot_download`` into ``<cache_root>/<org--name>``; a
    ``.download_complete`` marker makes subsequent calls offline + instant.
    """
    sanitized = repo_id if "/" not in repo_id else sanitize_model_id(repo_id)
    real_id = unsanitize_model_id(sanitized)
    dest = _cache_root() / sanitized
    marker = dest / ".download_complete"
    if marker.exists():
        return dest

    try:
        from huggingface_hub import snapshot_download  # lazy: optional at import
    except ImportError as e:  # pragma: no cover - dependency hint
        raise RuntimeError(
            f"huggingface_hub is required to download {real_id!r}. "
            "Install it, or pre-populate MODEL_CACHE_DIR. "
            "(pip install 'smantic[onnx]')"
        ) from e

    logger.info("Downloading %s -> %s", real_id, dest)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=real_id,
        local_dir=str(dest),
        allow_patterns=[
            "onnx/model.onnx",
            "onnx/model.onnx_data",
            "model.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "config.json",
            "special_tokens_map.json",
            "vocab.txt",
        ],
        token=os.getenv("HF_TOKEN"),  # only needed for gated/private repos
    )
    marker.touch()
    return dest


__all__ = [
    "resolve_model_dir",
    "sanitize_model_id",
    "unsanitize_model_id",
]
