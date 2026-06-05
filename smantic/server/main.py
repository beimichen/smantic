"""
FastAPI service for smantic.

A thin, self-hostable HTTP wrapper around the chunker:

  * ``GET  /healthz`` — liveness probe (does not touch any model).
  * ``GET  /info``    — version + embedder availability.
  * ``POST /chunk``   — chunk raw Markdown ``text`` or a parsed ``document``.

A single warm chunker is reused across requests (the ONNX embedder loads once),
guarded by a lock so requests serialize cleanly. Set
``SMANTIC_RELEASE_AFTER_REQUEST=1`` to drop the embedder after every request for
a near-zero idle footprint at the cost of a reload per call.

``smantic`` is imported lazily inside the handlers so this module imports even
when the optional ONNX deps are absent (handy for ``--reload`` dev and OpenAPI
generation in CI).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException

from .schemas import ChunkRequest, ChunkResponse, HealthResponse

logger = logging.getLogger("smantic.app")

_CHUNKER = None
_LOCK = threading.Lock()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _get_chunker():
    """Return the process-wide warm chunker, building it on first use."""
    global _CHUNKER
    if _CHUNKER is None:
        from smantic import make_chunker

        _CHUNKER = make_chunker()
    return _CHUNKER


def _release_chunker() -> None:
    global _CHUNKER
    if _CHUNKER is not None:
        try:
            _CHUNKER.release()
        except Exception:  # pragma: no cover - best effort
            logger.debug("chunker release failed", exc_info=True)
        _CHUNKER = None


def _safe_version() -> str:
    try:
        from smantic import __version__

        return __version__
    except Exception:  # pragma: no cover
        return "0"


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="smantic",
        version=_safe_version(),
        description=(
            "Structure-aware semantic chunking. POST Markdown or a parsed "
            "document to /chunk."
        ),
        license_info={
            "name": "Apache-2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0",
        },
    )

    @app.get("/healthz", response_model=HealthResponse, tags=["meta"])
    def healthz() -> HealthResponse:
        """Liveness probe. Cheap — does not touch the chunker or any model."""
        return HealthResponse(status="ok")

    @app.get("/info", tags=["meta"])
    def info() -> dict[str, Any]:
        """Version + whether the ONNX embedder is loaded/available."""
        # Snapshot the global once: a concurrent release can null it out.
        chunker = _CHUNKER
        embeddings = False
        if chunker is not None and getattr(chunker, "embedder", None) is not None:
            embeddings = chunker.embedder.available
        return {"version": _safe_version(), "embeddings_available": embeddings}

    @app.post("/chunk", response_model=ChunkResponse, tags=["chunk"])
    def chunk(req: ChunkRequest) -> ChunkResponse:
        """Chunk a document and return the chunks plus a small summary."""
        from smantic import config, from_markdown

        if req.text is not None:
            doc = from_markdown(req.text)
        else:
            # The schema types `document` as a dict, so it's already parsed.
            try:
                doc = _doc_from_dict(req.document)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"bad document: {exc}") from exc

        max_tokens = req.max_tokens if req.max_tokens is not None else config.DEFAULT_MAX_TOKENS
        overlap = req.overlap_tokens if req.overlap_tokens is not None else config.DEFAULT_OVERLAP_TOKENS

        with _LOCK:
            chunker = _get_chunker()
            # max_tokens / overlap are read at chunk time, so per-request tuning
            # just sets them on the warm chunker (calls are serialized by _LOCK).
            chunker.max_tokens = max_tokens
            chunker.overlap_tokens = overlap
            try:
                chunks = chunker.chunk_document(doc)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("chunk failed")
                raise HTTPException(status_code=500, detail=f"chunk failed: {exc}") from exc
            finally:
                if _env_flag("SMANTIC_RELEASE_AFTER_REQUEST", False):
                    _release_chunker()

        dicts = [c.to_dict() for c in chunks]
        avg = sum(c.token_count for c in chunks) / len(chunks) if chunks else 0.0
        return ChunkResponse(num_chunks=len(dicts), avg_tokens=avg, chunks=dicts)

    return app


def _doc_from_dict(data):
    from smantic import from_dict

    return from_dict(data)


# Module-level app instance for ``uvicorn smantic.server.main:app``.
app = create_app()


def run() -> None:
    """Console-script entry point (``smantic-serve``): boot uvicorn.

    Host/port via ``SMANTIC_HOST`` / ``SMANTIC_PORT`` (default 0.0.0.0:8000).
    """
    import uvicorn  # lazy: only needed for the [serve] extra

    uvicorn.run(
        "smantic.server.main:app",
        host=os.getenv("SMANTIC_HOST", "0.0.0.0"),
        port=int(os.getenv("SMANTIC_PORT", "8000")),
    )


__all__ = ["app", "create_app", "run"]
