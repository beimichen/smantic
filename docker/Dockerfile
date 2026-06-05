# ─────────────────────────────────────────────────────────────────────────────
# smantic — structure-aware semantic chunking, minus the heavyweight stack.
#
# Serves the FastAPI app (smantic.server.main:app) with the ONNX embedding
# stack: all-MiniLM-L6-v2 on raw onnxruntime + the Rust tokenizers lib. No
# torch, no transformers, no paddle, no GPU runtime — it runs anywhere
# python:3.12-slim runs, and needs no extra system libraries.
#
# Build context is the *repo root* (one dir up), so build with:
#     docker build -f docker/Dockerfile -t smantic .
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Quieter, snappier Python; unbuffered logs for `docker logs`.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── install ──────────────────────────────────────────────────────────────────
# Copy project metadata + source, then install with the ONNX (semantic
# embeddings) and serve (FastAPI/uvicorn) extras. The FastAPI service lives at
# smantic/server/, so the single smantic/ copy covers it. onnxruntime + numpy
# ship self-contained manylinux wheels, so no apt packages are needed.
COPY pyproject.toml README.md LICENSE ./
COPY smantic ./smantic

RUN pip install --no-cache-dir ".[onnx,serve]"

# ── model cache ──────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2 is pulled from the Hugging Face Hub on first use and cached
# here. Mount a volume at /models to persist the download across restarts.
ENV MODEL_CACHE_DIR=/models \
    HF_HOME=/models/hf
RUN mkdir -p /models
VOLUME ["/models"]

# ── serve ────────────────────────────────────────────────────────────────────
EXPOSE 8000

CMD ["uvicorn", "smantic.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
