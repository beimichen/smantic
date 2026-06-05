"""Pydantic request/response schemas for the smantic FastAPI service."""


from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: str = "ok"


class ChunkRequest(BaseModel):
    """A chunking request: supply EITHER raw ``text`` OR a parsed ``document``."""

    text: str | None = Field(
        None, description="Raw Markdown / VLM page text to chunk."
    )
    document: dict | None = Field(
        None,
        description="A parsed document dict (smantic IR, Docling, or NoPaddle "
        "shape — 'regions' and 'elements' both accepted).",
    )
    max_tokens: int | None = Field(
        None, ge=1, le=20_000, description="Soft max tokens per chunk."
    )
    overlap_tokens: int | None = Field(
        None, ge=0, le=10_000, description="Token overlap between prose chunks."
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "ChunkRequest":
        if (self.text is None) == (self.document is None):
            raise ValueError("provide exactly one of 'text' or 'document'")
        return self


class ChunkResponse(BaseModel):
    num_chunks: int
    avg_tokens: float
    chunks: list[dict]
