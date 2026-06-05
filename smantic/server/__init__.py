"""FastAPI service for smantic (optional ``[serve]`` extra)."""

from .main import app, create_app, run

__all__ = ["app", "create_app", "run"]
