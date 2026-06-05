"""
ONNX sentence embedder for semantic boundary detection.

Wraps the all-MiniLM-L6-v2 ONNX graph on raw ``onnxruntime`` plus the Rust
``tokenizers`` library, producing 384-dim L2-normalized embeddings. No torch,
no transformers, no MLX, so it runs the same on every OS/CPU.

The tokenizer also provides token counting (the chunker budgets chunks in
tokens), replacing what used to be a separate dependency.

Everything degrades gracefully: if ``onnxruntime``/``tokenizers`` are missing,
or the model can't be loaded (offline, no cache), the embedder reports
``available == False`` and returns uniform unit vectors so pairwise similarity
is 1.0 and no false semantic boundaries are introduced. The chunker still works
on structural boundaries alone.
"""

import logging
from pathlib import Path

import numpy as np

from . import config

logger = logging.getLogger(__name__)

# all-MiniLM-L6-v2 hidden size. Only used to shape the graceful-fallback
# vectors when the real model is unavailable.
_EMBED_DIM = 384

# Max sentence length (in tokens) fed to the embedder. Sentences longer than
# this are truncated for the embedding pass only; token *counting* is never
# truncated (the chunker needs true lengths to budget chunks).
_MAX_SEQ_LEN = 128


# ── optional heavy deps (graceful) ──────────────────────────────────────────
try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError as e:  # pragma: no cover - exercised in minimal installs
    ort = None
    _ONNX_AVAILABLE = False
    logger.debug("onnxruntime not available: %s", e)

try:
    from tokenizers import Tokenizer
    _TOKENIZERS_AVAILABLE = True
except ImportError as e:  # pragma: no cover - exercised in minimal installs
    Tokenizer = None
    _TOKENIZERS_AVAILABLE = False
    logger.debug("tokenizers not available: %s", e)


class OnnxSentenceEmbedder:
    """Sentence embedder using the all-MiniLM-L6-v2 ONNX model.

    Produces 384-dim L2-normalized embeddings and counts tokens with the same
    tokenizer. Falls back to uniform unit vectors + a char-based token estimate
    when the model is unavailable.
    """

    def __init__(self, model_dir: Path | None = None):
        self.session = None
        self.tokenizer = None
        self._input_names: set = set()
        self._available = False

        if not _ONNX_AVAILABLE:
            logger.warning("onnxruntime not available — ONNX embeddings disabled")
            return
        if not _TOKENIZERS_AVAILABLE:
            logger.warning("tokenizers not available — ONNX embeddings disabled")
            return

        try:
            self._load_model(model_dir)
        except Exception as e:
            logger.warning("Failed to load ONNX sentence embedder: %s — embeddings disabled", e)

    # ── loading ─────────────────────────────────────────────────────────────
    def _load_model(self, model_dir: Path | None):
        """Load from an explicit dir, else download the configured repo from HF."""
        # 1. Explicit model_dir
        if model_dir:
            model_dir = Path(model_dir)
            if model_dir.exists():
                onnx_path = self._find_onnx_file(model_dir)
                if onnx_path:
                    self._load_from_path(onnx_path, model_dir)
                    return

        # 2. Resolve from the Hugging Face Hub (cached on disk).
        from .models import resolve_model_dir
        resolved = resolve_model_dir(config.EMBED_REPO)
        onnx_path = self._find_onnx_file(resolved)
        if onnx_path:
            self._load_from_path(onnx_path, resolved)
            return

        raise FileNotFoundError(
            f"Could not find an ONNX model file under {resolved}"
        )

    @staticmethod
    def _find_onnx_file(directory: Path) -> Path | None:
        """Look for model.onnx (with or without an onnx/ subfolder)."""
        for candidate in (directory / "onnx" / "model.onnx", directory / "model.onnx"):
            if candidate.exists():
                return candidate
        return None

    def _load_from_path(self, onnx_path: Path, tokenizer_dir: Path):
        """Load the ONNX session + tokenizer from local files."""
        tokenizer_dir = Path(tokenizer_dir)
        tok_path = tokenizer_dir / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {tokenizer_dir}")

        logger.info("Loading ONNX sentence embedder from %s", onnx_path)

        # Prefer CoreML on macOS, fall back to CPU everywhere.
        providers = []
        if hasattr(ort, "get_available_providers"):
            if "CoreMLExecutionProvider" in ort.get_available_providers():
                providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self._input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        # all-MiniLM's tokenizer.json bakes in pad-to-128; that would make
        # count_tokens return 128 for everything and destroy the token budget.
        # Disable both and do truncation/padding ourselves in embed_sentences.
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self._available = True
        logger.info("ONNX sentence embedder loaded (providers: %s)", providers)

    # ── public API ──────────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        return self._available

    def count_tokens(self, text: str) -> int:
        """Count tokens with the model tokenizer (no special tokens, no truncation)."""
        if self.tokenizer is None:
            # Rough fallback: ~4 chars per token.
            return max(1, len(text) // 4)
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def embed_sentences(self, sentences: list[str]) -> np.ndarray:
        """Batch-embed sentences into (N, dim) L2-normalized vectors.

        Returns uniform unit vectors (pairwise similarity 1.0) when the model is
        unavailable, so the caller never sees a spurious semantic boundary.
        """
        if not self._available or not sentences:
            n = max(len(sentences), 1)
            uniform = np.ones((n, _EMBED_DIM), dtype=np.float32)
            return uniform / np.linalg.norm(uniform, axis=1, keepdims=True)

        encodings = self.tokenizer.encode_batch(sentences)  # special tokens on
        seq_len = min(_MAX_SEQ_LEN, max((len(e.ids) for e in encodings), default=1)) or 1

        input_ids = np.zeros((len(encodings), seq_len), dtype=np.int64)
        attention = np.zeros((len(encodings), seq_len), dtype=np.int64)
        for i, enc in enumerate(encodings):
            ids = enc.ids[:seq_len]
            # On truncation, keep the trailing special token ([SEP]) the way the
            # tokenizer's own truncation would, instead of a mid-sentence token.
            if len(enc.ids) > seq_len:
                ids = ids[:-1] + [enc.ids[-1]]
            input_ids[i, : len(ids)] = ids
            attention[i, : len(ids)] = 1

        ort_inputs = {"input_ids": input_ids, "attention_mask": attention}
        if "token_type_ids" in self._input_names:
            ort_inputs["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self.session.run(None, ort_inputs)

        # Mean pooling over token embeddings (first output = last_hidden_state).
        token_embeddings = outputs[0]  # (batch, seq_len, dim)
        mask = attention[..., np.newaxis].astype(np.float32)
        pooled = (token_embeddings * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return pooled / norms

    def embed_sentence(self, sentence: str) -> np.ndarray:
        """Single-sentence embedding."""
        return self.embed_sentences([sentence])[0]

    def release(self) -> None:
        """Drop the ONNX session + tokenizer so their memory can be reclaimed."""
        self.session = None
        self.tokenizer = None
        self._input_names = set()
        self._available = False


__all__ = ["OnnxSentenceEmbedder"]
