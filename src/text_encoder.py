"""Pluggable text encoders.

Three backends are exposed via :func:`get_text_encoder`:

  * ``"hash"``                — deterministic bag-of-words → random
                                projection. Zero dependencies; produces a
                                vector in whatever dimensionality the
                                visual features use, so cosine similarity
                                is well-defined out of the box. Good for
                                smoke tests; only weakly semantic.
  * ``"sentence-transformers"`` — lazy import of
                                  sentence-transformers. Default model
                                  ``all-MiniLM-L6-v2`` (~80MB, 384-d).
  * ``"clip"``               — lazy import of ``open_clip``. Default
                                ``ViT-B-32`` / ``openai`` weights
                                (~150MB, 512-d).

If the requested backend cannot be imported, falls back to ``"hash"``
with a printed warning so the pipeline still runs.
"""

from __future__ import annotations

import hashlib
import sys
from typing import List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class TextEncoder:
    """Minimal interface: encode a list of strings to a (N, D) array."""

    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Hash bag-of-words encoder (no dependencies)
# ---------------------------------------------------------------------------

class HashTextEncoder(TextEncoder):
    """Deterministic, dependency-free text encoder.

    Each token contributes a per-token Gaussian vector seeded by its hash.
    The sum is L2-normalized. This is *not* a strong semantic encoder, but
    it is a useful sanity baseline and is exactly what the synthetic
    dummy dataset is constructed against, so smoke-test metrics will be
    non-random.
    """

    def __init__(self, dim: int = 64, seed: int = 0):
        self.dim = int(dim)
        self.seed = int(seed)

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = [t for t in text.lower().replace(",", " ").split() if t]
        for tok in tokens:
            # blake2b makes the encoder deterministic across processes;
            # plain `hash()` is randomized unless PYTHONHASHSEED is set.
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest()
            token_seed = (int.from_bytes(h, "little") ^ self.seed) & 0xFFFFFFFF
            rng = np.random.default_rng(token_seed)
            vec += rng.standard_normal(self.dim).astype(np.float32)
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        return vec

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._embed_one(t) for t in texts], axis=0)


# ---------------------------------------------------------------------------
# Sentence-Transformers encoder (lazy import)
# ---------------------------------------------------------------------------

class SentenceTransformerEncoder(TextEncoder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # noqa: WPS433
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embs = self._model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True
        )
        return embs.astype(np.float32)


# ---------------------------------------------------------------------------
# CLIP text encoder (lazy import)
# ---------------------------------------------------------------------------

class ClipTextEncoder(TextEncoder):
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        import torch  # noqa: WPS433
        import open_clip  # noqa: WPS433

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._model = self._model.to(self._device).eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)
        # infer dim with a dummy forward
        with torch.no_grad():
            tok = self._tokenizer(["probe"]).to(self._device)
            emb = self._model.encode_text(tok)
        self.dim = int(emb.shape[-1])

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._torch.no_grad():
            tok = self._tokenizer(list(texts)).to(self._device)
            emb = self._model.encode_text(tok)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return emb.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_text_encoder(
    name: str = "hash",
    *,
    hash_dim: int = 64,
    st_model: str = "all-MiniLM-L6-v2",
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "openai",
) -> TextEncoder:
    """Return a text encoder, falling back to hash if the requested backend
    is not installed."""
    name = name.lower()
    if name in {"hash", "none", "stub"}:
        return HashTextEncoder(dim=hash_dim)
    if name in {"st", "sentence", "sentence-transformers", "sentence_transformers"}:
        try:
            return SentenceTransformerEncoder(model_name=st_model)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[text_encoder] could not load sentence-transformers ({exc!r}); "
                "falling back to HashTextEncoder.",
                file=sys.stderr,
            )
            return HashTextEncoder(dim=hash_dim)
    if name == "clip":
        try:
            return ClipTextEncoder(model_name=clip_model, pretrained=clip_pretrained)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[text_encoder] could not load CLIP ({exc!r}); "
                "falling back to HashTextEncoder.",
                file=sys.stderr,
            )
            return HashTextEncoder(dim=hash_dim)
    raise ValueError(f"unknown text encoder: {name!r}")


__all__ = [
    "TextEncoder",
    "HashTextEncoder",
    "SentenceTransformerEncoder",
    "ClipTextEncoder",
    "get_text_encoder",
]
