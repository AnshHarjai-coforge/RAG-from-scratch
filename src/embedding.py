"""Embedding models for passages and questions."""

import math
from typing import Protocol

from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    """Something that can embed passages and a search question.

    Tests supply a fake. The indexing script supplies ``QwenEmbedder``.
    """

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed document text. These calls do not use the query prompt."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a question with the model query prompt."""


class QwenEmbedder:
    """Local Qwen3 embedding model.

    The model loads once and is reused for every batch and every question.
    Passages are embedded without a prompt. Questions use the ``query`` prompt
    stored with the model. ``batch_size`` limits how many passages the model
    encodes at once.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return _as_lists(vectors)

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("question must not be empty")
        vector = self.model.encode(
            [text],
            prompt_name="query",
            show_progress_bar=False,
        )[0]
        values = vector.tolist()
        if any(not math.isfinite(value) for value in values):
            raise ValueError("query embedding contains a non-finite value")
        return values


def _as_lists(vectors) -> list[list[float]]:
    return [row.tolist() for row in vectors]
