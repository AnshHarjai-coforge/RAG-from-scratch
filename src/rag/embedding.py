"""Embed passages and questions with the local Qwen model."""

import math

from sentence_transformers import SentenceTransformer


class QwenEmbedder:
    """Local Qwen3 embedding model.

    The model loads once. Passages are embedded without a prompt. Questions
    use the model's ``query`` prompt. ``batch_size`` is how many passages
    ``encode`` handles in one call.
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
        return [row.tolist() for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            [text],
            prompt_name="query",
            show_progress_bar=False,
        )[0]
        values = vector.tolist()
        if any(not math.isfinite(value) for value in values):
            raise ValueError("query embedding contains a non-finite value")
        return values
