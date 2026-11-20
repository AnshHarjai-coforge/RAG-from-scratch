"""QwenEmbedder calls, without loading the real model."""

import pytest

from rag.embedding import QwenEmbedder


class _Row:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return list(self.values)


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls: list[tuple[list[str], dict]] = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return [_Row([float(len(text)), 1.0]) for text in texts]


def test_batch_size_must_be_positive(monkeypatch):
    def explode(model_name):
        raise AssertionError(model_name)

    monkeypatch.setattr("rag.embedding.SentenceTransformer", explode)

    with pytest.raises(ValueError, match="positive"):
        QwenEmbedder(batch_size=0)


def test_embed_passages_skips_an_empty_list(monkeypatch):
    monkeypatch.setattr("rag.embedding.SentenceTransformer", FakeSentenceTransformer)
    embedder = QwenEmbedder("example-model", batch_size=4)

    assert embedder.embed_passages([]) == []
    assert embedder.model.calls == []


def test_passages_and_queries_use_different_prompts(monkeypatch):
    monkeypatch.setattr("rag.embedding.SentenceTransformer", FakeSentenceTransformer)
    embedder = QwenEmbedder("example-model", batch_size=4)

    passages = embedder.embed_passages(["remote", "days"])
    query = embedder.embed_query("remote days")

    assert passages == [[6.0, 1.0], [4.0, 1.0]]
    assert query == [11.0, 1.0]
    assert embedder.model.calls[0] == (
        ["remote", "days"],
        {"batch_size": 4, "show_progress_bar": False},
    )
    assert embedder.model.calls[1][1]["prompt_name"] == "query"
    assert "batch_size" not in embedder.model.calls[1][1]


def test_non_finite_query_embedding_is_rejected(monkeypatch):
    class NonFinite(FakeSentenceTransformer):
        def encode(self, texts, **kwargs):
            return [_Row([float("nan"), 0.0])]

    monkeypatch.setattr("rag.embedding.SentenceTransformer", NonFinite)
    embedder = QwenEmbedder("example-model")

    with pytest.raises(ValueError, match="non-finite"):
        embedder.embed_query("remote")
