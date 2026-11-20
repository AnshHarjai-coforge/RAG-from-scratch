import pytest

from rag.rerank import rerank
from rag.vector_store import RetrievalHit


class CodeFirst:
    """Gives the EXP-214 chunk the higher score."""

    def predict(self, pairs):
        return [1.0 if "EXP-214" in text else 0.2 for _question, text in pairs]


def test_rerank_keeps_the_higher_score():
    hits = [
        RetrievalHit("a", "holiday calendar", 0.1, {}),
        RetrievalHit("b", "exception code EXP-214", 0.2, {}),
    ]

    ranked = rerank("EXP-214", hits, CodeFirst(), top_k=1)

    assert ranked[0][0].id == "b"
    assert ranked[0][1] == 1.0


def test_reranker_uses_the_named_model(monkeypatch):
    seen = {}

    class FakeCrossEncoder:
        def __init__(self, model_name: str) -> None:
            seen["model"] = model_name

        def predict(self, pairs):
            seen["pairs"] = pairs
            return [0.4]

    monkeypatch.setattr("rag.rerank.CrossEncoder", FakeCrossEncoder)
    from rag.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker("cross-encoder/example")

    assert reranker.predict([["question", "chunk"]]) == [0.4]
    assert seen == {"model": "cross-encoder/example", "pairs": [["question", "chunk"]]}


def test_rerank_returns_nothing_when_there_are_no_hits():
    assert rerank("EXP-214", [], CodeFirst(), top_k=1) == []


def test_rerank_rejects_a_top_k_below_one():
    with pytest.raises(ValueError):
        rerank("EXP-214", [], CodeFirst(), top_k=0)


def test_rerank_rejects_fewer_scores_than_hits():
    hits = [
        RetrievalHit("a", "holiday calendar", 0.1, {}),
        RetrievalHit("b", "exception code EXP-214", 0.2, {}),
    ]

    class OneScore:
        def predict(self, pairs):
            return [0.5]

    with pytest.raises(ValueError) as error:
        rerank("EXP-214", hits, OneScore(), top_k=1)

    message = str(error.value)
    assert "1" in message
    assert "2" in message


def test_rerank_rejects_more_scores_than_hits():
    hits = [
        RetrievalHit("a", "holiday calendar", 0.1, {}),
        RetrievalHit("b", "exception code EXP-214", 0.2, {}),
    ]

    class ThreeScores:
        def predict(self, pairs):
            return [0.2, 0.4, 0.6]

    with pytest.raises(ValueError) as error:
        rerank("EXP-214", hits, ThreeScores(), top_k=1)

    message = str(error.value)
    assert "3" in message
    assert "2" in message


def test_rerank_rejects_a_scalar_score():
    hit = RetrievalHit("a", "exception code EXP-214", 0.1, {})

    class Scalar:
        def predict(self, pairs):
            return 1.0

    with pytest.raises(ValueError) as error:
        rerank("EXP-214", [hit], Scalar(), top_k=1)

    message = str(error.value)
    assert "None" in message
    assert "1" in message


def test_rerank_prefers_the_chunk_that_shares_the_subject():
    hits = [
        RetrievalHit("equipment", "holiday calendar and Northstar may provide a laptop", 0.1, {}),
        RetrievalHit(
            "leave",
            "Parental leave is administered under a separate Northstar policy",
            0.2,
            {},
        ),
    ]

    class LaptopFirst:
        def predict(self, pairs):
            return [5.0 if "laptop" in text else 0.1 for _question, text in pairs]

    ranked = rerank(
        "How many weeks of parental leave does Northstar provide?",
        hits,
        LaptopFirst(),
        top_k=1,
    )

    assert ranked[0][0].id == "leave"
    assert ranked[0][1] == 0.1


def test_rerank_uses_the_score_when_overlap_ties():
    hits = [
        RetrievalHit("low", "exception code EXP-214", 0.2, {}),
        RetrievalHit("high", "see EXP-214 in the policy", 0.1, {}),
    ]

    class HighSecond:
        def predict(self, pairs):
            return [0.2, 0.9]

    ranked = rerank("EXP-214", hits, HighSecond(), top_k=1)

    assert ranked[0][0].id == "high"
    assert ranked[0][1] == 0.9


def test_rerank_accepts_a_numpy_score_vector():
    numpy = pytest.importorskip("numpy")
    hits = [
        RetrievalHit("a", "holiday calendar", 0.1, {}),
        RetrievalHit("b", "exception code EXP-214", 0.2, {}),
    ]

    class VectorScores:
        def predict(self, pairs):
            return numpy.array([0.2, 1.0])

    ranked = rerank("EXP-214", hits, VectorScores(), top_k=1)

    assert ranked[0][0].id == "b"
    assert ranked[0][1] == 1.0
