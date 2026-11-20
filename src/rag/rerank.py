"""Re-score retrieved chunks with a cross-encoder."""

import re

from sentence_transformers import CrossEncoder

from rag.vector_store import RetrievalHit

_TOKEN = re.compile(r"(?<![a-z0-9-])[a-z0-9]+(?:-[a-z0-9]+)*(?![a-z0-9-])")
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "can",
        "did",
        "do",
        "does",
        "for",
        "how",
        "in",
        "is",
        "many",
        "of",
        "on",
        "or",
        "should",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)


class CrossEncoderReranker:
    """Score question and chunk pairs. The model loads once."""

    def __init__(self, model_name: str) -> None:
        """Load the cross-encoder named in settings."""

        self.model = CrossEncoder(model_name)

    def predict(self, pairs: list[list[str]]):
        """Return one relevance score per ``[question, chunk]`` pair."""

        return self.model.predict(pairs)


def rerank(
    question: str,
    hits: list[RetrievalHit],
    reranker,
    top_k: int,
) -> list[tuple[RetrievalHit, float]]:
    """Order chunks by shared content words, then by the cross-encoder score.

    ``reranker.predict`` takes pairs of ``[question, chunk text]`` and returns
    one score per pair. A higher score is a better match. ``top_k`` is applied
    after that order.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not hits:
        return []

    scores = reranker.predict([[question, hit.text] for hit in hits])
    score_count = _score_count(scores)
    if score_count != len(hits):
        raise ValueError(f"reranker returned {score_count} scores for {len(hits)} hits")

    ranked = sorted(
        zip(hits, scores),
        key=lambda row: (_overlap(question, row[0].text), float(row[1])),
        reverse=True,
    )
    return [(hit, float(score)) for hit, score in ranked[:top_k]]


def _score_count(scores) -> int | None:
    """Return how many scores came back, or None when ``scores`` has no length."""

    try:
        return len(scores)
    except TypeError:
        return None


def _tokens(text: str) -> list[str]:
    """Lowercase content tokens.

    ``EXP-214`` stays one token, and a short word does not match inside a longer one.
    """

    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOP]


def _overlap(question: str, chunk: str) -> int:
    """How many question content tokens appear in the chunk as whole tokens."""

    chunk_tokens = set(_tokens(chunk))
    return sum(1 for token in _tokens(question) if token in chunk_tokens)
