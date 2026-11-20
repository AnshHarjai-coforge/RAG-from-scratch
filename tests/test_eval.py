"""Offline recall and answer checks for the fixed question list."""

from rag.answer import generate_answer
from rag.chunking import Chunk
from rag.config import get_settings
from rag.documents import DocumentSource
from rag.eval_cases import CASES
from rag.pipeline import IndexingPipeline
from rag.rerank import rerank
from rag.vector_store import ChromaStore


class KeywordEmbedder:
    """Maps a few keywords to fixed vectors so this harness stays local."""

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        if "remote" in lowered:
            return [1.0, 0.0, 0.0]
        if "expense" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


class LabelChunker:
    name = "label"

    def chunk(self, text: str) -> list[Chunk]:
        body = text.strip()
        if not body:
            return []
        return [Chunk(text=body, index=0, strategy=self.name)]


class KeepOrder:
    """Leaves the merged list in the order retrieval already produced."""

    def predict(self, pairs):
        return list(range(len(pairs), 0, -1))


def _fixtures(case: dict) -> dict[str, str]:
    question = case["question"].lower()
    phrase = case["expected_phrase"]
    if "expense" in question:
        body = f"This expense policy states {phrase}."
        other = "Employees may work remote for two days."
    elif "remote" in question:
        body = f"This remote policy states {phrase}."
        other = "Submit an expense report within thirty days."
    else:
        body = f"This policy states {phrase}."
        other = "Employees may work remote for two days."
    return {case["expected_source"]: body, "other.md": other}


def test_each_case_recalls_the_source_and_the_phrase(tmp_path):
    assert len(CASES) >= 8
    settings = get_settings()

    for case in CASES:
        policy_dir = tmp_path / case["id"]
        policy_dir.mkdir()
        for name, text in _fixtures(case).items():
            (policy_dir / name).write_text(text, encoding="utf-8")
        pipeline = IndexingPipeline(
            DocumentSource(policy_dir),
            LabelChunker(),
            KeywordEmbedder(),
            ChromaStore(policy_dir / "chroma"),
        )
        pipeline.index()

        ranked = rerank(
            case["question"],
            pipeline.retrieve_hybrid(case["question"], top_k=1),
            KeepOrder(),
            top_k=1,
        )
        hits = [hit for hit, _score in ranked]

        def complete(prompt: str, found=hits) -> str:
            return "\n".join(hit.text for hit in found)

        answer = generate_answer(case["question"], hits, settings, complete=complete)

        assert hits[0].metadata["source"] == case["expected_source"]
        assert case["expected_phrase"].lower() in answer.lower()
