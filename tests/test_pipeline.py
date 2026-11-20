import pytest

from chunking import Chunk, FixedSizeChunker
from documents import DocumentSource
from pipeline import IndexingPipeline
from vector_store import ChromaStore


class KeywordEmbedder:
    """Maps a few keywords to fixed vectors so retrieval tests stay local."""

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("question must not be empty")
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        if "BADVECTOR" in text:
            return [float("nan"), 0.0]
        if "remote" in text.lower():
            return [1.0, 0.0, 0.0]
        if "expense" in text.lower():
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


class LabelChunker:
    name = "label"

    def chunk(self, text: str) -> list[Chunk]:
        body = text.strip()
        if not body:
            return []
        return [Chunk(text=body, index=0, strategy=self.name)]


def _pipeline(tmp_path, policies, chunker=None):
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    for name, text in policies.items():
        (policy_dir / name).write_text(text, encoding="utf-8")
    return IndexingPipeline(
        DocumentSource(policy_dir),
        chunker or FixedSizeChunker(chunk_size=40, overlap=10),
        KeywordEmbedder(),
        ChromaStore(tmp_path / "chroma"),
    )


def test_index_and_retrieve_returns_the_matching_policy(tmp_path):
    pipeline = _pipeline(
        tmp_path,
        {
            "remote.md": "Employees may work remote for two days each week. " * 3,
            "expense.md": "Submit an expense report within thirty days. " * 3,
        },
    )

    report = pipeline.index()
    hits = pipeline.retrieve("How many remote days are allowed?")

    assert report.files_indexed == ["expense.md", "remote.md"]
    assert report.chunks_stored > 0
    assert hits[0].metadata["source"] == "remote.md"
    assert "remote" in hits[0].text


def test_reindex_does_not_grow_the_collection(tmp_path):
    pipeline = _pipeline(tmp_path, {"remote.md": "remote work " * 20})

    first = pipeline.index()
    second = pipeline.index()

    assert second.chunks_stored == first.chunks_stored
    assert pipeline.store.count() == first.chunks_stored


def test_shorter_file_drops_old_tail_ids(tmp_path):
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    path = policy_dir / "remote.md"
    path.write_text("remote work agreement " * 30, encoding="utf-8")
    store = ChromaStore(tmp_path / "chroma")
    pipeline = IndexingPipeline(
        DocumentSource(policy_dir),
        FixedSizeChunker(chunk_size=30, overlap=5),
        KeywordEmbedder(),
        store,
    )

    pipeline.index()
    assert any(chunk_id.endswith(":2") for chunk_id in store.ids_for("remote.md", "fixed"))

    path.write_text("remote work", encoding="utf-8")
    pipeline.index()

    ids = store.ids_for("remote.md", "fixed")
    assert ids == ["fixed:remote.md:0"]


def test_empty_file_on_reindex_removes_its_chunks(tmp_path):
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    path = policy_dir / "remote.md"
    path.write_text("remote work stays in the index", encoding="utf-8")
    store = ChromaStore(tmp_path / "chroma")
    pipeline = IndexingPipeline(
        DocumentSource(policy_dir),
        FixedSizeChunker(),
        KeywordEmbedder(),
        store,
    )

    pipeline.index()
    path.write_text("  \n", encoding="utf-8")
    report = pipeline.index()

    assert store.ids_for("remote.md", "fixed") == []
    assert report.skipped[0].reason == "empty"


def test_invalid_embeddings_skip_the_file_and_keep_the_others(tmp_path):
    pipeline = _pipeline(
        tmp_path,
        {
            "bad.md": "BADVECTOR should not be stored",
            "remote.md": "remote work is allowed",
        },
    )

    report = pipeline.index()

    assert report.files_indexed == ["remote.md"]
    assert any(
        item.reason == "invalid_embeddings" and item.name == "bad.md" for item in report.skipped
    )
    assert pipeline.store.ids_for("bad.md", "fixed") == []


def test_a_second_strategy_can_share_the_collection(tmp_path):
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    (policy_dir / "remote.md").write_text("remote work section", encoding="utf-8")
    store = ChromaStore(tmp_path / "chroma")
    fixed = IndexingPipeline(
        DocumentSource(policy_dir), FixedSizeChunker(), KeywordEmbedder(), store
    )
    labeled = IndexingPipeline(DocumentSource(policy_dir), LabelChunker(), KeywordEmbedder(), store)

    fixed.index()
    labeled.index()
    hits = labeled.retrieve("remote work", strategy="label")

    assert store.ids_for("remote.md", "fixed")
    assert hits[0].metadata["strategy"] == "label"
    assert hits[0].metadata["source"] == "remote.md"
    assert fixed.retrieve("remote work", strategy="fixed")[0].metadata["strategy"] == "fixed"


def test_blank_question_and_bad_top_k_are_rejected(tmp_path):
    pipeline = _pipeline(tmp_path, {"remote.md": "remote work"})
    pipeline.index()

    with pytest.raises(ValueError):
        pipeline.retrieve("  ")

    with pytest.raises(ValueError):
        pipeline.retrieve("remote", top_k=0)
