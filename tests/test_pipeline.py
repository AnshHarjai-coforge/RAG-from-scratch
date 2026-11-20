import pytest

from rag.chunking import Chunk, FixedSizeChunker, section_for
from rag.documents import DocumentSource
from rag.pipeline import IndexingPipeline
from rag.vector_store import ChromaStore


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


class FailOnRepeatEmbedder(KeywordEmbedder):
    """Raises ValueError when the same passage text is embedded again."""

    def __init__(self) -> None:
        self.seen: set[tuple[str, ...]] = set()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        key = tuple(texts)
        if key in self.seen:
            raise ValueError("embedding failed on reindex")
        self.seen.add(key)
        return super().embed_passages(texts)


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


def test_deleted_file_on_reindex_removes_its_chunks(tmp_path):
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    remote = policy_dir / "remote.md"
    expense = policy_dir / "expense.md"
    remote.write_text("remote work stays in the index", encoding="utf-8")
    expense.write_text("expense reports stay in the index", encoding="utf-8")
    store = ChromaStore(tmp_path / "chroma")
    pipeline = IndexingPipeline(
        DocumentSource(policy_dir),
        FixedSizeChunker(),
        KeywordEmbedder(),
        store,
    )

    pipeline.index()
    remote.unlink()
    report = pipeline.index()

    assert store.ids_for("remote.md", "fixed") == []
    assert store.ids_for("expense.md", "fixed")
    assert any(item.name == "remote.md" and item.reason == "absent" for item in report.skipped)


def test_deleted_file_does_not_remove_another_strategy(tmp_path):
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    path = policy_dir / "remote.md"
    path.write_text("remote work section", encoding="utf-8")
    store = ChromaStore(tmp_path / "chroma")
    fixed = IndexingPipeline(
        DocumentSource(policy_dir), FixedSizeChunker(), KeywordEmbedder(), store
    )
    labeled = IndexingPipeline(DocumentSource(policy_dir), LabelChunker(), KeywordEmbedder(), store)

    fixed.index()
    labeled.index()
    label_ids = store.ids_for("remote.md", "label")
    path.unlink()
    fixed.index()

    assert store.ids_for("remote.md", "fixed") == []
    assert store.ids_for("remote.md", "label") == label_ids


def test_invalid_utf8_on_reindex_keeps_the_previous_chunks(tmp_path):
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
    path.write_bytes(b"\xff\xfe not utf-8")
    report = pipeline.index()

    assert any(
        item.name == "remote.md" and item.reason == "invalid_utf8" for item in report.skipped
    )
    hits = pipeline.retrieve("remote work")
    assert "remote work stays in the index" in hits[0].text


def test_invalid_embeddings_on_reindex_keeps_the_previous_chunks_and_updates_the_sibling(tmp_path):
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    remote = policy_dir / "remote.md"
    expense = policy_dir / "expense.md"
    remote.write_text("remote work stays in the index", encoding="utf-8")
    expense.write_text("expense report original", encoding="utf-8")
    store = ChromaStore(tmp_path / "chroma")
    pipeline = IndexingPipeline(
        DocumentSource(policy_dir),
        LabelChunker(),
        FailOnRepeatEmbedder(),
        store,
    )

    pipeline.index()
    expense.write_text("expense report updated", encoding="utf-8")
    report = pipeline.index()

    assert any(
        item.name == "remote.md" and item.reason == "invalid_embeddings" for item in report.skipped
    )
    assert "remote.md" not in report.files_indexed
    assert "expense.md" in report.files_indexed
    assert pipeline.retrieve("remote work")[0].text == "remote work stays in the index"
    assert pipeline.retrieve("expense report")[0].text == "expense report updated"


def test_superseded_file_is_not_indexed_and_old_rows_are_deleted(tmp_path):
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    remote = policy_dir / "remote.md"
    remote.write_text("remote three days", encoding="utf-8")
    store = ChromaStore(tmp_path / "chroma")
    pipeline = IndexingPipeline(
        DocumentSource(policy_dir),
        LabelChunker(),
        KeywordEmbedder(),
        store,
    )

    pipeline.index()
    remote.write_text("---\nstatus: superseded\n---\n\nthree days", encoding="utf-8")
    (policy_dir / "current.md").write_text("remote two days", encoding="utf-8")
    report = pipeline.index()

    assert store.ids_for("remote.md", "label") == []
    assert "remote.md" not in report.files_indexed
    assert any(item.name == "remote.md" and item.reason == "superseded" for item in report.skipped)
    assert "current.md" in report.files_indexed
    assert store.ids_for("current.md", "label")


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


def test_index_stores_the_section_for_each_chunk(tmp_path):
    text = (
        "# Remote Work Policy\n\n"
        "## 2. Standard remote-work arrangement\n\n"
        "Employees may work remote for two days."
    )
    pipeline = _pipeline(tmp_path, {"remote.md": text}, chunker=LabelChunker())
    pipeline.index()

    hits = pipeline.retrieve("remote days")

    assert hits[0].metadata["section"] == section_for(text, hits[0].text)


def test_blank_question_and_bad_top_k_are_rejected(tmp_path):
    pipeline = _pipeline(tmp_path, {"remote.md": "remote work"})
    pipeline.index()

    with pytest.raises(ValueError):
        pipeline.retrieve("  ")

    with pytest.raises(ValueError):
        pipeline.retrieve("remote", top_k=0)


def test_embedder_that_returns_the_wrong_count_skips_the_file(tmp_path):
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    (policy_dir / "remote.md").write_text("remote work stays", encoding="utf-8")

    class ShortEmbedder:
        def embed_passages(self, texts: list[str]) -> list[list[float]]:
            return []

        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

    pipeline = IndexingPipeline(
        DocumentSource(policy_dir),
        FixedSizeChunker(),
        ShortEmbedder(),
        ChromaStore(tmp_path / "chroma"),
    )

    report = pipeline.index()

    assert report.files_indexed == []
    assert any(item.reason == "invalid_embeddings" for item in report.skipped)
    assert pipeline.store.ids_for("remote.md", "fixed") == []
