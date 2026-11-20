import chromadb
import pytest

from rag.vector_store import ChromaStore


def _store(tmp_path) -> ChromaStore:
    return ChromaStore(tmp_path)


def test_upsert_then_query_returns_the_closer_chunk(tmp_path):
    store = _store(tmp_path)
    store.upsert(
        ids=["fixed:a.md:0", "fixed:b.md:0"],
        documents=["remote work policy", "expense policy"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[
            {"source": "a.md", "chunk_index": 0, "strategy": "fixed"},
            {"source": "b.md", "chunk_index": 0, "strategy": "fixed"},
        ],
    )

    hits = store.query([1.0, 0.0], top_k=1)

    assert hits[0].text == "remote work policy"
    assert hits[0].metadata["source"] == "a.md"
    assert hits[0].distance < 0.1


def test_second_upsert_replaces_the_same_id(tmp_path):
    store = _store(tmp_path)
    metadata = {"source": "a.md", "chunk_index": 0, "strategy": "fixed"}
    store.upsert(["fixed:a.md:0"], ["first"], [[1.0, 0.0]], [metadata])
    store.upsert(["fixed:a.md:0"], ["second"], [[1.0, 0.0]], [metadata])

    hits = store.query([1.0, 0.0], top_k=2)

    assert store.count() == 1
    assert hits[0].text == "second"


def test_length_mismatch_is_rejected(tmp_path):
    store = _store(tmp_path)

    with pytest.raises(ValueError):
        store.upsert(
            ids=["fixed:a.md:0"],
            documents=["text"],
            embeddings=[[1.0, 0.0]],
            metadatas=[],
        )


def test_top_k_larger_than_the_collection_returns_what_exists(tmp_path):
    store = _store(tmp_path)
    store.upsert(
        ["fixed:a.md:0"],
        ["only"],
        [[1.0, 0.0]],
        [{"source": "a.md", "chunk_index": 0, "strategy": "fixed"}],
    )

    hits = store.query([1.0, 0.0], top_k=5)

    assert len(hits) == 1


def test_empty_collection_returns_no_hits(tmp_path):
    store = _store(tmp_path)

    assert store.query([1.0, 0.0], top_k=3) == []


def test_strategy_filter_ignores_other_strategies(tmp_path):
    store = _store(tmp_path)
    store.upsert(
        ids=["fixed:a.md:0", "structural:a.md:0"],
        documents=["fixed text", "structural text"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[
            {"source": "a.md", "chunk_index": 0, "strategy": "fixed"},
            {"source": "a.md", "chunk_index": 0, "strategy": "structural"},
        ],
    )

    hits = store.query([0.0, 1.0], top_k=3, strategy="structural")

    assert [hit.text for hit in hits] == ["structural text"]


def test_delete_source_removes_only_that_strategy(tmp_path):
    store = _store(tmp_path)
    store.upsert(
        ids=["fixed:a.md:0", "fixed:a.md:1", "structural:a.md:0"],
        documents=["one", "two", "other"],
        embeddings=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        metadatas=[
            {"source": "a.md", "chunk_index": 0, "strategy": "fixed"},
            {"source": "a.md", "chunk_index": 1, "strategy": "fixed"},
            {"source": "a.md", "chunk_index": 0, "strategy": "structural"},
        ],
    )

    store.delete_source("a.md", "fixed")

    assert store.ids_for("a.md", "fixed") == []
    assert store.ids_for("a.md", "structural") == ["structural:a.md:0"]


def test_existing_collection_with_another_distance_is_rejected(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection(
        "policies",
        metadata={"hnsw:space": "l2"},
        embedding_function=None,
    )

    with pytest.raises(ValueError, match="cosine"):
        ChromaStore(tmp_path)
