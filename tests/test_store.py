import chromadb
import pytest

from rag.vector_store import ChromaStore


def _store(tmp_path) -> ChromaStore:
    return ChromaStore(tmp_path)


def _batch_size_of_two(store, monkeypatch) -> None:
    monkeypatch.setattr(store.client, "get_max_batch_size", lambda: 2)


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


def test_rows_for_returns_text_for_one_strategy(tmp_path):
    store = _store(tmp_path)
    store.upsert(
        ids=["fixed:a.md:0", "structural:a.md:0"],
        documents=["fixed text", "structural text"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[
            {"source": "a.md", "chunk_index": 0, "strategy": "fixed", "section": "1. Purpose"},
            {"source": "a.md", "chunk_index": 0, "strategy": "structural"},
        ],
    )

    rows = store.rows_for("fixed")

    assert [row.id for row in rows] == ["fixed:a.md:0"]
    assert rows[0].text == "fixed text"
    assert rows[0].distance == 0.0
    assert rows[0].metadata["section"] == "1. Purpose"


def test_existing_collection_with_another_distance_is_rejected(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection(
        "policies",
        metadata={"hnsw:space": "l2"},
        embedding_function=None,
    )

    with pytest.raises(ValueError, match="cosine"):
        ChromaStore(tmp_path)


def test_mixed_dimensions_past_the_batch_size_write_nothing(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _batch_size_of_two(store, monkeypatch)
    ids = ["fixed:a.md:0", "fixed:b.md:0", "fixed:c.md:0"]
    documents = ["one", "two", "three"]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0, 0.0]]
    metadatas = [
        {"source": "a.md", "chunk_index": 0, "strategy": "fixed"},
        {"source": "b.md", "chunk_index": 0, "strategy": "fixed"},
        {"source": "c.md", "chunk_index": 0, "strategy": "fixed"},
    ]

    with pytest.raises(ValueError):
        store.upsert(ids, documents, embeddings, metadatas)

    assert store.count() == 0

    store.upsert(
        ["fixed:kept.md:0"],
        ["original"],
        [[1.0, 0.0]],
        [{"source": "kept.md", "chunk_index": 0, "strategy": "fixed"}],
    )

    with pytest.raises(ValueError):
        store.upsert(ids, documents, embeddings, metadatas)

    assert store.count() == 1
    assert [row.text for row in store.rows_for("fixed")] == ["original"]


def test_duplicate_id_past_the_batch_size_does_not_overwrite(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _batch_size_of_two(store, monkeypatch)
    kept = {"source": "a.md", "chunk_index": 0, "strategy": "fixed"}
    store.upsert(["fixed:a.md:0"], ["original"], [[1.0, 0.0]], [kept])

    with pytest.raises(ValueError, match="fixed:b\\.md:0"):
        store.upsert(
            ids=["fixed:a.md:0", "fixed:b.md:0", "fixed:b.md:0"],
            documents=["changed", "new", "new again"],
            embeddings=[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            metadatas=[
                kept,
                {"source": "b.md", "chunk_index": 0, "strategy": "fixed"},
                {"source": "b.md", "chunk_index": 1, "strategy": "fixed"},
            ],
        )

    assert store.count() == 1
    assert [row.text for row in store.rows_for("fixed")] == ["original"]
    assert store.ids_for("b.md", "fixed") == []


def test_valid_upsert_past_the_batch_size_stores_every_row(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _batch_size_of_two(store, monkeypatch)
    store.upsert(
        ids=["fixed:a.md:0", "fixed:b.md:0", "fixed:c.md:0"],
        documents=["one", "two", "three"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        metadatas=[
            {"source": "a.md", "chunk_index": 0, "strategy": "fixed"},
            {"source": "b.md", "chunk_index": 0, "strategy": "fixed"},
            {"source": "c.md", "chunk_index": 0, "strategy": "fixed"},
        ],
    )

    assert store.count() == 3
    assert sorted(row.id for row in store.rows_for("fixed")) == [
        "fixed:a.md:0",
        "fixed:b.md:0",
        "fixed:c.md:0",
    ]


def test_empty_upsert_writes_nothing(tmp_path):
    store = _store(tmp_path)

    store.upsert([], [], [], [])

    assert store.count() == 0


def test_reopen_reads_the_existing_cosine_collection(tmp_path):
    first = _store(tmp_path)
    first.upsert(
        ["fixed:a.md:0"],
        ["remote work"],
        [[1.0, 0.0]],
        [{"source": "a.md", "chunk_index": 0, "strategy": "fixed"}],
    )

    second = ChromaStore(tmp_path)

    assert second.count() == 1
    assert second.query([1.0, 0.0], top_k=1)[0].text == "remote work"


def test_sources_for_skips_rows_without_metadata(tmp_path, monkeypatch):
    store = _store(tmp_path)

    def get(*args, **kwargs):
        return {
            "ids": ["fixed:a.md:0", "fixed:none:0", "fixed:empty:0"],
            "metadatas": [{"source": "a.md"}, None, {}],
        }

    monkeypatch.setattr(store.collection, "get", get)

    assert store.sources_for("fixed") == ["a.md"]


def test_sources_for_and_count_for(tmp_path):
    store = _store(tmp_path)
    store.upsert(
        ids=["fixed:a.md:0", "fixed:b.md:0", "structural:c.md:0"],
        documents=["fixed a", "fixed b", "structural c"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        metadatas=[
            {"source": "a.md", "chunk_index": 0, "strategy": "fixed"},
            {"source": "b.md", "chunk_index": 0, "strategy": "fixed"},
            {"source": "c.md", "chunk_index": 0, "strategy": "structural"},
        ],
    )

    assert store.sources_for("fixed") == ["a.md", "b.md"]
    assert store.sources_for("structural") == ["c.md"]
    assert store.count_for("fixed") == 2
    assert store.count_for("structural") == 1
