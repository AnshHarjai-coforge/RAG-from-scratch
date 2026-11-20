"""Save policy chunks in Chroma and search them.

The database lives in a folder on disk, so the chunks are still there after
this script exits. Vectors are computed by the embedder and passed in.
Chroma does not embed the text again.
"""

from dataclasses import dataclass
from pathlib import Path

import chromadb

COLLECTION_NAME = "policies"


@dataclass(frozen=True)
class RetrievalHit:
    id: str
    text: str
    distance: float
    metadata: dict


class ChromaStore:
    """A cosine Chroma collection of precomputed embeddings.

    Saving an id that already exists replaces that row.
    """

    def __init__(self, persist_directory: Path, collection_name: str = COLLECTION_NAME) -> None:
        """Open the Chroma folder and the named collection. Rows stay on disk."""

        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self._open_collection(collection_name)

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Write chunks. An id that already exists replaces that row.

        The four lists line up: index 0 of each list is one chunk. Duplicate
        ids in one call, and embedding vectors of different lengths, are
        rejected before anything is written. A long write is split into batches
        Chroma will accept.
        """

        count = len(ids)
        if not (count == len(documents) == len(embeddings) == len(metadatas)):
            raise ValueError("ids, documents, embeddings, and metadata must be the same length")
        if count == 0:
            return
        if len(ids) != len(set(ids)):
            seen: set[str] = set()
            duplicates: list[str] = []
            for chunk_id in ids:
                if chunk_id in seen and chunk_id not in duplicates:
                    duplicates.append(chunk_id)
                seen.add(chunk_id)
            named = ", ".join(repr(chunk_id) for chunk_id in duplicates)
            raise ValueError(f"duplicate ids: {named}")
        lengths = {len(vector) for vector in embeddings}
        if len(lengths) != 1:
            raise ValueError("embedding vectors must all be the same length")

        batch_size = self.client.get_max_batch_size()
        for start in range(0, count, batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        strategy: str | None = None,
    ) -> list[RetrievalHit]:
        """Return the closest chunks for one question vector.

        Pass a strategy name to search only that chunker's rows. A smaller
        distance is a closer match.
        """

        where = {"strategy": strategy} if strategy is not None else None
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "distances", "metadatas"],
        )
        hits: list[RetrievalHit] = []
        for chunk_id, text, distance, metadata in zip(
            result["ids"][0],
            result["documents"][0],
            result["distances"][0],
            result["metadatas"][0],
        ):
            hits.append(
                RetrievalHit(
                    id=chunk_id,
                    text=text,
                    distance=distance,
                    metadata=dict(metadata or {}),
                )
            )
        return hits

    def ids_for(self, source: str, strategy: str) -> list[str]:
        """Return every stored id for one file and one chunking strategy.

        This looks up rows by metadata. It does not search the chunk text.
        """

        where = {"$and": [{"source": source}, {"strategy": strategy}]}
        result = self.collection.get(where=where, include=[])
        return list(result["ids"])

    def rows_for(self, strategy: str) -> list[RetrievalHit]:
        """Return every stored chunk for one chunking strategy.

        This reads rows by metadata. It does not rank them, so ``distance`` is 0.
        """

        result = self.collection.get(
            where={"strategy": strategy},
            include=["documents", "metadatas"],
        )
        documents = result["documents"] or []
        metadatas = result["metadatas"] or []
        hits: list[RetrievalHit] = []
        for chunk_id, text, metadata in zip(result["ids"], documents, metadatas):
            hits.append(
                RetrievalHit(
                    id=chunk_id,
                    text=text or "",
                    distance=0.0,
                    metadata=dict(metadata or {}),
                )
            )
        return hits

    def sources_for(self, strategy: str) -> list[str]:
        """Return the unique source names stored for one chunking strategy.

        Rows with no source are skipped. The names are sorted.
        """

        result = self.collection.get(where={"strategy": strategy}, include=["metadatas"])
        sources: set[str] = set()
        for metadata in result["metadatas"] or []:
            if not metadata:
                continue
            source = metadata.get("source")
            if isinstance(source, str) and source:
                sources.add(source)
        return sorted(sources)

    def count_for(self, strategy: str) -> int:
        """Return how many chunks are stored for one chunking strategy.

        This counts rows by metadata. It does not rank them.
        """

        result = self.collection.get(where={"strategy": strategy}, include=[])
        return len(result["ids"])

    def delete_ids(self, ids: list[str]) -> None:
        """Delete these rows. An empty list does nothing."""

        if not ids:
            return
        batch_size = self.client.get_max_batch_size()
        for start in range(0, len(ids), batch_size):
            self.collection.delete(ids=ids[start : start + batch_size])

    def delete_source(self, source: str, strategy: str) -> None:
        """Delete one file's chunks for one strategy. Other strategies stay.

        Used when a file comes back empty, so there are no new ids to compare.
        """

        self.delete_ids(self.ids_for(source, strategy))

    def count(self) -> int:
        """Return how many chunks are stored in the collection."""

        return self.collection.count()

    def _open_collection(self, name: str):
        """Create the collection, or reopen it if it uses cosine distance.

        Embeddings are passed in, so Chroma is told not to embed the text.
        """

        existing = [item for item in self.client.list_collections() if item.name == name]
        if not existing:
            return self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )

        collection = self.client.get_collection(name=name, embedding_function=None)
        space = (collection.metadata or {}).get("hnsw:space")
        if space != "cosine":
            raise ValueError(
                f"Collection {name!r} uses distance {space!r}. "
                "Create the store in a new directory so it can use cosine distance."
            )
        return collection
