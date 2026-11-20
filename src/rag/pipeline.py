"""Index policy files and retrieve the closest chunks."""

import math
from dataclasses import dataclass

from rag.chunking import Chunk, section_for
from rag.documents import DocumentSource, SkippedFile
from rag.vector_store import ChromaStore, RetrievalHit


@dataclass(frozen=True)
class IndexReport:
    """The files stored, the chunk count, and the files that were skipped."""

    files_indexed: list[str]
    chunks_stored: int
    skipped: list[SkippedFile]


class IndexingPipeline:
    """Load, chunk, embed, and store policy files, then search them."""

    def __init__(self, source: DocumentSource, chunker, embedder, store: ChromaStore) -> None:
        """Keep the reader, chunker, embedder, and store this run will use."""

        self.source = source
        self.chunker = chunker
        self.embedder = embedder
        self.store = store

    def index(self) -> IndexReport:
        """Chunk and embed each policy file, then save the chunks.

        A file that is gone, or an empty file, deletes that file's old rows
        for this chunker. A failed re-read keeps the rows already stored. A
        superseded file is deleted and not stored. A shorter file drops the
        chunk ids it no longer has. A bad embedding skips that file and leaves
        its previous rows in place.
        """

        documents, skipped = self.source.load()
        skipped = list(skipped)
        files_indexed: list[str] = []
        chunks_stored = 0

        seen = {document.name for document in documents}
        seen.update(item.name for item in skipped)
        for name in self.store.sources_for(self.chunker.name):
            if name not in seen:
                self.store.delete_source(name, self.chunker.name)
                skipped.append(SkippedFile(name, "absent"))

        for item in skipped:
            if item.reason == "empty":
                self.store.delete_source(item.name, self.chunker.name)

        for document in documents:
            if document.status == "superseded":
                self.store.delete_source(document.name, self.chunker.name)
                skipped.append(SkippedFile(document.name, "superseded"))
                continue

            chunks = self.chunker.chunk(document.text)
            try:
                vectors = self._embed(chunks)
            except ValueError:
                skipped.append(SkippedFile(document.name, "invalid_embeddings"))
                continue

            ids = [f"{chunk.strategy}:{document.name}:{chunk.index}" for chunk in chunks]
            self.store.upsert(
                ids=ids,
                documents=[chunk.text for chunk in chunks],
                embeddings=vectors,
                metadatas=[
                    {
                        "source": document.name,
                        "chunk_index": chunk.index,
                        "strategy": chunk.strategy,
                        "section": section_for(document.text, chunk.text, end=chunk.end),
                    }
                    for chunk in chunks
                ],
            )
            current = set(ids)
            stale = [
                chunk_id
                for chunk_id in self.store.ids_for(document.name, self.chunker.name)
                if chunk_id not in current
            ]
            self.store.delete_ids(stale)
            files_indexed.append(document.name)
            chunks_stored += len(chunks)

        return IndexReport(
            files_indexed=files_indexed,
            chunks_stored=chunks_stored,
            skipped=skipped,
        )

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
        strategy: str | None = None,
    ) -> list[RetrievalHit]:
        """Embed the question and return the closest stored chunks.

        A blank question or a top_k below 1 is rejected before the embedder runs.
        """

        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not question.strip():
            raise ValueError("question must not be empty")
        selected = self.chunker.name if strategy is None else strategy
        return self.store.query(
            self.embedder.embed_query(question),
            top_k=top_k,
            strategy=selected,
        )

    def _embed(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed these chunks. A missing or non-finite vector raises ValueError."""

        vectors = self.embedder.embed_passages([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("embedder returned a different number of vectors than chunks")
        for vector in vectors:
            if not vector or any(not math.isfinite(value) for value in vector):
                raise ValueError("embedding contains a non-finite value")
        return vectors
