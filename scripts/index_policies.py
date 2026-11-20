"""Index the policy documents and run one retrieval check."""

from rag.chunking import FixedSizeChunker
from rag.config import get_settings
from rag.documents import DocumentSource
from rag.embedding import QwenEmbedder
from rag.pipeline import IndexingPipeline
from rag.vector_store import ChromaStore

_QUESTION = "How many days per week may an employee work remotely?"


def main() -> None:
    settings = get_settings()
    source = DocumentSource(settings.data_dir, max_bytes=settings.max_file_bytes)
    chunker = FixedSizeChunker(settings.chunk_size, settings.chunk_overlap)
    embedder = QwenEmbedder(settings.embedding_model, batch_size=settings.embedding_batch_size)
    store = ChromaStore(settings.chroma_dir, collection_name=settings.collection_name)
    pipeline = IndexingPipeline(source, chunker, embedder, store)

    report = pipeline.index()
    print(f"Indexed {len(report.files_indexed)} files, {report.chunks_stored} chunks")
    for name in report.files_indexed:
        print(f"  {name}")
    for item in report.skipped:
        print(f"Skipped {item.name}: {item.reason}")
    print(f"Collection count: {store.count()}")

    print(f"\nQuestion: {_QUESTION}")
    for rank, hit in enumerate(pipeline.retrieve(_QUESTION, top_k=settings.top_k), start=1):
        print(f"\n#{rank} distance={hit.distance:.4f} id={hit.id}")
        print(hit.text[:400])


if __name__ == "__main__":
    main()
