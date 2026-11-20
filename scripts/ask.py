"""Ask one question and print retrieval, the answer, and the sources.

Pass ``--index`` to rebuild the vector store before answering. Without that
flag, the question is answered from the chunks already stored on disk.
"""

import argparse
import sys

from rag.answer import generate_answer
from rag.chunking import FixedSizeChunker
from rag.config import get_settings
from rag.documents import DocumentSource
from rag.embedding import QwenEmbedder
from rag.pipeline import IndexingPipeline
from rag.rerank import CrossEncoderReranker, rerank
from rag.vector_store import ChromaStore


def parse_args(argv: list[str] | None = None) -> tuple[bool, str]:
    """Return whether to index, and the question."""

    parser = argparse.ArgumentParser(
        description="Ask one question and print retrieval, the answer, and the sources.",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Rebuilds the vector store before answering.",
    )
    parser.add_argument("question", nargs="+")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return args.index, " ".join(args.question)


def main(argv: list[str] | None = None) -> None:
    """Answer the question from the command line. Index first only with ``--index``."""

    should_index, question = parse_args(argv)

    settings = get_settings()
    pipeline = IndexingPipeline(
        DocumentSource(settings.data_dir, max_bytes=settings.max_file_bytes),
        FixedSizeChunker(settings.chunk_size, settings.chunk_overlap),
        QwenEmbedder(settings.embedding_model, batch_size=settings.embedding_batch_size),
        ChromaStore(settings.chroma_dir, collection_name=settings.collection_name),
    )
    if should_index:
        report = pipeline.index()
        print(f"Indexed {len(report.files_indexed)} files, {report.chunks_stored} chunks")

    reranker = CrossEncoderReranker(settings.cross_encoder_model)
    print(f"\nQuestion: {question}")

    print("\nVector only:")
    for hit in pipeline.retrieve(question, top_k=settings.top_k):
        print(f"- {hit.id}")

    merged = pipeline.retrieve_hybrid(question, top_k=settings.top_k)
    print("\nHybrid:")
    for hit in merged:
        print(f"- {hit.id}")

    ranked = rerank(question, merged, reranker, settings.top_k)
    print("\nReranked:")
    for rank, (hit, score) in enumerate(ranked, start=1):
        print(f"#{rank} score={score:.4f} id={hit.id}")

    hits = [hit for hit, _score in ranked]
    answer = generate_answer(question, hits, settings)
    print("\nAnswer:")
    print(answer)
    print("\nSources:")
    for hit in hits:
        source = hit.metadata.get("source", "unknown")
        section = hit.metadata.get("section") or "body"
        print(f"- {source} § {section}")


if __name__ == "__main__":
    main()
