"""Score the fixed questions for retrieval recall and answer accuracy."""

import argparse
import sys

from rag.answer import generate_answer
from rag.chunking import FixedSizeChunker
from rag.config import get_settings
from rag.documents import DocumentSource
from rag.embedding import QwenEmbedder
from rag.eval_cases import CASES
from rag.pipeline import IndexingPipeline
from rag.rerank import CrossEncoderReranker, rerank
from rag.scoring import case_passed
from rag.vector_store import ChromaStore


def main(argv: list[str] | None = None) -> int:
    """Parse the command line, then score retrieval and the answers."""

    parser = argparse.ArgumentParser(
        description="score the fixed questions for retrieval recall and answer accuracy.",
    )
    parser.parse_args(sys.argv[1:] if argv is None else argv)
    return _run()


def _run() -> int:
    """Index the policies and print recall and answer accuracy."""

    settings = get_settings()
    pipeline = IndexingPipeline(
        DocumentSource(settings.data_dir, max_bytes=settings.max_file_bytes),
        FixedSizeChunker(settings.chunk_size, settings.chunk_overlap),
        QwenEmbedder(settings.embedding_model, batch_size=settings.embedding_batch_size),
        ChromaStore(settings.chroma_dir, collection_name=settings.collection_name),
    )
    report = pipeline.index()
    print(f"Indexed {len(report.files_indexed)} files, {report.chunks_stored} chunks")

    reranker = CrossEncoderReranker(settings.cross_encoder_model)
    recalled = 0
    answered = 0
    for case in CASES:
        ranked = rerank(
            case["question"],
            pipeline.retrieve_hybrid(case["question"], top_k=settings.top_k),
            reranker,
            settings.top_k,
        )
        hits = [hit for hit, _score in ranked]
        answer = generate_answer(case["question"], hits, settings)
        source_ok, phrase_ok = case_passed(case, hits, answer)
        recalled += int(source_ok)
        answered += int(phrase_ok)
        print(
            f"{'PASS' if source_ok and phrase_ok else 'FAIL'} {case['id']}: "
            f"source={source_ok} phrase={phrase_ok}"
        )

    total = len(CASES)
    print(f"\nRecall: {recalled}/{total}")
    print(f"Answer accuracy: {answered}/{total}")
    if recalled == total and answered == total:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
