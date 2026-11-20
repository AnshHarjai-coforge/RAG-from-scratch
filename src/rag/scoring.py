"""Decide whether a retrieved chunk and an answer support a case."""

import re

from rag.vector_store import RetrievalHit


def phrase_supported(answer: str, accept: str, reject: str) -> bool:
    """Return whether the answer states the fact and does not negate it.

    A reject match fails the check. Otherwise the accept pattern must match.
    Matching ignores case.
    """

    if re.search(reject, answer, re.IGNORECASE):
        return False
    return re.search(accept, answer, re.IGNORECASE) is not None


def source_supported(hits: list[RetrievalHit], expected_source: str, accept: str) -> bool:
    """Return whether one hit from the expected file states the fact.

    The file name and the accept pattern must match the same hit. Matching
    ignores case. The reject pattern is not applied to chunk text.
    """

    for hit in hits:
        if hit.metadata.get("source") != expected_source:
            continue
        if re.search(accept, hit.text, re.IGNORECASE):
            return True
    return False


def case_passed(
    case: dict[str, str],
    hits: list[RetrievalHit],
    answer: str,
) -> tuple[bool, bool]:
    """Return ``(source_ok, phrase_ok)`` for one evaluation case."""

    source_ok = source_supported(hits, case["expected_source"], case["accept"])
    phrase_ok = phrase_supported(answer, case["accept"], case["reject"])
    return source_ok, phrase_ok
