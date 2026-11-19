"""Chunking strategies for the policy index.

``IndexingPipeline`` depends on the ``Chunker`` protocol, not on one
strategy. ``FixedSizeChunker`` is the strategy used now. A later strategy
implements the same protocol and is passed into the pipeline in its place:

- structural chunking splits on document structure such as headings
- semantic chunking splits where the topic changes
- parent-child chunking returns small retrieval chunks and carries the
  parent context in ``Chunk.extra`` (for example ``parent_id`` or ``section``)

Strategy-specific metadata belongs in ``Chunk.extra``. The pipeline copies
those fields onto each stored row. Record ids include the strategy name, so
two strategies can live in one collection and be retrieved separately.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

MetadataValue = str | int | float | bool


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of a document.

    ``index`` is unique for a single document and strategy. ``extra`` holds
    fields a strategy needs beyond the text, such as a section title or a
    parent id. Values must be scalars Chroma can store.
    """

    text: str
    index: int
    strategy: str
    extra: Mapping[str, MetadataValue] = field(default_factory=dict)


class Chunker(Protocol):
    """A replaceable way to turn one document into chunks."""

    name: str

    def chunk(self, text: str) -> list[Chunk]:
        """Split ``text`` into chunks. Empty input returns an empty list."""


class FixedSizeChunker:
    """Split text into fixed character windows with overlap.

    A 500-character window is about one policy rule, which is the unit a
    question should retrieve. The 50-character overlap is 10 percent of that
    window, so a sentence that lands on a cut still appears whole in the next
    chunk. The model context window is far larger than 500 characters. The
    size is chosen for retrieval precision.

    The chunker ends a window on whitespace when that break leaves room for
    the overlap to advance. A span with no usable whitespace is cut at
    ``chunk_size`` so the window always moves forward.
    """

    name = "fixed"

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be zero or more and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[Chunk]:
        normalized = text.strip()
        if not normalized:
            return []
        if len(normalized) <= self.chunk_size:
            return [self._make(normalized, 0)]

        pieces: list[str] = []
        start = 0
        length = len(normalized)
        while start < length:
            end = min(start + self.chunk_size, length)
            if end < length:
                whitespace_at = self._last_whitespace(normalized, start, end)
                if whitespace_at is not None and whitespace_at - start > self.overlap:
                    end = whitespace_at
            piece = normalized[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= length:
                break
            next_start = end - self.overlap
            if next_start <= start:
                next_start = end
            start = next_start

        return [self._make(piece, index) for index, piece in enumerate(pieces)]

    def _make(self, text: str, index: int) -> Chunk:
        return Chunk(text=text, index=index, strategy=self.name)

    @staticmethod
    def _last_whitespace(text: str, start: int, end: int) -> int | None:
        for index in range(end - 1, start, -1):
            if text[index].isspace():
                return index
        return None
