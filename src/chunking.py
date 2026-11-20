"""Split a policy document into fixed-size pieces."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One piece of a document. ``index`` starts at 0 for each file."""

    text: str
    index: int
    strategy: str


class FixedSizeChunker:
    """Split text into character windows with overlap.

    A window ends on whitespace when that break leaves room for the overlap
    to move forward. A span with no usable whitespace is cut at ``chunk_size``.
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
            return [Chunk(text=normalized, index=0, strategy=self.name)]

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

        return [
            Chunk(text=piece, index=index, strategy=self.name) for index, piece in enumerate(pieces)
        ]

    @staticmethod
    def _last_whitespace(text: str, start: int, end: int) -> int | None:
        for index in range(end - 1, start, -1):
            if text[index].isspace():
                return index
        return None
