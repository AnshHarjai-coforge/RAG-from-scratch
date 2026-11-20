"""Split a policy document into fixed-size pieces."""

from dataclasses import dataclass


def section_for(document: str, chunk_text: str, end: int | None = None) -> str:
    """Return the last markdown heading at or before the end of this chunk.

    A heading is a line that starts with ``#``. The returned text has those
    marks removed. A chunk that ends before any heading is ``body``.
    When end is provided, a heading line already started by that index
    contributes its full title.
    """

    text = document.strip()
    if end is not None:
        heading = "body"
        line_start = 0
        for line in text.splitlines(keepends=True):
            if line_start >= end:
                break
            stripped = line.strip()
            line_start += len(line)
            if not stripped.startswith("#"):
                continue
            title = stripped.lstrip("#").strip()
            if title:
                heading = title
        return heading

    piece = chunk_text.strip()
    if not piece:
        return "body"

    start = text.find(piece)
    window = text[: start + len(piece)] if start >= 0 else piece
    heading = "body"
    for line in window.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        title = stripped.lstrip("#").strip()
        if title:
            heading = title
    return heading


@dataclass(frozen=True)
class Chunk:
    """One piece of a document. ``index`` starts at 0 for each file."""

    text: str
    index: int
    strategy: str
    start: int | None = None
    end: int | None = None


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
            return [
                Chunk(
                    text=normalized,
                    index=0,
                    strategy=self.name,
                    start=0,
                    end=len(normalized),
                )
            ]

        chunks: list[Chunk] = []
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
                chunks.append(
                    Chunk(
                        text=piece,
                        index=len(chunks),
                        strategy=self.name,
                        start=start,
                        end=end,
                    )
                )
            if end >= length:
                break
            next_start = end - self.overlap
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks

    @staticmethod
    def _last_whitespace(text: str, start: int, end: int) -> int | None:
        for index in range(end - 1, start, -1):
            if text[index].isspace():
                return index
        return None
