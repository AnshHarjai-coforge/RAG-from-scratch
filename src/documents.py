"""Read policy markdown files one at a time."""

from dataclasses import dataclass
from pathlib import Path

# Max File size is 64 MB
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class SourceDocument:
    """A markdown file that was read successfully."""

    name: str
    text: str


@dataclass(frozen=True)
class SkippedFile:
    """A file the indexer did not read, with the reason."""

    name: str
    reason: str


class DocumentSource:
    """Load every markdown file in a directory the same way.

    Each file is read completely with ``read_text``. Files at 5 KB and files
    at several megabytes use that same call. A file larger than ``max_bytes``
    is skipped. That limit is a safety cap, not a second reader.
    """

    def __init__(self, directory: Path, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.directory = Path(directory)
        self.max_bytes = max_bytes

    def load(self) -> tuple[list[SourceDocument], list[SkippedFile]]:
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Policy directory does not exist: {self.directory}")

        documents: list[SourceDocument] = []
        skipped: list[SkippedFile] = []
        for path in sorted(self.directory.glob("*.md")):
            try:
                if path.stat().st_size > self.max_bytes:
                    skipped.append(SkippedFile(path.name, "too_large"))
                    continue
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                skipped.append(SkippedFile(path.name, "invalid_utf8"))
                continue
            except OSError:
                skipped.append(SkippedFile(path.name, "unreadable"))
                continue
            if not text.strip():
                skipped.append(SkippedFile(path.name, "empty"))
                continue
            documents.append(SourceDocument(path.name, text))
        return documents, skipped
