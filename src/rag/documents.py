"""Read policy markdown files one at a time."""

import stat
from dataclasses import dataclass
from pathlib import Path

# Skip a file larger than this instead of reading it into memory.
DEFAULT_MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class SourceDocument:
    """A markdown file that was read successfully."""

    name: str
    text: str
    status: str = "current"


@dataclass(frozen=True)
class SkippedFile:
    """A file the indexer did not read, with the reason."""

    name: str
    reason: str


def _opening_status(text: str) -> str:
    """Return front-matter status, or current when it is absent."""

    lines = text.lstrip().splitlines()
    if not lines or lines[0] != "---":
        return "current"
    status = "current"
    for line in lines[1:]:
        if line == "---":
            break
        if line.startswith("status:"):
            value = line.removeprefix("status:").strip().lower()
            if value:
                status = value
    return status


class DocumentSource:
    """Load ``.md`` and ``.markdown`` files, any case, in one directory."""

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
        for path in sorted(self.directory.iterdir()):
            if path.suffix.lower() not in {".md", ".markdown"}:
                continue
            try:
                file_stat = path.stat()
                if not stat.S_ISREG(file_stat.st_mode):
                    skipped.append(SkippedFile(path.name, "not_a_regular_file"))
                    continue
                if file_stat.st_size > self.max_bytes:
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
            documents.append(SourceDocument(path.name, text, status=_opening_status(text)))
        return documents, skipped
