"""Settings read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# About one policy rule, so a question retrieves one rule instead of a whole file.
DEFAULT_CHUNK_SIZE = 500
# 10 percent of the window, so text on a cut is repeated in the next chunk.
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024
DEFAULT_EMBEDDING_BATCH_SIZE = 32


@dataclass(frozen=True)
class Settings:
    embedding_model: str
    embedding_batch_size: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_file_bytes: int
    data_dir: Path
    chroma_dir: Path
    collection_name: str


def get_settings() -> Settings:
    load_dotenv()
    root = Path(__file__).resolve().parents[2]
    data_dir = _under_root(root, os.getenv("DATA_DIR", "data/policies"))
    chroma_dir = _under_root(root, os.getenv("CHROMA_DIR", "data/chroma"))

    return Settings(
        embedding_model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        embedding_batch_size=int(
            os.getenv("EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE))
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP))),
        top_k=int(os.getenv("TOP_K", "3")),
        max_file_bytes=int(os.getenv("MAX_FILE_BYTES", str(DEFAULT_MAX_FILE_BYTES))),
        data_dir=data_dir,
        chroma_dir=chroma_dir,
        collection_name=os.getenv("COLLECTION_NAME", "policies"),
    )


def _under_root(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path
