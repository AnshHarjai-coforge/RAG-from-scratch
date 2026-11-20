from pathlib import Path

from rag.config import get_settings


def test_settings_use_env_values_and_resolve_relative_paths(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "16")
    monkeypatch.setenv("CHUNK_SIZE", "400")
    monkeypatch.setenv("CHUNK_OVERLAP", "40")
    monkeypatch.setenv("TOP_K", "4")
    monkeypatch.setenv("MAX_FILE_BYTES", "1000")
    monkeypatch.setenv("DATA_DIR", "data/policies")
    monkeypatch.setenv("CHROMA_DIR", "data/chroma")
    monkeypatch.setenv("COLLECTION_NAME", "policies")

    settings = get_settings()
    root = Path(__file__).resolve().parents[1]

    assert settings.embedding_model == "Qwen/Qwen3-Embedding-0.6B"
    assert settings.chunk_size == 400
    assert settings.chunk_overlap == 40
    assert settings.embedding_batch_size == 16
    assert settings.top_k == 4
    assert settings.max_file_bytes == 1000
    assert settings.data_dir == root / "data" / "policies"
    assert settings.chroma_dir == root / "data" / "chroma"
    assert settings.collection_name == "policies"
