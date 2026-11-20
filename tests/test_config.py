from pathlib import Path

import pytest

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


def test_foundry_model_falls_back_to_the_opus_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_OPUS_MODEL", "claude-opus-5")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "rg-test")
    monkeypatch.setenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    settings = get_settings()

    assert settings.foundry_model == "claude-opus-5"
    assert settings.foundry_api_key == "test-key"
    assert settings.foundry_resource == "rg-test"
    assert settings.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"


@pytest.mark.parametrize("blank", ["", " \t"])
def test_blank_required_settings_use_defaults(monkeypatch, blank):
    for name in (
        "EMBEDDING_MODEL",
        "EMBEDDING_BATCH_SIZE",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "TOP_K",
        "MAX_FILE_BYTES",
        "DATA_DIR",
        "CHROMA_DIR",
        "COLLECTION_NAME",
        "CROSS_ENCODER_MODEL",
    ):
        monkeypatch.setenv(name, blank)

    settings = get_settings()
    root = Path(__file__).resolve().parents[1]

    assert settings.data_dir == root / "data" / "policies"
    assert settings.chroma_dir == root / "data" / "chroma"
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.top_k == 3
    assert settings.max_file_bytes == 67108864
    assert settings.embedding_batch_size == 32
    assert settings.collection_name == "policies"
    assert settings.embedding_model == "Qwen/Qwen3-Embedding-0.6B"
    assert settings.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_chunk_size_rejects_non_integers(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "nope")

    with pytest.raises(ValueError) as error:
        get_settings()

    message = str(error.value)
    assert "CHUNK_SIZE" in message
    assert "nope" in message


def test_data_dir_keeps_surrounding_spaces(monkeypatch):
    monkeypatch.setenv("DATA_DIR", " policies ")

    settings = get_settings()
    root = Path(__file__).resolve().parents[1]

    assert settings.data_dir == root / " policies "
    assert settings.data_dir != root / "data" / "policies"


def test_absolute_chroma_dir_is_kept(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path))

    assert get_settings().chroma_dir == tmp_path


def test_unset_foundry_settings_stay_empty(monkeypatch):
    monkeypatch.setattr("rag.config.load_dotenv", lambda: None)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_DEFAULT_OPUS_MODEL", raising=False)

    settings = get_settings()

    assert settings.foundry_api_key is None
    assert settings.foundry_resource is None
    assert settings.foundry_model == "claude-opus-5"
