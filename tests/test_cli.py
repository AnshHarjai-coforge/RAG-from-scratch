"""Command-line parsing for ask and evaluate, without loading a model."""

import importlib.util
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(filename: str):
    path = ROOT / "scripts" / filename
    module_name = f"_script_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ASK = load_script("ask.py")
EVALUATE = load_script("evaluate.py")


def _forbid_setup(monkeypatch, module) -> None:
    """Replace constructors so a parse failure cannot open the store or a model."""

    def forbid(name):
        def constructor(*args, **kwargs):
            raise AssertionError(f"{name} was constructed")

        return constructor

    for name in (
        "get_settings",
        "IndexingPipeline",
        "QwenEmbedder",
        "ChromaStore",
        "CrossEncoderReranker",
    ):
        monkeypatch.setattr(module, name, forbid(name))


@pytest.mark.parametrize("argv", [["--help"], ["--index", "--help"]])
def test_ask_help_exits_before_the_store(argv, monkeypatch, capsys):
    _forbid_setup(monkeypatch, ASK)

    with pytest.raises(SystemExit) as caught:
        ASK.main(argv)

    assert caught.value.code == 0
    assert "Rebuilds the vector store before answering." in capsys.readouterr().out


def test_ask_unknown_flag_exits_2(monkeypatch):
    _forbid_setup(monkeypatch, ASK)

    with pytest.raises(SystemExit) as caught:
        ASK.main(["--bogus"])

    assert caught.value.code == 2


def test_ask_index_flag_parses_the_question():
    should_index, question = ASK.parse_args(["--index", "How", "many", "days"])
    assert should_index is True
    assert question == "How many days"

    should_index, question = ASK.parse_args(["How", "many", "days", "--index"])
    assert should_index is True
    assert question == "How many days"


@pytest.mark.parametrize("argv", [["--help"], ["-h"]])
def test_evaluate_help_exits_before_the_pipeline(argv, monkeypatch, capsys):
    _forbid_setup(monkeypatch, EVALUATE)

    with pytest.raises(SystemExit) as caught:
        EVALUATE.main(argv)

    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "score the fixed questions for retrieval recall and answer accuracy." in help_text


def test_evaluate_unknown_flag_exits_2(monkeypatch):
    _forbid_setup(monkeypatch, EVALUATE)

    with pytest.raises(SystemExit) as caught:
        EVALUATE.main(["--bogus"])

    assert caught.value.code == 2


class _Embedder:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        if "remote" in lowered:
            return [1.0, 0.0, 0.0]
        if "expense" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


class _Reranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(self, pairs):
        return [1.0 for _pair in pairs]


def _use_temp_dirs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "policies"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("COLLECTION_NAME", "coverage")
    monkeypatch.setenv("TOP_K", "3")


def _patch_models(monkeypatch) -> None:
    import rag.answer
    import rag.embedding
    import rag.rerank

    monkeypatch.setattr(rag.embedding, "QwenEmbedder", _Embedder)
    monkeypatch.setattr(rag.rerank, "CrossEncoderReranker", _Reranker)
    monkeypatch.setattr(ASK, "QwenEmbedder", _Embedder)
    monkeypatch.setattr(ASK, "CrossEncoderReranker", _Reranker)
    monkeypatch.setattr(EVALUATE, "QwenEmbedder", _Embedder)
    monkeypatch.setattr(EVALUATE, "CrossEncoderReranker", _Reranker)
    monkeypatch.setattr(
        rag.answer,
        "generate_answer",
        lambda question, hits, settings: "\n".join(hit.text for hit in hits),
    )
    monkeypatch.setattr(
        ASK,
        "generate_answer",
        lambda question, hits, settings: "\n".join(hit.text for hit in hits) or "no hits",
    )
    monkeypatch.setattr(
        EVALUATE,
        "generate_answer",
        lambda question, hits, settings: "\n".join(hit.text for hit in hits),
    )


def test_ask_prints_an_indexed_answer_and_skips_rebuild_without_the_flag(
    tmp_path, monkeypatch, capsys
):
    policies = tmp_path / "policies"
    policies.mkdir()
    (policies / "remote.md").write_text(
        "Employees may work remote for two days.\n",
        encoding="utf-8",
    )
    _use_temp_dirs(monkeypatch, tmp_path)
    _patch_models(monkeypatch)

    ASK.main(["--index", "How", "many", "remote", "days"])
    indexed = capsys.readouterr().out
    assert "Indexed 1 files, 1 chunks" in indexed
    assert "Question: How many remote days" in indexed
    assert "remote.md" in indexed
    assert "two days" in indexed

    ASK.main(["How many remote days"])
    reused = capsys.readouterr().out
    assert "Indexed" not in reused
    assert "Answer:" in reused
    assert "two days" in reused


def test_ask_module_entrypoint(tmp_path, monkeypatch, capsys):
    policies = tmp_path / "policies"
    policies.mkdir()
    (policies / "remote.md").write_text(
        "Employees may work remote for two days.\n", encoding="utf-8"
    )
    _use_temp_dirs(monkeypatch, tmp_path)
    _patch_models(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["ask.py", "--index", "remote", "days"])

    runpy.run_path(str(ROOT / "scripts" / "ask.py"), run_name="__main__")

    assert "Indexed 1 files, 1 chunks" in capsys.readouterr().out


def test_index_policies_prints_skips_and_a_hit(tmp_path, monkeypatch, capsys):
    policies = tmp_path / "policies"
    policies.mkdir()
    (policies / "remote.md").write_text(
        "Employees may work remote for two days.\n", encoding="utf-8"
    )
    (policies / "empty.md").write_text("  \n", encoding="utf-8")
    (policies / "old.md").write_text(
        "---\nstatus: superseded\n---\n\nthree days\n",
        encoding="utf-8",
    )
    _use_temp_dirs(monkeypatch, tmp_path)
    _patch_models(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["index_policies.py"])

    runpy.run_path(str(ROOT / "scripts" / "index_policies.py"), run_name="__main__")

    output = capsys.readouterr().out
    assert "Indexed 1 files, 1 chunks" in output
    assert "remote.md" in output
    assert "Skipped empty.md: empty" in output
    assert "Skipped old.md: superseded" in output
    assert "two days" in output
    assert "three days" not in output


def _write_eval_policies(policies: Path) -> None:
    policies.mkdir()
    (policies / "02_expense_reimbursement_policy.md").write_text(
        "This expense policy states 30 calendar days, $25 or more, "
        "$65 per person per day, and above $2,000.\n",
        encoding="utf-8",
    )
    (policies / "03_paid_time_off_policy.md").write_text(
        "Eligible employees receive 20 days (160 hours). "
        "They may carry over up to 40 hours. "
        "Submit a request at least two weeks in advance.\n",
        encoding="utf-8",
    )
    (policies / "01_remote_work_policy.md").write_text(
        "A remote employee must use the company-approved secure access method.\n",
        encoding="utf-8",
    )


def test_evaluate_module_entrypoint_passes_the_fixture_corpus(tmp_path, monkeypatch, capsys):
    policies = tmp_path / "policies"
    _write_eval_policies(policies)
    _use_temp_dirs(monkeypatch, tmp_path)
    _patch_models(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["evaluate.py"])

    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(ROOT / "scripts" / "evaluate.py"), run_name="__main__")

    assert caught.value.code == 0
    output = capsys.readouterr().out
    assert "Recall: 8/8" in output
    assert "Answer accuracy: 8/8" in output
    assert "FAIL" not in output


def test_evaluate_returns_1_when_the_answer_misses_the_fact(tmp_path, monkeypatch, capsys):
    policies = tmp_path / "policies"
    policies.mkdir()
    _use_temp_dirs(monkeypatch, tmp_path)
    _patch_models(monkeypatch)
    monkeypatch.setattr(EVALUATE, "generate_answer", lambda question, hits, settings: "nope")

    assert EVALUATE.main([]) == 1
    output = capsys.readouterr().out
    assert "FAIL expense_deadline" in output
    assert "Answer accuracy: 0/8" in output
