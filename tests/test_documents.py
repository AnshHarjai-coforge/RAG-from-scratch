import os
from pathlib import Path

import pytest

from rag.documents import DocumentSource


def test_missing_directory_raises(tmp_path):
    source = DocumentSource(tmp_path / "missing")

    with pytest.raises(FileNotFoundError):
        source.load()


def test_empty_directory_returns_nothing(tmp_path):
    documents, skipped = DocumentSource(tmp_path).load()

    assert documents == []
    assert skipped == []


def test_empty_whitespace_and_non_markdown_files_are_handled(tmp_path):
    (tmp_path / "empty.md").write_text("  \n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    (tmp_path / "b.md").write_text("second", encoding="utf-8")
    (tmp_path / "a.md").write_text("first", encoding="utf-8")

    documents, skipped = DocumentSource(tmp_path).load()

    assert [document.name for document in documents] == ["a.md", "b.md"]
    assert [item.name for item in skipped] == ["empty.md"]
    assert skipped[0].reason == "empty"


def test_invalid_utf8_is_skipped(tmp_path):
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe not text")
    (tmp_path / "good.md").write_text("readable", encoding="utf-8")

    documents, skipped = DocumentSource(tmp_path).load()

    assert [document.name for document in documents] == ["good.md"]
    assert skipped == [skipped[0]]
    assert skipped[0].name == "bad.md"
    assert skipped[0].reason == "invalid_utf8"


def test_unreadable_file_is_skipped_and_the_rest_are_read(tmp_path, monkeypatch):
    (tmp_path / "blocked.md").write_text("secret", encoding="utf-8")
    (tmp_path / "good.md").write_text("readable", encoding="utf-8")
    original = Path.read_text

    def read_text(self, *args, **kwargs):
        if self.name == "blocked.md":
            raise PermissionError("denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)

    documents, skipped = DocumentSource(tmp_path).load()

    assert [document.name for document in documents] == ["good.md"]
    assert [(item.name, item.reason) for item in skipped] == [("blocked.md", "unreadable")]


def test_files_over_the_cap_are_skipped_and_smaller_files_are_read(tmp_path):
    (tmp_path / "small.md").write_text("fits", encoding="utf-8")
    (tmp_path / "large.md").write_text("this file is too long", encoding="utf-8")
    cap = len(b"fits")

    documents, skipped = DocumentSource(tmp_path, max_bytes=cap).load()

    assert [document.name for document in documents] == ["small.md"]
    assert documents[0].text == "fits"
    assert skipped[0].name == "large.md"
    assert skipped[0].reason == "too_large"


def test_max_bytes_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        DocumentSource(tmp_path, max_bytes=0)


def test_unreadable_directory_raises_permission_error(tmp_path):
    blocked = tmp_path / "locked"
    blocked.mkdir()
    blocked.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            DocumentSource(blocked).load()
    finally:
        blocked.chmod(0o700)


def test_fifo_is_skipped_and_regular_markdown_is_loaded(tmp_path):
    os.mkfifo(tmp_path / "pipe.md")
    (tmp_path / "good.md").write_text("readable", encoding="utf-8")

    documents, skipped = DocumentSource(tmp_path).load()

    assert [document.name for document in documents] == ["good.md"]
    assert documents[0].text == "readable"
    assert [(item.name, item.reason) for item in skipped] == [("pipe.md", "not_a_regular_file")]


def test_markdown_suffixes_any_case_are_loaded_in_name_order(tmp_path):
    (tmp_path / "Report.MD").write_text("report", encoding="utf-8")
    (tmp_path / "note.markdown").write_text("note", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    (tmp_path / "Note.MD").write_text(" \n", encoding="utf-8")

    documents, skipped = DocumentSource(tmp_path).load()

    assert [document.name for document in documents] == ["Report.MD", "note.markdown"]
    assert [document.text for document in documents] == ["report", "note"]
    assert [(item.name, item.reason) for item in skipped] == [("Note.MD", "empty")]


def test_status_comes_from_opening_front_matter_only(tmp_path):
    (tmp_path / "superseded.md").write_text(
        "---\nstatus: superseded\n---\n\n# Title\n",
        encoding="utf-8",
    )
    (tmp_path / "current.md").write_text(
        "---\ntitle: Remote Work\nstatus: Current\n---\n\n# Title\n",
        encoding="utf-8",
    )
    (tmp_path / "plain.md").write_text("# Title\n", encoding="utf-8")
    (tmp_path / "body.md").write_text("# Title\n\nstatus: superseded\n", encoding="utf-8")
    (tmp_path / "later.md").write_text(
        "---\nstatus: current\n---\n\nstatus: superseded\n",
        encoding="utf-8",
    )

    documents, skipped = DocumentSource(tmp_path).load()
    by_name = {document.name: document for document in documents}

    assert skipped == []
    assert by_name["superseded.md"].status == "superseded"
    assert by_name["superseded.md"].text.startswith("---\nstatus: superseded\n")
    assert by_name["current.md"].status == "current"
    assert by_name["plain.md"].status == "current"
    assert by_name["body.md"].status == "current"
    assert by_name["later.md"].status == "current"
