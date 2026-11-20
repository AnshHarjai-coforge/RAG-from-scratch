import pytest

from rag.chunking import FixedSizeChunker, section_for


def test_empty_and_whitespace_return_no_chunks():
    chunker = FixedSizeChunker()

    assert chunker.chunk("") == []
    assert chunker.chunk("  \n\t ") == []


def test_short_text_is_one_chunk():
    chunker = FixedSizeChunker(chunk_size=500, overlap=50)

    chunks = chunker.chunk("  hello  ")

    assert len(chunks) == 1
    assert chunks[0].text == "hello"
    assert chunks[0].index == 0
    assert chunks[0].strategy == "fixed"
    assert chunks[0].start == 0
    assert chunks[0].end == len("hello")


def test_text_the_size_of_the_window_is_one_chunk():
    chunker = FixedSizeChunker(chunk_size=10, overlap=2)

    chunks = chunker.chunk("a" * 10)

    assert [chunk.text for chunk in chunks] == ["a" * 10]


def test_overlap_is_repeated_in_the_next_chunk():
    chunker = FixedSizeChunker(chunk_size=10, overlap=4)

    chunks = chunker.chunk("aaaaabbbbbccccc")

    assert [chunk.text for chunk in chunks] == ["aaaaabbbbb", "bbbbccccc"]
    assert chunks[0].text[-4:] == chunks[1].text[:4]
    assert [chunk.index for chunk in chunks] == [0, 1]


def test_window_ends_on_whitespace_when_a_break_is_available():
    chunker = FixedSizeChunker(chunk_size=12, overlap=3)

    chunks = chunker.chunk("hello beautiful world")

    assert chunks[0].text == "hello"
    assert all(len(chunk.text) <= 12 for chunk in chunks)


def test_span_without_whitespace_is_cut_at_chunk_size():
    chunker = FixedSizeChunker(chunk_size=10, overlap=2)

    chunks = chunker.chunk("a" * 25)

    assert chunks[0].text == "a" * 10
    assert all(len(chunk.text) <= 10 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks).count("a") > 25


def test_section_for_uses_the_heading_above_the_chunk():
    document = (
        "# Remote Work Policy\n\n"
        "## 2. Standard remote-work arrangement\n\n"
        "Employees may work remote for two days.\n\n"
        "## 5. Security and confidentiality\n\n"
        "Use the company-approved secure access method."
    )

    assert section_for(document, "Employees may work remote for two days.") == (
        "2. Standard remote-work arrangement"
    )
    security = (
        "## 5. Security and confidentiality\n\nUse the company-approved secure access method."
    )
    assert section_for(document, security) == "5. Security and confidentiality"
    spanned = (
        "This policy explains remote work.\n\n"
        "## 2. Standard remote-work arrangement\n\n"
        "Employees may work remote for two days."
    )
    assert section_for("# Remote Work Policy\n\n" + spanned, spanned) == (
        "2. Standard remote-work arrangement"
    )
    assert section_for(document, "this sentence is not in the document") == "body"
    assert section_for(document, "   ") == "body"


def test_overlap_must_be_smaller_than_the_chunk_size():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=10, overlap=10)

    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=0, overlap=0)


def test_repeated_chunk_uses_the_heading_at_its_end():
    document = "# A\n\nhello world\n\n# B\n\nhello world\n"
    chunks = FixedSizeChunker(chunk_size=11, overlap=0).chunk(document)
    chunk = chunks[-1]

    assert chunk.text == "world"
    assert all(piece.start is not None and piece.end is not None for piece in chunks)
    assert chunk.end > document.find("# B")
    assert section_for(document, chunk.text, end=chunk.end) == "B"


def test_cut_heading_uses_the_full_line_title():
    document = "# Remote Work Policy for every employee worldwide\n\n" + ("Body sentence. " * 5)
    chunk = FixedSizeChunker(chunk_size=24, overlap=5).chunk(document)[0]
    title = "Remote Work Policy for every employee worldwide"

    assert chunk.start == 0
    assert chunk.end < document.find("\n")
    assert "worldwide" not in chunk.text
    assert section_for(document, chunk.text, end=chunk.end) == title


def test_heading_that_starts_at_end_is_not_selected():
    document = "# Earlier title\n\nSome body text.\n\n# Later title\n\nMore text.\n"
    later = document.strip().find("# Later title")

    assert section_for(document, "anything", end=later) == "Earlier title"

    opening = "Some body text.\n\n# Later title\n\nMore text.\n"
    assert section_for(opening, "anything", end=opening.strip().find("#")) == "body"
