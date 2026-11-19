import pytest

from chunking import FixedSizeChunker


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


def test_overlap_must_be_smaller_than_the_chunk_size():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=10, overlap=10)

    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=0, overlap=0)
