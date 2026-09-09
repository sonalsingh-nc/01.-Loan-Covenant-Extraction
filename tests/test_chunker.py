"""Tests for ingestion/chunker.py."""
from __future__ import annotations

from covenant_extraction.ingestion import Block
from covenant_extraction.ingestion.chunker import chunk_blocks


def test_short_heading_merges_into_following_block():
    blocks = [
        Block(text="6.1", page=1, start_char=0, end_char=3),
        Block(
            text="The Borrower shall not permit the Consolidated Leverage Ratio to exceed 3.50:1.00.",
            page=1,
            start_char=4,
            end_char=90,
        ),
    ]

    chunks = chunk_blocks(blocks)

    assert len(chunks) == 1
    assert chunks[0].text.startswith("6.1\n")
    assert "3.50:1.00" in chunks[0].text
    assert chunks[0].page == 1


def test_normal_length_blocks_stay_separate():
    blocks = [
        Block(text="A" * 50, page=1, start_char=0, end_char=50),
        Block(text="B" * 50, page=1, start_char=51, end_char=101),
    ]

    chunks = chunk_blocks(blocks)

    assert len(chunks) == 2
    assert chunks[0].text == "A" * 50
    assert chunks[1].text == "B" * 50


def test_trailing_short_block_is_not_dropped():
    blocks = [
        Block(text="A" * 50, page=1, start_char=0, end_char=50),
        Block(text="short", page=1, start_char=51, end_char=56),
    ]

    chunks = chunk_blocks(blocks)

    assert len(chunks) == 2
    assert chunks[1].text == "short"


def test_empty_input_returns_empty_list():
    assert chunk_blocks([]) == []
