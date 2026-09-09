"""Chunking: groups ingested Blocks into slightly larger, paragraph-aware
chunks suitable for embedding and keyword scanning, while preserving the
page/section metadata needed for citations.

For loan contracts, the source Blocks from pdf_loader/html_loader already
tend to be section- or paragraph-sized, so chunking here is mostly a
pass-through with an optional merge step for very short blocks (e.g. short
headings) so they don't get scored/retrieved on their own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from covenant_extraction.ingestion import Block

MIN_CHUNK_CHARS = 40


@dataclass(frozen=True)
class Chunk:
    """A retrieval-ready piece of text plus citation metadata."""

    text: str
    page: Optional[int]
    section_label: Optional[str]
    start_char: int
    end_char: int


def chunk_blocks(blocks: List[Block]) -> List[Chunk]:
    """Merge short blocks (e.g. bare headings) into the following block so
    every chunk carries enough text to be meaningfully embedded/scanned.

    Args:
        blocks: ordered Blocks from a loader (pdf_loader or html_loader).

    Returns:
        Ordered list of Chunks, each backed by one or more source Blocks.
    """
    chunks: List[Chunk] = []
    pending: Optional[Block] = None

    for block in blocks:
        if pending is not None:
            merged_text = f"{pending.text}\n{block.text}"
            block = Block(
                text=merged_text,
                page=pending.page if pending.page is not None else block.page,
                section_label=pending.section_label if pending.section_label is not None else block.section_label,
                start_char=pending.start_char,
                end_char=block.end_char,
            )
            pending = None

        if len(block.text) < MIN_CHUNK_CHARS:
            pending = block
            continue

        chunks.append(
            Chunk(
                text=block.text,
                page=block.page,
                section_label=block.section_label,
                start_char=block.start_char,
                end_char=block.end_char,
            )
        )

    if pending is not None:
        chunks.append(
            Chunk(
                text=pending.text,
                page=pending.page,
                section_label=pending.section_label,
                start_char=pending.start_char,
                end_char=pending.end_char,
            )
        )

    return chunks
