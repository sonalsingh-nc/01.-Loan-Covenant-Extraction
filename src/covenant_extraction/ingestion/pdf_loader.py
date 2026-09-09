"""PDF loading using PyMuPDF (fitz).

Extracts text per page, tracking page numbers and running character offsets
so downstream chunks/covenants can cite back to a specific page.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pymupdf as fitz

from covenant_extraction.ingestion import Block


def load_pdf(path: str | Path) -> List[Block]:
    """Load a PDF file and return one Block per page.

    Args:
        path: path to the PDF file.

    Returns:
        List of Block objects, one per non-empty page, with `page` set and
        `start_char`/`end_char` reflecting offsets into the concatenation of
        all block texts (joined with a single newline).
    """
    path = Path(path)
    blocks: List[Block] = []
    offset = 0

    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            start = offset
            end = start + len(text)
            blocks.append(Block(text=text, page=page_index, start_char=start, end_char=end))
            offset = end + 1  # account for the newline joiner

    return blocks
