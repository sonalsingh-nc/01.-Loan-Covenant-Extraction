"""Document ingestion: converts raw PDF/HTML loan contracts into a list of
`Block` objects — plain text plus enough location metadata (page number or
nearest HTML section heading, and character offsets) to support citation back
to source later in the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Block:
    """A single contiguous piece of source text with location metadata."""

    text: str
    page: Optional[int] = None  # 1-indexed PDF page number, if applicable
    section_label: Optional[str] = None  # nearest section/subsection heading, if applicable
    start_char: int = 0  # offset of this block within the full document text
    end_char: int = 0
