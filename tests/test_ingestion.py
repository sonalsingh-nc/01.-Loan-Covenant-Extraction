"""Tests for ingestion/pdf_loader.py and ingestion/html_loader.py."""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
import pytest

from covenant_extraction.ingestion.html_loader import load_html
from covenant_extraction.ingestion.pdf_loader import load_pdf


@pytest.fixture
def two_page_pdf(tmp_path: Path) -> Path:
    """Builds a tiny 2-page PDF on the fly (no fixture binary needed)."""
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Page one: Consolidated Leverage Ratio shall not exceed 3.50:1.00.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Page two: standard boilerplate governing law clause.")
    pdf_path = tmp_path / "sample.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_load_pdf_returns_one_block_per_page(two_page_pdf: Path):
    blocks = load_pdf(two_page_pdf)

    assert len(blocks) == 2
    assert blocks[0].page == 1
    assert "Leverage Ratio" in blocks[0].text
    assert blocks[1].page == 2
    assert "boilerplate" in blocks[1].text


def test_load_pdf_offsets_are_monotonic(two_page_pdf: Path):
    blocks = load_pdf(two_page_pdf)

    assert blocks[0].start_char == 0
    assert blocks[0].end_char == len(blocks[0].text)
    assert blocks[1].start_char > blocks[0].end_char


def test_load_html_tracks_nearest_section_heading(sample_html_path: Path):
    blocks = load_html(sample_html_path)

    leverage_block = next(b for b in blocks if "3.50:1.00" in b.text)
    assert leverage_block.section_label == "Section 6.1"

    indebtedness_block = next(b for b in blocks if "create, incur, assume or suffer to exist any Indebtedness" in b.text)
    assert indebtedness_block.section_label == "Section 7.1"


def test_load_html_falls_back_to_heading_when_no_explicit_section_number(tmp_path: Path):
    html_path = tmp_path / "no_explicit.html"
    html_path.write_text(
        "<html><body><h2>SECTION 1. DEFINITIONS</h2>"
        "<p>Some definition text without an explicit section number prefix.</p>"
        "</body></html>",
        encoding="utf-8",
    )

    blocks = load_html(html_path)

    assert blocks[-1].section_label == "SECTION 1. DEFINITIONS"


def test_load_html_skips_empty_elements(tmp_path: Path):
    html_path = tmp_path / "empty.html"
    html_path.write_text("<html><body><p id='a'></p><p>Real text here.</p></body></html>", encoding="utf-8")

    blocks = load_html(html_path)

    assert len(blocks) == 1
    assert blocks[0].text == "Real text here."
