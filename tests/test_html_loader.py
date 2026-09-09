"""Tests for ingestion/html_loader.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from covenant_extraction.ingestion.html_loader import load_html


def _load(html: str):
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(html)
        path = Path(f.name)
    try:
        return load_html(path)
    finally:
        path.unlink()


def test_explicit_section_prefix_without_trailing_period():
    blocks = _load("<html><body><p>Section 6.1 Consolidated Leverage Ratio. The Borrower shall not.</p></body></html>")
    assert blocks[0].section_label == "Section 6.1"


def test_explicit_section_prefix_with_trailing_period_before_whitespace():
    # Word-exported agreements often render "SECTION 6.10.<many nbsp>Title"
    # -- the period sits before the whitespace run, not after it.
    blocks = _load(
        "<html><body><p>SECTION 6.10.    Financial Covenants. Commencing with the first "
        "fiscal quarter, the Borrower shall not.</p></body></html>"
    )
    assert blocks[0].section_label == "SECTION 6.10"


def test_explicit_section_prefix_becomes_running_label_for_later_blocks():
    blocks = _load(
        "<html><body>"
        "<p>SECTION 6.10.  Financial Covenants.</p>"
        "<p>(A) Consolidated Leverage Ratio shall not exceed 4.50:1.00.</p>"
        "</body></html>"
    )
    assert blocks[1].section_label == "SECTION 6.10"


def test_numeric_ratio_at_start_of_clause_is_not_mistaken_for_a_section():
    blocks = _load("<html><body><p>1.25 to 1.00 is the required Fixed Charge Coverage Ratio.</p></body></html>")
    assert blocks[0].section_label is None


def test_percentage_at_start_of_clause_is_not_mistaken_for_a_section():
    blocks = _load("<html><body><p>3.50% per annum shall apply during the covenant period.</p></body></html>")
    assert blocks[0].section_label is None


def test_hybrid_document_captures_leaf_div_alongside_p_tags():
    # Word-exported agreements sometimes mix semantic <p> tags for most of
    # the document with plain <div> runs for other sections (schedules,
    # riders) -- that div-wrapped text shouldn't be silently dropped just
    # because <p> tags exist elsewhere in the same document.
    blocks = _load(
        "<html><body>"
        "<p>This Agreement is entered into among the parties.</p>"
        "<div>The aggregate principal amount of the advances shall not exceed $15,000,000.</div>"
        "<p>Notices shall be delivered by hand or overnight courier.</p>"
        "</body></html>"
    )
    texts = [b.text for b in blocks]
    assert "The aggregate principal amount of the advances shall not exceed $15,000,000." in texts


def test_hybrid_document_preserves_reading_order():
    blocks = _load(
        "<html><body>"
        "<p>First paragraph.</p>"
        "<div>Second, div-wrapped paragraph.</div>"
        "<p>Third paragraph.</p>"
        "</body></html>"
    )
    texts = [b.text for b in blocks]
    assert texts == ["First paragraph.", "Second, div-wrapped paragraph.", "Third paragraph."]


def test_hybrid_document_does_not_duplicate_div_wrapping_a_p_tag():
    # A <div> that itself contains a <p> is a leaf div (no nested <div>),
    # but its text is already captured via that inner <p> -- it must not
    # also be emitted as a second, duplicate block.
    blocks = _load("<html><body><div><p>Only once, please.</p></div></body></html>")
    texts = [b.text for b in blocks]
    assert texts == ["Only once, please."]


def test_bare_article_header_sets_section_label():
    # Some agreements (commonly CMBS/commercial mortgage loans) have no
    # h1-4 tags at all and number top-level sections as a bare "ARTICLE N."
    # paragraph, with the descriptive title in a separate following
    # paragraph -- unlike SECTION X.Y, there's no title text to require in
    # the same element.
    blocks = _load(
        "<html><body>"
        "<p>ARTICLE 4.</p>"
        "<p>BORROWER COVENANTS</p>"
        "<p>Borrower will continuously maintain its existence.</p>"
        "</body></html>"
    )
    assert blocks[0].section_label == "ARTICLE 4"
    assert blocks[2].section_label == "ARTICLE 4"


def test_article_header_is_not_confused_with_decimal_section():
    # A later "4.1 Existence" style sub-heading should still take over as
    # the more precise running label, same as the SECTION X.Y case.
    blocks = _load(
        "<html><body>"
        "<p>ARTICLE 4.</p>"
        "<p>4.1 Existence. Borrower shall maintain its existence.</p>"
        "</body></html>"
    )
    assert blocks[1].section_label == "4.1"


def test_div_only_document_still_falls_back_to_leaf_divs():
    # No semantic h1-4/p/li tags at all -- every leaf <div> is a paragraph
    # (e.g. SEC EDGAR filings exported from Word with no <p> tags).
    blocks = _load(
        "<html><body>"
        "<div>9.4 Financial Covenant. The Borrower shall maintain a minimum EBITDA.</div>"
        "<div>9.5 Reporting. The Borrower shall deliver quarterly financials.</div>"
        "</body></html>"
    )
    assert len(blocks) == 2
    assert blocks[0].section_label == "9.4"


def test_table_rows_are_merged_into_the_preceding_block():
    # A covenant threshold schedule (test date -> corresponding number) is
    # commonly rendered as an HTML table -- without explicit handling, none
    # of it (no section-like tag covers <table>/<tr>/<td>) reaches the rest
    # of the pipeline at all. It's merged into the *preceding* block (not
    # emitted as its own) since chunk_blocks() only merges short neighboring
    # blocks -- a table long enough to clear that bar would otherwise land
    # in its own chunk with no covenant-name/keyword context to be found by
    # (this is exactly what broke Flux Power's Minimum EBITDA schedule).
    blocks = _load(
        "<html><body>"
        "<p>10.1 Minimum EBITDA. The Borrower shall have EBITDA of no less than the amount set forth below:</p>"
        "<table>"
        "<tr><td>For the month ending July 31, 2023</td><td>$</td><td>(4,750,000)</td></tr>"
        "<tr><td>For the month ending August 31, 2023</td><td>$</td><td>(4,650,000)</td></tr>"
        "</table>"
        "<p>10.2 Capital Expenditures. The Borrower shall not incur unfinanced Capital Expenditures.</p>"
        "</body></html>"
    )
    texts = [b.text for b in blocks]
    assert len(blocks) == 2
    assert "10.1 Minimum EBITDA" in texts[0]
    assert "For the month ending July 31, 2023 | $ | (4,750,000)" in texts[0]
    assert "For the month ending August 31, 2023 | $ | (4,650,000)" in texts[0]
    assert "10.2 Capital Expenditures" in texts[1]


def test_table_merged_into_preceding_block_keeps_that_block_section_label():
    blocks = _load(
        "<html><body>"
        "<p>10.1 Minimum EBITDA. Amount set forth below:</p>"
        "<table><tr><td>July 2023</td><td>(4,750,000)</td></tr></table>"
        "</body></html>"
    )
    assert len(blocks) == 1
    assert blocks[0].section_label == "10.1"
    assert "July 2023 | (4,750,000)" in blocks[0].text


def test_table_cell_text_is_not_duplicated_via_orphan_div_or_p():
    # A <td> containing a <p> or a <div> must not ALSO surface as its own
    # separate block -- that would duplicate the row's text and strip it of
    # its row/column context (which number belongs to which test date).
    blocks = _load(
        "<html><body>"
        "<table><tr><td><p>July 2023</p></td><td><div>(4,750,000)</div></td></tr></table>"
        "</body></html>"
    )
    assert len(blocks) == 1
    assert blocks[0].text == "July 2023 | (4,750,000)"


def test_empty_table_cells_are_dropped_not_kept_as_blank_segments():
    blocks = _load(
        "<html><body>"
        "<table><tr><td>July 2023</td><td>&nbsp;</td><td>(4,750,000)</td></tr></table>"
        "</body></html>"
    )
    assert blocks[0].text == "July 2023 | (4,750,000)"
