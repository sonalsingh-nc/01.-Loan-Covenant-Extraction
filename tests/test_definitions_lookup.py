"""Tests for retrieval/definitions_lookup.py."""
from __future__ import annotations

from covenant_extraction.ingestion.chunker import Chunk
from covenant_extraction.retrieval.definitions_lookup import extract_definitions_text, find_definition


def _chunk(text: str, section_label) -> Chunk:
    return Chunk(text=text, page=None, section_label=section_label, start_char=0, end_char=len(text))


def test_extract_definitions_text_picks_only_definitions_section():
    chunks = [
        _chunk('"Consolidated EBITDA" means Consolidated Net Income plus add-backs.', "SECTION 1. DEFINITIONS"),
        _chunk("The Borrower shall not exceed 3.50:1.00.", "Section 6.1"),
    ]

    text = extract_definitions_text(chunks)

    assert "Consolidated EBITDA" in text
    assert "3.50:1.00" not in text


def test_extract_definitions_text_matches_additional_definitions_heading():
    chunks = [_chunk('"Permitted Liens" means Liens described on Schedule X.', "ADDITIONAL DEFINITIONS")]

    text = extract_definitions_text(chunks)

    assert "Permitted Liens" in text


def test_find_definition_returns_matching_term():
    definitions_text = (
        '"Consolidated EBITDA" means, for any period, Consolidated Net Income for such period '
        'plus depreciation and amortization expense. "Consolidated Net Income" means net income '
        "determined in accordance with GAAP."
    )

    definition = find_definition("Consolidated EBITDA", definitions_text)

    assert definition is not None
    assert "Consolidated Net Income" in definition
    assert "Consolidated Net Income determined in accordance" not in definition


def test_find_definition_returns_none_when_term_not_defined():
    definitions_text = '"Consolidated EBITDA" means Consolidated Net Income plus add-backs.'

    assert find_definition("Consolidated Leverage Ratio", definitions_text) is None


def test_find_definition_returns_none_for_empty_text():
    assert find_definition("Consolidated EBITDA", "") is None


# --- curly quotes + <U>-tag-split terms (e.g. BeautifulSoup output from
# `&ldquo;<U>Term</U>&rdquo;`-style HTML) ------------------------------------


def test_find_definition_handles_curly_quotes_and_newline_split_term():
    # Mirrors what BeautifulSoup's get_text(" ", strip=True) produces from
    # `&ldquo;<U>Consolidated Leverage\nRatio</U>&rdquo; means ...`: curly
    # quotes, a space between the quote and the term, and a literal newline
    # inside the term where the source HTML wrapped it.
    definitions_text = (
        "“ Consolidated Leverage\nRatio ” means, as at any day, the ratio of "
        "(a) Consolidated Total Debt on such day to (b) Consolidated EBITDA. "
        "“ Consolidated Secured\nLeverage Ratio ” means, as at any day, the ratio of "
        "(a) Consolidated Secured Debt on such day to (b) Consolidated EBITDA."
    )

    definition = find_definition("Consolidated Leverage Ratio", definitions_text)

    assert definition is not None
    assert "Consolidated Total Debt" in definition
    assert "Consolidated Secured Debt" not in definition


def test_find_definition_still_matches_plain_straight_quotes():
    # Regression: the curly-quote/whitespace tolerance must not break the
    # original straight-quote, no-gap case.
    definitions_text = '"Consolidated EBITDA" means Consolidated Net Income plus add-backs.'

    assert find_definition("Consolidated EBITDA", definitions_text) is not None


def test_extract_definitions_text_falls_back_to_content_when_heading_missing():
    # Real-world case: the "Definitions" heading is a plain, unstyled
    # paragraph the HTML loader doesn't recognize as a heading, so
    # section_label never contains "definitions" -- the chunk should still
    # be picked up because it opens with a `"<Term>" means` declaration.
    chunks = [
        _chunk(
            "“ Consolidated Leverage\nRatio ” means, as at any day, the ratio of "
            "(a) Consolidated Total Debt to (b) Consolidated EBITDA.",
            "ARTICLE I",
        ),
        _chunk("The Borrower shall not permit the ratio to exceed 3.50:1.00.", "Section 6.1"),
    ]

    text = extract_definitions_text(chunks)

    assert "Consolidated Leverage" in text
    assert "3.50:1.00" not in text


# --- singular/plural tolerance (e.g. a reference to "Restricted
# Subsidiaries" should still find a term declared as "Restricted
# Subsidiary") ----------------------------------------------------------


def test_find_definition_matches_singular_search_term_against_plural_declaration():
    definitions_text = '"Restricted Subsidiaries" means any subsidiary of a Borrower other than an Unrestricted Subsidiary.'

    assert find_definition("Restricted Subsidiary", definitions_text) is not None


def test_find_definition_matches_plural_search_term_against_singular_declaration():
    definitions_text = '"Restricted Subsidiary" means any subsidiary of a Borrower other than an Unrestricted Subsidiary.'

    assert find_definition("Restricted Subsidiaries", definitions_text) is not None


def test_find_definition_matches_plain_trailing_s_plural():
    definitions_text = '"Permitted Lien" means any Lien described on Schedule X.'

    assert find_definition("Permitted Liens", definitions_text) is not None


def test_find_definition_does_not_match_unrelated_term_via_pluralization():
    definitions_text = '"Consolidated EBITDA" means Consolidated Net Income plus add-backs.'

    assert find_definition("Consolidated Debts", definitions_text) is None


# --- multi-chunk definitions (a lead-in sentence + itemized (a)/(b)/(c)...
# list, each its own HTML paragraph/chunk) --------------------------------


def test_extract_definitions_text_includes_continuation_chunks_sharing_a_section_label():
    # Only the first chunk itself opens with `"Term" means`; the itemized
    # sub-clauses are separate chunks that don't individually look like a
    # definition, but share the same section_label -- they must still be
    # included so find_definition doesn't truncate the definition.
    chunks = [
        _chunk('"Consolidated EBITDA" means, for any period, the sum of (without duplication):', "SECTION 1.01"),
        _chunk("(a) operating income as reported in each Borrower's financial statements, plus", "SECTION 1.01"),
        _chunk("(b) depreciation, amortization, and interest expense.", "SECTION 1.01"),
        _chunk("Section 6.1 Consolidated Leverage Ratio shall not exceed 3.50:1.00.", "SECTION 6.1"),
    ]

    text = extract_definitions_text(chunks)

    assert "operating income" in text
    assert "depreciation, amortization" in text
    assert "3.50:1.00" not in text


def test_find_definition_captures_full_multi_chunk_definition():
    chunks = [
        _chunk('"Consolidated EBITDA" means, for any period, the sum of (without duplication):', "SECTION 1.01"),
        _chunk("(a) operating income, plus", "SECTION 1.01"),
        _chunk("(b) depreciation and amortization.", "SECTION 1.01"),
        _chunk('"Consolidated Net Income" means the net income of the Borrower.', "SECTION 1.01"),
    ]
    definitions_text = extract_definitions_text(chunks)

    definition = find_definition("Consolidated EBITDA", definitions_text)

    assert definition is not None
    assert "operating income" in definition
    assert "depreciation and amortization" in definition
    assert "net income of the Borrower" not in definition


def test_extract_definitions_text_does_not_over_include_unlabeled_chunks():
    # A content-matched chunk with NO section_label shouldn't sweep in other
    # unrelated, also-unlabeled chunks (there's no reliable grouping key).
    chunks = [
        _chunk('"Consolidated EBITDA" means Consolidated Net Income plus add-backs.', None),
        _chunk("This is unrelated boilerplate with no section label.", None),
    ]

    text = extract_definitions_text(chunks)

    assert "Consolidated EBITDA" in text
    assert "unrelated boilerplate" not in text


# --- missing closing quote / qualifier-before-"means" (e.g. Old Plank
# Trail's `"“ Fixed Charges for any period, means ...` -- no closing
# quote at all, and the qualifier sits before "means" instead of after) -----


def test_find_definition_handles_missing_closing_quote():
    definitions_text = "“ Fixed Charges for any period, means, the sum of cash Interest Expense and scheduled principal payments."

    definition = find_definition("Fixed Charges", definitions_text)

    assert definition is not None
    assert "cash Interest Expense" in definition


def test_find_definition_handles_qualifier_before_means_with_proper_quotes():
    # Same qualifier-before-means phrasing, but with a normal closing quote
    # this time -- both cases should work independently.
    definitions_text = '"Fixed Charges" for any period, means, the sum of cash Interest Expense and scheduled principal payments.'

    definition = find_definition("Fixed Charges", definitions_text)

    assert definition is not None
    assert "cash Interest Expense" in definition


def test_extract_definitions_text_content_fallback_handles_missing_closing_quote():
    chunks = [
        _chunk("“ Fixed Charges for any period, means, the sum of cash Interest Expense.", "SECTION 1.01"),
        _chunk("The Borrower shall not permit the ratio to exceed 3.50:1.00.", "SECTION 6.1"),
    ]

    text = extract_definitions_text(chunks)

    assert "Fixed Charges" in text
    assert "3.50:1.00" not in text
