"""Tests for definitions/regex_discovery.py -- the LLM-free, cost-bounded
dependency-chain discovery used to wire definitions-chaining into pipeline.py."""
from __future__ import annotations

from covenant_extraction.definitions.regex_discovery import discover_term_chain

DEFINITIONS_TEXT = (
    '"Consolidated Leverage Ratio" means, as at any day, the ratio of (a) Consolidated Total Debt '
    "on such day to (b) Consolidated EBITDA for the most recent four fiscal quarter period. "
    '"Consolidated Total Debt" means, at any date, the aggregate principal amount of all '
    "Indebtedness of the Borrowers determined in accordance with GAAP. "
    '"Consolidated EBITDA" means, for any period, Consolidated Net Income plus Consolidated '
    "Interest Expense and depreciation and amortization expense. "
    '"Consolidated Net Income" means the net income of the Borrower determined in accordance '
    "with GAAP. "
    '"Consolidated Interest Expense" means the total interest expense of the Borrower and its '
    "Restricted Subsidiaries."
)


def test_discover_term_chain_finds_direct_references():
    graph, terms = discover_term_chain("Consolidated Leverage Ratio", DEFINITIONS_TEXT, max_depth=1)

    assert "Consolidated Leverage Ratio" in terms
    assert graph.get_children("Consolidated Leverage Ratio") >= {
        "Consolidated Total Debt",
        "Consolidated EBITDA",
    }


def test_discover_term_chain_recurses_within_max_depth():
    graph, terms = discover_term_chain("Consolidated Leverage Ratio", DEFINITIONS_TEXT, max_depth=2, max_terms=10)

    # Depth 2: EBITDA's own references should now be resolved too.
    assert "Consolidated Net Income" in terms
    assert "Consolidated Interest Expense" in terms


def test_discover_term_chain_stops_at_max_depth():
    graph, terms = discover_term_chain("Consolidated Leverage Ratio", DEFINITIONS_TEXT, max_depth=1, max_terms=10)

    # depth-1 references (Total Debt, EBITDA) resolved, but not their own
    # depth-2 references.
    assert "Consolidated EBITDA" in terms
    assert "Consolidated Net Income" not in terms


def test_discover_term_chain_stops_at_max_terms():
    graph, terms = discover_term_chain("Consolidated Leverage Ratio", DEFINITIONS_TEXT, max_depth=5, max_terms=2)

    assert len(terms) <= 2


def test_discover_term_chain_skips_false_candidate_phrases():
    # "GAAP" and "Restricted Subsidiaries" aren't defined anywhere in
    # DEFINITIONS_TEXT (single-word GAAP is filtered by the 2+-word
    # candidate regex; "Restricted Subsidiaries" fails the find_definition
    # existence check) -- neither should end up in `terms`.
    graph, terms = discover_term_chain("Consolidated Leverage Ratio", DEFINITIONS_TEXT, max_depth=5, max_terms=20)

    assert "GAAP" not in terms
    assert "Restricted Subsidiaries" not in terms


def test_discover_term_chain_returns_empty_terms_for_undefined_root():
    graph, terms = discover_term_chain("Nonexistent Term", DEFINITIONS_TEXT)

    assert terms == {}
