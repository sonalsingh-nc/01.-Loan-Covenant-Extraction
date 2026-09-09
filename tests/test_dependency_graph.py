"""Tests for definitions/dependency_graph.py (Phase 2: pure string matching,
no LLM)."""
from __future__ import annotations

from covenant_extraction.definitions.dependency_graph import (
    DAG,
    build_dependency_graph,
    detect_cycles,
    find_term_references,
    topological_sort,
)
from covenant_extraction.definitions.schema import DefinedTerm


def _term(name: str, definition_text: str) -> DefinedTerm:
    return DefinedTerm(term=name, definition_text=definition_text, source_clause=None)


def test_find_term_references_matches_known_terms_in_text():
    known = {"Consolidated Net Income", "Consolidated Interest Expense", "Unrelated Term"}
    refs = find_term_references(
        "Consolidated EBITDA means Consolidated Net Income plus Consolidated Interest Expense.",
        known,
    )
    assert set(refs) == {"Consolidated Net Income", "Consolidated Interest Expense"}


def test_build_dependency_graph_chains_composite_to_leaves():
    terms = [
        _term("Consolidated EBITDA", "Consolidated Net Income plus Consolidated Interest Expense."),
        _term("Consolidated Net Income", "the net income of the Borrower determined in accordance with GAAP."),
        _term("Consolidated Interest Expense", "total interest expense of the Borrower."),
    ]

    graph = build_dependency_graph(terms)

    assert graph.get_children("Consolidated EBITDA") == {
        "Consolidated Net Income",
        "Consolidated Interest Expense",
    }
    assert graph.get_children("Consolidated Net Income") == set()
    assert graph.get_children("Consolidated Interest Expense") == set()


def test_build_dependency_graph_skips_self_reference():
    terms = [_term("Applicable Margin", "the Applicable Margin set forth in the pricing grid below.")]

    graph = build_dependency_graph(terms)

    assert graph.get_children("Applicable Margin") == set()


def test_topological_sort_orders_leaves_before_composite():
    terms = [
        _term("Consolidated EBITDA", "Consolidated Net Income plus Consolidated Interest Expense."),
        _term("Consolidated Net Income", "net income of the Borrower."),
        _term("Consolidated Interest Expense", "interest expense of the Borrower."),
    ]
    graph = build_dependency_graph(terms)

    order = topological_sort(graph, root="Consolidated EBITDA")

    assert order[-1] == "Consolidated EBITDA"
    assert set(order[:-1]) == {"Consolidated Net Income", "Consolidated Interest Expense"}
    assert order.index("Consolidated Net Income") < order.index("Consolidated EBITDA")
    assert order.index("Consolidated Interest Expense") < order.index("Consolidated EBITDA")


def test_detect_cycles_finds_circular_definition():
    # build_dependency_graph itself already breaks cycles, so detect_cycles
    # is exercised directly against a raw (unbroken) graph here.
    raw_graph = DAG()
    raw_graph.add_edge("Term A", "Term B")
    raw_graph.add_edge("Term B", "Term A")

    cycles = detect_cycles(raw_graph)

    assert len(cycles) == 1
    assert set(cycles[0].terms) == {"Term A", "Term B"}


def test_build_dependency_graph_breaks_cycles_so_traversal_terminates():
    terms = [
        _term("Term A", "has the meaning given to Term B."),
        _term("Term B", "has the meaning given to Term A."),
    ]

    graph = build_dependency_graph(terms)

    # Should not raise/hang -- topological_sort must terminate even though
    # the raw definitions were circular.
    order = topological_sort(graph, root="Term A")
    assert "Term A" in order
