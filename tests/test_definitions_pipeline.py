"""Tests for the definitions/ package: Phase 1 term extraction, Phase 3
formula resolution, and the full resolve_defined_term chain."""
from __future__ import annotations

from typing import Dict, List, Type

from covenant_extraction.definitions.dependency_graph import build_dependency_graph
from covenant_extraction.definitions.formula_resolver import (
    extract_carve_outs,
    resolve_covenant_formula,
    search_document_for_term,
)
from covenant_extraction.definitions.pipeline import resolve_defined_term
from covenant_extraction.definitions.schema import DefinedTerm, FormulaFragment
from covenant_extraction.definitions.term_extractor import chunk_by_term_boundary, extract_defined_terms
from covenant_extraction.ingestion.chunker import Chunk


class FakeStructuredLLMClient:
    """Canned responses keyed by (schema, substring-of-prompt), so tests can
    verify orchestration without a real LLM."""

    def __init__(self, term_responses=None, formula_responses=None):
        self.term_responses: Dict[str, DefinedTerm] = term_responses or {}
        self.formula_responses: Dict[str, FormulaFragment] = formula_responses or {}
        self.calls: List[tuple] = []

    def complete(self, system_prompt: str, user_prompt: str, schema: Type):
        self.calls.append((schema, user_prompt))
        table = self.term_responses if schema is DefinedTerm else self.formula_responses
        for marker, response in table.items():
            if marker in user_prompt:
                return response
        raise AssertionError(f"no canned {schema.__name__} response for prompt: {user_prompt[:120]}")


DEFINITIONS_TEXT = (
    '"Consolidated EBITDA" means, for any period, Consolidated Net Income for such period '
    "plus Consolidated Interest Expense and taxes; provided that there shall be excluded "
    "therefrom any extraordinary, non-recurring or unusual gains. "
    '"Consolidated Net Income" means the net income of the Borrower determined in accordance '
    "with GAAP. "
    '"Consolidated Interest Expense" means the total interest expense of the Borrower and its '
    "Restricted Subsidiaries."
)


def _chunks() -> List[Chunk]:
    return [Chunk(text=DEFINITIONS_TEXT, page=1, section_label="Section 1.01 Definitions", start_char=0, end_char=len(DEFINITIONS_TEXT))]


# --- Phase 1: term_extractor -------------------------------------------------

def test_chunk_by_term_boundary_splits_on_each_defined_term():
    chunks = chunk_by_term_boundary(DEFINITIONS_TEXT)

    assert len(chunks) == 3
    assert chunks[0].startswith('"Consolidated EBITDA" means')
    assert chunks[1].startswith('"Consolidated Net Income" means')
    assert chunks[2].startswith('"Consolidated Interest Expense" means')


def test_chunk_by_term_boundary_empty_input_returns_empty_list():
    assert chunk_by_term_boundary("") == []
    assert chunk_by_term_boundary("   ") == []


def test_extract_defined_terms_makes_one_llm_call_per_term():
    fake = FakeStructuredLLMClient(
        term_responses={
            "Consolidated EBITDA": DefinedTerm(
                term="Consolidated EBITDA",
                definition_text=(
                    "Consolidated Net Income plus Consolidated Interest Expense; provided that there "
                    "shall be excluded therefrom any extraordinary gains."
                ),
                source_clause="Section 1.01(a)",
            ),
            "Consolidated Net Income": DefinedTerm(
                term="Consolidated Net Income",
                definition_text="the net income of the Borrower determined in accordance with GAAP.",
                source_clause="Section 1.01(b)",
            ),
            "Consolidated Interest Expense": DefinedTerm(
                term="Consolidated Interest Expense",
                definition_text="the total interest expense of the Borrower.",
                source_clause="Section 1.01(c)",
            ),
        }
    )

    terms = extract_defined_terms(_chunks(), fake)

    assert {t.term for t in terms} == {
        "Consolidated EBITDA",
        "Consolidated Net Income",
        "Consolidated Interest Expense",
    }
    assert len(fake.calls) == 3
    assert all(schema is DefinedTerm for schema, _ in fake.calls)


# --- Phase 3: formula_resolver ----------------------------------------------

def test_extract_carve_outs_finds_proviso_language():
    carve_outs = extract_carve_outs(
        "Consolidated Net Income plus addbacks; provided that there shall be excluded therefrom "
        "any extraordinary, non-recurring or unusual gains."
    )
    assert len(carve_outs) == 1
    assert "excluded therefrom" in carve_outs[0]


def test_extract_carve_outs_returns_empty_list_when_none_present():
    assert extract_carve_outs("the net income of the Borrower determined in accordance with GAAP.") == []


def test_search_document_for_term_finds_inline_definition():
    doc = 'Some boilerplate. "Applicable Margin" means the percentage set forth in the pricing grid below. More text.'
    found = search_document_for_term(doc, "Applicable Margin")
    assert found is not None
    assert "pricing grid" in found


def test_search_document_for_term_returns_none_when_absent():
    assert search_document_for_term("nothing relevant here", "Applicable Margin") is None


def test_resolve_covenant_formula_chains_leaves_into_composite():
    terms = {
        "Consolidated EBITDA": DefinedTerm(
            term="Consolidated EBITDA",
            definition_text=(
                "Consolidated Net Income plus Consolidated Interest Expense; provided that there "
                "shall be excluded therefrom any extraordinary gains."
            ),
            source_clause="Section 1.01(a)",
        ),
        "Consolidated Net Income": DefinedTerm(
            term="Consolidated Net Income",
            definition_text="the net income of the Borrower determined in accordance with GAAP.",
            source_clause="Section 1.01(b)",
        ),
        "Consolidated Interest Expense": DefinedTerm(
            term="Consolidated Interest Expense",
            definition_text="the total interest expense of the Borrower.",
            source_clause="Section 1.01(c)",
        ),
    }
    graph = build_dependency_graph(list(terms.values()))

    fake = FakeStructuredLLMClient(
        formula_responses={
            "net income of the Borrower": FormulaFragment(formula="Net Income"),
            "total interest expense": FormulaFragment(formula="Interest Expense"),
            "Parent definition": FormulaFragment(formula="Net Income + Interest Expense"),
        }
    )

    resolved = resolve_covenant_formula(
        root_term="Consolidated EBITDA",
        graph=graph,
        terms=terms,
        document_text=DEFINITIONS_TEXT,
        llm_client=fake,
    )

    assert resolved.term == "Consolidated EBITDA"
    assert resolved.formula == "Net Income + Interest Expense"
    assert resolved.status == "resolved"
    assert set(resolved.components.keys()) == {"Consolidated Net Income", "Consolidated Interest Expense"}
    assert resolved.components["Consolidated Net Income"].formula == "Net Income"
    assert resolved.provenance == ["Section 1.01(a)"]
    assert len(resolved.carve_outs) == 1
    assert "excluded therefrom" in resolved.carve_outs[0]


def test_resolve_covenant_formula_composes_best_effort_when_a_child_is_missing():
    # A child the graph found an edge to (e.g. an inline nickname like
    # "Expected Cost Savings" that was never a standalone definition, or a
    # term definitions-lookup simply missed) shouldn't block the PARENT from
    # composing a formula from whatever else it has -- only an explicit
    # "has the meaning set forth in Section X" cross-reference should still
    # gate on scoped_resolver (see the next test).
    terms = {
        "Consolidated EBITDA": DefinedTerm(
            term="Consolidated EBITDA",
            definition_text="Consolidated Net Income plus addbacks.",
            source_clause="Section 1.01(a)",
        ),
        # "Consolidated Net Income" is referenced but was never extracted.
    }
    graph = build_dependency_graph([terms["Consolidated EBITDA"]])
    graph.add_node("Consolidated Net Income")
    graph.add_edge("Consolidated EBITDA", "Consolidated Net Income")

    fake = FakeStructuredLLMClient(
        formula_responses={"Parent definition": FormulaFragment(formula="Net Income + addbacks")}
    )

    resolved = resolve_covenant_formula(
        root_term="Consolidated EBITDA",
        graph=graph,
        terms=terms,
        document_text="",
        llm_client=fake,
    )

    # The missing child is honestly flagged on its own component entry...
    assert resolved.components["Consolidated Net Income"].status == "requires_human_review"
    assert resolved.components["Consolidated Net Income"].reason == "term_not_found_in_definitions_section"
    # ...but that doesn't block the parent from composing best-effort.
    assert resolved.status == "resolved"
    assert resolved.formula == "Net Income + addbacks"


def test_resolve_covenant_formula_still_gates_on_unresolved_section_cross_reference():
    # An explicit '"<Term>" has the meaning set forth in Section X'
    # cross-reference that scoped_resolver can't find anywhere in the
    # document IS still a hard block -- unlike a plain missing DAG child,
    # there's no formula fragment to fall back on at all here. "Base Rate"
    # gives the node a real DAG child too, so this exercises the
    # cross-reference branch rather than the leaf branch (which triggers
    # whenever a term has zero children, regardless of an xref elsewhere in
    # its text).
    terms = {
        "Applicable Margin": DefinedTerm(
            term="Applicable Margin",
            definition_text='means Base Rate plus "Pricing Grid" has the meaning set forth in Section 2.05.',
            source_clause="Section 1.01(b)",
        ),
        "Base Rate": DefinedTerm(
            term="Base Rate",
            definition_text="the prime rate announced by the Administrative Agent.",
            source_clause="Section 1.01(c)",
        ),
    }
    graph = build_dependency_graph(list(terms.values()))

    fake = FakeStructuredLLMClient(formula_responses={"prime rate": FormulaFragment(formula="Prime Rate")})

    resolved = resolve_covenant_formula(
        root_term="Applicable Margin",
        graph=graph,
        terms=terms,
        document_text="no such section exists in this document",
        llm_client=fake,
    )

    assert resolved.status == "requires_human_review"
    assert resolved.reason == "unresolved_reference"
    assert resolved.formula == ""


# --- Full chain: pipeline.resolve_defined_term -------------------------------

def test_resolve_defined_term_runs_full_three_phase_chain():
    fake = FakeStructuredLLMClient(
        term_responses={
            "Consolidated EBITDA": DefinedTerm(
                term="Consolidated EBITDA",
                definition_text=(
                    "Consolidated Net Income plus Consolidated Interest Expense; provided that there "
                    "shall be excluded therefrom any extraordinary gains."
                ),
                source_clause="Section 1.01(a)",
            ),
            "Consolidated Net Income": DefinedTerm(
                term="Consolidated Net Income",
                definition_text="the net income of the Borrower determined in accordance with GAAP.",
                source_clause="Section 1.01(b)",
            ),
            "Consolidated Interest Expense": DefinedTerm(
                term="Consolidated Interest Expense",
                definition_text="the total interest expense of the Borrower.",
                source_clause="Section 1.01(c)",
            ),
        },
        formula_responses={
            "net income of the Borrower": FormulaFragment(formula="Net Income"),
            "total interest expense": FormulaFragment(formula="Interest Expense"),
            "Parent definition": FormulaFragment(formula="Net Income + Interest Expense"),
        },
    )

    resolved = resolve_defined_term(
        root_term="Consolidated EBITDA",
        chunks=_chunks(),
        document_text=DEFINITIONS_TEXT,
        llm_client=fake,
    )

    assert resolved.term == "Consolidated EBITDA"
    assert resolved.formula == "Net Income + Interest Expense"
    assert resolved.status == "resolved"
    assert resolved.components["Consolidated Interest Expense"].formula == "Interest Expense"
