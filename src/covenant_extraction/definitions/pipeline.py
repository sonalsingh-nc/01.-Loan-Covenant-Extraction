"""End-to-end orchestration for Definitions of Financial Terms construction:
extract every defined term from the contract's Definitions section (Phase 1)
-> build the term-dependency graph (Phase 2) -> resolve one covenant term's
formula by chaining bottom-up through its full dependency chain (Phase 3).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from covenant_extraction.definitions.dependency_graph import build_dependency_graph
from covenant_extraction.definitions.formula_resolver import resolve_covenant_formula
from covenant_extraction.definitions.regex_discovery import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_TERMS,
    discover_term_chain,
)
from covenant_extraction.definitions.schema import DefinedTerm, ResolvedFormula
from covenant_extraction.definitions.term_extractor import extract_defined_terms
from covenant_extraction.extraction.llm_client import StructuredLLMClient
from covenant_extraction.ingestion.chunker import Chunk


def resolve_defined_term(
    root_term: str,
    chunks: List[Chunk],
    document_text: str,
    llm_client: StructuredLLMClient,
) -> ResolvedFormula:
    """Extract every defined term from the contract's Definitions section,
    build its dependency graph, then resolve `root_term` (e.g. "Consolidated
    EBITDA") bottom-up into a single fully-substituted formula with its full
    chain of components, carve-outs, and provenance attached.

    Args:
        root_term: the covenant term to resolve, e.g. "Consolidated EBITDA".
        chunks: all chunks from the document (see ingestion/chunker.py).
        document_text: the full document text, for scoped_resolver's
            outside-the-definitions-section lookups.
        llm_client: a StructuredLLMClient (real LMStudioClient or a test fake).

    Returns:
        A ResolvedFormula for `root_term`; `status` is "requires_human_review"
        (rather than a silent guess) wherever the chain couldn't be
        completed.
    """
    terms = extract_defined_terms(chunks, llm_client)
    terms_by_name: Dict[str, DefinedTerm] = {t.term: t for t in terms}
    graph = build_dependency_graph(terms)

    return resolve_covenant_formula(
        root_term=root_term,
        graph=graph,
        terms=terms_by_name,
        document_text=document_text,
        llm_client=llm_client,
    )


def resolve_defined_term_bounded(
    root_term: str,
    definitions_text: str,
    document_text: str,
    llm_client: StructuredLLMClient,
    resolved_cache: Optional[Dict[str, ResolvedFormula]] = None,
    max_terms: int = DEFAULT_MAX_TERMS,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> ResolvedFormula:
    """Cost-bounded alternative to `resolve_defined_term`, for wiring
    definitions-chaining into the production pipeline (pipeline.py) against
    large real-world documents.

    Discovers `root_term`'s dependency chain via regex_discovery's cheap,
    LLM-free lookups instead of term_extractor's bulk, whole-document
    Phase 1 extraction, then spends the LLM only on formula_resolver's
    leaf/compose steps for the small set of terms actually reached (capped
    at `max_terms` total, `max_depth` hops deep).

    Args:
        root_term: the covenant term to resolve, e.g. "Consolidated Leverage
            Ratio".
        definitions_text: the contract's definitions-section text (see
            retrieval/definitions_lookup.extract_definitions_text).
        document_text: the full document text, for scoped_resolver's
            outside-the-definitions-section lookups.
        llm_client: a StructuredLLMClient (real LMStudioClient or a test fake).
        resolved_cache: optional dict shared across multiple calls (e.g. two
            covenants that both reference "Consolidated EBITDA"), so a term
            already resolved by an earlier call is reused rather than
            re-resolved. Pass the same dict across calls in one pipeline run
            to get this reuse; omit for a one-off, uncached call.
        max_terms: stop discovery after this many distinct terms resolve.
        max_depth: stop discovery after this many hops from `root_term`.

    Returns:
        A ResolvedFormula for `root_term`; `status` is "requires_human_review"
        wherever the bounded chain couldn't be completed (including simply
        running past `max_terms`/`max_depth`).
    """
    graph, terms = discover_term_chain(root_term, definitions_text, max_terms=max_terms, max_depth=max_depth)

    return resolve_covenant_formula(
        root_term=root_term,
        graph=graph,
        terms=terms,
        document_text=document_text,
        llm_client=llm_client,
        resolved=resolved_cache,
    )
