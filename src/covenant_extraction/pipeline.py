"""End-to-end orchestration: load a contract -> chunk -> select candidate
clauses -> LLM extraction -> deterministic validation -> JSON-ready export.

This is the single entry point scripts/run_extraction.py calls into.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from covenant_extraction.definitions.pipeline import resolve_defined_term_bounded
from covenant_extraction.definitions.regex_discovery import candidate_references
from covenant_extraction.definitions.schema import ResolvedFormula
from covenant_extraction.extraction.extractor import extract_from_candidates
from covenant_extraction.extraction.llm_client import LLMClient
from covenant_extraction.extraction.loan_terms import extract_loan_terms
from covenant_extraction.extraction.schema import (
    CitedText,
    CovenantExtractionResult,
    EbitdaAddback,
    FinancialCovenant,
    LoanTerms,
    NegativeCovenant,
)
from covenant_extraction.ingestion.chunker import chunk_blocks
from covenant_extraction.ingestion.html_loader import load_html
from covenant_extraction.ingestion.pdf_loader import load_pdf
from covenant_extraction.retrieval.candidate_selector import select_candidates
from covenant_extraction.retrieval.definitions_lookup import extract_definitions_text, find_definition
from covenant_extraction.retrieval.embeddings import Embedder
from covenant_extraction.validation.calculators import check_financial_covenant
from covenant_extraction.validation.citation_audit import audit_citation

SUPPORTED_SUFFIXES = {".pdf": load_pdf, ".html": load_html, ".htm": load_html}


def load_document(path: str | Path) -> List:
    """Load a contract file into Blocks, dispatching on file extension."""
    path = Path(path)
    loader = SUPPORTED_SUFFIXES.get(path.suffix.lower())
    if loader is None:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )
    return loader(path)


def _export_citation_item(item, full_text: str) -> Dict[str, Any]:
    audit = audit_citation(item.citation.source_text, full_text)
    payload = item.model_dump()
    payload["audit"] = {
        "citation_verified": audit.verified,
        "citation_similarity": round(audit.similarity, 3),
    }
    return payload


def _export_financial_covenant(
    item: FinancialCovenant,
    full_text: str,
    definitions_text: str,
    llm_client: LLMClient,
    resolved_cache: Dict[str, ResolvedFormula],
) -> Dict[str, Any]:
    payload = _export_citation_item(item, full_text)
    check = check_financial_covenant(item)
    payload["audit"]["calculator_checked"] = check.recomputed is not None
    payload["audit"]["calculator_match"] = check.matches
    payload["audit"]["citation_grounded"] = check.citation_grounded
    if check.recomputed is not None:
        payload["audit"]["recomputed_threshold"] = check.recomputed.threshold
        payload["audit"]["recomputed_comparison"] = check.recomputed.comparison.value
    payload["definition"] = find_definition(item.name, definitions_text)

    if payload["definition"] is not None:
        # Chain into whatever terms this covenant's own definition
        # references (e.g. "Consolidated EBITDA", "Consolidated Total
        # Debt"), rather than stopping at one level -- see
        # definitions/regex_discovery.py for why this is bounded/cheap
        # even against a large real-world definitions section.
        resolved = resolve_defined_term_bounded(
            root_term=item.name,
            definitions_text=definitions_text,
            document_text=full_text,
            llm_client=llm_client,
            resolved_cache=resolved_cache,
        )
        payload["resolved_definition"] = resolved.model_dump()

    return payload


def _find_resolvable_term(value: str, definitions_text: str) -> Optional[str]:
    """First 2+-word Title-Case phrase in `value` that's itself an
    actually-defined term in this document (per find_definition) -- catches
    an LLM-extracted loan_terms value that names a defined term instead of
    resolving it (e.g. end_date coming back as "Term Loan Maturity Date" or
    "Maturity Date or termination of Revolving Commitments" instead of the
    date that term resolves to).

    Skipped entirely when `value` already contains a digit: a value like
    "a per annum fixed rate equal to 12.0%; provided that at any time after
    the Initial Maturity Date has been extended..." is already a complete,
    concrete answer -- resolving the incidental "Initial Maturity Date"
    mention buried inside it would replace a correct value with an
    unrelated one (that clause's own trigger date, not the rate itself).
    A value with no digits at all is the actual failure mode this targets:
    it's standing in for a number/date the LLM never substituted."""
    if not definitions_text or any(ch.isdigit() for ch in value):
        return None
    for candidate in candidate_references(value):
        if find_definition(candidate, definitions_text) is not None:
            return candidate
    return None


def _export_cited_text(
    item: Optional[CitedText],
    full_text: str,
    definitions_text: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    resolved_cache: Optional[Dict[str, ResolvedFormula]] = None,
) -> Optional[Dict[str, Any]]:
    if item is None:
        return None
    payload = _export_citation_item(item, full_text)

    if definitions_text and llm_client is not None:
        candidate = _find_resolvable_term(item.value, definitions_text)
        if candidate is not None:
            resolved = resolve_defined_term_bounded(
                root_term=candidate,
                definitions_text=definitions_text,
                document_text=full_text,
                llm_client=llm_client,
                resolved_cache=resolved_cache,
            )
            if resolved.status == "resolved":
                payload["resolved_value"] = resolved.model_dump()

    return payload


def _export_loan_terms(
    loan_terms: LoanTerms,
    full_text: str,
    definitions_text: str = "",
    llm_client: Optional[LLMClient] = None,
    resolved_cache: Optional[Dict[str, ResolvedFormula]] = None,
) -> Dict[str, Any]:
    # Definitions-chaining is only wired in for the three fields where an
    # LLM-returned defined-term name (instead of its actual value) has been
    # observed in practice -- end_date, loan_amount, interest_rate. The
    # other fields (borrowers/lenders/loan_type/start_date/repayment terms)
    # don't get this treatment: it would cost an extra LLM round-trip per
    # field for a failure mode that hasn't shown up there.
    return {
        "borrowers": [_export_cited_text(b, full_text) for b in loan_terms.borrowers],
        "lenders": [_export_cited_text(l, full_text) for l in loan_terms.lenders],
        "loan_type": _export_cited_text(loan_terms.loan_type, full_text),
        "loan_amount": _export_cited_text(
            loan_terms.loan_amount, full_text, definitions_text, llm_client, resolved_cache
        ),
        "interest_rate": _export_cited_text(
            loan_terms.interest_rate, full_text, definitions_text, llm_client, resolved_cache
        ),
        "start_date": _export_cited_text(loan_terms.start_date, full_text),
        "end_date": _export_cited_text(
            loan_terms.end_date, full_text, definitions_text, llm_client, resolved_cache
        ),
        "principal_repayment_terms": _export_cited_text(loan_terms.principal_repayment_terms, full_text),
    }


def to_export_dict(
    result: CovenantExtractionResult,
    full_text: str,
    definitions_text: str,
    llm_client: LLMClient,
    resolved_cache: Optional[Dict[str, ResolvedFormula]] = None,
) -> Dict[str, Any]:
    """Convert a CovenantExtractionResult into a JSON-ready dict, with every
    item annotated with citation-audit and (where applicable) calculator
    cross-check results. Financial covenants additionally get a `definition`
    field looked up from the contract's Definitions section, plus (when a
    definition was found) a `resolved_definition` chaining into whatever
    other defined terms that definition itself references.

    `resolved_cache` may be passed in shared with a sibling
    `_export_loan_terms` call for the same document, so a term resolved by
    one (e.g. "Maturity Date") is reused by the other instead of re-resolved.
    Defaults to a fresh dict per call."""
    if resolved_cache is None:
        resolved_cache = {}
    return {
        "financial_covenants": [
            _export_financial_covenant(c, full_text, definitions_text, llm_client, resolved_cache)
            for c in result.financial_covenants
        ],
        "ebitda_addbacks": [_export_citation_item(c, full_text) for c in result.ebitda_addbacks],
        "negative_covenants": [_export_citation_item(c, full_text) for c in result.negative_covenants],
    }


def run_pipeline(
    path: str | Path,
    embedder: Embedder,
    llm_client: LLMClient,
    max_candidates: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the full extraction pipeline on a single contract file.

    Args:
        path: path to a .pdf or .html/.htm loan contract.
        embedder: an Embedder (real BgeM3Embedder/TfidfEmbedder, or a test fake).
        llm_client: an LLMClient (real LMStudioClient or a test fake).
        max_candidates: override for select_candidates' cap on how many
            clauses get sent to the LLM (default: settings.max_candidates).
            Verified TfidfEmbedder needs a higher cap than BgeM3Embedder to
            reach the same recall -- see candidate_selector.py.

    Returns:
        JSON-ready dict, see `to_export_dict`.
    """
    blocks = load_document(path)
    chunks = chunk_blocks(blocks)
    full_text = "\n".join(block.text for block in blocks)
    definitions_text = extract_definitions_text(chunks)

    candidates = select_candidates(chunks, embedder, max_candidates=max_candidates)
    result = extract_from_candidates(candidates, llm_client)
    loan_terms = extract_loan_terms(chunks, llm_client)

    # Shared across both calls so a term resolved for one covenant (or for
    # loan_terms' own end_date/loan_amount/interest_rate) is reused if
    # another covenant or field ends up referencing that same term.
    resolved_cache: Dict[str, ResolvedFormula] = {}
    export = to_export_dict(result, full_text, definitions_text, llm_client, resolved_cache)
    export["loan_terms"] = _export_loan_terms(loan_terms, full_text, definitions_text, llm_client, resolved_cache)
    return export
