"""Phase 3: resolve a covenant formula by walking the term-dependency DAG
bottom-up (topological order), substituting resolved child definitions into
parent formulas. The LLM is called per-node, not per-decision -- no free
reasoning about what to look up next.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from covenant_extraction.definitions.dependency_graph import DAG, topological_sort
from covenant_extraction.definitions.prompts import (
    COMPOSE_SYSTEM_PROMPT,
    LEAF_SYSTEM_PROMPT,
    TERM_SYSTEM_PROMPT,
    build_compose_user_prompt,
    build_leaf_user_prompt,
    build_term_user_prompt,
)
from covenant_extraction.definitions.schema import DefinedTerm, FormulaFragment, ResolvedFormula
from covenant_extraction.extraction.llm_client import StructuredLLMClient

MAX_SCOPED_HOPS = 2

# Carve-out / exception language, e.g. "...; provided that there shall be
# excluded therefrom (a) ... (b) ...".
_CARVE_OUT_RE = re.compile(
    r"\b(?:provided(?:,)?\s+(?:however,\s+)?that|excluding|other than|except(?:\s+for)?)\b.{0,300}",
    re.IGNORECASE,
)

# "<Term>" has the meaning set forth/specified in Section X -- a
# cross-reference the dependency graph's string matching can't follow on its
# own, since the referenced term's *name* (not its definition) is what's
# quoted here.
_SECTION_XREF_RE = re.compile(
    r'"([A-Z][^"]{0,80})"\s+has\s+the\s+meaning\s+(?:set\s+forth|specified)\s+in\s+Section\s+([\d.()a-zA-Z]+)',
    re.IGNORECASE,
)


def extract_carve_outs(definition_text: str) -> List[str]:
    """Pull carve-out/exception clauses out of definition prose so they're
    preserved as explicit conditions rather than collapsed into the formula."""
    return [m.group(0).strip() for m in _CARVE_OUT_RE.finditer(definition_text)]


def _find_missing_reference(definition_text: str) -> Optional[str]:
    match = _SECTION_XREF_RE.search(definition_text)
    return match.group(1) if match else None


def has_unresolved_cross_reference(node: DefinedTerm, child_formulas: Dict[str, ResolvedFormula]) -> bool:
    """True if the definition has an explicit '"<Term>" has the meaning set
    forth in Section X' cross-reference not already covered by a resolved
    child -- the ONE case that still routes to scoped_resolver rather than
    composing best-effort from whatever children DID resolve.

    A child that simply failed to resolve (e.g. a real-world addback list
    referencing an inline nickname -- "Expected Cost Savings" -- that was
    never its own standalone definition) does NOT block composition: a
    real credit agreement's EBITDA definition routinely has a few such
    minor unresolvable references, and blanking out the entire parent
    formula over one of them would hide an otherwise-useful result. See
    resolve_covenant_formula -- each such child's own `status` stays
    visible in `components`, so nothing is hidden, just not gated on.
    """
    missing_reference = _find_missing_reference(node.definition_text)
    return missing_reference is not None and missing_reference not in child_formulas


def search_document_for_term(document_text: str, term: str) -> Optional[str]:
    """Find the paragraph containing `"<term>" means ...` (or "shall mean")
    anywhere in the full document, not just the definitions section -- covers
    terms defined inline elsewhere (e.g. inside a pricing-grid section)."""
    pattern = re.compile(
        r'"' + re.escape(term) + r'"\s+(?:means|shall mean)\s*,?\s*(.{0,1000}?)(?:\.\s|\Z)',
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(document_text)
    if not match:
        return None
    return document_text[match.start() : match.end()].strip()


def _try_alternate_phrasing(term: str) -> str:
    """Cheap fallback phrasing when the exact term isn't found verbatim --
    strip a leading qualifier, the most common reason an exact match misses
    (prose sometimes drops "Consolidated"/"Applicable" on later mentions)."""
    for prefix in ("Consolidated ", "Applicable "):
        if term.startswith(prefix):
            return term[len(prefix) :]
    return term


def scoped_resolver(
    term: str,
    missing_ref: str,
    document_text: str,
    llm_client: StructuredLLMClient,
    max_hops: int = MAX_SCOPED_HOPS,
) -> Optional[DefinedTerm]:
    """Bounded search for a cross-referenced term the dependency graph
    couldn't resolve -- the ONE place agent-like behavior is allowed here.
    Hard cap on hops; fails loud (returns None -> human review) rather than
    looping or guessing."""
    ref = missing_ref
    for _ in range(max_hops):
        location = search_document_for_term(document_text, ref)
        if location:
            # Reuses the Phase 1 term-extraction prompt/schema -- a scoped
            # lookup is extracting one more DefinedTerm, just from outside
            # the definitions section.
            return llm_client.complete(
                system_prompt=TERM_SYSTEM_PROMPT,
                user_prompt=build_term_user_prompt(location),
                schema=DefinedTerm,
            )
        ref = _try_alternate_phrasing(ref)
    return None


def _provenance(node: DefinedTerm) -> List[str]:
    return [node.source_clause] if node.source_clause else []


def resolve_covenant_formula(
    root_term: str,
    graph: DAG,
    terms: Dict[str, DefinedTerm],
    document_text: str,
    llm_client: StructuredLLMClient,
    resolved: Optional[Dict[str, ResolvedFormula]] = None,
) -> ResolvedFormula:
    """Walk the DAG bottom-up, substituting resolved child definitions into
    parent formulas, and return the fully-substituted formula for
    `root_term` with its full dependency chain, carve-outs, and provenance
    attached.

    `resolved` may be passed in as a cache shared across multiple calls (e.g.
    several covenants that both reference "Consolidated EBITDA"), so a term
    already resolved by an earlier call is reused rather than re-resolved --
    saving an LLM call per repeat. Defaults to a fresh dict per call.
    """
    order = topological_sort(graph, root=root_term)
    if resolved is None:
        resolved = {}

    for term_name in order:
        if term_name in resolved:
            continue

        node = terms.get(term_name)
        if node is None:
            resolved[term_name] = ResolvedFormula(
                term=term_name,
                formula="",
                status="requires_human_review",
                reason="term_not_found_in_definitions_section",
            )
            continue

        children = graph.get_children(term_name)
        child_formulas = {c: resolved[c] for c in children if c in resolved}

        if not children:
            fragment: FormulaFragment = llm_client.complete(
                system_prompt=LEAF_SYSTEM_PROMPT,
                user_prompt=build_leaf_user_prompt(node.definition_text),
                schema=FormulaFragment,
            )
            resolved[term_name] = ResolvedFormula(
                term=term_name, formula=fragment.formula, provenance=_provenance(node)
            )

        elif has_unresolved_cross_reference(node, child_formulas):
            # Explicit "has the meaning set forth in Section X" cross-reference
            # -- scoped, capped lookup, NOT an open-ended agent loop.
            missing_ref = _find_missing_reference(node.definition_text)
            supplement = (
                scoped_resolver(term_name, missing_ref, document_text, llm_client) if missing_ref else None
            )

            if supplement is None:
                resolved[term_name] = ResolvedFormula(
                    term=term_name,
                    formula="",
                    components=child_formulas,
                    status="requires_human_review",
                    reason="unresolved_reference",
                    provenance=_provenance(node),
                )
                resolved[term_name].carve_outs = extract_carve_outs(node.definition_text)
                continue

            fragment = llm_client.complete(
                system_prompt=COMPOSE_SYSTEM_PROMPT,
                user_prompt=build_compose_user_prompt(
                    node.definition_text, child_formulas, supplement=supplement.definition_text
                ),
                schema=FormulaFragment,
            )
            resolved[term_name] = ResolvedFormula(
                term=term_name,
                formula=fragment.formula,
                components=child_formulas,
                provenance=_provenance(node),
            )

        else:
            # Best-effort compose from whatever children DID resolve, even
            # if one or more are themselves flagged `requires_human_review`
            # (e.g. an inline nickname that was never a standalone
            # definition). Nothing is hidden -- each component's own status
            # stays visible under `components` -- but one such minor gap no
            # longer blanks out an otherwise-composable parent formula.
            fragment = llm_client.complete(
                system_prompt=COMPOSE_SYSTEM_PROMPT,
                user_prompt=build_compose_user_prompt(node.definition_text, child_formulas),
                schema=FormulaFragment,
            )
            resolved[term_name] = ResolvedFormula(
                term=term_name,
                formula=fragment.formula,
                components=child_formulas,
                provenance=_provenance(node),
            )

        # Preserve carve-outs/exceptions as conditions, not collapsed away.
        resolved[term_name].carve_outs = extract_carve_outs(node.definition_text)

    return resolved[root_term]
