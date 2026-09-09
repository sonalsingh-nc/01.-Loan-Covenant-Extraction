"""Regex-first, cost-bounded discovery of a covenant term's dependency
chain -- the on-ramp used to wire definitions-chaining into the production
pipeline (pipeline.py) without paying the cost of term_extractor's bulk,
whole-document Phase 1 extraction.

A real credit agreement's Definitions section can hold hundreds of terms;
extracting every one via the LLM (term_extractor.extract_defined_terms)
before we even know which are relevant would mean hundreds of LLM calls just
to resolve a couple of financial covenants. Since retrieval/definitions_lookup
already has a fast, free, regex-based `find_definition`, this module reuses
it twice over instead:

1. To fetch each term's own definition text (no LLM call).
2. To verify whether a Title-Case phrase found INSIDE that text is itself a
   genuinely-defined term, by simply trying the same lookup on it -- a false
   candidate (e.g. "Loan Documents" if it happens not to be separately
   defined) just fails the lookup and is dropped, no LLM spent either way.

The LLM is only ever spent downstream, in formula_resolver's leaf/compose
steps, for the small, bounded set of terms this discovery actually reaches.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from covenant_extraction.definitions.dependency_graph import DAG
from covenant_extraction.definitions.schema import DefinedTerm
from covenant_extraction.retrieval.definitions_lookup import find_definition

DEFAULT_MAX_TERMS = 8
DEFAULT_MAX_DEPTH = 2

# 2+ consecutive Title-Case words: long enough to plausibly be one of these
# documents' multi-word defined terms (e.g. "Consolidated Total Debt"),
# which rules out common single-word capitalized terms (GAAP, Borrower,
# Section) that would otherwise flood the candidate set.
_CANDIDATE_TERM_RE = re.compile(r"(?:[A-Z][a-zA-Z]*\s+){1,6}[A-Z][a-zA-Z]*")


def candidate_references(text: str) -> List[str]:
    """Every distinct 2+ word Title-Case phrase in `text`, in first-seen
    order -- candidates for "this might be another defined term". Public
    since pipeline.py also reuses it to spot a bare defined-term name an LLM
    returned as a field's *value* instead of resolving it (e.g. "loan_terms"'
    end_date coming back as "Term Loan Maturity Date")."""
    seen: List[str] = []
    for match in _CANDIDATE_TERM_RE.finditer(text):
        phrase = re.sub(r"\s+", " ", match.group(0)).strip()
        if phrase not in seen:
            seen.append(phrase)
    return seen


def discover_term_chain(
    root_term: str,
    definitions_text: str,
    max_terms: int = DEFAULT_MAX_TERMS,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Tuple[DAG, Dict[str, DefinedTerm]]:
    """Bounded breadth-first walk over `find_definition`, discovering
    `root_term`'s dependency chain with zero LLM calls.

    Stops once `max_terms` distinct terms have been successfully resolved or
    `max_depth` hops from `root_term` have been exhausted. A DAG node with no
    corresponding `terms` entry (a candidate phrase that didn't actually
    resolve) is left for formula_resolver.resolve_covenant_formula to flag as
    `requires_human_review` -- same as any other term_extractor gap.
    """
    graph = DAG()
    terms: Dict[str, DefinedTerm] = {}
    queue: List[Tuple[str, int]] = [(root_term, 0)]
    seen_nodes = {root_term}

    while queue and len(terms) < max_terms:
        term_name, depth = queue.pop(0)
        graph.add_node(term_name)

        definition_text = find_definition(term_name, definitions_text)
        if definition_text is None:
            continue
        terms[term_name] = DefinedTerm(term=term_name, definition_text=definition_text, source_clause=None)

        if depth >= max_depth:
            continue

        for candidate in candidate_references(definition_text):
            if candidate == term_name or candidate in seen_nodes:
                continue
            seen_nodes.add(candidate)
            graph.add_edge(parent=term_name, child=candidate)
            queue.append((candidate, depth + 1))

    return graph, terms
