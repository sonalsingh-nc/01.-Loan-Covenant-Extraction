"""Phase 2: build a dependency graph of defined terms (algorithmic, no LLM
call). For each term's definition text, find references to other defined
terms via string matching against the set of known term names, then expose
that graph in dependency order for Phase 3's bottom-up traversal.

Cycles are a real risk in amended/restated agreements (e.g. two terms whose
"as amended" definitions each reference the other's prior definition) -- they
are detected and flagged for human review rather than silently resolved.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set

from covenant_extraction.definitions.schema import DefinedTerm


@dataclass
class DAG:
    """Directed graph over term names: edge parent -> child means "parent's
    definition references child"."""

    nodes: Set[str] = field(default_factory=set)
    edges: Dict[str, Set[str]] = field(default_factory=dict)  # parent -> children

    def add_node(self, name: str) -> None:
        self.nodes.add(name)
        self.edges.setdefault(name, set())

    def add_edge(self, parent: str, child: str) -> None:
        self.add_node(parent)
        self.add_node(child)
        self.edges[parent].add(child)

    def get_children(self, name: str) -> Set[str]:
        return self.edges.get(name, set())

    def remove_edge(self, parent: str, child: str) -> None:
        self.edges.get(parent, set()).discard(child)

    def copy(self) -> "DAG":
        return DAG(nodes=set(self.nodes), edges={k: set(v) for k, v in self.edges.items()})


@dataclass(frozen=True)
class Cycle:
    terms: List[str]  # e.g. ["A", "B", "C", "A"] -- last entry closes the loop back to the first


def _term_pattern(term: str) -> re.Pattern:
    """Match `term` as a quoted phrase or bare capitalized phrase, allowing a
    trailing "s" for simple pluralization (e.g. "Restricted Subsidiaries")."""
    return re.compile(r'"?' + re.escape(term) + r'"?s?\b')


def term_appears_in_text(term: str, text: str) -> bool:
    return _term_pattern(term).search(text) is not None


def find_term_references(text: str, known_terms: Iterable[str]) -> List[str]:
    """Match known defined terms inside another definition's text.

    Handles exact match and simple plural forms via `term_appears_in_text`.
    Cross-references like "the meaning set forth in Section 6.02" are NOT
    resolved here -- an unmatched reference falls through to Phase 3's
    scoped_resolver instead.
    """
    return [t for t in known_terms if term_appears_in_text(t, text)]


def detect_cycles(graph: DAG) -> List[Cycle]:
    """DFS-based cycle detection; returns every distinct cycle found."""
    cycles: List[Cycle] = []
    visiting: Set[str] = set()
    visited: Set[str] = set()
    stack: List[str] = []

    def visit(node: str) -> None:
        visiting.add(node)
        stack.append(node)
        for child in sorted(graph.get_children(node)):
            if child in visiting:
                cycle_start = stack.index(child)
                cycles.append(Cycle(terms=stack[cycle_start:] + [child]))
            elif child not in visited:
                visit(child)
        stack.pop()
        visiting.discard(node)
        visited.add(node)

    for node in sorted(graph.nodes):
        if node not in visited:
            visit(node)

    return cycles


def break_cycles_for_traversal(graph: DAG, cycles: List[Cycle]) -> DAG:
    """Return a COPY of `graph` with the closing edge of each cycle removed,
    so topological_sort can proceed. Doesn't mutate `graph` -- callers that
    need the full (un-broken) picture, e.g. for human review, keep it."""
    broken = graph.copy()
    for cycle in cycles:
        parent, child = cycle.terms[-2], cycle.terms[-1]
        broken.remove_edge(parent, child)
    return broken


def topological_sort(graph: DAG, root: str) -> List[str]:
    """Return the nodes reachable from `root`, ordered leaves-first, suitable
    for Phase 3's bottom-up traversal. `graph` must already be acyclic (run
    it through break_cycles_for_traversal first if it isn't)."""
    order: List[str] = []
    visited: Set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        visited.add(node)
        for child in sorted(graph.get_children(node)):
            visit(child)
        order.append(node)

    visit(root)
    return order


def flag_for_human_review(cycle: Cycle, reason: str) -> None:
    """Surface a circular-definition finding. A separate hook (rather than
    raising) so callers can decide how to surface it without this module
    depending on any particular reporting mechanism."""
    print(f"[human review] {reason}: {' -> '.join(cycle.terms)}", file=sys.stderr, flush=True)


def build_dependency_graph(terms: List[DefinedTerm]) -> DAG:
    """For each term's definition text, find references to other defined
    terms (pure string matching -- no LLM call). Detects cycles and flags
    them for human review; the returned graph has cycle-closing edges
    removed so it's always safe to run topological_sort against it.
    """
    term_names = {t.term for t in terms}
    graph = DAG()

    for term in terms:
        graph.add_node(term.term)
        for ref in find_term_references(term.definition_text, term_names):
            if ref != term.term:
                graph.add_edge(parent=term.term, child=ref)

    cycles = detect_cycles(graph)
    if not cycles:
        return graph

    for cycle in cycles:
        flag_for_human_review(cycle, reason="circular_definition")

    return break_cycles_for_traversal(graph, cycles)
