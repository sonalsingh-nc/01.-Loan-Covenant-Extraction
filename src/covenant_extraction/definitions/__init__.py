"""Construction of Definitions of Financial Terms, with chaining.

Three phases:
    1. term_extractor  -- locate the Definitions section, extract every
       defined term (one schema-constrained LLM call per term).
    2. dependency_graph -- build a DAG of which defined terms reference which
       other defined terms (pure string matching, no LLM).
    3. formula_resolver -- walk the DAG bottom-up, substituting resolved
       child definitions into parent formulas, to produce a single fully
       chained ResolvedFormula for a given covenant term (e.g. "Consolidated
       EBITDA") with carve-outs and provenance attached.

See pipeline.py for the single entry point tying all three together.
"""
from __future__ import annotations
