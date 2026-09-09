"""Structured output schemas for defined-term extraction (Phase 1) and
formula resolution (Phase 3).

`ResolvedFormula` is assembled by orchestration code in formula_resolver.py
(components/carve_outs/provenance are attached programmatically) -- only
`DefinedTerm` and `FormulaFragment` are ever produced directly by the LLM.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DefinedTerm(BaseModel):
    term: str = Field(..., description='e.g. "Consolidated EBITDA" (without surrounding quotes)')
    definition_text: str = Field(..., description="Raw legal prose defining the term, verbatim from the contract.")
    source_clause: Optional[str] = Field(
        None, description='e.g. "Section 1.01(a)", if visible in the source text; otherwise null.'
    )


class FormulaFragment(BaseModel):
    """LLM-produced formula fragment for a single definition, before
    carve-outs/provenance are attached by formula_resolver.py."""

    formula: str = Field(
        ..., description='e.g. "Net Income + Interest Expense + Taxes + Depreciation and Amortization"'
    )
    notes: Optional[str] = Field(None, description="Any ambiguity or assumption the model had to make, if any.")


class ResolvedFormula(BaseModel):
    """Fully substituted formula for one defined term, with its full
    dependency chain, carve-outs, and provenance attached."""

    term: str
    formula: str
    components: Dict[str, "ResolvedFormula"] = Field(default_factory=dict)
    carve_outs: List[str] = Field(default_factory=list)
    provenance: List[str] = Field(default_factory=list)
    status: str = "resolved"  # "resolved" | "requires_human_review"
    reason: Optional[str] = None


ResolvedFormula.model_rebuild()
