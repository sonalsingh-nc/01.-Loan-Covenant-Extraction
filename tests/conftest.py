"""Shared pytest fixtures: sample contract paths plus fake embedder/LLM
client implementations so the test suite runs fully offline without any
real model downloads or a running LM Studio server.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Type

import numpy as np
import pytest

from covenant_extraction.definitions.schema import FormulaFragment
from covenant_extraction.extraction.schema import (
    Citation,
    Comparison,
    CovenantExtractionResult,
    CovenantType,
    FinancialCovenant,
    LoanTerms,
    NegativeCovenant,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_HTML_PATH = PROJECT_ROOT / "data" / "sample" / "sample_credit_agreement.html"

_TOKEN_RE = re.compile(r"[a-z]+")
_VECTOR_DIM = 256


class FakeEmbedder:
    """Deterministic bag-of-words "embedding" (hashing trick), so cosine
    similarity is meaningful (shares-vocabulary => higher score) without
    downloading or running a real model."""

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), _VECTOR_DIM), dtype=np.float64)
        for i, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                vectors[i, hash(token) % _VECTOR_DIM] += 1.0
        return vectors


class FakeLLMClient:
    """Canned extraction responses keyed on simple substring rules, so
    extractor tests can verify orchestration without a real LLM.

    Also implements StructuredLLMClient.complete() for FormulaFragment
    (definitions/formula_resolver.py's only schema, besides DefinedTerm via
    scoped_resolver -- which these tests don't exercise) and LoanTerms
    (extraction/loan_terms.py, always called by run_pipeline), so
    pipeline.py can run entirely against this same fake."""

    def __init__(
        self,
        canned: Dict[str, CovenantExtractionResult] | None = None,
        canned_formulas: Dict[str, FormulaFragment] | None = None,
        canned_loan_terms: Dict[str, LoanTerms] | None = None,
    ):
        self._canned = canned or {}
        self._canned_formulas = canned_formulas or {}
        self._canned_loan_terms = canned_loan_terms or {}
        self.calls: List[str] = []
        self.complete_calls: List[str] = []

    def complete(self, system_prompt: str, user_prompt: str, schema: Type):
        self.complete_calls.append(user_prompt)

        for marker, formula in self._canned_formulas.items():
            if marker in user_prompt:
                return formula
        for marker, loan_terms in self._canned_loan_terms.items():
            if marker in user_prompt:
                return loan_terms

        if schema is FormulaFragment:
            return FormulaFragment(formula="(unresolved)")
        if schema is LoanTerms:
            return LoanTerms()
        raise AssertionError(
            f"FakeLLMClient.complete() has no canned response for schema {schema} / prompt: {user_prompt[:120]}"
        )

    def extract(self, clause_text: str) -> CovenantExtractionResult:
        self.calls.append(clause_text)

        for marker, result in self._canned.items():
            if marker in clause_text:
                return result

        if "Leverage Ratio" in clause_text and "3.50" in clause_text:
            return CovenantExtractionResult(
                financial_covenants=[
                    FinancialCovenant(
                        name="Consolidated Leverage Ratio",
                        comparison=Comparison.NOT_TO_EXCEED,
                        threshold=3.50,
                        unit="ratio",
                        test_frequency="quarterly",
                        covenant_type=CovenantType.MAINTENANCE,
                        citation=Citation(
                            source_text="the Consolidated Leverage Ratio as of the last day of "
                            "any fiscal quarter to exceed 3.50:1.00"
                        ),
                    )
                ]
            )

        if "Indebtedness" in clause_text:
            return CovenantExtractionResult(
                negative_covenants=[
                    NegativeCovenant(
                        name="Indebtedness",
                        summary="Borrower may not incur indebtedness except limited carve-outs.",
                        exceptions=["purchase money Indebtedness not to exceed $5,000,000"],
                        citation=Citation(source_text="create, incur, assume or suffer to exist any Indebtedness"),
                    )
                ]
            )

        return CovenantExtractionResult()


@pytest.fixture
def sample_html_path() -> Path:
    return SAMPLE_HTML_PATH


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm_client() -> FakeLLMClient:
    return FakeLLMClient()
