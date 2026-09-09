"""Tests for extraction/schema.py Pydantic models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from covenant_extraction.extraction.schema import (
    Citation,
    Comparison,
    CovenantExtractionResult,
    CovenantType,
    FinancialCovenant,
)


def test_financial_covenant_valid():
    covenant = FinancialCovenant(
        name="Consolidated Leverage Ratio",
        comparison=Comparison.NOT_TO_EXCEED,
        threshold=3.50,
        unit="ratio",
        covenant_type=CovenantType.MAINTENANCE,
        citation=Citation(source_text="shall not exceed 3.50:1.00"),
    )
    assert covenant.comparison == Comparison.NOT_TO_EXCEED
    assert covenant.test_frequency is None
    assert covenant.covenant_type == CovenantType.MAINTENANCE


def test_financial_covenant_rejects_invalid_covenant_type():
    with pytest.raises(ValidationError):
        FinancialCovenant(
            name="X",
            comparison=Comparison.NOT_TO_EXCEED,
            threshold=1.0,
            unit="ratio",
            covenant_type="not_a_real_value",
            citation=Citation(source_text="x"),
        )


def test_financial_covenant_requires_covenant_type():
    with pytest.raises(ValidationError):
        FinancialCovenant(
            name="X",
            comparison=Comparison.NOT_TO_EXCEED,
            threshold=1.0,
            unit="ratio",
            citation=Citation(source_text="x"),
        )


def test_financial_covenant_rejects_invalid_comparison():
    with pytest.raises(ValidationError):
        FinancialCovenant(
            name="X",
            comparison="not_a_real_value",
            threshold=1.0,
            unit="ratio",
            citation=Citation(source_text="x"),
        )


def test_financial_covenant_requires_citation():
    with pytest.raises(ValidationError):
        FinancialCovenant(
            name="X",
            comparison=Comparison.NOT_TO_EXCEED,
            threshold=1.0,
            unit="ratio",
        )


def test_covenant_extraction_result_defaults_to_empty_lists():
    result = CovenantExtractionResult()
    assert result.financial_covenants == []
    assert result.ebitda_addbacks == []
    assert result.negative_covenants == []


def test_covenant_extraction_result_round_trips_through_json():
    result = CovenantExtractionResult(
        financial_covenants=[
            FinancialCovenant(
                name="Consolidated Leverage Ratio",
                comparison=Comparison.NOT_TO_EXCEED,
                threshold=3.50,
                unit="ratio",
                covenant_type=CovenantType.MAINTENANCE,
                citation=Citation(source_text="shall not exceed 3.50:1.00"),
            )
        ]
    )

    restored = CovenantExtractionResult.model_validate_json(result.model_dump_json())
    assert restored == result
