"""Tests for validation/citation_audit.py and validation/calculators.py."""
from __future__ import annotations

from covenant_extraction.extraction.schema import Citation, Comparison, CovenantType, FinancialCovenant
from covenant_extraction.validation.calculators import (
    check_financial_covenant,
    citation_contains_threshold,
    recompute_threshold,
)
from covenant_extraction.validation.citation_audit import audit_citation


# --- citation_audit ---------------------------------------------------------

def test_audit_citation_exact_substring_match():
    source = "The Borrower shall not permit the Consolidated Leverage Ratio to exceed 3.50:1.00."
    result = audit_citation("Consolidated Leverage Ratio to exceed 3.50:1.00", source)
    assert result.verified is True
    assert result.similarity == 1.0


def test_audit_citation_tolerates_whitespace_differences():
    source = "The Borrower   shall not   permit the Consolidated Leverage Ratio\nto exceed 3.50:1.00."
    result = audit_citation("Consolidated Leverage Ratio to exceed 3.50:1.00", source)
    assert result.verified is True


def test_audit_citation_flags_fabricated_quote():
    source = "The Borrower shall not permit the Consolidated Leverage Ratio to exceed 3.50:1.00."
    result = audit_citation("The Borrower shall maintain a minimum EBITDA of $50,000,000", source)
    assert result.verified is False


def test_audit_citation_empty_quote_is_not_verified():
    result = audit_citation("", "some source text")
    assert result.verified is False


# --- calculators -------------------------------------------------------------

def test_recompute_threshold_parses_ratio_not_to_exceed():
    recomputed = recompute_threshold("shall not permit the ratio to exceed 3.50:1.00")
    assert recomputed is not None
    assert recomputed.threshold == 3.50
    assert recomputed.comparison == Comparison.NOT_TO_EXCEED


def test_recompute_threshold_parses_dollar_not_less_than():
    recomputed = recompute_threshold("shall maintain Consolidated Liquidity of not less than $10,000,000")
    assert recomputed is not None
    assert recomputed.threshold == 10_000_000.0
    assert recomputed.comparison == Comparison.NOT_LESS_THAN


def test_recompute_threshold_returns_none_when_no_direction_keyword():
    assert recompute_threshold("Consolidated EBITDA means Consolidated Net Income plus taxes.") is None


def test_check_financial_covenant_matches_when_llm_agrees_with_regex():
    covenant = FinancialCovenant(
        name="Consolidated Leverage Ratio",
        comparison=Comparison.NOT_TO_EXCEED,
        threshold=3.50,
        unit="ratio",
        covenant_type=CovenantType.MAINTENANCE,
        citation=Citation(source_text="the ratio shall not exceed 3.50:1.00"),
    )
    check = check_financial_covenant(covenant)
    assert check.matches is True
    assert check.recomputed is not None


def test_check_financial_covenant_flags_mismatch():
    covenant = FinancialCovenant(
        name="Consolidated Leverage Ratio",
        comparison=Comparison.NOT_TO_EXCEED,
        threshold=4.00,  # LLM said 4.00, but source text says 3.50
        unit="ratio",
        covenant_type=CovenantType.MAINTENANCE,
        citation=Citation(source_text="the ratio shall not exceed 3.50:1.00"),
    )
    check = check_financial_covenant(covenant)
    assert check.matches is False
    assert check.recomputed.threshold == 3.50


def test_check_financial_covenant_passes_when_nothing_to_cross_check():
    covenant = FinancialCovenant(
        name="Some Covenant",
        comparison=Comparison.NOT_TO_EXCEED,
        threshold=3.50,
        unit="ratio",
        covenant_type=CovenantType.INCURRENCE_CONDITION,
        citation=Citation(source_text="a vague clause with no explicit numeric restatement"),
    )
    check = check_financial_covenant(covenant)
    assert check.recomputed is None
    assert check.matches is True
    assert check.citation_grounded is False


# --- citation_contains_threshold / citation_grounded -------------------------


def test_citation_contains_threshold_matches_ratio_format():
    assert citation_contains_threshold(4.5, "does not exceed either (x) 4.50:1.00") is True


def test_citation_contains_threshold_matches_whole_dollar_amount():
    assert citation_contains_threshold(10_000_000.0, "of not less than $10,000,000") is True


def test_citation_contains_threshold_false_when_number_absent():
    # Mirrors a real extraction bug found in the Mizuho-QVC agreement: a
    # covenant citation that trails off before ever stating a number
    # (a dangling "(y) solely in the case of ..." clause), where the LLM
    # nonetheless filled in a threshold -- almost certainly copied from a
    # different, nearby covenant's number.
    source_text = (
        "solely in the case of any such Incremental Facility or Pari Passu Indebtedness being "
        "incurred to finance an acquisition or other similar Investment permitted hereunder, the "
        "Consolidated Leverage Ratio for the most recent four fiscal quarter period for which "
        "financial statements have been delivered pursuant to Section 5.01"
    )
    assert citation_contains_threshold(3.5, source_text) is False


def test_check_financial_covenant_flags_ungrounded_threshold():
    covenant = FinancialCovenant(
        name="Consolidated Leverage Ratio",
        comparison=Comparison.NOT_TO_EXCEED,
        threshold=3.5,
        unit="ratio",
        covenant_type=CovenantType.INCURRENCE_CONDITION,
        citation=Citation(source_text="solely in the case of an acquisition, the ratio for such period"),
    )
    check = check_financial_covenant(covenant)
    assert check.citation_grounded is False


def test_check_financial_covenant_grounded_when_threshold_present():
    covenant = FinancialCovenant(
        name="Consolidated Leverage Ratio",
        comparison=Comparison.NOT_TO_EXCEED,
        threshold=4.5,
        unit="ratio",
        covenant_type=CovenantType.MAINTENANCE,
        citation=Citation(source_text="shall not exceed 4.50:1.00"),
    )
    check = check_financial_covenant(covenant)
    assert check.citation_grounded is True
