"""Structured output schema for covenant extraction.

Every extracted item carries a `Citation` so a reviewer can trace the number
back to its source clause — this is deliberately non-optional per the
"LLM shouldn't be the system of record" guardrail: nothing is trusted without
a pointer back to the contract text.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Comparison(str, Enum):
    NOT_TO_EXCEED = "not_to_exceed"  # "shall not exceed"
    NOT_LESS_THAN = "not_less_than"  # "shall not be less than" / "shall maintain ... of not less than"


class CovenantType(str, Enum):
    # Tested on a standing schedule (e.g. "as of the last day of each fiscal
    # quarter") regardless of what the borrower does -- a breach on its own
    # is an Event of Default.
    MAINTENANCE = "maintenance"
    # Only tested as a condition precedent to a specific voluntary action
    # (incurring debt, making an investment/restricted payment, etc.) --
    # e.g. "solely in the case of any such Incremental Facility ... being
    # incurred". Failing the test just blocks that action; it isn't itself
    # a standing breach.
    INCURRENCE_CONDITION = "incurrence_condition"


class Citation(BaseModel):
    """Pointer back to the exact source clause an extraction came from."""

    source_text: str = Field(..., description="Verbatim quote from the contract supporting this extraction.")
    page: Optional[int] = Field(None, description="PDF page number, if the source document is a PDF.")
    section_label: Optional[str] = Field(
        None, description='Nearest section/subsection heading (e.g. "Section 6.1"), if the source document is HTML.'
    )


class FinancialCovenant(BaseModel):
    name: str = Field(..., description='e.g. "Consolidated Leverage Ratio", "Fixed Charge Coverage Ratio"')
    comparison: Comparison
    threshold: float = Field(..., description="Numeric threshold, e.g. 3.50 for a 3.50:1.00 ratio, or 10000000 for a $10,000,000 minimum.")
    unit: str = Field(..., description='e.g. "ratio", "USD"')
    test_frequency: Optional[str] = Field(None, description='e.g. "quarterly", "at all times"')
    covenant_type: CovenantType = Field(
        ...,
        description=(
            'Whether this is a "maintenance" covenant tested on a standing schedule regardless of borrower '
            'activity (e.g. a quarterly leverage ratio test in a dedicated Financial Covenants section), or an '
            '"incurrence_condition" only tested as a precondition to a specific action like incurring new debt '
            '(e.g. "solely in the case of any such Incremental Facility ... being incurred, the ratio would not '
            'be greater than X").'
        ),
    )
    citation: Citation


class EbitdaAddback(BaseModel):
    name: str = Field(..., description='e.g. "depreciation and amortization", "non-cash stock-based compensation"')
    description: Optional[str] = None
    citation: Citation


class NegativeCovenant(BaseModel):
    name: str = Field(..., description='e.g. "Indebtedness", "Liens", "Restricted Payments"')
    summary: str = Field(..., description="One-sentence plain-English summary of the restriction.")
    exceptions: List[str] = Field(default_factory=list, description="Carve-outs/baskets, e.g. dollar-amount exceptions.")
    citation: Citation


class CovenantExtractionResult(BaseModel):
    financial_covenants: List[FinancialCovenant] = Field(default_factory=list)
    ebitda_addbacks: List[EbitdaAddback] = Field(default_factory=list)
    negative_covenants: List[NegativeCovenant] = Field(default_factory=list)


class CitedText(BaseModel):
    """A single extracted fact, in the contract's own words, plus its citation."""

    value: str = Field(..., description="The extracted value, in the contract's own words.")
    citation: Citation


class LoanTerms(BaseModel):
    """Loan-level deal terms -- the fixed facts of the transaction (who,
    how much, what kind, when), as opposed to its ongoing covenants (see
    FinancialCovenant/EbitdaAddback/NegativeCovenant above). Each field is
    independently citable since these facts typically live in different
    parts of the document (parties/amount/type/date usually in the
    preamble; interest rate, maturity, and repayment terms usually each in
    their own dedicated section)."""

    borrowers: List[CitedText] = Field(
        default_factory=list, description='Each named borrower entity, e.g. "Acme Borrower LLC".'
    )
    lenders: List[CitedText] = Field(
        default_factory=list,
        description='Each named lender/agent entity, e.g. "Example Bank, N.A., as Administrative Agent".',
    )
    loan_type: Optional[CitedText] = Field(
        None,
        description='The facility\'s seniority/security characterization, e.g. "Senior Secured Term Loan", '
        '"Senior Unsecured Revolving Credit Facility".',
    )
    loan_amount: Optional[CitedText] = Field(
        None,
        description='The principal commitment amount(s), e.g. "$50,000,000", or a multi-tranche breakdown '
        "if more than one facility amount is stated together.",
    )
    interest_rate: Optional[CitedText] = Field(
        None, description='The rate or rate-setting mechanism, e.g. "Term SOFR plus 2.50% per annum".'
    )
    start_date: Optional[CitedText] = Field(
        None, description='The Effective Date / Closing Date the agreement was entered into, e.g. "January 15, 2024".'
    )
    end_date: Optional[CitedText] = Field(
        None, description='The Maturity Date / Termination Date, e.g. "January 15, 2029".'
    )
    principal_repayment_terms: Optional[CitedText] = Field(
        None,
        description='The amortization schedule, e.g. "equal quarterly installments of 1.25% of the original '
        'principal amount, with the remaining balance due at maturity", or "bullet repayment at maturity" if no '
        "scheduled amortization is stated.",
    )
