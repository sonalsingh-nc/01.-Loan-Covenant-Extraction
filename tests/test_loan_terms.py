"""Tests for extraction/loan_terms.py."""
from __future__ import annotations

from typing import Dict, List, Type

from covenant_extraction.extraction.loan_terms import extract_loan_terms, select_loan_terms_candidates
from covenant_extraction.extraction.schema import Citation, CitedText, LoanTerms
from covenant_extraction.ingestion.chunker import Chunk


def _chunk(text: str, page=None, section_label=None) -> Chunk:
    return Chunk(text=text, page=page, section_label=section_label, start_char=0, end_char=len(text))


class FakeStructuredLLMClient:
    """Canned LoanTerms responses keyed by substring-of-prompt, so tests can
    verify orchestration without a real LLM."""

    def __init__(self, responses: Dict[str, LoanTerms] | None = None):
        self._responses = responses or {}
        self.calls: List[str] = []

    def complete(self, system_prompt: str, user_prompt: str, schema: Type):
        self.calls.append(user_prompt)
        for marker, response in self._responses.items():
            if marker in user_prompt:
                return response
        return LoanTerms()


# --- select_loan_terms_candidates -------------------------------------------


def test_select_loan_terms_candidates_includes_preamble_up_to_budget():
    chunks = [_chunk("A" * 1000), _chunk("B" * 1000), _chunk("C" * 1000), _chunk("D" * 1000)]

    candidates = select_loan_terms_candidates(chunks)

    # 3000-char budget: first 3 chunks (3000 chars) satisfy it; the 4th
    # isn't needed for the preamble and doesn't match any keyword either.
    assert chunks[0] in candidates
    assert chunks[1] in candidates
    assert chunks[2] in candidates
    assert chunks[3] not in candidates


def test_select_loan_terms_candidates_adds_keyword_matched_chunks_outside_preamble():
    # Large enough on its own to exhaust the preamble char budget, so the
    # other chunks are only included if they keyword-match (mirrors a real
    # document, where the preamble is a sizeable recital paragraph and
    # everything else is a separate, much later section).
    preamble = _chunk("This Agreement is entered into among the parties. " * 70)
    interest_chunk = _chunk("The Applicable Margin plus Term SOFR shall constitute the interest rate.")
    maturity_chunk = _chunk("The Maturity Date shall be five years from the Closing Date.")
    repayment_chunk = _chunk("Principal shall be repaid in equal quarterly installments of the outstanding principal amount.")
    unrelated = _chunk("Notices shall be delivered by hand or overnight courier.")

    chunks = [preamble, interest_chunk, maturity_chunk, repayment_chunk, unrelated]
    candidates = select_loan_terms_candidates(chunks)

    assert preamble in candidates  # within char budget
    assert interest_chunk in candidates
    assert maturity_chunk in candidates
    assert repayment_chunk in candidates
    assert unrelated not in candidates


def test_select_loan_terms_candidates_empty_input():
    assert select_loan_terms_candidates([]) == []


def test_select_loan_terms_candidates_ignores_accounting_amortization():
    # "amortization" bare is ambiguous -- EBITDA-addback (depreciation &
    # amortization) accounting language shouldn't be mistaken for a loan
    # principal repayment schedule.
    preamble = _chunk("This Agreement is entered into among the parties. " * 70)
    ebitda_addback_chunk = _chunk(
        "Consolidated EBITDA means Net Income plus (i) depreciation, (ii) amortization, (iii) interest expense."
    )
    real_repayment_chunk = _chunk("The Term Loans shall be repaid in scheduled installments of principal each quarter.")

    chunks = [preamble, ebitda_addback_chunk, real_repayment_chunk]
    candidates = select_loan_terms_candidates(chunks)

    assert ebitda_addback_chunk not in candidates
    assert real_repayment_chunk in candidates


def test_select_loan_terms_candidates_prefers_applicable_margin_over_generic_base_rate():
    # A specific pricing-grid term ("Applicable Margin") earlier only in
    # document position after a generic "Base Rate" mention should still
    # win -- the specific tier is tried across the whole document before
    # falling back to the generic one.
    preamble = _chunk("This Agreement is entered into among the parties. " * 70)
    generic_rate_chunk = _chunk('"ABR" means a rate determined by reference to the Base Rate.')
    specific_rate_chunk = _chunk("Loans shall bear interest at Term SOFR plus the Applicable Margin.")

    chunks = [preamble, generic_rate_chunk, specific_rate_chunk]
    candidates = select_loan_terms_candidates(chunks)

    assert specific_rate_chunk in candidates
    assert generic_rate_chunk not in candidates


def test_select_loan_terms_candidates_adds_loan_type_and_amount_chunks():
    # Neither the facility-type nor the headline amount is stated in the
    # preamble here (unlike a real credit agreement's title/recital), so
    # both should be picked up from their own keyword-matched chunks.
    preamble = _chunk("This Agreement is entered into among the parties. " * 70)
    loan_type_chunk = _chunk("The Loans constitute a senior secured revolving credit facility.")
    loan_amount_chunk = _chunk("The aggregate principal amount of the Commitments shall not exceed $50,000,000.")
    unrelated = _chunk("Notices shall be delivered by hand or overnight courier.")

    chunks = [preamble, loan_type_chunk, loan_amount_chunk, unrelated]
    candidates = select_loan_terms_candidates(chunks)

    assert loan_type_chunk in candidates
    assert loan_amount_chunk in candidates
    assert unrelated not in candidates


def test_select_loan_terms_candidates_loan_amount_ignores_bare_commitment():
    # A bare "commitment" mention (an individual lender's piece, not the
    # deal's headline amount) shouldn't be mistaken for the loan-amount
    # clause -- only the specific "aggregate/total ... amount|commitment"
    # phrasing should match.
    preamble = _chunk("This Agreement is entered into among the parties. " * 70)
    individual_commitment_chunk = _chunk("Each Lender's Commitment is set forth on Schedule 2.01.")
    real_amount_chunk = _chunk("The Total Commitments in the aggregate amount of $100,000,000 shall be available.")

    chunks = [preamble, individual_commitment_chunk, real_amount_chunk]
    candidates = select_loan_terms_candidates(chunks)

    assert individual_commitment_chunk not in candidates
    assert real_amount_chunk in candidates


def test_select_loan_terms_candidates_loan_amount_falls_back_to_not_to_exceed():
    # No "aggregate"/"total" phrasing anywhere -- the generic fallback tier
    # ("not to exceed $") should still catch the headline figure.
    preamble = _chunk("This Agreement is entered into among the parties. " * 70)
    amount_chunk = _chunk("The Loans shall not to exceed $25,000,000 in outstanding principal at any time.")

    chunks = [preamble, amount_chunk]
    candidates = select_loan_terms_candidates(chunks)

    assert amount_chunk in candidates


def test_select_loan_terms_candidates_loan_amount_ignores_usury_boilerplate():
    # "the maximum amount permitted by Applicable Law" is ordinary late-fee
    # boilerplate, not the deal's headline figure -- "credit"/"loan"/
    # "facility" must appear between maximum/total and amount.
    preamble = _chunk("This Agreement is entered into among the parties. " * 70)
    usury_chunk = _chunk("Borrower shall pay the lesser of 5% or the maximum amount permitted by Applicable Law.")
    real_amount_chunk = _chunk("The Maximum Credit Amount available under this facility is $20,000,000.")

    chunks = [preamble, usury_chunk, real_amount_chunk]
    candidates = select_loan_terms_candidates(chunks)

    assert usury_chunk not in candidates
    assert real_amount_chunk in candidates


def test_select_loan_terms_candidates_loan_amount_falls_back_to_principal_amount_of():
    # Commercial-mortgage-style agreements often state the headline figure
    # as a plain Note/Loan definition with no "aggregate"/"commitment"
    # language at all.
    preamble = _chunk("This Agreement is entered into among the parties. " * 70)
    amount_chunk = _chunk(
        "Note shall mean that certain Promissory Note in the principal amount of $30,000,000, made by Borrower."
    )

    chunks = [preamble, amount_chunk]
    candidates = select_loan_terms_candidates(chunks)

    assert amount_chunk in candidates


def test_select_loan_terms_candidates_loan_amount_bare_mention_without_dollar_sign_ignored():
    # "principal amount of" without an immediately-following dollar figure
    # is too generic (e.g. referring to a different note's balance) to
    # trust as the deal's headline amount.
    preamble = _chunk("This Agreement is entered into among the parties. " * 70)
    bare_mention_chunk = _chunk("Lender may request a statement setting forth the unpaid principal amount of the Note.")

    chunks = [preamble, bare_mention_chunk]
    candidates = select_loan_terms_candidates(chunks)

    assert bare_mention_chunk not in candidates


def test_select_loan_terms_candidates_adds_start_date_chunk():
    # "dated as of" is outside the preamble budget here, unlike a real
    # document where it's usually the very first sentence.
    preamble = _chunk("This Agreement governs the relationship between the parties. " * 70)
    start_date_chunk = _chunk("This Credit Agreement, dated as of March 3, 2022, is made among the parties hereto.")
    unrelated = _chunk("Notices shall be delivered by hand or overnight courier.")

    chunks = [preamble, start_date_chunk, unrelated]
    candidates = select_loan_terms_candidates(chunks)

    assert start_date_chunk in candidates
    assert unrelated not in candidates


def test_select_loan_terms_candidates_maturity_falls_back_to_expiration_date():
    # Neither "Maturity Date" nor "Termination Date" appears -- the
    # secondary maturity-phrasing tier should still find "Expiration Date".
    preamble = _chunk("This Agreement is entered into among the parties. " * 70)
    maturity_chunk = _chunk("The Revolving Credit Facility shall be available until the Expiration Date.")

    chunks = [preamble, maturity_chunk]
    candidates = select_loan_terms_candidates(chunks)

    assert maturity_chunk in candidates


# --- extract_loan_terms -------------------------------------------------------


def test_extract_loan_terms_merges_across_candidates_and_pins_citations():
    preamble = _chunk(
        'This Credit Agreement is entered into as of January 15, 2024, among Acme Borrower LLC '
        '("Borrower") and Example Bank, N.A. ("Lender").',
        page=1,
    )
    rate_chunk = _chunk("Interest Rate: Term SOFR plus 2.50% per annum.", page=12)

    fake = FakeStructuredLLMClient(
        responses={
            "January 15, 2024": LoanTerms(
                borrowers=[CitedText(value="Acme Borrower LLC", citation=Citation(source_text="Acme Borrower LLC"))],
                lenders=[CitedText(value="Example Bank, N.A.", citation=Citation(source_text="Example Bank, N.A."))],
                start_date=CitedText(value="January 15, 2024", citation=Citation(source_text="as of January 15, 2024")),
            ),
            "Term SOFR plus 2.50%": LoanTerms(
                interest_rate=CitedText(
                    value="Term SOFR plus 2.50% per annum", citation=Citation(source_text="Term SOFR plus 2.50% per annum")
                )
            ),
        }
    )

    result = extract_loan_terms([preamble, rate_chunk], fake)

    assert len(result.borrowers) == 1
    assert result.borrowers[0].value == "Acme Borrower LLC"
    assert result.borrowers[0].citation.page == 1  # pinned from the real chunk, not model-guessed

    assert result.interest_rate is not None
    assert result.interest_rate.value == "Term SOFR plus 2.50% per annum"
    assert result.interest_rate.citation.page == 12

    assert result.start_date is not None
    assert result.start_date.value == "January 15, 2024"


def test_extract_loan_terms_dedupes_repeated_party_names():
    preamble = _chunk("Acme Borrower LLC is the Borrower under this Agreement with Example Bank, N.A.", page=1)
    rate_chunk = _chunk("Acme Borrower LLC shall pay interest at Term SOFR plus 2.50%.", page=12)

    borrower_a = LoanTerms(
        borrowers=[CitedText(value="Acme Borrower LLC", citation=Citation(source_text="Acme Borrower LLC is the Borrower"))]
    )
    borrower_b = LoanTerms(
        borrowers=[CitedText(value="Acme Borrower LLC", citation=Citation(source_text="Acme Borrower LLC shall pay"))]
    )
    fake = FakeStructuredLLMClient(responses={"Borrower under this": borrower_a, "shall pay interest": borrower_b})

    result = extract_loan_terms([preamble, rate_chunk], fake)

    assert len(result.borrowers) == 1


def test_extract_loan_terms_returns_empty_when_nothing_found():
    chunks = [_chunk("Notices shall be delivered by hand or overnight courier.")]
    fake = FakeStructuredLLMClient()

    result = extract_loan_terms(chunks, fake)

    assert result.borrowers == []
    assert result.lenders == []
    assert result.loan_type is None
    assert result.loan_amount is None
    assert result.interest_rate is None
    assert result.start_date is None
    assert result.end_date is None
    assert result.principal_repayment_terms is None
