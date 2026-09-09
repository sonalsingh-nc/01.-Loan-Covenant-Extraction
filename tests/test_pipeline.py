"""Integration-style test for pipeline.py, still fully mocked (fake embedder
+ fake LLM client) so it runs without any real model or network access.
"""
from __future__ import annotations

from pathlib import Path

from covenant_extraction.definitions.schema import FormulaFragment
from covenant_extraction.pipeline import run_pipeline, to_export_dict
from covenant_extraction.extraction.schema import (
    Citation,
    CitedText,
    Comparison,
    CovenantExtractionResult,
    CovenantType,
    FinancialCovenant,
    LoanTerms,
)


def test_run_pipeline_on_sample_html(sample_html_path: Path, fake_embedder, fake_llm_client):
    result = run_pipeline(sample_html_path, embedder=fake_embedder, llm_client=fake_llm_client)

    assert "financial_covenants" in result
    assert "ebitda_addbacks" in result
    assert "negative_covenants" in result

    fc = result["financial_covenants"]
    assert any(item["name"] == "Consolidated Leverage Ratio" for item in fc)
    leverage = next(item for item in fc if item["name"] == "Consolidated Leverage Ratio")
    assert leverage["audit"]["citation_verified"] is True
    assert leverage["audit"]["calculator_match"] is True
    # "Consolidated Leverage Ratio" itself isn't a defined term in the sample
    # contract's Definitions section, so lookup correctly finds nothing --
    # and since there's no definition to chain from, no definitions-chaining
    # LLM call happens. extract_loan_terms's calls still happen though (it
    # always runs, regardless of covenant definitions).
    assert "definition" in leverage
    assert leverage["definition"] is None
    assert "resolved_definition" not in leverage
    assert fake_llm_client.complete_calls
    assert all(call.startswith("Extract any loan-level deal terms") for call in fake_llm_client.complete_calls)

    assert "loan_terms" in result
    assert result["loan_terms"]["borrowers"] == []
    assert result["loan_terms"]["loan_type"] is None


def test_run_pipeline_forwards_max_candidates_to_select_candidates(
    sample_html_path: Path, fake_embedder, fake_llm_client
):
    # Verifies the max_candidates override actually reaches
    # candidate_selector.select_candidates() -- needed so callers using
    # TfidfEmbedder can raise the cap (see retrieval/candidate_selector.py's
    # TFIDF_RECOMMENDED_MAX_CANDIDATES: TF-IDF's weaker ranking pushes real
    # covenant clauses below the bge-m3-tuned default of 5).
    run_pipeline(sample_html_path, embedder=fake_embedder, llm_client=fake_llm_client, max_candidates=1)

    # Exactly one clause was sent to extraction, not the 3 the sample
    # contract would otherwise yield (SECTION 6.1/6.2/6.3).
    assert len(fake_llm_client.calls) == 1


def test_to_export_dict_chains_into_referenced_terms(fake_llm_client):
    # A covenant whose own definition references another defined term
    # ("Consolidated Net Income") -- to_export_dict should chain into that
    # component, not just stop at the covenant's own definition text.
    fake_llm_client._canned_formulas = {
        "net income": FormulaFragment(formula="Net Income"),
        "Parent definition": FormulaFragment(formula="Net Income + addbacks"),
    }
    result = CovenantExtractionResult(
        financial_covenants=[
            FinancialCovenant(
                name="Consolidated EBITDA",
                comparison=Comparison.NOT_TO_EXCEED,
                threshold=1.0,
                unit="USD",
                covenant_type=CovenantType.MAINTENANCE,
                citation=Citation(source_text="Consolidated EBITDA"),
            )
        ]
    )
    definitions_text = (
        '"Consolidated EBITDA" means Consolidated Net Income plus addbacks. '
        '"Consolidated Net Income" means the net income of the Borrower.'
    )

    export = to_export_dict(result, full_text=definitions_text, definitions_text=definitions_text, llm_client=fake_llm_client)

    covenant = export["financial_covenants"][0]
    assert covenant["definition"] == "Consolidated Net Income plus addbacks."
    assert "resolved_definition" in covenant
    resolved = covenant["resolved_definition"]
    assert resolved["term"] == "Consolidated EBITDA"
    assert resolved["formula"] == "Net Income + addbacks"
    assert "Consolidated Net Income" in resolved["components"]
    assert resolved["components"]["Consolidated Net Income"]["formula"] == "Net Income"


_LOAN_TERMS_SAMPLE_HTML = (
    "<html><body>"
    "<p>This Loan Agreement is dated as of January 1, 2024, between Acme Corp as Borrower "
    "and Test Bank as Lender.</p>"
    "<p>Loans hereunder shall be repaid in full on the Maturity Date.</p>"
    '<p>"Maturity Date" means October 16, 2030.</p>'
    "</body></html>"
)


def test_run_pipeline_resolves_loan_terms_end_date_from_bare_term_name(
    tmp_path: Path, fake_embedder, fake_llm_client
):
    # The extractor returned the defined term's *name* as end_date's value
    # (the recurring loan_terms failure mode -- see extraction/loan_terms.py)
    # instead of resolving it; to_export_dict-style definitions-chaining
    # should catch that and resolve it to the actual date.
    html_path = tmp_path / "contract.html"
    html_path.write_text(_LOAN_TERMS_SAMPLE_HTML, encoding="utf-8")

    fake_llm_client._canned_loan_terms = {
        "repaid in full on the Maturity Date": LoanTerms(
            end_date=CitedText(
                value="Maturity Date", citation=Citation(source_text="repaid in full on the Maturity Date")
            )
        )
    }
    # A marker unique to the leaf-resolution prompt (build_leaf_user_prompt)
    # -- "October 16, 2030" alone would also match the loan-terms-extraction
    # prompt, since that same document text is a preamble candidate too.
    fake_llm_client._canned_formulas = {
        "Convert this definition into a formula": FormulaFragment(formula="October 16, 2030")
    }

    result = run_pipeline(html_path, embedder=fake_embedder, llm_client=fake_llm_client)

    end_date = result["loan_terms"]["end_date"]
    assert end_date["value"] == "Maturity Date"  # original extraction left untouched
    assert "resolved_value" in end_date
    assert end_date["resolved_value"]["term"] == "Maturity Date"
    assert end_date["resolved_value"]["formula"] == "October 16, 2030"


def test_run_pipeline_leaves_loan_terms_end_date_alone_when_not_a_defined_term(
    tmp_path: Path, fake_embedder, fake_llm_client
):
    # end_date is already a real, resolved-looking value (not a bare defined
    # term name) -- no defined term matches it, so no resolution is
    # attempted and no `resolved_value` key is added.
    html_path = tmp_path / "contract.html"
    html_path.write_text(_LOAN_TERMS_SAMPLE_HTML, encoding="utf-8")

    fake_llm_client._canned_loan_terms = {
        "repaid in full on the Maturity Date": LoanTerms(
            end_date=CitedText(
                value="October 16, 2030", citation=Citation(source_text="repaid in full on the Maturity Date")
            )
        )
    }

    result = run_pipeline(html_path, embedder=fake_embedder, llm_client=fake_llm_client)

    end_date = result["loan_terms"]["end_date"]
    assert end_date["value"] == "October 16, 2030"
    assert "resolved_value" not in end_date


def test_run_pipeline_skips_resolution_when_value_already_has_a_digit(
    tmp_path: Path, fake_embedder, fake_llm_client
):
    # interest_rate is already a complete, concrete answer that happens to
    # mention a defined term in passing ("...after the Maturity Date has
    # been extended...") -- resolving that incidental mention would replace
    # a correct value with an unrelated one, so it must be skipped outright
    # once the value contains any digit.
    html_path = tmp_path / "contract.html"
    html_path.write_text(_LOAN_TERMS_SAMPLE_HTML, encoding="utf-8")

    fake_llm_client._canned_loan_terms = {
        "repaid in full on the Maturity Date": LoanTerms(
            interest_rate=CitedText(
                value="12.0% per annum; provided that after the Maturity Date has been extended, 17.5%",
                citation=Citation(source_text="repaid in full on the Maturity Date"),
            )
        )
    }
    fake_llm_client._canned_formulas = {
        "Convert this definition into a formula": FormulaFragment(formula="October 16, 2030")
    }

    result = run_pipeline(html_path, embedder=fake_embedder, llm_client=fake_llm_client)

    interest_rate = result["loan_terms"]["interest_rate"]
    assert interest_rate["value"] == "12.0% per annum; provided that after the Maturity Date has been extended, 17.5%"
    assert "resolved_value" not in interest_rate


def test_run_pipeline_does_not_attempt_resolution_for_start_date_or_loan_type(
    tmp_path: Path, fake_embedder, fake_llm_client
):
    # Definitions-chaining is only wired in for end_date/loan_amount/
    # interest_rate -- start_date (and other fields) keep their value as-is
    # even if it happens to look like a defined term name.
    html_path = tmp_path / "contract.html"
    html_path.write_text(_LOAN_TERMS_SAMPLE_HTML, encoding="utf-8")

    fake_llm_client._canned_loan_terms = {
        "repaid in full on the Maturity Date": LoanTerms(
            start_date=CitedText(
                value="Maturity Date", citation=Citation(source_text="repaid in full on the Maturity Date")
            )
        )
    }
    # A marker unique to the leaf-resolution prompt (build_leaf_user_prompt)
    # -- "October 16, 2030" alone would also match the loan-terms-extraction
    # prompt, since that same document text is a preamble candidate too.
    fake_llm_client._canned_formulas = {
        "Convert this definition into a formula": FormulaFragment(formula="October 16, 2030")
    }

    result = run_pipeline(html_path, embedder=fake_embedder, llm_client=fake_llm_client)

    start_date = result["loan_terms"]["start_date"]
    assert start_date["value"] == "Maturity Date"
    assert "resolved_value" not in start_date


def test_run_pipeline_rejects_unsupported_file_type(tmp_path: Path, fake_embedder, fake_llm_client):
    bad_file = tmp_path / "contract.txt"
    bad_file.write_text("not a supported type", encoding="utf-8")

    import pytest

    with pytest.raises(ValueError):
        run_pipeline(bad_file, embedder=fake_embedder, llm_client=fake_llm_client)
