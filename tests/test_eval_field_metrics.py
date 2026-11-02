"""Tests for evals/field_metrics.py, schedule/optional matching, and the
eval CLI's scoring + report rendering (all offline)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from covenant_extraction.evals import cli
from covenant_extraction.evals.field_metrics import (
    aggregate_fields,
    names_match,
    normalize_frequency,
    normalize_unit,
    score_fields,
    thresholds_close,
)
from covenant_extraction.evals.matching import match_covenants, normalize_text
from covenant_extraction.evals.report import render_markdown
from covenant_extraction.evals.scoring import score_matches

GOLD_DIR = Path(__file__).resolve().parent.parent / "evals" / "gold"


def _ext(name, threshold, source_text, comparison="not_to_exceed", covenant_type="maintenance", unit="ratio", freq="quarterly"):
    return {
        "name": name,
        "threshold": threshold,
        "comparison": comparison,
        "covenant_type": covenant_type,
        "unit": unit,
        "test_frequency": freq,
        "citation": {"source_text": source_text},
    }


STEP_GOLD = {
    "name": "Total Net Leverage Ratio",
    "comparison": "not_to_exceed",
    "threshold": 4.75,
    "unit": "ratio",
    "covenant_type": "maintenance",
    "test_frequency": "quarterly",
    "citation_contains": "Total Net Leverage Ratio as of the end of each fiscal quarter",
    "schedule": [{"period": "Q4 2025", "threshold": 4.75}, {"period": "Q1 2026+", "threshold": 4.50}],
}


# --- numeric helpers -----------------------------------------------------------


def test_thresholds_close_uses_relative_tolerance():
    assert thresholds_close(3.5, 3.50)
    assert thresholds_close(10_000_000, 10_050_000)  # 0.5% off
    assert not thresholds_close(10_000_000, 10_200_000)  # 2% off
    assert not thresholds_close(-4_750_000, 4_750_000)  # sign flip is an error
    assert not thresholds_close(None, 3.5)


def test_normalizers():
    assert normalize_unit("$") == normalize_unit("USD") == "USD"
    assert normalize_unit("Ratio") == "ratio"
    assert normalize_frequency("as of the last day of each fiscal quarter") == "quarterly"
    assert normalize_frequency("at all times") == "continuous"
    assert normalize_frequency("for any fiscal year") == "annual"
    assert normalize_frequency("pro forma") is None


def test_names_match_ignores_filler_but_not_substance():
    assert names_match("Consolidated Leverage Ratio", "Leverage Ratio")
    assert not names_match("Interest Coverage Ratio", "Fixed Charge Coverage Ratio")
    assert names_match("Capex Limit", "Capital Expenditures", aliases=["Capex Limit"])


def test_normalize_text_drops_html_punctuation_spacing():
    assert normalize_text("Ratio . The  Borrower ( x )") == normalize_text("ratio. the borrower (x)")


# --- schedules -----------------------------------------------------------------


def test_schedule_rows_are_absorbed_not_false_positives():
    lead = "The Borrower shall not permit the Total Net Leverage Ratio as of the end of each fiscal quarter to exceed"
    extracted = [_ext("Total Net Leverage Ratio", 4.75, lead + " (i) 4.75"), _ext("Total Net Leverage Ratio", 4.5, lead + " (ii) 4.50")]

    matches, unmatched = match_covenants([STEP_GOLD], extracted)
    fields = score_fields(matches[0])

    assert unmatched == []
    assert len(matches[0].all_extracted) == 2
    assert fields.threshold_exact
    assert fields.schedule_levels == 2 and fields.schedule_levels_found == 2
    assert fields.schedule_complete is True


def test_partial_schedule_is_flagged():
    extracted = [_ext("Total Net Leverage Ratio", 4.75, "Total Net Leverage Ratio as of the end of each fiscal quarter ... 4.75")]
    matches, _ = match_covenants([STEP_GOLD], extracted)
    fields = score_fields(matches[0])
    assert fields.schedule_levels_found == 1
    assert fields.schedule_complete is False


# --- optional gold --------------------------------------------------------------


def test_optional_gold_counts_neither_way():
    gold = [
        {"name": "A", "threshold": 3.0, "comparison": "not_to_exceed", "covenant_type": "maintenance", "citation_contains": "anchor-a"},
        {"name": "B", "threshold": 1.1, "comparison": "not_less_than", "covenant_type": "incurrence_condition",
         "citation_contains": "anchor-b", "required": False},
        {"name": "C", "threshold": 9.0, "comparison": "not_less_than", "covenant_type": "incurrence_condition",
         "citation_contains": "anchor-c", "required": False},
    ]
    extracted = [_ext("A", 3.0, "x anchor-a y"), _ext("B", 1.1, "x anchor-b y")]

    score = score_matches(*match_covenants(gold, extracted))

    assert score.num_gold == 1  # only A is required
    assert score.recall == 1.0  # missing optional C is not a miss
    assert score.num_false_positives == 0  # extracting optional B is not a false positive
    assert score.precision == 1.0


def test_anchor_list_matches_any_alternative():
    gold = [{"name": "A", "threshold": 1.0, "comparison": "not_to_exceed", "covenant_type": "maintenance",
             "citation_contains": ["first phrase", "second phrase"]}]
    matches, _ = match_covenants(gold, [_ext("A", 1.0, "only the second phrase is here")])
    assert matches[0].extracted is not None


# --- field scoring --------------------------------------------------------------


def test_field_scoring_flags_each_wrong_field():
    gold = [{"name": "Minimum Liquidity", "threshold": 15_000_000, "comparison": "not_less_than", "unit": "USD",
             "covenant_type": "maintenance", "test_frequency": "continuous", "citation_contains": "liquidity"}]
    extracted = [_ext("Minimum Liquidity", 15_000_000, "Liquidity shall not be less than $15,000,000",
                      comparison="not_to_exceed", unit="ratio", freq="quarterly")]
    f = score_fields(match_covenants(gold, extracted)[0][0])
    assert f.threshold_exact and f.name and f.covenant_type_correct
    assert not f.comparison and not f.unit and f.test_frequency is False
    assert not f.all_fields

    agg = aggregate_fields([f])
    assert agg.accuracy("threshold_exact") == 1.0
    assert agg.accuracy("comparison") == 0.0


def test_frequency_not_scored_for_incurrence_conditions():
    gold = [{"name": "Leverage", "threshold": 2.5, "comparison": "not_to_exceed", "unit": "ratio",
             "covenant_type": "incurrence_condition", "citation_contains": "2.50"}]
    f = score_fields(match_covenants(gold, [_ext("Leverage", 2.5, "exceed 2.50 to 1.00", covenant_type="incurrence_condition", freq=None)])[0][0])
    assert f.test_frequency is None


# --- CLI scoring + report ---------------------------------------------------------


def test_run_eval_and_render_markdown_end_to_end():
    gold = {
        "_id": "doc1",
        "document": "data/sample/x.html",
        "financial_covenants": [STEP_GOLD],
    }
    empty_gold = {"_id": "doc2", "document": "data/sample/y.html", "financial_covenants": []}
    extracted = {
        "doc1": [_ext("Total Net Leverage Ratio", 4.75, "Total Net Leverage Ratio as of the end of each fiscal quarter ... 4.75")],
        "doc2": [_ext("Made Up Ratio", 9.9, "nothing")],
    }

    report = cli.run_eval([gold, empty_gold], lambda g: extracted[g["_id"]], rel_tol=0.01)
    report["meta"] = {"label": "t", "source": "test", "date": "2026-01-01", "gold_files": 2}
    s = report["summary"]

    assert s["true_positives"] == 1 and s["false_positives"] == 1 and s["false_negatives"] == 0
    assert s["schedule_levels"] == 2 and s["schedule_levels_found"] == 1
    assert s["no_covenant_documents"] == 1 and s["no_covenant_documents_clean"] == 0

    md = render_markdown(report)
    assert "Incomplete schedule" in md
    assert "False positive" in md and "Made Up Ratio" in md


def test_run_eval_skips_documents_without_extractions():
    gold = {"_id": "doc1", "document": "d", "financial_covenants": []}
    report = cli.run_eval([gold], lambda g: None, rel_tol=0.01)
    assert report["summary"]["documents_skipped"] == ["doc1"]


# --- the gold set itself -----------------------------------------------------------


@pytest.mark.parametrize("path", sorted(GOLD_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_gold_files_are_well_formed(path):
    gold = json.loads(path.read_text(encoding="utf-8"))
    assert (GOLD_DIR.parent.parent / gold["document"]).exists()
    for cov in gold["financial_covenants"]:
        assert cov["comparison"] in {"not_to_exceed", "not_less_than"}
        assert cov["covenant_type"] in {"maintenance", "incurrence_condition"}
        assert cov["citation_contains"]
        if cov.get("required", True):
            assert cov["threshold"] is not None
        if cov.get("schedule"):
            assert cov["schedule"][0]["threshold"] == cov["threshold"]


def test_gold_set_size():
    assert 10 <= len(list(GOLD_DIR.glob("*.json"))) <= 20
