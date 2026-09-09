"""Tests for evals/matching.py and evals/scoring.py."""
from __future__ import annotations

from covenant_extraction.evals.matching import match_covenants
from covenant_extraction.evals.scoring import score_matches


def _gold(name, threshold, comparison, covenant_type, citation_contains):
    return {
        "name": name,
        "threshold": threshold,
        "comparison": comparison,
        "covenant_type": covenant_type,
        "citation_contains": citation_contains,
    }


def _extracted(name, threshold, comparison, covenant_type, source_text):
    return {
        "name": name,
        "threshold": threshold,
        "comparison": comparison,
        "covenant_type": covenant_type,
        "citation": {"source_text": source_text},
    }


# --- match_covenants ---------------------------------------------------------


def test_match_covenants_matches_on_citation_substring_not_name():
    gold = [_gold("Leverage Ratio", 3.5, "not_to_exceed", "maintenance", "shall not exceed 3.50:1.00")]
    extracted = [
        _extracted(
            "Consolidated Leverage Ratio",  # different name -- shouldn't matter
            3.5,
            "not_to_exceed",
            "maintenance",
            "The Borrower shall not exceed 3.50:1.00 as of quarter end.",
        )
    ]

    matches, unmatched = match_covenants(gold, extracted)

    assert len(matches) == 1
    assert matches[0].extracted is not None
    assert matches[0].threshold_correct is True
    assert unmatched == []


def test_match_covenants_reports_miss_when_no_citation_overlap():
    gold = [_gold("Leverage Ratio", 3.5, "not_to_exceed", "maintenance", "a phrase that never appears")]
    extracted = [_extracted("Leverage Ratio", 3.5, "not_to_exceed", "maintenance", "totally unrelated citation text")]

    matches, unmatched = match_covenants(gold, extracted)

    assert matches[0].extracted is None
    assert len(unmatched) == 1  # the extracted covenant matched nothing


def test_match_covenants_flags_wrong_threshold_on_an_otherwise_matched_citation():
    gold = [_gold("Leverage Ratio", 3.5, "not_to_exceed", "maintenance", "shall not exceed")]
    extracted = [_extracted("Leverage Ratio", 4.0, "not_to_exceed", "maintenance", "the ratio shall not exceed 4.00:1.00")]

    matches, unmatched = match_covenants(gold, extracted)

    assert matches[0].extracted is not None  # citation matched
    assert matches[0].threshold_correct is False  # but the number is wrong
    assert unmatched == []


def test_match_covenants_flags_wrong_covenant_type():
    gold = [_gold("Leverage Ratio", 3.5, "not_to_exceed", "maintenance", "shall not exceed")]
    extracted = [_extracted("Leverage Ratio", 3.5, "not_to_exceed", "incurrence_condition", "the ratio shall not exceed")]

    matches, unmatched = match_covenants(gold, extracted)

    assert matches[0].threshold_correct is True
    assert matches[0].covenant_type_correct is False


def test_match_covenants_extra_extraction_is_unmatched_false_positive():
    gold = [_gold("Leverage Ratio", 3.5, "not_to_exceed", "maintenance", "shall not exceed 3.50")]
    extracted = [
        _extracted("Leverage Ratio", 3.5, "not_to_exceed", "maintenance", "the ratio shall not exceed 3.50:1.00"),
        _extracted("Fabricated Ratio", 9.9, "not_to_exceed", "incurrence_condition", "some other clause entirely"),
    ]

    matches, unmatched = match_covenants(gold, extracted)

    assert len(matches) == 1
    assert matches[0].extracted is not None
    assert len(unmatched) == 1
    assert unmatched[0]["name"] == "Fabricated Ratio"


def test_match_covenants_does_not_double_match_the_same_extraction():
    # Two gold entries with overlapping anchor text shouldn't both claim
    # the same single extracted covenant.
    gold = [
        _gold("A", 3.5, "not_to_exceed", "maintenance", "shall not exceed 3.50"),
        _gold("B", 3.5, "not_to_exceed", "maintenance", "shall not exceed 3.50"),
    ]
    extracted = [_extracted("A", 3.5, "not_to_exceed", "maintenance", "the ratio shall not exceed 3.50:1.00")]

    matches, unmatched = match_covenants(gold, extracted)

    matched_count = sum(1 for m in matches if m.extracted is not None)
    assert matched_count == 1  # only one gold entry can claim the single extraction
    assert unmatched == []


# --- score_matches -------------------------------------------------------------


def test_score_matches_perfect_extraction():
    gold = [_gold("A", 3.5, "not_to_exceed", "maintenance", "anchor")]
    extracted = [_extracted("A", 3.5, "not_to_exceed", "maintenance", "text with anchor in it")]
    matches, unmatched = match_covenants(gold, extracted)

    score = score_matches(matches, unmatched)

    assert score.recall == 1.0
    assert score.precision == 1.0
    assert score.threshold_accuracy == 1.0
    assert score.covenant_type_accuracy == 1.0
    assert score.num_misses == 0
    assert score.num_false_positives == 0


def test_score_matches_penalizes_missed_covenant():
    gold = [
        _gold("A", 3.5, "not_to_exceed", "maintenance", "anchor-a"),
        _gold("B", 4.0, "not_to_exceed", "maintenance", "anchor-b"),
    ]
    extracted = [_extracted("A", 3.5, "not_to_exceed", "maintenance", "text with anchor-a in it")]
    matches, unmatched = match_covenants(gold, extracted)

    score = score_matches(matches, unmatched)

    assert score.num_gold == 2
    assert score.num_hits == 1
    assert score.num_misses == 1
    assert score.recall == 0.5
    assert score.precision == 1.0  # what WAS extracted is real


def test_score_matches_penalizes_fabricated_extraction():
    gold = [_gold("A", 3.5, "not_to_exceed", "maintenance", "anchor-a")]
    extracted = [
        _extracted("A", 3.5, "not_to_exceed", "maintenance", "text with anchor-a in it"),
        _extracted("Fabricated", 9.9, "not_to_exceed", "incurrence_condition", "made up clause"),
    ]
    matches, unmatched = match_covenants(gold, extracted)

    score = score_matches(matches, unmatched)

    assert score.recall == 1.0  # the real one was found
    assert score.precision == 0.5  # but half of what we extracted is fake
    assert score.num_false_positives == 1


def test_score_matches_empty_gold_and_empty_extraction_is_trivially_perfect():
    matches, unmatched = match_covenants([], [])
    score = score_matches(matches, unmatched)

    assert score.recall == 1.0
    assert score.precision == 1.0
    assert score.num_gold == 0


# --- anchor robustness across models with different quoting boundaries ------
#
# Real finding from a cross-model eval run (Mizuho-QVC gold set): two models
# extracted the SAME real covenant correctly, but one quoted a longer span
# (lead-in text through the number) and the other quoted a shorter span
# (starting right at the number). A `citation_contains` anchor built only
# from the first model's lead-in text spuriously missed the second model's
# equally-correct, more tightly-scoped citation, AND simultaneously flagged
# its citation as an unmatched false positive -- one real covenant, double
# counted as an error. The fix: anchor on the fact being verified (the
# number-bearing phrase itself), not on incidental quoting-boundary choices.


def test_match_covenants_anchor_on_leadin_text_misses_a_narrower_correct_citation():
    # Demonstrates the bug: an anchor built from one model's full quote
    # (lead-in text) fails to match a second, equally-correct model whose
    # citation starts closer to the actual number.
    gold = [
        _gold(
            "Consolidated Leverage Ratio", 4.50, "not_to_exceed", "incurrence_condition",
            citation_contains="in the case of any Incremental Facility or Pari Passu Indebtedness that is unsecured",
        )
    ]
    # A real, correct citation from a model that quoted more narrowly.
    extracted = [
        _extracted(
            "Consolidated Leverage Ratio", 4.50, "not_to_exceed", "incurrence_condition",
            source_text="the Consolidated Leverage Ratio does not exceed either (x) 4.50:1.00",
        )
    ]

    matches, unmatched = match_covenants(gold, extracted)

    # This is the bug being documented, not the desired behavior: a correct
    # extraction gets counted as both a miss AND a false positive.
    assert matches[0].extracted is None
    assert len(unmatched) == 1


def test_match_covenants_anchor_on_the_number_bearing_phrase_matches_either_quoting_style():
    # The fix: anchor on the phrase that states the actual fact (the
    # threshold), which any correct citation -- narrow or broad -- must
    # include, rather than on lead-in text that's an artifact of one
    # model's specific quoting boundary.
    gold = [
        _gold(
            "Consolidated Leverage Ratio", 4.50, "not_to_exceed", "incurrence_condition",
            citation_contains="does not exceed either (x) 4.50:1.00",
        )
    ]
    broad_quote = _extracted(
        "Consolidated Leverage Ratio", 4.50, "not_to_exceed", "incurrence_condition",
        source_text=(
            "in the case of any Incremental Facility or Pari Passu Indebtedness that is unsecured, "
            "the Consolidated Leverage Ratio does not exceed either (x) 4.50:1.00"
        ),
    )
    narrow_quote = _extracted(
        "Consolidated Leverage Ratio", 4.50, "not_to_exceed", "incurrence_condition",
        source_text="the Consolidated Leverage Ratio does not exceed either (x) 4.50:1.00",
    )

    for candidate in (broad_quote, narrow_quote):
        matches, unmatched = match_covenants(gold, [candidate])
        assert matches[0].extracted is not None
        assert unmatched == []
