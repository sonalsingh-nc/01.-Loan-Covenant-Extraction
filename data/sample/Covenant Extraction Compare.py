#!/usr/bin/env python3
"""
covenant_extraction_compare.py

Compares two covenant-extraction JSON outputs (conforming to the
financial_covenants / ebitda_addbacks / negative_covenants schema) and
produces a field-level agreement report.

Typical use: run the same extraction prompt through two different
models (e.g. Claude vs. a local Qwen2.5-32B pipeline) against the same
source document, then diff the two JSON outputs to see where they
agree, where they diverge, and whether either hallucinated a citation.

Usage:
    python covenant_extraction_compare.py output_a.json output_b.json \
        [--source-doc agreement.txt] [--name-threshold 0.6] [--out report.json]

Notes on design choices:
  - Items are matched between the two outputs by name similarity, not by
    list position -- extraction order is not meaningful and a naive
    zip() comparison would misalign items whenever one model finds an
    extra covenant or orders things differently.
  - Basket/exception strings ("greater of $50,000,000 or 2.1% of
    Consolidated EBITDA") are parsed into structured tuples before
    comparison. String-equality on these fields is close to useless --
    the same basket gets phrased a dozen ways -- so structural
    comparison is what actually tells you if the two extractions agree.
  - Citation/provenance checking is optional and only runs if you pass
    --source-doc: it flags any source_text that does NOT appear
    verbatim in the source document, which is the sharpest signal for
    hallucination in this kind of task.
  - This script does not have a notion of "ground truth" -- it compares
    A against B. Feed it a human-annotated gold file as one of the two
    inputs to turn this into an accuracy check rather than an agreement
    check.
"""

import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_extraction(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("financial_covenants", "ebitda_addbacks", "negative_covenants"):
        data.setdefault(key, [])
    return data


# ---------------------------------------------------------------------
# Name matching (bipartite greedy match by string similarity)
# ---------------------------------------------------------------------

def _similarity(a, b):
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def match_items(list_a, list_b, name_key="name", threshold=0.6):
    """
    Greedy best-first matching between two lists of dicts, using
    name_key for similarity. Returns (matched_pairs, unmatched_a, unmatched_b).
    Greedy-by-best-score avoids the common failure mode of matching two
    items just because they happen to appear at the same list index.
    """
    candidates = []
    for i, ia in enumerate(list_a):
        for j, ib in enumerate(list_b):
            score = _similarity(ia.get(name_key, ""), ib.get(name_key, ""))
            if score >= threshold:
                candidates.append((score, i, j))
    candidates.sort(reverse=True)

    used_a, used_b = set(), set()
    pairs = []
    for score, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((list_a[i], list_b[j], score))

    unmatched_a = [x for i, x in enumerate(list_a) if i not in used_a]
    unmatched_b = [x for i, x in enumerate(list_b) if i not in used_b]
    return pairs, unmatched_a, unmatched_b


# ---------------------------------------------------------------------
# Basket / formula normalization
# ---------------------------------------------------------------------

_DOLLAR_RE = re.compile(r"\$([\d,]+(?:\.\d+)?)\s*(million|mm|thousand|k)?", re.I)
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_BASE_RE = re.compile(
    r"%\s*of\s+([A-Za-z][A-Za-z0-9 \-\u2019']*?)(?:\s+for|\s*[\.,;)]|$)", re.I
)

_MULTIPLIERS = {"million": 1_000_000, "mm": 1_000_000, "thousand": 1_000, "k": 1_000}


def parse_basket(text):
    """
    Parse a basket/exception string into a structured, comparable form:
      {
        "fixed_amount": float or None,   # dollar figure, normalized to units
        "percent": float or None,        # percentage figure
        "percent_base": str or None,     # what the percentage is "of"
        "is_grower": bool,               # "greater of $X or Y% of Z" pattern
        "raw": original text
      }
    This is intentionally permissive -- the goal is a rough structural
    fingerprint for comparison, not a legal-grade parser.
    """
    fixed_amount = None
    m = _DOLLAR_RE.search(text)
    if m:
        amount = float(m.group(1).replace(",", ""))
        mult = _MULTIPLIERS.get((m.group(2) or "").lower(), 1)
        fixed_amount = amount * mult

    percent = None
    pm = _PERCENT_RE.search(text)
    if pm:
        percent = float(pm.group(1))

    percent_base = None
    bm = _BASE_RE.search(text)
    if bm:
        percent_base = bm.group(1).strip().rstrip(".,;")

    is_grower = bool(
        re.search(r"greater of", text, re.I) and fixed_amount is not None and percent is not None
    )

    return {
        "fixed_amount": fixed_amount,
        "percent": percent,
        "percent_base": percent_base,
        "is_grower": is_grower,
        "raw": text,
    }


def baskets_equivalent(a, b, dollar_tol=0.01, percent_tol=0.01):
    """
    Two parsed baskets are considered equivalent if their structural
    fields match within tolerance. None == None counts as a match
    (both extractions agree the field is absent).
    """
    def close(x, y, tol):
        if x is None and y is None:
            return True
        if x is None or y is None:
            return False
        if x == 0:
            return y == 0
        return abs(x - y) / abs(x) <= tol

    fixed_ok = close(a["fixed_amount"], b["fixed_amount"], dollar_tol)
    percent_ok = close(a["percent"], b["percent"], percent_tol)
    base_ok = (
        a["percent_base"] is None
        or b["percent_base"] is None
        or _similarity(a["percent_base"], b["percent_base"]) >= 0.6
    )
    return fixed_ok and percent_ok and base_ok


def compare_exception_lists(exceptions_a, exceptions_b):
    """
    Set-style comparison of two exception/basket lists after normalization.
    Returns matched pairs, and the leftovers from each side.
    """
    parsed_a = [parse_basket(x) for x in exceptions_a]
    parsed_b = [parse_basket(x) for x in exceptions_b]

    used_b = set()
    matched = []
    for pa in parsed_a:
        for j, pb in enumerate(parsed_b):
            if j in used_b:
                continue
            if baskets_equivalent(pa, pb):
                matched.append((pa["raw"], pb["raw"]))
                used_b.add(j)
                break

    matched_raw_a = {m[0] for m in matched}
    only_a = [p["raw"] for p in parsed_a if p["raw"] not in matched_raw_a]
    only_b = [pb["raw"] for j, pb in enumerate(parsed_b) if j not in used_b]

    return matched, only_a, only_b


# ---------------------------------------------------------------------
# Category comparisons
# ---------------------------------------------------------------------

def compare_financial_covenants(list_a, list_b, threshold, numeric_tol=0.01):
    pairs, only_a, only_b = match_items(list_a, list_b, threshold=threshold)
    field_agreement = {"comparison": 0, "threshold": 0, "unit": 0, "test_frequency": 0}
    details = []
    for a, b, score in pairs:
        row = {"name_a": a.get("name"), "name_b": b.get("name"), "name_similarity": round(score, 3)}
        cmp_ok = a.get("comparison") == b.get("comparison")
        unit_ok = (a.get("unit") or "").strip().lower() == (b.get("unit") or "").strip().lower()
        freq_ok = (a.get("test_frequency") or None) == (b.get("test_frequency") or None)
        ta, tb = a.get("threshold"), b.get("threshold")
        thr_ok = (
            ta is not None and tb is not None and ta != 0
            and abs(ta - tb) / abs(ta) <= numeric_tol
        ) or (ta == tb)

        field_agreement["comparison"] += cmp_ok
        field_agreement["unit"] += unit_ok
        field_agreement["test_frequency"] += freq_ok
        field_agreement["threshold"] += thr_ok

        row.update(
            comparison_match=cmp_ok, threshold_match=thr_ok,
            unit_match=unit_ok, test_frequency_match=freq_ok,
            threshold_a=ta, threshold_b=tb,
        )
        details.append(row)

    n = max(len(pairs), 1)
    return {
        "matched_count": len(pairs),
        "only_in_a": [x.get("name") for x in only_a],
        "only_in_b": [x.get("name") for x in only_b],
        "field_agreement_rate": {k: round(v / n, 3) for k, v in field_agreement.items()},
        "details": details,
    }


def compare_ebitda_addbacks(list_a, list_b, threshold):
    pairs, only_a, only_b = match_items(list_a, list_b, threshold=threshold)
    total = len(list_a) + len(list_b)
    denom = total - len(pairs) if total - len(pairs) > 0 else 1
    jaccard = len(pairs) / (len(pairs) + len(only_a) + len(only_b)) if (
        len(pairs) + len(only_a) + len(only_b) > 0
    ) else 1.0
    return {
        "matched_count": len(pairs),
        "only_in_a": [x.get("name") for x in only_a],
        "only_in_b": [x.get("name") for x in only_b],
        "jaccard_overlap": round(jaccard, 3),
        "matches": [
            {"name_a": a.get("name"), "name_b": b.get("name"), "similarity": round(s, 3)}
            for a, b, s in pairs
        ],
    }


def compare_negative_covenants(list_a, list_b, threshold):
    pairs, only_a, only_b = match_items(list_a, list_b, threshold=threshold)
    details = []
    for a, b, score in pairs:
        matched_exc, only_exc_a, only_exc_b = compare_exception_lists(
            a.get("exceptions", []), b.get("exceptions", [])
        )
        summary_sim = _similarity(a.get("summary", ""), b.get("summary", ""))
        n_exc = max(len(a.get("exceptions", [])), len(b.get("exceptions", [])), 1)
        details.append({
            "name_a": a.get("name"),
            "name_b": b.get("name"),
            "name_similarity": round(score, 3),
            "summary_similarity": round(summary_sim, 3),
            "exception_match_rate": round(len(matched_exc) / n_exc, 3),
            "exceptions_only_in_a": only_exc_a,
            "exceptions_only_in_b": only_exc_b,
        })
    return {
        "matched_count": len(pairs),
        "only_in_a": [x.get("name") for x in only_a],
        "only_in_b": [x.get("name") for x in only_b],
        "details": details,
    }


# ---------------------------------------------------------------------
# Provenance / hallucination check
# ---------------------------------------------------------------------

def check_provenance(data, source_text, label):
    """
    Flags any citation.source_text that is NOT found verbatim in the
    supplied source document -- the sharpest available signal for a
    hallucinated citation in this pipeline.
    """
    flat_source = " ".join(source_text.split())
    issues = []
    for category in ("financial_covenants", "ebitda_addbacks", "negative_covenants"):
        for item in data.get(category, []):
            quote = (item.get("citation") or {}).get("source_text", "")
            flat_quote = " ".join(quote.split())
            if flat_quote and flat_quote not in flat_source:
                issues.append({
                    "category": category,
                    "item_name": item.get("name"),
                    "source_text": quote,
                })
    return {"model": label, "unverified_citation_count": len(issues), "unverified_citations": issues}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file_a", help="First extraction JSON (e.g. Claude output)")
    ap.add_argument("file_b", help="Second extraction JSON (e.g. local Qwen2.5-32B output)")
    ap.add_argument("--source-doc", help="Path to the original source agreement text, for citation verification")
    ap.add_argument("--name-threshold", type=float, default=0.6, help="Min name similarity to consider two items a match (0-1)")
    ap.add_argument("--out", help="Write full JSON report to this path (in addition to printing summary)")
    args = ap.parse_args()

    data_a = load_extraction(args.file_a)
    data_b = load_extraction(args.file_b)

    report = {
        "inputs": {"a": args.file_a, "b": args.file_b},
        "financial_covenants": compare_financial_covenants(
            data_a["financial_covenants"], data_b["financial_covenants"], args.name_threshold
        ),
        "ebitda_addbacks": compare_ebitda_addbacks(
            data_a["ebitda_addbacks"], data_b["ebitda_addbacks"], args.name_threshold
        ),
        "negative_covenants": compare_negative_covenants(
            data_a["negative_covenants"], data_b["negative_covenants"], args.name_threshold
        ),
    }

    if args.source_doc:
        with open(args.source_doc, "r", encoding="utf-8") as f:
            source_text = f.read()
        report["provenance_check"] = {
            "a": check_provenance(data_a, source_text, args.file_a),
            "b": check_provenance(data_b, source_text, args.file_b),
        }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    # ---- console summary ----
    fc = report["financial_covenants"]
    ea = report["ebitda_addbacks"]
    nc = report["negative_covenants"]

    print("=== Financial Covenants ===")
    print(f"Matched: {fc['matched_count']}  Only in A: {fc['only_in_a']}  Only in B: {fc['only_in_b']}")
    print(f"Field agreement rate (on matched items): {fc['field_agreement_rate']}")

    print("\n=== EBITDA Addbacks ===")
    print(f"Matched: {ea['matched_count']}  Jaccard overlap: {ea['jaccard_overlap']}")
    if ea["only_in_a"]:
        print(f"Only in A: {ea['only_in_a']}")
    if ea["only_in_b"]:
        print(f"Only in B: {ea['only_in_b']}")

    print("\n=== Negative Covenants ===")
    print(f"Matched: {nc['matched_count']}  Only in A: {nc['only_in_a']}  Only in B: {nc['only_in_b']}")
    for d in nc["details"]:
        print(
            f"  - {d['name_a']} <-> {d['name_b']}: "
            f"summary_sim={d['summary_similarity']}, exception_match_rate={d['exception_match_rate']}"
        )
        if d["exceptions_only_in_a"]:
            print(f"      only in A: {d['exceptions_only_in_a']}")
        if d["exceptions_only_in_b"]:
            print(f"      only in B: {d['exceptions_only_in_b']}")

    if "provenance_check" in report:
        print("\n=== Provenance Check ===")
        for side in ("a", "b"):
            pc = report["provenance_check"][side]
            print(f"{pc['model']}: {pc['unverified_citation_count']} unverified citation(s)")
            for issue in pc["unverified_citations"]:
                print(f"    [{issue['category']}] {issue['item_name']}: {issue['source_text'][:80]}...")

    if args.out:
        print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()