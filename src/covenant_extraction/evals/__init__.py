"""Evaluation harness: scores extraction output against a small,
hand-annotated gold set (see evals/gold/*.json at the project root), so
extraction quality can be measured against ground truth -- not just the
pipeline's own self-consistency checks (citation_verified, calculator_match,
citation_grounded), which only validate what WAS extracted, never what was
missed or fabricated.
"""
from __future__ import annotations
