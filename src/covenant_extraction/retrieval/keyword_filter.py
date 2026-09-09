"""Cheap, fast keyword/regex flags used to pre-filter candidate clauses
before (or alongside) embedding-based ranking.

Scope is intentionally limited to financial-metric/ratio covenant language
(leverage ratio, coverage ratios, minimum liquidity, etc.) — negative
covenants (Indebtedness, Liens, Restricted Payments) and generic EBITDA
add-back language are out of scope for candidate selection.
"""
from __future__ import annotations

import re
from typing import List, Pattern

COVENANT_KEYWORD_PATTERNS: List[Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"leverage ratio",
        r"fixed charge coverage",
        r"debt service coverage",
        r"interest coverage ratio",
        r"minimum liquidity",
        r"consolidated liquidity",
        r"financial covenants?",
    ]
]


def matches_keyword(text: str) -> bool:
    """Return True if any financial-metric covenant keyword pattern matches `text`."""
    return any(pattern.search(text) for pattern in COVENANT_KEYWORD_PATTERNS)
