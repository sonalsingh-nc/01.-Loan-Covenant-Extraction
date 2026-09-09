"""Phase 1: locate a contract's Definitions section and extract every
defined term -- one schema-constrained LLM call per term, no iteration, no
agentic loop.
"""
from __future__ import annotations

import re
from typing import List

from covenant_extraction.definitions.prompts import TERM_SYSTEM_PROMPT, build_term_user_prompt
from covenant_extraction.definitions.schema import DefinedTerm
from covenant_extraction.extraction.llm_client import StructuredLLMClient
from covenant_extraction.ingestion.chunker import Chunk
from covenant_extraction.retrieval.definitions_lookup import extract_definitions_text

# Splits on each quoted, title-cased defined-term marker (e.g.
# `"Consolidated EBITDA" means`), so every resulting chunk covers exactly one
# term's definition prose.
_TERM_BOUNDARY_RE = re.compile(r'(?="[A-Z][^"]{0,80}"\s+(?:means|shall mean)\b)')


def locate_definitions_section(chunks: List[Chunk]) -> str:
    """Find and concatenate the contract's Definitions / Additional
    Definitions section. Thin wrapper over the existing heading-based lookup
    in retrieval/definitions_lookup.py."""
    return extract_definitions_text(chunks)


def chunk_by_term_boundary(definitions_section: str) -> List[str]:
    """Split definitions-section text on defined-term markers, so each chunk
    contains exactly one term's raw definition prose."""
    if not definitions_section.strip():
        return []
    pieces = _TERM_BOUNDARY_RE.split(definitions_section)
    return [p.strip() for p in pieces if p.strip()]


def extract_defined_terms(chunks: List[Chunk], llm_client: StructuredLLMClient) -> List[DefinedTerm]:
    """Locate the definitions section, extract every defined term.

    Single decoder call per chunk, schema-constrained via
    StructuredLLMClient.complete. No iteration, no agentic loop.
    """
    definitions_section = locate_definitions_section(chunks)
    term_chunks = chunk_by_term_boundary(definitions_section)

    terms: List[DefinedTerm] = []
    for chunk_text in term_chunks:
        term = llm_client.complete(
            system_prompt=TERM_SYSTEM_PROMPT,
            user_prompt=build_term_user_prompt(chunk_text),
            schema=DefinedTerm,
        )
        terms.append(term)

    return terms
