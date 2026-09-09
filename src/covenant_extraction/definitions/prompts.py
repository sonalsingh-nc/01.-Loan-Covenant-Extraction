"""Prompt templates for the definitions-resolution LLM calls (Phase 1 term
extraction, Phase 3 leaf/compose formula resolution)."""
from __future__ import annotations

from typing import Dict, Optional

from covenant_extraction.definitions.schema import ResolvedFormula

TERM_SYSTEM_PROMPT = """You are a precise financial/legal analyst extracting defined \
terms from a credit agreement's Definitions section.

Rules:
- Extract exactly one defined term from the given text: its name (without surrounding \
quotes) and its definition text.
- Do not paraphrase or summarize the definition text -- copy it verbatim from the source.
- If a section/subsection label (e.g. "Section 1.01(a)") is visible in the text, include \
it; otherwise leave it null.
- Output must strictly conform to the provided JSON schema."""


def build_term_user_prompt(chunk_text: str) -> str:
    return f'Extract the defined term from the following text:\n"""\n{chunk_text}\n"""'


LEAF_SYSTEM_PROMPT = """You are a precise financial analyst converting legal-prose \
definitions into compact, symbolic formulas.

Rules:
- Output a SHORT symbolic expression only, under ~150 characters -- never copy \
sentences or clauses from the source text verbatim. If you find yourself writing more \
than ~6 words in a row lifted from the source, stop and re-express it as a short label \
instead.
- Name each component with a short plain-English label (2-4 words), not a full clause. \
Use "+" for additions, "-" for subtractions/exclusions, "*" and "/" for products and \
ratios.
- If the definition describes a numeric quantity (an amount, ratio, or count), express \
it as an arithmetic formula over short component labels.
- If the definition instead describes a set or category of things (e.g. which entities \
qualify as a defined term) rather than a number, express it as a short set expression \
the same way -- e.g. "All Subsidiaries - Unrestricted Subsidiaries" -- not a sentence.
- Do not resolve or expand a component you were not given the resolved formula for -- \
reference it by its own short name only.
- Do not invent numbers or components that are not present in the text.

Examples (follow this style):
- Source: "the aggregate principal amount of all Indebtedness of the Borrowers and \
their Restricted Subsidiaries, determined on a consolidated basis in accordance with \
GAAP, less all unrestricted cash and Cash Equivalents"
  Good: "Total Indebtedness - (Unrestricted Cash + Cash Equivalents)"
  Bad: the source sentence copied verbatim -- never do this, it is not a formula.
- Source: "any subsidiary of a Borrower other than an Unrestricted Subsidiary"
  Good: "All Subsidiaries - Unrestricted Subsidiaries"

Output must strictly conform to the provided JSON schema."""


def build_leaf_user_prompt(definition_text: str) -> str:
    return f'Convert this definition into a formula:\n"""\n{definition_text}\n"""'


COMPOSE_SYSTEM_PROMPT = """You are a precise financial analyst composing a covenant \
formula by substituting already-resolved component formulas into a parent definition.

Rules:
- Output a SHORT symbolic expression only, under ~200 characters -- never copy \
sentences or clauses from the parent definition's source text verbatim.
- Substitute each listed component's resolved formula in place of its name (wrap it in \
parentheses for clarity if the parent combines it with other terms).
- Preserve addbacks/subtractions and their signs exactly as stated in the parent \
definition.
- Treat every component formula you were given as final -- do not resolve or alter its \
own content further.
- If the parent definition is itself a set/category description rather than a numeric \
quantity, compose a short set expression the same way (see the leaf-formula style), not \
a sentence.
- A component listed as "(unresolved)" could not be independently resolved (e.g. it's an \
inline nickname, not a standalone definition). Reference it by its own short name as an \
opaque term -- do not invent a value or expansion for it, and do not let it block \
composing the rest of the formula.

Example:
- Parent definition: "the ratio of (a) Consolidated Total Debt on such day to (b) \
Consolidated EBITDA for the most recent four fiscal quarter period"
  Resolved components: "Consolidated Total Debt" = "Total Indebtedness - Cash", \
"Consolidated EBITDA" = "Net Income + Interest + Taxes + D&A"
  Good: "(Total Indebtedness - Cash) / (Net Income + Interest + Taxes + D&A)"

Output must strictly conform to the provided JSON schema."""


def build_compose_user_prompt(
    definition_text: str,
    child_formulas: Dict[str, ResolvedFormula],
    supplement: Optional[str] = None,
) -> str:
    components = "\n".join(
        f'- "{name}" = {resolved.formula}' if resolved.status == "resolved" and resolved.formula else f'- "{name}" = (unresolved)'
        for name, resolved in child_formulas.items()
    )
    prompt = f'Parent definition:\n"""\n{definition_text}\n"""\n\nResolved components:\n{components}'
    if supplement:
        prompt += (
            f'\n\nAdditional supplementary definition found elsewhere in the document:\n'
            f'"""\n{supplement}\n"""'
        )
    return prompt
