"""Thin client around LM Studio's OpenAI-compatible local server.

LM Studio exposes an OpenAI-compatible /v1/chat/completions endpoint with
support for JSON-schema-constrained output ("Structured Output"). We reuse
the official `openai` SDK pointed at the local base_url.
"""
from __future__ import annotations

import json
from typing import Protocol, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from covenant_extraction.config import settings
from covenant_extraction.extraction.prompts import SYSTEM_PROMPT, build_user_prompt
from covenant_extraction.extraction.schema import CovenantExtractionResult

T = TypeVar("T", bound=BaseModel)


class StructuredLLMClient(Protocol):
    """Generic single-shot, schema-constrained completion. Shared by covenant
    extraction (LLMClient.extract) and the definitions-resolution pipeline
    (covenant_extraction.definitions), so each new schema-constrained call
    site doesn't need to reimplement the request/response-format boilerplate.
    """

    def complete(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:
        ...


class LLMClient(StructuredLLMClient, Protocol):
    """Minimal interface required by extractor.py."""

    def extract(self, clause_text: str) -> CovenantExtractionResult:
        ...


class LMStudioClient:
    """Real client that talks to a running LM Studio server."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self._base_url = base_url or settings.lm_studio_base_url
        self._model = model or settings.llm_model_name
        self._client = OpenAI(base_url=self._base_url, api_key="lm-studio")

    def complete(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:
        json_schema = schema.model_json_schema()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema.__name__.lower(), "schema": json_schema, "strict": True},
            },
            temperature=0,
        )
        content = response.choices[0].message.content
        return schema.model_validate(json.loads(content))

    def extract(self, clause_text: str) -> CovenantExtractionResult:
        return self.complete(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(clause_text),
            schema=CovenantExtractionResult,
        )


class ClaudeClient:
    """Real client that talks to the Anthropic API (Claude models) --
    same LLMClient interface as LMStudioClient, different backend, so the
    two are interchangeable drop-ins. Built for A/B evaluation of
    extraction quality/cost across models (see evals/).

    Credentials resolve from the environment the same way the `anthropic`
    SDK/CLI does (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, an `ant auth
    login` profile, etc.) -- nothing is read from `settings` for auth.
    `anthropic` is imported lazily so it stays an optional dependency for
    anyone only using LM Studio.
    """

    def __init__(self, model: str | None = None, max_tokens: int = 4096):
        import anthropic

        self._model = model or settings.claude_model_name
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def complete(self, system_prompt: str, user_prompt: str, schema: Type[T]) -> T:
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_format=schema,
        )
        return response.parsed_output

    def extract(self, clause_text: str) -> CovenantExtractionResult:
        return self.complete(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(clause_text),
            schema=CovenantExtractionResult,
        )
