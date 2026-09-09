# Loan Covenant Extraction

Extracts financial covenants (leverage ratios, coverage ratios, minimum
liquidity), EBITDA add-back definitions, and negative covenants (with
carve-outs) from loan/credit agreements (PDF or HTML), with source citations
and deterministic cross-checks on every extracted number.

## Architecture

Two-stage, local-only pipeline:

1. **Narrow (cheap, fast)** — `retrieval/`: keyword regex flags (e.g. "Leverage
   Ratio", "shall not exceed") OR BGE-M3 embedding similarity to a fixed set
   of reference queries select a short list of candidate clauses out of the
   full contract.
2. **Extract (LLM, structured)** — `extraction/`: each candidate clause is sent
   to a local LLM (via LM Studio's OpenAI-compatible API) with a JSON-schema
   response format, so output is a validated Pydantic object, not free text.
3. **Validate (deterministic)** — `validation/`: every extracted item's
   citation quote is fuzzy-matched back into the source document
   (`citation_audit.py`), and every financial covenant's numeric threshold is
   independently re-parsed via regex and cross-checked against the LLM's
   answer (`calculators.py`). The LLM is never the system of record — every
   number ships with an audit trail a reviewer can check.

```
Document (PDF/HTML)
  -> ingestion (load + chunk, keeps page/section citations)
  -> retrieval (keyword + BGE-M3 candidate selection)
  -> extraction (LM Studio, JSON-schema constrained)
  -> validation (citation audit + regex cross-check)
  -> JSON export
```

## Project Layout

```
data/
  raw/        your real loan contracts (gitignored)
  processed/  cached intermediate output (gitignored)
  sample/     synthetic sample contract used by tests/demo
models/       local embedding model cache (gitignored)
src/covenant_extraction/
  config.py        settings (env-driven)
  ingestion/        pdf_loader.py, html_loader.py, chunker.py
  retrieval/        keyword_filter.py, embeddings.py, candidate_selector.py
  extraction/       schema.py, prompts.py, llm_client.py, extractor.py
  validation/       citation_audit.py, calculators.py
  pipeline.py       end-to-end orchestration
scripts/
  run_extraction.py    CLI: extract covenants from one contract
  download_models.py   one-time BGE-M3 download/cache
tests/          fully offline unit tests (fake embedder + fake LLM client)
```

## Setup

### 1. Python 3.11 environment

Your system Python may be newer than what `torch`/`sentence-transformers`
currently ship wheels for. Use Python 3.11 specifically:

```
py install 3.11
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

(`pip install -e .` makes the `covenant_extraction` package importable from anywhere — needed to run `scripts/*.py` directly, not just via `pytest`.)

### 2. LM Studio (local LLM server)

This machine has an AMD Radeon 890M iGPU (no CUDA) and ~93 GB RAM, so
inference runs via LM Studio's CPU/Vulkan backend rather than CUDA — slower
per-token than a discrete GPU, but entirely adequate for offline/batch
covenant extraction.

1. Download and install LM Studio (GUI installer) from lmstudio.ai.
2. In LM Studio's model search, download **Qwen2.5-32B-Instruct**, **Q4_K_M**
   GGUF quantization (~20 GB) — fits comfortably in 93 GB RAM. If you want
   faster responses at some accuracy cost, use the 14B-Instruct variant
   instead.
3. In LM Studio settings, optionally enable GPU offload (Vulkan) for the
   890M iGPU; CPU-only also works fine for this workload.
4. Start the local server (Developer tab -> Start Server), default
   `http://localhost:1234`. Ensure "Structured Output" / JSON schema mode is
   enabled if your LM Studio version exposes that toggle.

### 3. Embedding model

```
python scripts/download_models.py
```

Downloads and caches `BAAI/bge-m3` under `models/` (one-time, needs
internet). All later runs work fully offline.

### 4. Configuration

```
copy .env.example .env
```

Adjust `LLM_MODEL_NAME` if it doesn't match the exact model name shown in
LM Studio.

## Usage

```
python scripts/run_extraction.py data/sample/sample_credit_agreement.html
python scripts/run_extraction.py path\to\your\contract.pdf --out result.json
```

Output is JSON with `financial_covenants`, `ebitda_addbacks`, and
`negative_covenants`, each item including an `audit` block:
`citation_verified`, `citation_similarity`, and (for financial covenants)
`calculator_match` against an independently regex-parsed threshold.

**Every run's output should be treated as a first-pass draft for human
review** — check any item where `citation_verified` or `calculator_match`
is `false` first.

## Testing

```
pytest -q
```

All tests run fully offline: `tests/conftest.py` provides a deterministic
`FakeEmbedder` and a rule-based `FakeLLMClient`, so no GPU, model download,
or running LM Studio server is required to run the suite.

## Known Limitations

- No OCR — scanned/image-only PDFs are not supported (text-layer PDFs and
  HTML only).
- The regex-based calculator cross-check only covers ratio (`X:1.00`) and
  dollar-amount thresholds; it flags items it can't parse as
  "not independently verified" rather than blocking them.
- Candidate selection uses fixed reference queries; for contracts with very
  unusual covenant phrasing, consider lowering the keyword-only reliance by
  tuning `retrieval/keyword_filter.py`.
