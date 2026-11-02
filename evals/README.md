# Covenant extraction evals

Scores the financial-covenant extraction against a hand-labeled gold set of
10 public loan and credit agreements (the SEC EDGAR filings in
`data/sample/`, plus one synthetic fixture).

Latest results: [`results/cached.md`](results/cached.md).

## Run it

```
pip install -e .

covenant-eval --check-gold                 # every gold anchor exists in its document
covenant-eval --backend cached             # score the committed outputs in data/processed/ (no LLM)
covenant-eval --backend lmstudio           # run the pipeline live, then score
covenant-eval --backend claude --model claude-haiku-4-5
covenant-eval --from-results evals/results/<run>.json   # re-score a saved run, no LLM
```

`python evals/run_eval.py ...` takes the same arguments. Each run writes
`evals/results/<label>.json` (full detail, including the raw extractions so
it can be re-scored later) and `evals/results/<label>.md` (the report).
Useful options: `--gold <file>` for one document, `--rel-tol 0.01` for the
numeric tolerance, `--label` to name the run.

## Gold set

| File | Document | Required covenants | What it tests |
|---|---|---|---|
| `sample_credit_agreement` | synthetic fixture | 3 | leverage, FCCR, minimum liquidity |
| `astronics_credit_agreement` | Astronics credit agreement | 4 | 3 maintenance tests, a 4.75x to 4.50x step-down, an incurrence test in a definition |
| `mizuho_qvc` | QVC / Mizuho credit agreement | 6 | maintenance leverage plus 5 incurrence tests in debt, investment and restricted-payment baskets |
| `old_plank_trail_halo_pets` | Halo Purely for Pets loan agreement | 1 | single FCCR |
| `beyond_meat_unprocessed_foods` | Beyond Meat / Unprocessed Foods | 2 | minimum liquidity and a dollar cap on junior debt service |
| `flux_power_loan_agreement` | Flux Power loan agreement | 2 | springing minimum EBITDA with a 25-step monthly schedule and negative values, capex cap |
| `runway_growth_aspen_university` | Aspen University / Runway Growth | 2 | current ratio, minimum net worth |
| `innventure_wti_fund` | Innventure / WTI venture loan | 0 | agreement explicitly has no financial covenants; any extraction is a false positive |
| `lone_star_ast_science` | AST & Science equipment loan | 0 | no covenants; one optional springing cash deposit |
| `wells_fargo_retailco` | CMBS mortgage loan | 0 | no covenants; three optional DSCR / debt-yield triggers |

Each gold file has a `notes` field recording how the document was checked
(usually: every `N.NN to 1.00` ratio and every Financial Covenants section
read by hand).

### Gold entry fields

- `name`, `comparison`, `threshold`, `unit`, `covenant_type`: same meaning as
  `FinancialCovenant` in `extraction/schema.py`.
- `test_frequency`: one of `quarterly`, `monthly`, `annual`, `continuous`
  (maintenance covenants only).
- `citation_contains`: a phrase (or list of alternatives) from the clause.
  An extracted covenant matches when its quoted citation contains the
  phrase. Anchor on the number-bearing phrase, not lead-in text.
- `schedule`: step-down/step-up levels, `[{period, threshold}]`. `threshold`
  is the first level. The extractor emits one row per level, and all rows
  quoting the clause are credited to the one covenant.
- `required: false`: borderline items. Reporting them is not a false
  positive and missing them is not a miss.
- `name_aliases`: other acceptable names.
- `cached_output`: the committed pipeline output for `--backend cached`.

## Metrics

- **Detection**: precision, recall and F1 over required covenants,
  micro-averaged across documents, plus recall split by maintenance vs
  incurrence condition.
- **Field-level** (on matched covenants): threshold exact, threshold within
  tolerance (default 1% relative, so a sign flip or a dropped `000` still
  fails), comparison direction, covenant type, unit, name, test frequency,
  and "all fields" (threshold, direction, type and unit all right).
- **Step-downs**: share of schedule levels whose threshold was extracted,
  and how many schedules were captured completely.
- **Abstention**: documents with no required covenants that produced zero
  false positives.
