# Covenant extraction eval: cached

- Source of extractions: committed pipeline outputs in `data/processed/`
- Gold set: 10 documents in `evals/gold/`, 10 scored
- Numeric tolerance: 1% relative
- Date: 2026-11-02

## Headline

| Metric | Value |
|---|---|
| Precision | 88% (15 of 17) |
| Recall | 75% (15 of 20) |
| F1 | 0.81 |
| Matched covenants fully correct | 100% (15 of 15) |
| Step-down schedule levels captured | 3 of 27 (1 of 2 schedules complete) |
| Documents with no required covenants and zero false positives | 3 of 3 |

## Recall by covenant type

| Type | Found | Gold | Recall |
|---|---|---|---|
| incurrence_condition | 4 | 6 | 67% |
| maintenance | 11 | 14 | 79% |

## Field-level accuracy (matched, required covenants)

| Field | Correct | Scored | Accuracy |
|---|---|---|---|
| Threshold, exact | 15 | 15 | 100% |
| Threshold, within tolerance | 15 | 15 | 100% |
| Comparison direction | 15 | 15 | 100% |
| Covenant type (maintenance vs incurrence) | 15 | 15 | 100% |
| Unit | 15 | 15 | 100% |
| Name | 15 | 15 | 100% |
| Test frequency (maintenance only) | 11 | 11 | 100% |
| All of threshold + direction + type + unit | 15 | 15 | 100% |

## Per document

| Document | Gold | Found | Missed | False pos. | Precision | Recall | Fully correct |
|---|---|---|---|---|---|---|---|
| astronics_credit_agreement | 4 | 4 | 0 | 0 | 100% | 100% | 4/4 |
| beyond_meat_unprocessed_foods | 2 | 1 | 1 | 0 | 100% | 50% | 1/1 |
| flux_power_loan_agreement | 2 | 1 | 1 | 0 | 100% | 50% | 1/1 |
| innventure_wti_fund | 0 | 0 | 0 | 0 | 100% | 100% | 0/0 |
| lone_star_ast_science | 0 | 0 | 0 | 0 | 100% | 100% | 0/0 |
| mizuho_qvc | 6 | 4 | 2 | 2 | 67% | 67% | 4/4 |
| old_plank_trail_halo_pets | 1 | 1 | 0 | 0 | 100% | 100% | 1/1 |
| runway_growth_aspen_university | 2 | 1 | 1 | 0 | 100% | 50% | 1/1 |
| sample_credit_agreement | 3 | 3 | 0 | 0 | 100% | 100% | 3/3 |
| wells_fargo_retailco | 0 | 0 | 0 | 0 | 100% | 100% | 0/0 |

## Errors

### beyond_meat_unprocessed_foods

- **Missed**: Junior Debt Service Cap (maintenance, $20,000,000)

### flux_power_loan_agreement

- **Missed**: Capital Expenditures (maintenance, $1,200,000)
- **Incomplete schedule** on Minimum EBITDA: 1 of 25 step levels extracted

### mizuho_qvc

- **Missed**: Consolidated Secured Leverage Ratio (incurrence_condition, 3.75)
- **Missed**: Consolidated Leverage Ratio (incurrence_condition, 4)
- **False positive**: Consolidated Leverage Ratio 4.5 (incurrence_condition) citing "solely in the case of any such Incremental Facility or Pari Passu Indebtedness being incurred to finance an a…"
- **False positive**: Consolidated Secured Leverage Ratio 3.5 (incurrence_condition) citing "solely in the case of any such Incremental Facility or Pari Passu Indebtedness being incurred to finance an a…"

### runway_growth_aspen_university

- **Missed**: Minimum Consolidated Net Worth (maintenance, $5,750,000)

## Optional items the extractor reported

These are borderline items (gold `required: false`). They count neither for nor against the scores.

- wells_fargo_retailco: Debt Service Coverage Ratio (1.1)
