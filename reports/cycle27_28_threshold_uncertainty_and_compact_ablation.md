# Threshold Uncertainty And Compact Ablation (Cycles 27-28)

Date: 2026-02-20

## Goal
1) Add uncertainty intervals to disagreement-threshold policy gains.  
2) Test whether baseline confounds still help after score-level combining.

## Cycle 27: Threshold uncertainty

### Script
- `implementation/scripts/run_outer_split_threshold_uncertainty.py`

### Outputs
- `implementation/outputs/cycle27_outer_split_threshold_uncertainty/outer_split_disagreement_policy_bootstrap_ci.csv`
- `implementation/outputs/cycle27_outer_split_threshold_uncertainty/outer_split_disagreement_policy_bootstrap_distribution.tsv`
- `implementation/outputs/cycle27_outer_split_threshold_uncertainty/outer_split_disagreement_policy_recommended_thresholds.csv`

### Key CI results (`compact - best single` AUC delta)
- Immune:
  - `tau=0.10`: `+0.0346` (95% CI `[+0.0113, +0.0505]`)
  - `tau=0.15`: `+0.0467` (95% CI `[+0.0167, +0.0656]`)
- External-lung:
  - `tau=0.05`: `+0.0191` (95% CI `[+0.0105, +0.0225]`)
  - `tau=0.10`: `+0.0288` (95% CI `[+0.0125, +0.0381]`)
- Lung:
  - `tau=0.05`: `+0.0151` (95% CI `[+0.0084, +0.0223]`)
  - Higher thresholds show wide CIs crossing zero.

Interpretation:
- Moderate thresholds produce statistically stable gains.
- Very high thresholds are low-coverage and unstable.

## Cycle 28: Compact feature ablation

### Script
- `implementation/scripts/run_outer_split_compact_ablation.py`

### Outputs
- `implementation/outputs/cycle28_outer_split_compact_ablation/outer_split_compact_ablation_summary.csv`
- `implementation/outputs/cycle28_outer_split_compact_ablation/outer_split_compact_ablation_oof.tsv`

### Key results
- `compact(prob-only)` vs best single:
  - immune `+0.0149`
  - lung `+0.0071`
  - external-lung `+0.0082`
- `compact(prob+baseline)` vs best single:
  - immune `+0.0170`
  - lung `+0.0098`
  - external-lung `+0.0109`
- Increment from adding baseline confounds to prob-only compact:
  - immune `+0.0021`
  - lung `+0.0027`
  - external-lung `+0.0027`

Interpretation:
- Score-only combination already helps, but baseline confounds still contribute small consistent gains.

## Conclusion
- Threshold-policy gains are credible at moderate disagreement cutoffs with bootstrap support.
- Deployment compact model should keep baseline confounds unless strict simplicity constraints outweigh a ~0.002-0.003 AUROC gain.
