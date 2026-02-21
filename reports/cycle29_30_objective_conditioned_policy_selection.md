# Objective-Conditioned Policy Selection (Cycles 29-30)

Date: 2026-02-20

## Goal
Select deployment policies under explicit objectives (ranking-first vs reliability-first), quantify joint uncertainty (`delta AUROC`, `delta ECE`), and test stability under stricter coverage constraints.

## Script
- `implementation/scripts/run_objective_conditioned_policy_selection.py`

## Key Outputs
- Coverage floor 0.05:
  - `implementation/outputs/cycle29b_objective_policy_selection_cov05/policy_candidates.csv`
  - `implementation/outputs/cycle29b_objective_policy_selection_cov05/policy_selected_by_objective.csv`
  - `implementation/outputs/cycle29b_objective_policy_selection_cov05/policy_selected_with_bootstrap.csv`
  - `implementation/outputs/cycle29b_objective_policy_selection_cov05/cross_domain_default_policy_selected.csv`
  - `implementation/outputs/cycle29b_objective_policy_selection_cov05/cross_domain_default_vs_domain_specific.csv`
- Coverage sensitivity:
  - `implementation/outputs/cycle30_policy_selection_cov10/policy_selected_by_objective.csv`
  - `implementation/outputs/cycle30_policy_selection_cov20/policy_selected_by_objective.csv`
  - `implementation/outputs/cycle30_policy_selection_cov_sensitivity_summary.csv`

## Objective-Conditioned Results (coverage floor 0.05)

### Ranking-first (constraint: `delta ECE <= +0.01`)
- External-lung: `isotonic`, `tau=0.05`, coverage `0.397`, delta AUROC `+0.0099`, delta ECE `-0.0019`.
- Immune: `isotonic`, `tau=0.05`, coverage `0.701`, delta AUROC `+0.0257`, delta ECE `+0.0056`.
- Lung: `isotonic`, `tau=0.05`, coverage `0.449`, delta AUROC `+0.0100`, delta ECE `+0.0016`.

### Reliability-first (constraint: `delta AUROC >= -0.01`)
- External-lung: `isotonic`, `tau=0.05`.
- Immune: `platt`, `tau=0.05`.
- Lung: `platt`, `tau=0.05`.

Interpretation:
- Ranking-first and default policy strongly favor `isotonic + tau=0.05`.
- Reliability-first differs by domain (platt for immune/lung), trading some calibration behavior for ranking stability.

## Joint Uncertainty (bootstrap on selected policies)
- External-lung ranking-first:
  - delta AUROC CI `[+0.0065, +0.0142]`
  - delta ECE CI `[-0.0058, +0.0043]`
- Immune ranking-first:
  - delta AUROC CI `[+0.0127, +0.0380]`
  - delta ECE CI `[-0.0118, +0.0213]`
- Lung ranking-first:
  - delta AUROC CI `[+0.0056, +0.0142]`
  - delta ECE CI `[-0.0044, +0.0062]`

Interpretation:
- AUROC gains are consistently positive for ranking-first selections.
- ECE impact is small and can cross zero in some domains (expected for mixed ranking/reliability tradeoffs).

## Cross-Domain Default Policy
- Selected conservative default:
  - `calibration=isotonic`, `disagreement_threshold=0.05`
  - mean delta AUROC `+0.0152`, min delta AUROC `+0.0099`
  - max delta ECE `+0.0056`
- For ranking-first objective, this default exactly matched domain-specific selections in all three domains.

## Coverage-Floor Sensitivity (Cycle 30)
- Re-ran policy selection with minimum coverage floors `0.10` and `0.20`.
- Selected policies remained unchanged from coverage floor `0.05`.
- Consolidated sensitivity table:
  - `implementation/outputs/cycle30_policy_selection_cov_sensitivity_summary.csv`

Interpretation:
- Policy recommendations are stable under stricter practical coverage requirements.

## Conclusion
- A simple default policy (`isotonic`, `tau=0.05`) is robust and competitive across domains.
- Objective-conditioned alternatives are still useful when reliability is prioritized domain-by-domain.
