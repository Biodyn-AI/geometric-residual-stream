# Cost, Calibration, And Policy Simplification (Cycles 32-34)

Date: 2026-02-21

## Goals
1. Add referral-cost-aware policy optimization.  
2. Lock calibration regime with direct `raw vs isotonic` comparison at fixed threshold.  
3. Test whether a two-rule policy can approximate domain-specific optimal assignments.

## Cycle 32: Referral-cost utility sweep

### Script
- `implementation/scripts/run_policy_cost_utility_sweep.py`

### Outputs
- `implementation/outputs/cycle32_policy_cost_utility_sweep/policy_cost_utility_scored_grid.csv`
- `implementation/outputs/cycle32_policy_cost_utility_sweep/policy_cost_utility_best_by_domain_lambda.csv`
- `implementation/outputs/cycle32_policy_cost_utility_sweep/policy_cost_utility_best_default_by_lambda.csv`
- `implementation/outputs/cycle32_policy_cost_utility_sweep/policy_cost_utility_default_vs_domain_gap.csv`
- `implementation/outputs/cycle32_policy_cost_utility_sweep/policy_cost_utility_selection_stability.csv`

### Utility definition
- `U = delta_auc_vs_best_single - lambda * coverage_fraction`
- `lambda` grid: `0.0` to `0.03`.

### Key findings
- Isotonic remains selected throughout the sweep.
- As `lambda` increases, optimal thresholds move upward:
  - early region: `tau=0.05`,
  - mid region: `tau=0.10`,
  - highest-cost region (immune): `tau=0.15`.
- Cross-domain default transitions:
  - `tau=0.05` for low cost,
  - `tau=0.10` for higher cost.

Interpretation:
- Threshold tuning, not model-family changes, carries most cost-aware adaptation.

## Cycle 33: Calibration regime lock at fixed threshold

### Script
- `implementation/scripts/run_calibration_regime_lock.py`

### Outputs
- `implementation/outputs/cycle33_calibration_regime_lock/calibration_regime_policy_metrics.csv`
- `implementation/outputs/cycle33_calibration_regime_lock/calibration_regime_lock_comparisons.csv`

### Setup
- Fixed threshold: `tau=0.05`.
- Methods compared: `raw`, `isotonic`, `platt`.

### Key findings (`isotonic - raw`)
- Policy AUC decreases across all domains:
  - external-lung: `-0.00443` (CI `[-0.00594, -0.00282]`)
  - immune: `-0.00647` (CI `[-0.01222, -0.00129]`)
  - lung: `-0.00235` (CI `[-0.00429, -0.00024]`)
- Policy ECE and Brier improve strongly:
  - ECE drops by about `0.16` to `0.22`
  - Brier drops by about `0.034` to `0.051`

### Key findings (`platt - raw`)
- Small AUC decrease (near zero), minimal ECE/Brier change.

Interpretation:
- `isotonic` is reliability-first.
- `raw/platt` are ranking-first.
- This validates objective-dependent calibration choice.

## Cycle 34: Two-rule policy simplification

### Script
- `implementation/scripts/run_two_rule_policy_simplification.py`

### Outputs
- `implementation/outputs/cycle34_two_rule_policy_simplification/two_rule_policy_summary_by_lambda.csv`
- `implementation/outputs/cycle34_two_rule_policy_simplification/two_rule_policy_domain_assignments.csv`
- `implementation/outputs/cycle34_two_rule_policy_simplification/two_rule_policy_pattern_stability.csv`

### Key findings
- Best two-rule family is essentially lossless vs domain-specific optimum:
  - mean utility gap is `0` (or numerically tiny) across lambda.
- Stable rule patterns:
  - low cost: one rule `isotonic@0.05`,
  - medium cost: two rules `isotonic@0.05` and `isotonic@0.10`,
  - high cost: two rules `isotonic@0.10` and `isotonic@0.15`.

Interpretation:
- Policy complexity can be reduced to at most two rules with negligible utility loss.

## Conclusion
- Cost-aware deployment should tune threshold with `lambda`, while keeping calibration regime aligned to objective:
  - ranking-first: favor `raw/platt`,
  - reliability-first: favor `isotonic`.
- A compact two-rule policy is a practical near-optimal deployment approximation.
