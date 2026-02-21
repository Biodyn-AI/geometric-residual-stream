# Dual-Mode Deployment Specification (Cycle 35)

Date: 2026-02-21

## Goal
Finalize practical deployment modes and provide a cost-aware default threshold schedule.

## Script
- `implementation/scripts/run_dual_mode_deployment_spec.py`

## Outputs
- `implementation/outputs/cycle35_dual_mode_deployment_spec/dual_mode_domain_policy_table.csv`
- `implementation/outputs/cycle35_dual_mode_deployment_spec/dual_mode_summary.csv`
- `implementation/outputs/cycle35_dual_mode_deployment_spec/cost_aware_default_schedule.csv`

## Domain-Level Mode Decisions
- Ranking-first mode:
  - all domains: `raw`, `tau=0.05`
- Reliability-first mode:
  - all domains: `isotonic`, `tau=0.05`

## Mode Summary
- Ranking-first:
  - mean policy AUC: `0.6588`
  - mean policy ECE: `0.2010`
  - mean coverage: `0.5756`
- Reliability-first:
  - mean policy AUC: `0.6544`
  - mean policy ECE: `0.0089`
  - mean coverage: `0.5158`

Interpretation:
- Ranking-first gives higher AUC.
- Reliability-first drastically improves calibration with modest AUC tradeoff.

## Cost-Aware Default Schedule
- Low-to-mid referral cost (`lambda` in `[0.000, 0.0125]`):
  - `isotonic@0.05`
- Higher referral cost (`lambda` in `[0.015, 0.030]`):
  - `isotonic@0.10`

Interpretation:
- Cost sensitivity is captured by threshold escalation while keeping calibration family fixed (`isotonic`) in the default schedule.

## Conclusion
- Deployment can now use an explicit two-mode specification:
  - `raw@0.05` for ranking-first,
  - `isotonic@0.05` for reliability-first.
- If a single default is required with cost-awareness, use the two-segment isotonic schedule above.
