# Pre-Implementation Dry-Run Checklist (Cycle 36)

Date: 2026-02-21

## Goal
Translate the completed policy analysis into an implementation-ready checklist with concrete inputs, transforms, thresholds, and monitoring metrics.

## Dry-run basis
- Mode table: `implementation/outputs/cycle35_dual_mode_deployment_spec/dual_mode_domain_policy_table.csv`
- Mode summary: `implementation/outputs/cycle35_dual_mode_deployment_spec/dual_mode_summary.csv`
- Cost schedule: `implementation/outputs/cycle35_dual_mode_deployment_spec/cost_aware_default_schedule.csv`

## Deployment checklist
1. Inputs
- Confirm runtime receives per-edge probabilities for:
  - `scgpt_prob`
  - `geneformer_prob`
  - `compact_prob`
- Confirm domain label is available (`immune`, `lung`, `external_lung`).

2. Calibration transform
- `ranking-first` mode: use `raw`.
- `reliability-first` mode: use `isotonic`.
- If unknown mode, default to `reliability-first`.

3. Referral threshold rule
- Base threshold: `tau=0.05`.
- Cost-aware default schedule:
  - `lambda in [0.000, 0.0125]`: `tau=0.05`
  - `lambda in [0.015, 0.030]`: `tau=0.10`

4. Policy logic
- Compute disagreement: `abs(geneformer_prob - scgpt_prob)`.
- If disagreement `>= tau`, output `compact_prob`.
- Else output best single-model probability for the chosen mode.

5. Monitoring metrics (per domain, rolling window)
- `policy_auc`
- `policy_ece`
- `policy_brier`
- `coverage_fraction` (referral rate)
- drift checks on disagreement distribution

6. Alert thresholds (initial)
- `coverage_fraction` drift > ±20% from calibration baseline.
- `policy_ece` increase > +0.01 from baseline in reliability-first mode.
- `policy_auc` drop > -0.01 from baseline in ranking-first mode.

7. Rollout steps
- Start with shadow mode logging only.
- Compare live telemetry against cycle35 baseline tables.
- Enable serving after one stable window with no alert breach.

## Conclusion
- Implementation path is now explicit for both deployment modes and the cost-aware default schedule.
- Remaining work is engineering integration and telemetry validation, not research uncertainty reduction.
