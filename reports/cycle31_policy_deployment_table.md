# Policy Deployment Table (Cycle 31)

Date: 2026-02-20

## Goal
Convert the objective-conditioned selection outputs into a compact deployment-ready table for three profiles:
- ranking-first,
- reliability-first,
- cross-domain default.

## Output
- `implementation/outputs/cycle31_policy_deployment_table.csv`

## Result Snapshot
- External-lung:
  - ranking-first: isotonic + `tau=0.05`
  - reliability-first: isotonic + `tau=0.05`
  - cross-domain default: isotonic + `tau=0.05`
- Immune:
  - ranking-first: isotonic + `tau=0.05`
  - reliability-first: platt + `tau=0.05`
  - cross-domain default: isotonic + `tau=0.05`
- Lung:
  - ranking-first: isotonic + `tau=0.05`
  - reliability-first: platt + `tau=0.05`
  - cross-domain default: isotonic + `tau=0.05`

The table includes delta AUROC and delta ECE with bootstrap CIs for each policy/profile/domain row.

## Interpretation
- A single default policy is viable and stable.
- Reliability-first profile differs only by calibration mode in immune/lung.
- This provides a directly usable handoff artifact for deployment-rule implementation.
