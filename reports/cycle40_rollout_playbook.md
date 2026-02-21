# Cycle 40: Rollout Playbook (Shadow -> Canary -> Full)

Date: 2026-02-21

## Goal
Define an executable rollout process using validated policy logic (cycle 37) and tuned monitoring thresholds (cycles 38-39).

## Baseline policy to deploy
1. Mode selection:
- `ranking-first`: `raw@0.05`
- `reliability-first`: `isotonic@0.05`
2. Cost-aware default schedule:
- low cost: `isotonic@0.05`
- higher cost: `isotonic@0.10`
- note small lambda handoff gap (`0.0125 -> 0.015`): treat as explicit product decision boundary.

## Monitoring profile (recommended initial)

Window:
- primary: `500` edges per domain-mode window
- fallback for low-volume domains: `1000` edges

Hard-alert thresholds (window `500`, tuned to ~5% false alert):
- `coverage_rel_drift_abs > 0.20` (keep)
- ranking-first: `auc_drop > 0.046`
- reliability-first: `ece_uplift > 0.050`

Soft-alert thresholds (early signal only, no rollback alone):
- ranking-first: `auc_drop > 0.020`
- reliability-first: `ece_uplift > 0.020`

Escalation logic:
- hard alert if threshold breach in `>= 2` consecutive windows for the same domain-mode.
- soft alert if `>= 3` soft breaches in 5 consecutive windows.

## Staged rollout

### Stage 0: Offline certification (complete)
- Conformance replay exact match vs policy table.
- Row-wise and vectorized serving outputs identical.

### Stage 1: Shadow mode
- Duration: at least `10` windows per domain-mode.
- Actions:
  - log all policy outputs and referral decisions,
  - compute rolling AUC/ECE/Brier/coverage and drift.
- Exit criteria:
  - no hard alerts,
  - soft alerts not persistent (no `3-of-5` pattern).

### Stage 2: Canary serving
- Traffic fraction: `10%` then `25%`.
- Minimum duration per step: `5` windows/domain-mode.
- Guardrails:
  - same hard/soft alert logic,
  - compare canary metrics against shadow baseline.
- Rollback criteria:
  - any hard alert persists for `2` consecutive windows,
  - or manual review confirms systematic calibration degradation.

### Stage 3: Full rollout
- Move to `100%` after stable canary.
- Keep monitoring thresholds unchanged for first post-rollout week.
- Refit thresholds only after collecting real live-window distributions.

## Implementation checklist additions
1. Add explicit lambda boundary behavior for the schedule gap (`0.0125, 0.015`).
2. Implement separate hard vs soft alert channels.
3. Store per-window metrics with immutable config snapshot (mode, threshold, calibration).
4. Add rollback hook to switch all traffic to best single model per mode.

## Expected impact
- Reduced false-alarm burden vs original monitoring spec.
- Preserves sensitivity to sustained degradation rather than one-window noise.
- Keeps deployment policy simple and auditable.
