# Cycle 41: Monitoring Incident-Injection Evaluation

Date: 2026-02-21

## Goal
Estimate alert sensitivity and detection delay under synthetic incidents, and compare:
- current strict thresholds,
- tuned thresholds targeting ~5% false-alert rate.

## Script and outputs

Script:
- `implementation/scripts/run_monitoring_incident_injection.py`

Outputs:
- `implementation/outputs/cycle41_monitoring_incident_injection/monitoring_incident_power.csv`
- `implementation/outputs/cycle41_monitoring_incident_injection/monitoring_incident_sequence_detection.csv`
- `implementation/outputs/cycle41_monitoring_incident_injection/monitoring_incident_performance_shift_matrix.csv`

## Setup
- Source windows: cycle 38 pseudo-rolling backtest windows.
- Profiles:
  - `current_strict`: coverage cap `0.20`, performance cap `0.01`.
  - `tuned_5pct`: coverage cap `0.20`, performance caps from cycle 39.
- Incident axes:
  - performance shift: `0.0, 0.005, 0.01, 0.02, 0.03, 0.05`
  - coverage shift: `0.0, 0.05, 0.10, 0.15`
- Sequence detection rule (rollout playbook hard-alert logic):
  - trigger if two consecutive hard-alert windows occur within a 20-window horizon.

## Key results

### 1) False-alert baseline under no incident (performance shift `0`, coverage shift `0`)
- `current_strict` remains high:
  - ranking-first: `0.298-0.397` hard-alert rate (window `1000` to `250`)
  - reliability-first: `0.833-0.990` hard-alert rate
- `tuned_5pct` remains near design target:
  - both modes: `~0.046-0.049` hard-alert rate

### 2) Detection sensitivity at window `500` (no extra coverage shift)
- hard-alert rate under tuned profile:
  - ranking-first:
    - `+0.01` performance shift -> `0.096`
    - `+0.02` -> `0.173`
    - `+0.03` -> `0.283`
    - `+0.05` -> `0.563`
  - reliability-first:
    - `+0.01` -> `0.156`
    - `+0.02` -> `0.396`
    - `+0.03` -> `0.735`
    - `+0.05` -> `0.999`

### 3) Two-consecutive-window detection within 20 windows (window `500`, tuned profile, no coverage shift)
- detection probability:
  - ranking-first:
    - `+0.01` shift -> `0.157`
    - `+0.02` -> `0.394`
    - `+0.03` -> `0.730`
    - `+0.05` -> `0.996`
  - reliability-first:
    - `+0.01` shift -> `0.341`
    - `+0.02` -> `0.926`
    - `+0.03` -> `1.000`
- median first detection window:
  - ranking-first: around windows `11 -> 8 -> 4` as shift grows (`0.01 -> 0.03 -> 0.05`)
  - reliability-first: around windows `10 -> 2 -> 2`.

## Interpretation
- Tuned thresholds achieve the intended false-alert control while preserving useful sensitivity to moderate/large incidents.
- Reliability-first mode detects ECE degradation faster than ranking-first mode detects comparable AUC drops.
- For small incidents (`~0.01` shift), two-consecutive-window criteria are intentionally conservative (lower short-horizon detection), which matches the false-alert minimization objective.

## Practical implication
- Keep tuned thresholds for initial rollout.
- Use soft alerts to surface weak early shifts before hard-alert persistence criteria are met.
