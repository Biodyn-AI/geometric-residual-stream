# Cycle 37-39: Deployment Conformance and Monitoring Backtest

Date: 2026-02-21

## Goal
Validate that deployment logic exactly matches selected policies, then stress-test monitoring thresholds before rollout.

## Inputs
- `implementation/outputs/cycle25_outer_split_compact_model/outer_split_oof_predictions.tsv`
- `implementation/outputs/cycle35_dual_mode_deployment_spec/dual_mode_domain_policy_table.csv`
- `implementation/outputs/cycle35_dual_mode_deployment_spec/cost_aware_default_schedule.csv`

## Cycle 37: Deterministic conformance replay

Script:
- `implementation/scripts/run_deployment_policy_conformance.py`

Outputs:
- `implementation/outputs/cycle37_deployment_policy_conformance/conformance_mode_metrics.csv`
- `implementation/outputs/cycle37_deployment_policy_conformance/conformance_row_equivalence.csv`
- `implementation/outputs/cycle37_deployment_policy_conformance/deployment_replay_predictions.tsv`
- `implementation/outputs/cycle37_deployment_policy_conformance/cost_schedule_integrity.csv`
- `implementation/outputs/cycle37_deployment_policy_conformance/cost_schedule_rule_replay_metrics.csv`

Key results:
- Aggregate conformance is exact up to floating-point precision:
  - `policy_auc_diff = 0.0` in all domain-mode rows.
  - `coverage_fraction_diff ~= 0` in all domain-mode rows.
  - `policy_ece_diff ~= 0` and `policy_brier_diff ~= 0`.
- Row-level implementation equivalence is exact:
  - `prob_mismatch_count = 0`
  - `referral_mismatch_count = 0`
  - `exact_match = True` for all rows.
- Cost schedule integrity check:
  - no overlaps,
  - one small gap between segments: `0.0125 -> 0.015` (`gap = 0.0025`).
- Schedule rule replay:
  - `isotonic@0.05`: mean AUC `0.6544`, mean ECE `0.00893`, mean coverage `0.5158`
  - `isotonic@0.10`: mean AUC `0.6510`, mean ECE `0.00964`, mean coverage `0.2428`

Interpretation:
- Serving logic is implementation-consistent with the selected policy table.
- Cost schedule gap should be treated as an explicit handoff region in implementation (`lambda` policy decision boundary handling).

## Cycle 38: Monitoring threshold backtest

Script:
- `implementation/scripts/run_monitoring_backtest.py`

Outputs:
- `implementation/outputs/cycle38_monitoring_backtest/monitoring_backtest_window_metrics.csv`
- `implementation/outputs/cycle38_monitoring_backtest/monitoring_backtest_summary.csv`
- `implementation/outputs/cycle38_monitoring_backtest/monitoring_backtest_mode_aggregate_summary.csv`
- `implementation/outputs/cycle38_monitoring_backtest/monitoring_backtest_threshold_recommendations.csv`

Setup:
- Permutation-based pseudo-rolling windows.
- Window sizes: `250`, `500`, `1000`.
- Alert thresholds from cycle 36 checklist:
  - coverage drift: `|delta| > 20%`
  - ranking-first AUC drop: `> 0.01`
  - reliability-first ECE uplift: `> 0.01`

Key results:
- Coverage alerts are rare at all window sizes (near zero).
- Performance alerts dominate false positives:
  - ranking-first performance alert rate:
    - `39.6%` (`w=250`), `35.4%` (`w=500`), `29.8%` (`w=1000`)
  - reliability-first performance alert rate:
    - `99.0%` (`w=250`), `95.5%` (`w=500`), `83.3%` (`w=1000`)
- Therefore any-alert rates are too high for production use with current performance caps.

Interpretation:
- Existing monitoring thresholds are over-sensitive to normal stochastic variation in held-out replay.
- Reliability-first alerting is especially miscalibrated when using `ECE uplift > 0.01`.

## Cycle 39: Threshold tuning for target false-alert rate

Script:
- `implementation/scripts/run_monitoring_threshold_tuning.py`

Outputs:
- `implementation/outputs/cycle39_monitoring_threshold_tuning/monitoring_threshold_tuning_grid.csv`
- `implementation/outputs/cycle39_monitoring_threshold_tuning/monitoring_threshold_tuning_best.csv`
- `implementation/outputs/cycle39_monitoring_threshold_tuning/monitoring_threshold_tuning_perf_only_with_current_cov.csv`

Target:
- `any_alert_rate <= 0.05`.

Key tuned results:
- Feasible threshold pairs exist for all tested window sizes/modes.
- If coverage cap remains fixed at `0.20`, tuned performance caps that achieve ~5% any-alert:
  - `w=250`: ranking AUC drop `~0.066`, reliability ECE uplift `~0.073`
  - `w=500`: ranking AUC drop `~0.046`, reliability ECE uplift `~0.050`
  - `w=1000`: ranking AUC drop `~0.032`, reliability ECE uplift `~0.034`
- Sensitivity check (intermediate strictness):
  - using performance cap `0.02` still yields high any-alert rates:
    - `w=500`: ranking `23.0%`, reliability `72.4%`.

Interpretation:
- To control false-alert rate, performance thresholds must be materially wider than `0.01`.
- Larger windows reduce required alert caps, but increase detection latency.

## Updated conclusion
- Policy implementation is correct and deterministic.
- Monitoring policy needed retuning; the original `0.01` performance caps are not deployment-stable.
- The project is ready for a staged rollout with tuned alert thresholds and explicit handling of the small cost-schedule lambda gap.
