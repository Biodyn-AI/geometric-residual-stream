# Outer-Split Calibration Intervention (Cycle 26)

Date: 2026-02-20

## Goal
Test calibration-only interventions (Platt and isotonic) on leakage-resistant outer-split predictions from Cycle 25.

## Script
- `implementation/scripts/run_outer_split_calibration_intervention.py`

## Outputs
- `implementation/outputs/cycle26_outer_split_calibration_intervention/outer_split_calibration_summary.csv`
- `implementation/outputs/cycle26_outer_split_calibration_intervention/outer_split_calibration_reliability_bins.csv`
- `implementation/outputs/cycle26_outer_split_calibration_intervention/outer_split_compact_calibration_focus.csv`

## Key Results

### Compact model calibration tradeoff
- `raw` compact AUC:
  - immune `0.7276`
  - lung `0.6218`
  - external-lung `0.6297`
- `isotonic` compact AUC:
  - immune `0.7152` (`-0.0124`)
  - lung `0.6177` (`-0.0041`)
  - external-lung `0.6254` (`-0.0043`)
- `isotonic` compact ECE:
  - immune `0.0309` (from `0.1772`)
  - lung `0.0039` (from `0.1965`)
  - external-lung `0.0024` (from `0.2241`)
- `isotonic` compact Brier also improves strongly across all domains.

### Platt scaling behavior
- Minimal AUC change (near-zero) but little/no ECE improvement.
- In some cases ECE slightly worsens vs raw.

Interpretation:
- Isotonic is substantially better for calibration quality, with a modest ranking (AUC) penalty.
- Platt is nearly rank-preserving but too weak for large calibration correction in this setting.

## Conclusion
- Calibration-only changes can materially improve reliability without changing representation features.
- If the deployment objective prioritizes probabilistic reliability (ECE/Brier), isotonic is favorable.
- If ranking AUC is primary, raw or lightly calibrated outputs remain preferable; next step should evaluate this tradeoff under threshold-policy CIs.
