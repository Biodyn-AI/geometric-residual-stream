# Hard-Edge Calibration And Margin Sensitivity (Cycle 24)

Date: 2026-02-20

## Goal
Quantify how hard-edge behavior changes with disagreement margin and test whether hard-edge calibration gaps persist after bundle improvements.

## Script
- `implementation/scripts/run_hard_edge_calibration_margin_sweep.py`

## Outputs
- `implementation/outputs/cycle24_hard_edge_calibration_margin_sweep/hard_edge_margin_summary.csv`
- `implementation/outputs/cycle24_hard_edge_calibration_margin_sweep/hard_edge_calibration_summary.csv`
- `implementation/outputs/cycle24_hard_edge_calibration_margin_sweep/hard_edge_reliability_bins.csv`
- `implementation/outputs/cycle24_hard_edge_calibration_margin_sweep/hard_edge_calibration_gap_summary.csv`
- `implementation/outputs/cycle24_hard_edge_calibration_margin_sweep/hard_edge_residual_margin_sweep_summary.csv`
- `implementation/outputs/cycle24_hard_edge_calibration_margin_sweep/hard_edge_residual_hardgroup_compact_view.csv`

## Setup
- Domain set: immune, lung, external-lung.
- Feature regime: seed-ensemble scGPT L0-11 bundle vs Geneformer centered-cosine setup on shared valid edges.
- Margins swept: `0.1`, `0.2`, `0.3`.
- Calibration metric: ECE with 10 equal-width bins.

## Key Results

### 1) Hard-edge coverage drops sharply with stricter margin
- Immune hard fraction: `21.78%` (`m=0.1`) -> `6.45%` (`m=0.2`) -> `1.70%` (`m=0.3`).
- Lung hard fraction: `8.79%` -> `0.87%` -> `0.08%`.
- External-lung hard fraction: `8.62%` -> `0.60%` -> `0.06%`.

Interpretation:
- Large margins isolate very small high-confidence disagreement subsets, especially in lung/external-lung.

### 2) Hard subsets are calibration hotspots, especially for scGPT
Examples at `m=0.2`:
- Immune:
  - scGPT ECE: hard `0.4329` vs non-hard `0.1776`
  - Geneformer ECE: hard `0.2140` vs non-hard `0.1890`
- Lung:
  - scGPT ECE: hard `0.5144` vs non-hard `0.1986`
  - Geneformer ECE: hard `0.3352` vs non-hard `0.1998`
- External-lung:
  - scGPT ECE: hard `0.4009` vs non-hard `0.2270`
  - Geneformer ECE: hard `0.2738` vs non-hard `0.2274`

Interpretation:
- Hard-edge degradation is systematic for both models, but substantially larger for scGPT.

### 3) Residual-model margin sensitivity
Additional runs:
- `m=0.1`: `implementation/outputs/cycle24_hard_edge_residual_model_m01/hard_edge_residual_model_summary.csv`
- `m=0.3`: `implementation/outputs/cycle24_hard_edge_residual_model_m03/hard_edge_residual_model_summary.csv`
- Combined with prior `m=0.2` into: `hard_edge_residual_margin_sweep_summary.csv`.

Observed pattern:
- Combined-score residual model (`baseline + scProb + gfProb`) remains strongest on hard subsets for all margins.
- Hard-subset gains over baseline remain large:
  - Immune: `+0.418` (`m=0.1`), `+0.455` (`m=0.2`), `+0.484` (`m=0.3`)
  - Lung: `+0.406`, `+0.292`, `+0.350`
  - External-lung: `+0.434`, `+0.357`, `+0.292`
- At `m=0.3`, lung/external-lung hard sets are tiny (`n=12` and `n=11`), so AUC values are high-variance.

## Caveat
- Hard edges are defined from model probability disagreement and label agreement, then evaluated within that subset; this can make hard-subset performance optimistic.
- Next iteration should use an outer-split protocol where hard-edge selection is learned on train folds and evaluated on held-out folds only.

## Conclusion
- Hard-edge behavior is margin-consistent: stricter margins reduce coverage but isolate stronger calibration failure zones.
- Residual modeling remains a strong strategy, but the next priority is leakage-resistant deployment evaluation rather than further feature expansion.
