# Hard-Edge Analysis After Bundle Improvements (Cycles 22-23)

Date: 2026-02-20

## Goal
Profile residual difficult edges after the multi-layer/multi-seed scGPT improvement and quantify remaining recoverable signal.

## Cycle 22: Hard-edge profiling

### Script
- `implementation/scripts/run_hard_edge_profiling_after_bundle.py`

### Outputs
- `implementation/outputs/cycle22_hard_edge_profiling_after_bundle_m02/hard_edge_summary.csv`
- `implementation/outputs/cycle22_hard_edge_profiling_after_bundle_m02/hard_edge_baseline_profile.csv`
- `implementation/outputs/cycle22_hard_edge_profiling_after_bundle_m02/hard_edges_detailed.tsv`

### Setup
- Bundle regime: seed-ensemble L0-11 scGPT vs Geneformer.
- Hard-edge threshold: margin = 0.2 on model-probability disagreement.

### Key results
- Hard-edge fractions are small:
  - Immune: 6.45% (114 / 1768)
  - Lung: 0.87% (134 / 15384)
  - External lung: 0.60% (104 / 17214)
- Hard subsets are enriched for positives in lung/external-lung:
  - Lung hard positive rate: 0.50 vs non-hard 0.293
  - External-lung hard positive rate: 0.567 vs non-hard 0.262

Interpretation:
- After bundle improvements, remaining disagreement is concentrated in a small hard-edge region.
- Residual hard edges tend to be enriched for signal-carrying positives in lung/external-lung.

## Cycle 23: Residual model on hard edges

### Script
- `implementation/scripts/run_hard_edge_residual_model.py`

### Output
- `implementation/outputs/cycle23_hard_edge_residual_model/hard_edge_residual_model_summary.csv`

### Key results

| Domain | Group | Baseline AUC | +scProb AUC | +gfProb AUC | +bothProb AUC | both - best single |
|---|---|---:|---:|---:|---:|---:|
| Immune | hard | 0.545 | 0.906 | 0.970 | 1.000 | +0.030 |
| Immune | non-hard | 0.650 | 0.746 | 0.673 | 0.746 | -0.000 |
| Lung | hard | 0.708 | 0.997 | 0.992 | 1.000 | +0.003 |
| Lung | non-hard | 0.571 | 0.617 | 0.591 | 0.620 | +0.004 |
| External lung | hard | 0.643 | 0.968 | 0.998 | 1.000 | +0.002 |
| External lung | non-hard | 0.593 | 0.621 | 0.610 | 0.626 | +0.005 |

Interpretation:
- Hard-edge regions still contain highly recoverable signal from model-score features.
- Combining both model outputs consistently helps, especially on hard edges.

## Conclusion
- The bundling strategy removes most global gap, but a compact hard-edge subset remains.
- This residual subset is highly tractable with combined model signals, supporting a targeted residual-model strategy rather than broad pipeline changes.
