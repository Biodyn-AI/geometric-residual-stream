# Cross-Domain Generalization and Null Controls (Cycles 4-8)

Date: 2026-02-20

## Goal
Test whether residual-stream geometric signal beyond gene-level confounds generalizes across datasets and remains detectable under stronger null controls.

## Domain-level aggregate results (3 seeds where available)

| Domain | Aggregate file | Best layer | Mean delta CV AUROC | Std | Range | Frac CI lower > 0 |
|---|---|---:|---:|---:|---:|---:|
| Kidney | `implementation/outputs/cycle1_aggregate/cycle1_layer_aggregate.csv` | L4 | 0.08953 | 0.02438 | [0.06396, 0.11251] | 1.00 |
| Immune subset | `implementation/outputs/cycle4_immune_aggregate/cycle4_immune_layer_aggregate.csv` | L0 | 0.03329 | 0.00736 | [0.02512, 0.03938] | 1.00 |
| Lung | `implementation/outputs/cycle6_lung_aggregate/cycle6_lung_layer_aggregate.csv` | L0 | 0.00073 | 0.00137 | [-0.00035, 0.00227] | 0.00 |
| External lung | `implementation/outputs/cycle7_external_lung_aggregate/cycle7_external_lung_layer_aggregate.csv` | L3 | 0.00183 | 0.00108 | [0.00074, 0.00290] | 0.67 |

## Stronger null controls

### Geometry-feature shuffle null
- Kidney (L5): true delta 0.07460 vs shuffle mean 0.00303, p `< 1/60`.
  - File: `implementation/outputs/cycle5_kidney_geom_shuffle_null/geometry_shuffle_summary.csv`
- Immune (L0): true delta 0.02344 vs shuffle mean -0.00134, p `< 1/60`.
  - File: `implementation/outputs/cycle5_immune_geom_shuffle_null/geometry_shuffle_summary.csv`
- Lung (L0): true delta -0.00039, p = 0.65.
  - File: `implementation/outputs/cycle8_lung_geom_shuffle_null/geometry_shuffle_summary.csv`
- External lung (L3): true delta 0.00065, p = 0.0167.
  - File: `implementation/outputs/cycle8_external_lung_geom_shuffle_null/geometry_shuffle_summary.csv`

### Label-permutation null
- Kidney (L5): p = 0.025.
  - File: `implementation/outputs/cycle3_null/label_permutation_summary.csv`
- Immune (L0): p = 0.025.
  - File: `implementation/outputs/cycle4_immune_null/label_permutation_summary.csv`
- Lung (L0): p = 0.425.
  - File: `implementation/outputs/cycle8_lung_label_perm_null/label_permutation_summary.csv`
- External lung (L3): p = 0.325.
  - File: `implementation/outputs/cycle8_external_lung_label_perm_null/label_permutation_summary.csv`

## Interpretation
- Kidney and immune show consistent positive incremental geometry signal.
- Lung is negative under both aggregate robustness and null controls.
- External lung shows at most tiny effects, with mixed control outcomes and negligible practical lift.
- Current evidence supports domain-dependent geometry signal rather than a universal effect.

## Update note
- Subsequent Cycle 9 metric-sensitivity tests found that lung is recoverable with `centered_cosine` (see `reports/cycle9_alternative_geometry_features.md`), indicating part of the earlier domain gap is metric-induced.

## Immediate next steps
1. Quantify whether domain dependence tracks baseline separability (`baseline_cv_auroc`) and mapped TRRUST coverage.
2. Add alternative geometry features (e.g., centered cosine, low-rank projection distances) and re-run on lung/external lung.
3. Evaluate donor-aware and cell-type-stratified splits to test whether the kidney/immune signal is composition-driven.
