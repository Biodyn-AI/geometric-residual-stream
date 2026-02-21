# Alternative Geometry Feature Audit (Cycle 9)

Date: 2026-02-20

## Question
Are lung-domain failures under raw cosine caused by metric choice rather than absence of geometric signal?

## Method
- Added `implementation/scripts/evaluate_alt_geometry_features.py`.
- Evaluated four geometry features on fixed best layers per domain:
  - `cosine`
  - `centered_cosine`
  - `neg_l2`
  - `dot`
- Compared baseline vs baseline+feature using repeated stratified CV and bootstrap CI for delta AUROC.

## Seed-42 feature comparison by domain

| Domain | Layer | Best feature | Delta CV AUROC | Bootstrap 95% CI |
|---|---:|---|---:|---:|
| Kidney (`cycle1_main`) | 4 | `dot` | 0.08875 | [0.03387, 0.13387] |
| Immune (`cycle4_immune_main`) | 0 | `centered_cosine` | 0.02622 | [0.00716, 0.04407] |
| Lung (`cycle6_lung_main`) | 0 | `centered_cosine` | 0.00793 | [0.00308, 0.01327] |
| External lung (`cycle7_external_lung_main`) | 3 | `neg_l2` | 0.00136 | [-0.00018, 0.00281] |

## Lung rescue check (centered cosine)
- Runs:
  - `cycle6_lung_main/alt_geometry_metrics_layer0_lung.csv`
  - `cycle6_lung_seed43/alt_geometry_metrics_layer0_lung_seed43.csv`
  - `cycle6_lung_seed44/alt_geometry_metrics_layer0_lung_seed44.csv`
- Centered-cosine deltas:
  - seed42: 0.00793
  - seed43: 0.00970
  - seed44: 0.01163
- Aggregate:
  - mean delta: 0.00975
  - std: 0.00185
  - range: [0.00793, 0.01163]
  - fraction with bootstrap CI lower > 0: 1.00

## Null controls for alternative features
- Lung centered-cosine (seed42):
  - Geometry-shuffle null: p `< 1/60`
    - `implementation/outputs/cycle9_lung_centered_cosine_shuffle_null/geometry_shuffle_summary.csv`
  - Label-permutation null: p `< 1/40`
    - `implementation/outputs/cycle9_lung_centered_cosine_label_perm_null/label_permutation_summary.csv`
- External-lung neg_l2:
  - seed42:
    - Geometry-shuffle p `< 1/60`
    - Label-permutation p = 0.30
  - seed43:
    - Geometry-shuffle p `< 1/60`
    - Label-permutation p = 0.05

## Interpretation
- Lung is not universally null: raw cosine misses a recoverable signal that appears under centered-cosine.
- This indicates part of the earlier domain dependence was metric-induced.
- External-lung remains weak and unstable: effects are tiny and only borderline under label permutation.
