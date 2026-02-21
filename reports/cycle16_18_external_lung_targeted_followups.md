# External-Lung Targeted Follow-Ups (Cycles 16-18)

Date: 2026-02-20

## Goal
Execute the next focused diagnostics on external lung:
1. source-targeted scGPT seed/layer sweep,
2. calibration diagnostics,
3. disagreement-source ablation.

## Cycle 16: Source-targeted seed/layer sweep

### Script
- `implementation/scripts/run_external_lung_source_targeted_sweep.py`

### Outputs
- `implementation/outputs/cycle16_external_lung_source_targeted_sweep/source_seed_layer_detail.csv`
- `implementation/outputs/cycle16_external_lung_source_targeted_sweep/source_layer_scgpt_aggregate.csv`
- `implementation/outputs/cycle16_external_lung_source_targeted_sweep/source_best_scgpt_vs_geneformer.csv`
- `implementation/outputs/cycle16_external_lung_source_targeted_sweep/source_sweep_stability_summary.csv`

### Setup
- Targeted top-12 disagreement-enriched external-lung sources:
  `DR1, CRX, XPC, E2F6, KAT5, SP4, TRAF6, SNIP1, ATF1, SNW1, GFI1, HCFC1`.
- scGPT evaluated across seeds 42/43/44 and layers 0-11 (`pca64_centered_cosine`).

### Result summary
- Source-level deltas are highly variable and often extreme due small sample sizes per source (many sources have ~25-50 edges and only ~3-7 positives).
- Best-layer selection frequently favors optimistic scGPT deltas under these tiny subsets, so these values are not stable effect-size estimates.

Interpretation:
- The sweep is useful for locating candidate problematic sources/layers, but not for robust per-source ranking without stronger sample constraints.

## Cycle 17: Calibration diagnostics on targeted sources

### Script
- `implementation/scripts/run_external_lung_source_calibration_diagnostics.py`

### Outputs
- `implementation/outputs/cycle17_external_lung_source_calibration/source_calibration_summary.csv`
- `implementation/outputs/cycle17_external_lung_source_calibration/source_reliability_bins.csv`
- `implementation/outputs/cycle17_external_lung_source_calibration/grouped_source_calibration_summary.csv`

### Key source-level caveat
- Per-source calibration remains sample-limited; one targeted source (`CRX`) has only 1 positive and is not estimable for AUROC-based calibration.

### Grouped stability check (more reliable)

| Group | n_pairs | n_positive | scGPT AUROC | Geneformer AUROC | Gap (GF - scGPT) |
|---|---:|---:|---:|---:|---:|
| top10 disagreement sources | 312 | 52 | 0.5401 | 0.5752 | +0.0351 |
| other sources | 20353 | 5368 | 0.5860 | 0.6091 | +0.0231 |
| all | 20665 | 5420 | 0.5848 | 0.6077 | +0.0229 |

Interpretation:
- On stable grouped slices, Geneformer remains better calibrated/predictive than scGPT.
- The gap is present both inside and outside top disagreement source groups.

## Cycle 18: Disagreement-source ablation test

### Script
- `implementation/scripts/run_external_lung_source_ablation_gap.py`

### Output
- `implementation/outputs/cycle18_external_lung_source_ablation_gap/external_lung_source_ablation_gap_summary.csv`

### Result
- Baseline scenario:
  - scGPT delta: 0.00308
  - Geneformer delta: 0.02596
  - Gap: +0.02288
- After excluding disagreement-enriched sources:
  - `exclude_top10`: gap +0.02307
  - `exclude_top20`: gap +0.02291
  - `exclude_top50`: gap +0.02218
  - `exclude_ratio_ge_2`: gap +0.02333

Interpretation:
- Removing high-disagreement source groups does not materially reduce the cross-model gap.
- The external-lung gap appears diffuse across the edge space, not confined to a small source subset.

## Current takeaway
- Targeted follow-ups reinforce the earlier conclusion:
  - the external-lung scGPT vs Geneformer gap is real on shared edges,
  - it is not explained by gene coverage mismatch,
  - it is not removed by simple source-group ablation,
  - and grouped calibration checks still favor Geneformer.
