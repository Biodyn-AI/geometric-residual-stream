# Cross-Model Divergence Diagnostics (Cycle 13)

Date: 2026-02-20

## Goal
Test whether the scGPT vs Geneformer gap (especially in external lung) is explained by:
1. Gene coverage mismatch.
2. Edge sampling mismatch.
3. Feature-level disagreement on shared edges.

## Part A: Matched-gene restriction test

### Script and output
- Script: `implementation/scripts/run_cross_model_matched_gene_diagnostic.py`
- Output: `implementation/outputs/cycle13_cross_model_matched_gene_diagnostic/cross_model_matched_gene_summary.csv`

### Result (`scGPT pca64_centered_cosine`)

| Domain | Matched pair fraction | Delta (all valid) | Delta (matched genes) | Shift (matched - all) |
|---|---:|---:|---:|---:|
| Immune | 1.0000 | 0.02758 | 0.02758 | +0.00000 |
| Lung | 0.9906 | 0.01298 | 0.01353 | +0.00055 |
| External lung | 0.9923 | 0.00285 | 0.00239 | -0.00046 |

Interpretation:
- Geneformer-mapped coverage is already near-complete for lung/external-lung edges under this setup.
- Restricting scGPT to matched genes does not recover external-lung signal.
- Coverage mismatch is not the primary cause of the cross-model gap.

## Part B: Same-edge concordance and delta gap

### Script and output
- Script: `implementation/scripts/run_cross_model_edge_score_concordance.py`
- Output: `implementation/outputs/cycle13_cross_model_edge_concordance/cross_model_edge_concordance_summary.csv`

### Evaluation setup
- Used Geneformer edge datasets as the common edge universe.
- Compared:
  - `baseline + scGPT pca64_centered_cosine`
  - `baseline + Geneformer centered_cosine`
- Computed score correlations between scGPT and Geneformer features.

### Result

| Domain | scGPT delta | Geneformer delta | Delta gap (GF - scGPT) | Spearman(all edges) |
|---|---:|---:|---:|---:|
| Immune | 0.02758 | 0.05023 | +0.02265 | 0.47565 |
| Lung | 0.01285 | 0.02591 | +0.01306 | 0.47996 |
| External lung | 0.00308 | 0.02596 | +0.02289 | 0.46698 |

Interpretation:
- Features are moderately aligned (Spearman ~0.47), but Geneformer has materially larger incremental predictive lift.
- External-lung divergence persists even on identical edges, so it is not a negative-sampling artifact.
- The gap is now best framed as a model-specific representation/calibration difference rather than coverage.

## Current diagnostic conclusion
- Coverage mismatch: mostly ruled out.
- Edge sampling mismatch: mostly ruled out.
- Remaining explanation: model-specific feature quality/calibration on the same regulatory pairs.

## Immediate next experiments
1. Calibrated feature fusion:
- Evaluate `baseline + scGPT + Geneformer` on shared edges to test complementarity vs redundancy.
2. Error stratification:
- Partition edges by agreement/disagreement quantiles and inspect TF-family enrichment.
3. External-lung targeted robustness:
- Repeat concordance with multiple scGPT seeds/layers to test whether the gap is stable or layer-sensitive.
