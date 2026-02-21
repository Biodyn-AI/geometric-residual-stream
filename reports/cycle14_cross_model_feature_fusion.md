# Cross-Model Feature Fusion on Shared Edges (Cycle 14)

Date: 2026-02-20

## Goal
Test whether scGPT and Geneformer geometry features are complementary on the same edge sets.

## Method
- Script: `implementation/scripts/run_cross_model_feature_fusion.py`
- Shared edge universe: Geneformer edge datasets per domain.
- Models compared:
  - baseline only
  - baseline + scGPT (`pca64_centered_cosine`)
  - baseline + Geneformer (`centered_cosine`)
  - baseline + both features
- Output: `implementation/outputs/cycle14_cross_model_feature_fusion/cross_model_feature_fusion_summary.csv`

## Results

| Domain | Baseline+scGPT delta | Baseline+Geneformer delta | Baseline+Both delta | Both vs best single |
|---|---:|---:|---:|---:|
| Immune | 0.02758 | 0.05023 | 0.04929 | -0.00094 |
| Lung | 0.01285 | 0.02591 | 0.02677 | +0.00087 |
| External lung | 0.00308 | 0.02596 | 0.02630 | +0.00034 |

## Interpretation
- Geneformer remains the dominant single feature across all domains.
- Fusion adds only small incremental value in lung/external-lung and none in immune.
- Combined with Cycle 13 concordance, this suggests:
  - moderate shared information between models,
  - limited extra information from scGPT on top of Geneformer in current setup.

## Conclusion
- Cross-model fusion does not remove the external-lung gap; it only yields marginal gains.
- The next high-value step is disagreement-stratified analysis (which edges benefit from each model) rather than broader feature mixing.
