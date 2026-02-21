# External-Lung Layer-Bundle Test (Cycle 19)

Date: 2026-02-20

## Goal
Test whether using stacked scGPT geometry from multiple layers reduces the external-lung gap vs Geneformer on shared edges.

## Script and output
- Script: `implementation/scripts/run_external_lung_layer_bundle_test.py`
- Output: `implementation/outputs/cycle19_external_lung_layer_bundle_test/external_lung_layer_bundle_summary.csv`

## Results

| scGPT bundle | scGPT delta | Geneformer delta | Gap (GF - scGPT) | Both delta | Both - Geneformer |
|---|---:|---:|---:|---:|---:|
| L3 | 0.00308 | 0.02596 | +0.02288 | 0.02630 | +0.00034 |
| L1-4 | 0.01193 | 0.02596 | +0.01403 | 0.03009 | +0.00413 |
| L0-5 | 0.01656 | 0.02596 | +0.00940 | 0.03338 | +0.00742 |
| L0-11 | 0.02265 | 0.02596 | +0.00332 | 0.03804 | +0.01208 |

## Interpretation
- Multi-layer scGPT features substantially improve external-lung performance.
- The scGPT-vs-Geneformer gap shrinks from +0.02288 (single L3) to +0.00332 (L0-11 bundle).
- Combined model (`baseline + scGPT-bundle + Geneformer`) now provides clear additional gain over Geneformer alone, indicating complementary information in multi-layer scGPT geometry.

## Conclusion
- The external-lung gap is not fixed; a richer scGPT representation (layer bundle) largely closes it.
- Next high-value step is testing whether this improvement is stable across scGPT seeds (seed-ensemble / multi-seed bundle check).
