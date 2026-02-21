# Cross-Model Disagreement Stratification (Cycle 15)

Date: 2026-02-20

## Goal
Identify where scGPT vs Geneformer edge-score disagreement concentrates and whether high-disagreement regions have distinct label behavior.

## Method
- Script: `implementation/scripts/run_cross_model_disagreement_stratification.py`
- Shared edge universe: Geneformer edge datasets (immune/lung/external-lung).
- Score inputs:
  - scGPT: `pca64_centered_cosine` at domain-specific layer (immune/lung L0, external-lung L3)
  - Geneformer: centered-cosine input-embedding score
- Disagreement metric: `abs(z_scgpt - z_geneformer)`.
- Outputs:
  - `implementation/outputs/cycle15_cross_model_disagreement_stratification/disagreement_quantile_label_rates.csv`
  - `implementation/outputs/cycle15_cross_model_disagreement_stratification/source_enrichment_top_disagreement_bin.csv`

## Label-rate pattern by disagreement decile

### Immune
- Positive rate fluctuates, with highest-disagreement bin still high (`0.3349` in bin 9).
- Interpretation: disagreement in immune is not simply a failure zone.

### Lung
- Positive rate declines from low to high disagreement:
  - bin 0: `0.3065`
  - bin 9: `0.2542`
- Interpretation: high-disagreement edges are enriched for negatives in lung.

### External lung
- Similar decline with mild rebound at top bin:
  - bin 0: `0.2830`
  - bin 8: `0.2522`
  - bin 9: `0.2627`
- Interpretation: high disagreement generally tracks harder/less-positive edges in external lung.

## Source-TF enrichment in top disagreement bin

Examples of high-ratio sources (`top_bin_share / overall_share`):
- Immune: `GATA1` (3.49x), `KLF1` (3.11x), `NR5A1` (2.30x)
- Lung: `SLA2` (6.92x), `ZNF175` (5.77x), `HNF4A` (5.38x)
- External lung: `DR1` (4.00x), `CRX` (3.79x), `XPC` (3.44x)

## Interpretation
- Disagreement is structured, not random, and concentrates around specific source TFs.
- In lung/external-lung, high disagreement coincides with lower positive-edge prevalence, consistent with a model-calibration gap on difficult edges.
- The enrichment tables provide concrete candidates for targeted per-TF diagnostics.

## Next step
- Run per-TF stability sweeps across scGPT layers/seeds for top disagreement-enriched sources, then re-evaluate the shared-edge delta gap.
