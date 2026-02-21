# Cell-Type Stratification and Geneformer Replication (Cycles 11-12)

Date: 2026-02-20

## Goal
1. Test whether low-rank geometric signal is stable within major cell-type strata.
2. Replicate the geometry effect in a second model family (Geneformer embeddings).
3. Add null controls for the Geneformer effect.

## Part A: Cell-type stratified scGPT checks (Cycle 11)

### Setup
- Script for subset construction: `implementation/scripts/make_obs_subsets.py`
- Analysis protocol: full layerwise audit + low-rank feature scan on layer 0 (`pca64_centered_cosine`).
- Subsets:
  - Immune: `b_cell` (3762 cells), `cd4_positive_alpha_beta_t_cell` (3373), `cd8_positive_alpha_beta_t_cell` (2547)
  - Lung: `macrophage` (4547), `pulmonary_alveolar_type_2_cell` (3238), `pulmonary_alveolar_type_1_cell` (1798)

### Layer-0 low-rank results (`pca64_centered_cosine`)

| Domain | Cell type | Delta CV AUROC | Bootstrap 95% CI |
|---|---|---:|---:|
| Immune | B cell | 0.03370 | [0.01781, 0.05212] |
| Immune | CD4 T | 0.04025 | [0.01617, 0.06806] |
| Immune | CD8 T | 0.06725 | [0.04291, 0.09331] |
| Lung | Macrophage | 0.00793 | [0.00295, 0.01367] |
| Lung | Alveolar type 2 | 0.00637 | [-0.00152, 0.01377] |
| Lung | Alveolar type 1 | -0.00077 | [-0.00451, 0.00260] |

Aggregate summaries (`implementation/outputs/cycle11_celltype_pca64_centered_aggregate.csv`):
- Immune mean delta: 0.04707 (std 0.01778), CI-lower>0 fraction: 1.00
- Lung mean delta: 0.00451 (std 0.00464), CI-lower>0 fraction: 0.333

Interpretation:
- Immune signal remains clearly positive across major immune cell types.
- Lung signal remains weak and composition-dependent under this scGPT protocol.

## Part B: External-lung low-rank nulls across seeds (Cycle 11 extension)

Summary file: `implementation/outputs/cycle11_external_lung_pca64_null_summary_all_seeds.csv`

| Seed | True delta | Shuffle p-value | Label-permutation p-value |
|---:|---:|---:|---:|
| 42 | 0.002846 | 0.000 | 0.175 |
| 43 | 0.000272 | 0.050 | 0.225 |
| 44 | 0.002238 | 0.000 | 0.175 |

Interpretation:
- External-lung scGPT effects stay tiny.
- Evidence is unstable under label permutation despite frequent separation from geometry shuffles.

## Part C: Geneformer embedding replication (Cycle 12)

### Setup
- Script: `implementation/scripts/run_geneformer_embedding_geometry_audit.py`
- Features: `cosine`, `centered_cosine`, `dot` on Geneformer input embeddings.
- Best feature in all domains: `centered_cosine`.

| Domain | Genes mapped | Best delta CV AUROC | Bootstrap 95% CI |
|---|---:|---:|---:|
| Immune | 2047 / 4941 (41.43%) | 0.04794 | [0.02862, 0.06369] |
| Lung | 6803 / 8181 (83.16%) | 0.03934 | [0.03326, 0.04533] |
| External lung | 6355 / 8229 (77.23%) | 0.04626 | [0.03970, 0.05251] |

Summary file: `implementation/outputs/cycle12_geneformer_best_feature_summary.csv`

Interpretation:
- Geneformer shows strong positive geometry signal in all three domains, including external lung.

## Part D: Geneformer null controls (Cycle 12 extension)

### New script
- `implementation/scripts/run_geneformer_feature_null_controls.py`
- Nulls:
  - Geometry-feature shuffle (`n_shuffles=60`)
  - Label permutation (`n_perm=40`)

Summary file: `implementation/outputs/cycle12_geneformer_centered_null_summary_all_domains.csv`

| Domain | True delta | Shuffle p-value | Label-permutation p-value |
|---|---:|---:|---:|
| Immune | 0.047939 | 0.000 | 0.000 |
| Lung | 0.039342 | 0.000 | 0.000 |
| External lung | 0.046262 | 0.000 | 0.000 |

Interpretation:
- Under the tested null grid, Geneformer centered-cosine gains are strongly non-null in every domain.
- This sharply contrasts with weak external-lung scGPT results and motivates targeted model-comparison diagnostics.

## Current conclusion
- scGPT: strong kidney, moderate immune, weak lung/external-lung with meaningful metric and subgroup dependence.
- Geneformer embeddings: robust positive geometry signal across immune, lung, and external lung with strong null-control support.
- Next research focus should explain the cross-model divergence, especially on external lung.
