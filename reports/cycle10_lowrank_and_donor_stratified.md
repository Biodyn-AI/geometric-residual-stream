# Low-Rank Feature Extension and Donor-Stratified Check (Cycle 10)

Date: 2026-02-20

## Goal
1. Test whether low-rank projected geometry features improve signal quality.
2. Check whether immune signal remains stable when evaluating major donors separately.

## Part A: Low-rank feature extension

### Method
- Extended `implementation/scripts/evaluate_alt_geometry_features.py` with PCA-derived features:
  - `pca{k}_centered_cosine`
  - `pca{k}_neg_l2`
- Evaluated `k in {32, 64, 128}` on previously selected domain/layer pairs.

### Seed-42 best feature per domain

| Domain | Layer | Best feature | Delta CV AUROC | Bootstrap 95% CI |
|---|---:|---|---:|---:|
| Kidney (`cycle1_main`) | 4 | `pca32_centered_cosine` | 0.09863 | [0.05563, 0.14602] |
| Immune (`cycle4_immune_main`) | 0 | `pca128_centered_cosine` | 0.02886 | [0.01076, 0.04511] |
| Lung (`cycle6_lung_main`) | 0 | `pca64_centered_cosine` | 0.01298 | [0.00755, 0.01836] |
| External lung (`cycle7_external_lung_main`) | 3 | `pca64_centered_cosine` | 0.00285 | [0.00019, 0.00557] |

Interpretation:
- Low-rank centered-cosine improved every tested domain relative to raw cosine.
- Largest practical gain remains in kidney; lung now shows clear positive lift under low-rank centered geometry.

## Part B: Donor-stratified immune check (top-3 donors)

### Setup
- Built donor subsets from `tabula_sapiens_immune_subset_hpn_processed.h5ad`:
  - `TSP14` (4513 cells), `TSP25` (3261 cells), `TSP21` (3223 cells)
- Ran full layerwise audit on each donor:
  - `cycle10_immune_donor_TSP14_seed42`
  - `cycle10_immune_donor_TSP25_seed42`
  - `cycle10_immune_donor_TSP21_seed42`

### Raw-geometry best layer per donor
- `TSP14`: L5, delta 0.02886, CI [0.01244, 0.04376]
- `TSP25`: L1, delta 0.02028, CI [0.00323, 0.03696]
- `TSP21`: L0, delta 0.01836, CI [0.00017, 0.03647]

### Layer-0 low-rank feature (pca64 centered-cosine)
- `TSP14`: delta 0.03290, CI [0.01326, 0.05372]
- `TSP25`: delta 0.01580, CI [0.00215, 0.03051]
- `TSP21`: delta 0.03300, CI [0.01046, 0.05503]
- Aggregate across these 3 donors:
  - mean delta: 0.02723
  - std: 0.00990
  - fraction with CI lower > 0: 1.00

### Donor-level permutation nulls for pca64 centered-cosine (layer 0, n_perm=120)
- `TSP14`: p = 0.0167
- `TSP25`: p = 0.075
- `TSP21`: p = 0.0167

Interpretation:
- Donor stratification preserves positive effect direction and CI-positive deltas.
- Statistical strength varies by donor; `TSP25` is weaker/borderline under permutation.
- The donor analysis supports robustness with heterogeneity, not full invariance.

## Additional null check (lung, pca64 centered-cosine)
- Geometry-shuffle null (n=60): p `< 1/60`
- Label-permutation null (n=40): p `< 1/40`

Interpretation:
- Lung recovery under low-rank centered-cosine is not explained by trivial feature shuffling or label randomization in the tested run.
