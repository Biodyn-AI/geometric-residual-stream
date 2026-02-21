# Research Log

## 2026-02-20

### Setup
- Created subproject workspace `subproject_38_geometric_residual_stream_interpretability`.
- Wrote a concrete plan in `planning/research_plan.md` focused on layer-wise geometric signal in scGPT residual stream.
- Provisioned dedicated runtime environment `conda` env: `subproject38-geo` (separate from base env).

### Environment hardening update
- Created a fresh isolated env: `subproject38-geo-v2`.
- Added reproducible manifests under `implementation/environment/`:
  - `environment.yml` (portable spec with pinned channels + versions)
  - `conda-explicit-osx-arm64.txt` (exact lock for this machine class)
  - `environment.from-history.yml` (historical requested set)
- Re-ran the geometry audit smoke test fully in `subproject38-geo-v2`:
  - Output root: `implementation/outputs/smoke_v2_env/`
  - End-to-end execution succeeded (data load, model load, residual extraction, metrics writeout).
- Re-ran full Cycle 1 in `subproject38-geo-v2` with production settings (`max_cells=256`, `max_genes=512`, seed 42):
  - Output root: `implementation/outputs/cycle1_main_v2env_seed42/`
  - Result matches original Cycle 1 metrics exactly (row-wise equality in `cycle1_layer_geometry_metrics.csv`).

### Cycle 1: Layer-wise geometric audit (seed 42, max_genes=512)
- Run artifact root: `implementation/outputs/cycle1_main/`
- Dataset: 256 sampled kidney cells, 4803 genes (processed matrix), 12 transformer layers.
- Edge dataset: 288 TRRUST positives + 864 matched random negatives.
- Best layer (by delta CV AUROC): L5
  - Baseline CV AUROC: 0.5153
  - Baseline+Geometry CV AUROC: 0.5795
  - Delta CV AUROC: +0.0642
  - Bootstrap 95% CI for delta: [0.0183, 0.1108]

Interpretation:
- Residual geometry adds measurable predictive signal beyond gene-level confounds in this setting.
- Signal is strongest in early-mid layers (roughly L3-L6), then declines in late layers.

### Cycle 1b: Seeded robustness replicates (seeds 43, 44; max_genes=512)
- Run artifacts: `implementation/outputs/cycle1_seed43/`, `implementation/outputs/cycle1_seed44/`
- Best layer per run:
  - Seed 42: L5, delta +0.0642
  - Seed 43: L0, delta +0.1216
  - Seed 44: L5, delta +0.0925
- Aggregate across seeds (see `implementation/outputs/cycle1_aggregate/`):
  - Best mean layer: L4
  - Mean delta CV AUROC: +0.0895
  - Std: 0.0244
  - Range: [0.0640, 0.1125]
  - Fraction replicates with bootstrap CI lower bound > 0 at L4: 1.00

Interpretation:
- Positive geometric incremental value replicates across seeds.
- Exact peak layer shifts modestly, but high-signal band remains early-mid layers.

### Cycle 2: Sensitivity to token budget (max_genes=1024)
- Run artifact root: `implementation/outputs/cycle2_maxgenes1024/`
- Best layer: L4
  - Baseline CV AUROC: 0.5077
  - Baseline+Geometry CV AUROC: 0.5565
  - Delta CV AUROC: +0.0488
  - Bootstrap 95% CI: [0.0037, 0.0910]

Interpretation:
- Increasing token budget attenuates effect size versus max_genes=512 but does not remove it.
- The geometric signal remains positive and still peaks in the same early-mid zone.

### Cycle 3: Label-permutation null control
- Run artifact root: `implementation/outputs/cycle3_null/`
- Evaluated layer: L5 (from cycle1 main run)
- Null setup: 40 label permutations with identical feature/model protocol.
- Results:
  - True delta CV AUROC: 0.0746
  - Null mean delta: 0.0078 (std 0.0201)
  - Empirical right-tail p-value: 0.025

Interpretation:
- Observed improvement is unlikely to be explained by modeling artifact alone under this null.

### Cycle 4: Cross-domain extension on immune subset
- Main run artifact: `implementation/outputs/cycle4_immune_main/`
- Dataset: `tabula_sapiens_immune_subset_hpn_processed.h5ad` (256 sampled cells, 4941 genes).
- Best layer in main run: L0
  - Baseline CV AUROC: 0.6308
  - Baseline+Geometry CV AUROC: 0.6559
  - Delta CV AUROC: +0.0251
  - Bootstrap 95% CI: [0.0062, 0.0412]
- Replicates: `cycle4_immune_seed43`, `cycle4_immune_seed44`
- Aggregate (`cycle4_immune_aggregate`):
  - Best mean layer: L0
  - Mean delta CV AUROC: +0.0333 (std 0.0074; range [0.0251, 0.0394])
  - Fraction replicates with CI lower bound > 0: 1.00

Interpretation:
- Geometry signal generalizes to an immune-focused dataset, but with materially smaller effect size than kidney.
- Peak layer shifts earlier (L0-L2 region).

### Cycle 5: Geometry-feature shuffle null (stronger control)
- Added script: `implementation/scripts/geometry_feature_shuffle_null.py`
- Kidney (layer 5): `cycle5_kidney_geom_shuffle_null`
  - True delta: 0.0746
  - Shuffle mean delta: 0.0030 (std 0.0128)
  - Empirical right-tail p: `< 1/60` (reported as 0.0 due finite shuffle grid)
- Immune (layer 0): `cycle5_immune_geom_shuffle_null`
  - True delta: 0.0234
  - Shuffle mean delta: -0.0013 (std 0.0016)
  - Empirical right-tail p: `< 1/60` (reported as 0.0 due finite shuffle grid)

Interpretation:
- Pairwise alignment structure in the geometry feature, not just its marginal distribution, carries signal in kidney and immune.

### Cycle 6: Lung domain extension
- Main + replicate artifacts:
  - `implementation/outputs/cycle6_lung_main/`
  - `implementation/outputs/cycle6_lung_seed43/`
  - `implementation/outputs/cycle6_lung_seed44/`
- Aggregate (`cycle6_lung_aggregate`):
  - Best mean layer: L0
  - Mean delta CV AUROC: +0.00073 (std 0.00137; range [-0.00035, 0.00227])
  - Fraction replicates with CI lower bound > 0: 0.00

Interpretation:
- No practically useful or robust incremental geometry signal was found in this lung dataset under the current protocol.

### Cycle 7: External lung extension
- Main + replicate artifacts:
  - `implementation/outputs/cycle7_external_lung_main/`
  - `implementation/outputs/cycle7_external_lung_seed43/`
  - `implementation/outputs/cycle7_external_lung_seed44/`
- Aggregate (`cycle7_external_lung_aggregate`):
  - Best mean layer: L3
  - Mean delta CV AUROC: +0.00183 (std 0.00108; range [0.00074, 0.00290])
  - Fraction replicates with CI lower bound > 0: 0.67

Interpretation:
- External lung shows at most a very small effect size; statistical detectability may occur due large edge counts, but practical lift is negligible.

### Cycle 8: Additional nulls on lung datasets
- Lung controls:
  - Geometry-shuffle: `cycle8_lung_geom_shuffle_null` -> p = 0.65
  - Label-permutation: `cycle8_lung_label_perm_null` -> p = 0.425
- External lung controls:
  - Geometry-shuffle: `cycle8_external_lung_geom_shuffle_null` -> p = 0.0167
  - Label-permutation: `cycle8_external_lung_label_perm_null` -> p = 0.325

Interpretation:
- Lung remains null by both controls.
- External lung exhibits weak/inconsistent evidence: detectable against geometry-shuffle in one run, but not against label permutation.

### Cycle 9: Alternative geometry feature audit (metric sensitivity)
- Added script: `implementation/scripts/evaluate_alt_geometry_features.py`
- Evaluated `cosine`, `centered_cosine`, `neg_l2`, `dot` on fixed best layers.
- Seed-42 best feature by domain:
  - Kidney (L4): `dot`, delta 0.0887, bootstrap CI [0.0339, 0.1339]
  - Immune (L0): `centered_cosine`, delta 0.0262, CI [0.0072, 0.0441]
  - Lung (L0): `centered_cosine`, delta 0.0079, CI [0.0031, 0.0133]
  - External lung (L3): `neg_l2`, delta 0.00136, CI [-0.00018, 0.00281]
- Lung centered-cosine replicate check across seeds 42/43/44:
  - Deltas: 0.00793, 0.00970, 0.01163
  - Mean 0.00975, std 0.00185, range [0.00793, 0.01163]
  - Fraction with bootstrap CI lower > 0: 1.00
- Alternative-feature null controls:
  - Lung centered-cosine: geometry-shuffle p `< 1/60`, label-permutation p `< 1/40`
  - External-lung neg_l2:
    - seed42: shuffle p `< 1/60`, label-permutation p = 0.30
    - seed43: shuffle p `< 1/60`, label-permutation p = 0.05

Interpretation:
- A substantial part of the lung failure under raw cosine was metric-induced; centered-cosine recovers consistent positive signal.
- External-lung remains weak with tiny absolute effect sizes and unstable permutation evidence.

### Cycle 10: Low-rank feature extension + donor-stratified immune checks
- Extended feature audit script with PCA-derived variants:
  - `pca{k}_centered_cosine`, `pca{k}_neg_l2` for `k in {32, 64, 128}`.
- Seed-42 best low-rank feature outcomes:
  - Kidney (L4): `pca32_centered_cosine`, delta 0.0986, CI [0.0556, 0.1460]
  - Immune (L0): `pca128_centered_cosine`, delta 0.0289, CI [0.0108, 0.0451]
  - Lung (L0): `pca64_centered_cosine`, delta 0.0130, CI [0.0075, 0.0184]
  - External lung (L3): `pca64_centered_cosine`, delta 0.00285, CI [0.00019, 0.00557]
- Donor-stratified immune subset analysis (top donors: TSP14/TSP25/TSP21):
  - Built donor-specific h5ad subsets and reran full layerwise pipeline.
  - Layer-0 `pca64_centered_cosine` deltas:
    - TSP14: 0.0329, CI [0.0133, 0.0537]
    - TSP25: 0.0158, CI [0.0022, 0.0305]
    - TSP21: 0.0330, CI [0.0105, 0.0550]
  - Aggregate donor mean: 0.0272 (std 0.0099), fraction CI lower > 0: 1.00
- Null checks:
  - Lung `pca64_centered_cosine`:
    - geometry-shuffle p `< 1/60`
    - label-permutation p `< 1/40`
  - Donor permutation null for `pca64_centered_cosine` (n=120):
    - TSP14 p = 0.0167
    - TSP25 p = 0.075
    - TSP21 p = 0.0167

Interpretation:
- Low-rank centered-cosine strengthens or preserves signal in all tested domains.
- Immune donor splits maintain positive effect direction but show expected donor heterogeneity in statistical strength.

### Cycle 11: Cell-type stratified low-rank checks + external-lung all-seed nulls
- Added subset builder script: `implementation/scripts/make_obs_subsets.py`
- Built top cell-type subsets:
  - Immune: B cell (3762 cells), CD4 T (3373), CD8 T (2547)
  - Lung: macrophage (4547), alveolar type 2 (3238), alveolar type 1 (1798)
- Ran full layerwise audits per subset plus layer-0 low-rank feature scan.
- `pca64_centered_cosine` results:
  - Immune:
    - B cell: delta 0.03370, CI [0.01781, 0.05212]
    - CD4 T: delta 0.04025, CI [0.01617, 0.06806]
    - CD8 T: delta 0.06725, CI [0.04291, 0.09331]
    - Aggregate mean 0.04707, std 0.01778, CI-lower>0 fraction 1.00
  - Lung:
    - Macrophage: delta 0.00793, CI [0.00295, 0.01367]
    - Alveolar type 2: delta 0.00637, CI [-0.00152, 0.01377]
    - Alveolar type 1: delta -0.00077, CI [-0.00451, 0.00260]
    - Aggregate mean 0.00451, std 0.00464, CI-lower>0 fraction 0.333
- External-lung low-rank null controls across all seeds (`pca64_centered_cosine`, layer 3):
  - seed42: delta 0.002846, shuffle p 0.000, label-perm p 0.175
  - seed43: delta 0.000272, shuffle p 0.050, label-perm p 0.225
  - seed44: delta 0.002238, shuffle p 0.000, label-perm p 0.175
- Added consolidated summaries:
  - `implementation/outputs/cycle11_celltype_pca64_centered_summary.csv`
  - `implementation/outputs/cycle11_celltype_pca64_centered_aggregate.csv`
  - `implementation/outputs/cycle11_external_lung_pca64_null_summary_all_seeds.csv`

Interpretation:
- Immune low-rank signal is robust across major cell types.
- Lung remains weak and heterogeneous.
- External-lung scGPT effect stays tiny and unstable under label permutation.

### Cycle 12: Geneformer embedding replication
- Added script: `implementation/scripts/run_geneformer_embedding_geometry_audit.py`
- Environment adjustments for compatibility:
  - Installed `transformers==4.41.2`, `huggingface-hub==0.36.2`, `tokenizers==0.19.1` in `subproject38-geo-v2`.
- Loaded model `ctheodoris/Geneformer` and mapped genes to token IDs via published dictionaries.
- Best feature per domain: `centered_cosine`
  - Immune:
    - Mapped genes: 2047 / 4941 (41.43%)
    - Delta 0.047939, CI [0.028616, 0.063690]
  - Lung:
    - Mapped genes: 6803 / 8181 (83.16%)
    - Delta 0.039342, CI [0.033264, 0.045330]
  - External lung:
    - Mapped genes: 6355 / 8229 (77.23%)
    - Delta 0.046262, CI [0.039696, 0.052511]
- Added consolidated summary:
  - `implementation/outputs/cycle12_geneformer_best_feature_summary.csv`

Interpretation:
- Geneformer reproduces a strong positive geometry lift across all three domains, including external lung.

### Cycle 12 extension: Geneformer null controls
- Added script: `implementation/scripts/run_geneformer_feature_null_controls.py`
- Ran centered-cosine null controls for immune/lung/external-lung:
  - Geometry-feature shuffle (`n_shuffles=60`)
  - Label permutation (`n_perm=40`)
- Results (`implementation/outputs/cycle12_geneformer_centered_null_summary_all_domains.csv`):
  - Immune: true delta 0.047939, shuffle p 0.000, perm p 0.000
  - Lung: true delta 0.039342, shuffle p 0.000, perm p 0.000
  - External-lung: true delta 0.046262, shuffle p 0.000, perm p 0.000

Interpretation:
- Geneformer gains are non-null across all tested domains under both null families.

### Cycle 13: Cross-model divergence diagnostics

#### Matched-gene restriction diagnostic
- Added script: `implementation/scripts/run_cross_model_matched_gene_diagnostic.py`
- Output: `implementation/outputs/cycle13_cross_model_matched_gene_diagnostic/cross_model_matched_gene_summary.csv`
- scGPT `pca64_centered_cosine` deltas (all valid vs matched-gene subset):
  - Immune:
    - matched fraction: 1.0000
    - delta all: 0.02758
    - delta matched: 0.02758
  - Lung:
    - matched fraction: 0.9906
    - delta all: 0.01298
    - delta matched: 0.01353
  - External-lung:
    - matched fraction: 0.9923
    - delta all: 0.00285
    - delta matched: 0.00239

Interpretation:
- Restricting to Geneformer-mapped genes does not recover external-lung scGPT signal.
- Coverage mismatch is not the main explanation for the scGPT vs Geneformer gap.

#### Shared-edge concordance diagnostic
- Added script: `implementation/scripts/run_cross_model_edge_score_concordance.py`
- Output: `implementation/outputs/cycle13_cross_model_edge_concordance/cross_model_edge_concordance_summary.csv`
- On identical Geneformer edge sets (same baseline protocol):
  - Immune:
    - scGPT delta: 0.02758
    - Geneformer delta: 0.05023
    - gap (Geneformer - scGPT): 0.02265
    - Spearman(score concordance): 0.47565
  - Lung:
    - scGPT delta: 0.01285
    - Geneformer delta: 0.02591
    - gap: 0.01306
    - Spearman: 0.47996
  - External-lung:
    - scGPT delta: 0.00308
    - Geneformer delta: 0.02596
    - gap: 0.02289
    - Spearman: 0.46698

Interpretation:
- Edge-score alignment is moderate, but Geneformer keeps a sizable predictive lift advantage.
- External-lung divergence persists even on identical edge sets, ruling out a sampling-only explanation.

### Cycle 14: Cross-model feature fusion on shared edges
- Added script: `implementation/scripts/run_cross_model_feature_fusion.py`
- Output: `implementation/outputs/cycle14_cross_model_feature_fusion/cross_model_feature_fusion_summary.csv`
- Results:
  - Immune:
    - baseline + scGPT delta: 0.02758
    - baseline + Geneformer delta: 0.05023
    - baseline + both delta: 0.04929
    - both vs best single: -0.00094
  - Lung:
    - baseline + scGPT delta: 0.01285
    - baseline + Geneformer delta: 0.02591
    - baseline + both delta: 0.02677
    - both vs best single: +0.00087
  - External-lung:
    - baseline + scGPT delta: 0.00308
    - baseline + Geneformer delta: 0.02596
    - baseline + both delta: 0.02630
    - both vs best single: +0.00034

Interpretation:
- Fusion provides only marginal extra lift in lung/external-lung and no gain in immune.
- Geneformer remains the dominant single feature across all domains.

### Cycle 15: Cross-model disagreement stratification
- Added script: `implementation/scripts/run_cross_model_disagreement_stratification.py`
- Outputs:
  - `implementation/outputs/cycle15_cross_model_disagreement_stratification/disagreement_quantile_label_rates.csv`
  - `implementation/outputs/cycle15_cross_model_disagreement_stratification/source_enrichment_top_disagreement_bin.csv`
- Key findings:
  - Lung positive rate decreases with disagreement (bin0: 0.3065 -> bin9: 0.2542).
  - External-lung positive rate is also lower in high-disagreement bins (bin0: 0.2830; bin8: 0.2522; bin9: 0.2627).
  - Immune does not follow the same decline pattern (high-disagreement bin remains relatively positive: 0.3349).
- Example high-disagreement source enrichments:
  - Immune: `GATA1` (3.49x), `KLF1` (3.11x)
  - Lung: `SLA2` (6.92x), `ZNF175` (5.77x), `HNF4A` (5.38x)
  - External-lung: `DR1` (4.00x), `CRX` (3.79x), `XPC` (3.44x)

Interpretation:
- Disagreement is structured and source-specific, not random.
- In lung/external-lung, high disagreement aligns with harder, lower-positive-rate edge regions.

### Cycle 16: External-lung targeted source seed/layer sweep
- Added script: `implementation/scripts/run_external_lung_source_targeted_sweep.py`
- Outputs:
  - `implementation/outputs/cycle16_external_lung_source_targeted_sweep/source_seed_layer_detail.csv`
  - `implementation/outputs/cycle16_external_lung_source_targeted_sweep/source_layer_scgpt_aggregate.csv`
  - `implementation/outputs/cycle16_external_lung_source_targeted_sweep/source_best_scgpt_vs_geneformer.csv`
  - `implementation/outputs/cycle16_external_lung_source_targeted_sweep/source_sweep_stability_summary.csv`
- Scope: top-12 disagreement-enriched external-lung sources, seeds 42/43/44, layers 0-11.

Interpretation:
- Per-source results are highly unstable and often extreme because each source subset is tiny (often ~25-50 edges with few positives).
- The sweep is useful for identifying candidate difficult sources/layers, but not for stable source-level effect-size ranking.

### Cycle 17: External-lung source calibration diagnostics
- Added script: `implementation/scripts/run_external_lung_source_calibration_diagnostics.py`
- Outputs:
  - `implementation/outputs/cycle17_external_lung_source_calibration/source_calibration_summary.csv`
  - `implementation/outputs/cycle17_external_lung_source_calibration/source_reliability_bins.csv`
  - `implementation/outputs/cycle17_external_lung_source_calibration/grouped_source_calibration_summary.csv`
- Grouped calibration summary (more stable than per-source):
  - Top10 disagreement sources:
    - scGPT AUROC 0.5401
    - Geneformer AUROC 0.5752
    - gap +0.0351
  - Other sources:
    - scGPT AUROC 0.5860
    - Geneformer AUROC 0.6091
    - gap +0.0231

Interpretation:
- Even where source-level estimates are noisy, grouped calibration still favors Geneformer.
- The gap is not confined to only top disagreement sources.

### Cycle 18: External-lung disagreement-source ablation
- Added script: `implementation/scripts/run_external_lung_source_ablation_gap.py`
- Output:
  - `implementation/outputs/cycle18_external_lung_source_ablation_gap/external_lung_source_ablation_gap_summary.csv`
- Gap results (Geneformer delta - scGPT delta):
  - none: +0.02288
  - exclude_top10: +0.02307
  - exclude_top20: +0.02291
  - exclude_top50: +0.02218
  - exclude_ratio_ge_2: +0.02333

Interpretation:
- Excluding disagreement-enriched source groups does not materially close the external-lung gap.
- The gap appears distributed across a broad set of edges/sources.

### Cycle 19: External-lung layer-bundle test
- Added script: `implementation/scripts/run_external_lung_layer_bundle_test.py`
- Output:
  - `implementation/outputs/cycle19_external_lung_layer_bundle_test/external_lung_layer_bundle_summary.csv`
- Results:
  - L3 (single layer):
    - scGPT delta 0.00308
    - Geneformer delta 0.02596
    - gap +0.02288
  - L1-4 bundle:
    - scGPT delta 0.01193
    - gap +0.01403
  - L0-5 bundle:
    - scGPT delta 0.01656
    - gap +0.00940
  - L0-11 bundle:
    - scGPT delta 0.02265
    - gap +0.00332
    - combined model (`scGPT bundle + Geneformer`) delta 0.03804

Interpretation:
- Multi-layer scGPT representation substantially improves external-lung signal.
- The cross-model gap is almost closed under L0-11 bundling, and combined modeling shows clear complementarity.

### Cycle 20: External-lung seed-ensemble bundle test
- Added script: `implementation/scripts/run_external_lung_seed_ensemble_bundle_test.py`
- Outputs:
  - `implementation/outputs/cycle20_external_lung_seed_ensemble_bundle_test/external_lung_seed_ensemble_bundle_detail.csv`
  - `implementation/outputs/cycle20_external_lung_seed_ensemble_bundle_test/external_lung_seed_ensemble_bundle_seed_aggregate.csv`
- Mean single-seed results (42/43/44):
  - L3:
    - mean scGPT delta 0.00305
    - mean GF-scGPT gap +0.02464
  - L0-11:
    - mean scGPT delta 0.02633
    - mean GF-scGPT gap +0.00135
- Seed-ensemble (averaged features across seeds):
  - L0-11:
    - scGPT delta 0.02578
    - Geneformer delta 0.02438
    - gap (GF - scGPT) -0.00140
    - combined (`scGPT + Geneformer`) delta 0.03958

Interpretation:
- Multi-layer improvements are robust across seeds.
- With seed-ensemble bundling, scGPT slightly surpasses Geneformer on external-lung under the shared-edge protocol.
- Combined modeling remains strongest, confirming remaining complementarity.

### Cycle 21: Cross-domain seed-ensemble bundle transfer
- Added script: `implementation/scripts/run_cross_domain_seed_ensemble_bundle_transfer.py`
- Outputs:
  - `implementation/outputs/cycle21_cross_domain_seed_ensemble_bundle_transfer/cross_domain_seed_ensemble_bundle_detail.csv`
  - `implementation/outputs/cycle21_cross_domain_seed_ensemble_bundle_transfer/cross_domain_seed_ensemble_bundle_seed_aggregate.csv`
- Mean single-seed results (42/43/44):
  - Immune:
    - L3 mean scGPT delta: 0.01774, mean gap +0.03098
    - L0-11 mean scGPT delta: 0.05263, mean gap -0.00391
  - Lung:
    - L3 mean scGPT delta: 0.00326, mean gap +0.02409
    - L0-11 mean scGPT delta: 0.03521, mean gap -0.00787
- Seed-ensemble highlights:
  - Immune L0-11:
    - scGPT delta 0.06634
    - Geneformer delta 0.05361
    - gap -0.01273
  - Lung L0-11:
    - scGPT delta 0.03937
    - Geneformer delta 0.02813
    - gap -0.01125

Interpretation:
- The bundling/ensembling representation fix generalizes beyond external-lung to immune and lung.
- Deep bundles reverse the Geneformer-minus-scGPT gap in all tested domains.

### Cycle 22: Hard-edge profiling after bundle improvements
- Added script: `implementation/scripts/run_hard_edge_profiling_after_bundle.py`
- Outputs:
  - `implementation/outputs/cycle22_hard_edge_profiling_after_bundle_m02/hard_edge_summary.csv`
  - `implementation/outputs/cycle22_hard_edge_profiling_after_bundle_m02/hard_edge_baseline_profile.csv`
  - `implementation/outputs/cycle22_hard_edge_profiling_after_bundle_m02/hard_edges_detailed.tsv`
- Using margin 0.2:
  - hard-edge fraction:
    - immune: 6.45%
    - lung: 0.87%
    - external-lung: 0.60%

Interpretation:
- Remaining disagreement is sparse after bundle improvements and concentrated in a compact hard-edge region.

### Cycle 23: Residual model on hard-edge subsets
- Added script: `implementation/scripts/run_hard_edge_residual_model.py`
- Output:
  - `implementation/outputs/cycle23_hard_edge_residual_model/hard_edge_residual_model_summary.csv`
- Hard-edge AUC (baseline vs combined-score residual model):
  - immune hard: 0.545 -> 1.000
  - lung hard: 0.708 -> 1.000
  - external-lung hard: 0.643 -> 1.000
- Non-hard subsets also improve but with smaller increments.

Interpretation:
- Hard-edge subsets retain high recoverable signal from combined model-score features.
- A targeted residual model is more promising than further global feature tweaks.

### Cycle 24: Hard-edge calibration + margin sensitivity
- Added script: `implementation/scripts/run_hard_edge_calibration_margin_sweep.py`
- Outputs:
  - `implementation/outputs/cycle24_hard_edge_calibration_margin_sweep/hard_edge_margin_summary.csv`
  - `implementation/outputs/cycle24_hard_edge_calibration_margin_sweep/hard_edge_calibration_summary.csv`
  - `implementation/outputs/cycle24_hard_edge_calibration_margin_sweep/hard_edge_reliability_bins.csv`
  - `implementation/outputs/cycle24_hard_edge_calibration_margin_sweep/hard_edge_calibration_gap_summary.csv`
- Additional margin-sensitivity runs:
  - `implementation/outputs/cycle24_hard_edge_residual_model_m01/hard_edge_residual_model_summary.csv`
  - `implementation/outputs/cycle24_hard_edge_residual_model_m03/hard_edge_residual_model_summary.csv`
  - Consolidated table: `implementation/outputs/cycle24_hard_edge_calibration_margin_sweep/hard_edge_residual_margin_sweep_summary.csv`
- Hard-edge fractions by margin:
  - immune: 21.78% (`m=0.1`), 6.45% (`m=0.2`), 1.70% (`m=0.3`)
  - lung: 8.79% (`m=0.1`), 0.87% (`m=0.2`), 0.08% (`m=0.3`)
  - external-lung: 8.62% (`m=0.1`), 0.60% (`m=0.2`), 0.06% (`m=0.3`)
- Calibration summary:
  - On hard subsets, scGPT ECE is consistently much worse than non-hard (e.g., immune `0.433` vs `0.178` at `m=0.2`; lung `0.514` vs `0.199`; external-lung `0.401` vs `0.227`).
  - Geneformer also degrades on hard subsets, but with smaller ECE penalties than scGPT.
- Residual-model margin sensitivity:
  - Combined-score model remains strongest on hard subsets across margins.
  - At `m=0.3`, lung/external-lung hard subsets are very small (`n=12` / `n=11`), so perfect AUC estimates are unstable and should be treated as sample-limited.

Interpretation:
- Remaining disagreement is concentrated in calibration-heavy hard regions, not broad global failure.
- Margin dependence is coherent: higher margins isolate higher-confidence but lower-coverage hard subsets.
- The main remaining task is a deployment-style outer-split evaluation to remove optimism from coupled hard-edge selection and evaluation.

### Cycle 25: Outer-split compact model (deployment-style)
- Added script: `implementation/scripts/run_outer_split_compact_model.py`
- Outputs:
  - `implementation/outputs/cycle25_outer_split_compact_model/outer_split_compact_model_summary.csv`
  - `implementation/outputs/cycle25_outer_split_compact_model/outer_split_disagreement_policy_summary.csv`
  - `implementation/outputs/cycle25_outer_split_compact_model/outer_split_disagreement_policy_best_thresholds.csv`
  - `implementation/outputs/cycle25_outer_split_compact_model/outer_split_oof_predictions.tsv`
- Protocol:
  - Outer repeated CV (`5x3`) with inner-fold OOF stacking for the compact combiner.
  - Compact model inputs: baseline confounds + scGPT base probability + Geneformer base probability.
- Global outer-split AUROC:
  - immune: best single 0.7106, compact 0.7276, gain `+0.0170`
  - lung: best single 0.6120, compact 0.6218, gain `+0.0098`
  - external-lung: best single 0.6188, compact 0.6297, gain `+0.0109`
- Disagreement-threshold policy highlights:
  - immune: best threshold `0.15`, coverage `28.7%`, gain `+0.0467`
  - external-lung: best threshold `0.10`, coverage `18.1%`, gain `+0.0288`
  - lung: highest robust coverage-gain point at threshold `0.05`, coverage `50.1%`, gain `+0.0151`

Interpretation:
- The compact combined model remains beneficial after leakage-resistant outer-split evaluation.
- Gains are moderate (not extreme), which is consistent with stricter evaluation and confirms that prior hard-edge-only estimates were optimistic.
- Disagreement gating remains useful, but threshold tuning must avoid very small high-threshold subsets.

### Cycle 26: Calibration-only intervention on outer-split predictions
- Added script: `implementation/scripts/run_outer_split_calibration_intervention.py`
- Outputs:
  - `implementation/outputs/cycle26_outer_split_calibration_intervention/outer_split_calibration_summary.csv`
  - `implementation/outputs/cycle26_outer_split_calibration_intervention/outer_split_calibration_reliability_bins.csv`
  - `implementation/outputs/cycle26_outer_split_calibration_intervention/outer_split_compact_calibration_focus.csv`
- Compact model calibration tradeoff:
  - Isotonic sharply improves calibration quality:
    - external-lung ECE `0.2241 -> 0.0024`, Brier `0.2370 -> 0.1862`
    - lung ECE `0.1965 -> 0.0039`, Brier `0.2384 -> 0.2000`
    - immune ECE `0.1772 -> 0.0309`, Brier `0.2117 -> 0.1775`
  - Isotonic has small AUC penalty:
    - external-lung `-0.0043`, lung `-0.0041`, immune `-0.0124`
  - Platt scaling is near rank-preserving but yields little calibration improvement.

Interpretation:
- Calibration-only adjustments can strongly improve probability reliability without representation changes.
- There is a clear ranking-vs-calibration tradeoff, especially in immune.

### Cycle 27: Disagreement-threshold uncertainty quantification
- Added script: `implementation/scripts/run_outer_split_threshold_uncertainty.py`
- Outputs:
  - `implementation/outputs/cycle27_outer_split_threshold_uncertainty/outer_split_disagreement_policy_bootstrap_ci.csv`
  - `implementation/outputs/cycle27_outer_split_threshold_uncertainty/outer_split_disagreement_policy_bootstrap_distribution.tsv`
  - `implementation/outputs/cycle27_outer_split_threshold_uncertainty/outer_split_disagreement_policy_recommended_thresholds.csv`
- Bootstrap CI highlights for `compact - best single`:
  - immune:
    - `tau=0.10`: `+0.0346`, CI `[+0.0113, +0.0505]`
    - `tau=0.15`: `+0.0467`, CI `[+0.0167, +0.0656]`
  - external-lung:
    - `tau=0.05`: `+0.0191`, CI `[+0.0105, +0.0225]`
    - `tau=0.10`: `+0.0288`, CI `[+0.0125, +0.0381]`
  - lung:
    - `tau=0.05`: `+0.0151`, CI `[+0.0084, +0.0223]`
    - higher thresholds become CI-wide and often include zero.

Interpretation:
- Moderate disagreement thresholds have statistically supported uplift.
- Very high thresholds are unstable due low coverage.

### Cycle 28: Compact feature ablation
- Added script: `implementation/scripts/run_outer_split_compact_ablation.py`
- Outputs:
  - `implementation/outputs/cycle28_outer_split_compact_ablation/outer_split_compact_ablation_summary.csv`
  - `implementation/outputs/cycle28_outer_split_compact_ablation/outer_split_compact_ablation_oof.tsv`
- AUC results:
  - `compact(prob-only)` minus best single:
    - immune `+0.0149`, lung `+0.0071`, external-lung `+0.0082`
  - `compact(prob+baseline)` minus best single:
    - immune `+0.0170`, lung `+0.0098`, external-lung `+0.0109`
  - Increment from adding baseline confounds:
    - immune `+0.0021`, lung `+0.0027`, external-lung `+0.0027`

Interpretation:
- Score-only combining is already useful.
- Baseline confounds still add a small consistent incremental lift.

### Cycle 29: Objective-conditioned policy selection + joint uncertainty
- Added script: `implementation/scripts/run_objective_conditioned_policy_selection.py`
- Outputs (coverage floor 0.05):
  - `implementation/outputs/cycle29b_objective_policy_selection_cov05/policy_candidates.csv`
  - `implementation/outputs/cycle29b_objective_policy_selection_cov05/policy_selected_by_objective.csv`
  - `implementation/outputs/cycle29b_objective_policy_selection_cov05/policy_selected_with_bootstrap.csv`
  - `implementation/outputs/cycle29b_objective_policy_selection_cov05/cross_domain_default_policy_grid.csv`
  - `implementation/outputs/cycle29b_objective_policy_selection_cov05/cross_domain_default_policy_selected.csv`
  - `implementation/outputs/cycle29b_objective_policy_selection_cov05/cross_domain_default_vs_domain_specific.csv`
- Ranking-first (`delta_ece <= 0.01`) selected:
  - external-lung: isotonic, `tau=0.05`, delta AUC `+0.0099`
  - immune: isotonic, `tau=0.05`, delta AUC `+0.0257`
  - lung: isotonic, `tau=0.05`, delta AUC `+0.0100`
- Reliability-first (`delta_auc >= -0.01`) selected:
  - external-lung: isotonic, `tau=0.05`
  - immune: platt, `tau=0.05`
  - lung: platt, `tau=0.05`
- Cross-domain conservative default:
  - isotonic + `tau=0.05`
  - mean delta AUC `+0.0152`, min delta AUC `+0.0099`, max delta ECE `+0.0056`
  - Default matches all domain-specific ranking-first selections.
- Joint uncertainty (ranking-first):
  - external-lung delta AUC CI `[+0.0065, +0.0142]`
  - immune delta AUC CI `[+0.0127, +0.0380]`
  - lung delta AUC CI `[+0.0056, +0.0142]`

Interpretation:
- Objective-conditioned policy search converges to a simple robust default.
- AUROC gains remain consistently positive under bootstrap while ECE effects are small/mixed by domain.

### Cycle 30: Coverage-floor sensitivity on policy selection
- Re-ran cycle-29 selection at stricter minimum coverage floors:
  - `implementation/outputs/cycle30_policy_selection_cov10/`
  - `implementation/outputs/cycle30_policy_selection_cov20/`
- Consolidated summary:
  - `implementation/outputs/cycle30_policy_selection_cov_sensitivity_summary.csv`
- Result:
  - Selected policies are unchanged across coverage floors `0.05`, `0.10`, and `0.20`.

Interpretation:
- The recommended policy family is stable under practical coverage constraints, reducing deployment sensitivity risk.

### Cycle 31: Objective-specific deployment table
- Output:
  - `implementation/outputs/cycle31_policy_deployment_table.csv`
- Consolidated three deployment profiles per domain:
  - ranking-first,
  - reliability-first,
  - cross-domain default.
- Policy pattern:
  - External-lung: isotonic + `tau=0.05` for all profiles.
  - Immune/lung:
    - ranking-first + default: isotonic + `tau=0.05`
    - reliability-first: platt + `tau=0.05`

Interpretation:
- Deployment recommendations are now explicit and operationally simple.
- Remaining tuning is primarily cost-aware thresholding, not additional feature/model complexity.

### Cycle 32: Referral-cost utility sweep
- Added script: `implementation/scripts/run_policy_cost_utility_sweep.py`
- Outputs:
  - `implementation/outputs/cycle32_policy_cost_utility_sweep/policy_cost_utility_scored_grid.csv`
  - `implementation/outputs/cycle32_policy_cost_utility_sweep/policy_cost_utility_best_by_domain_lambda.csv`
  - `implementation/outputs/cycle32_policy_cost_utility_sweep/policy_cost_utility_best_default_by_lambda.csv`
  - `implementation/outputs/cycle32_policy_cost_utility_sweep/policy_cost_utility_default_vs_domain_gap.csv`
  - `implementation/outputs/cycle32_policy_cost_utility_sweep/policy_cost_utility_selection_stability.csv`
- Utility objective:
  - `U = delta_auc_vs_best_single - lambda * coverage_fraction`
  - `lambda` grid: `0.0` to `0.03`
- Results:
  - Isotonic remains selected throughout.
  - Thresholds increase with referral cost:
    - low cost: `tau=0.05`
    - medium cost: `tau=0.10`
    - high cost (immune): `tau=0.15`

Interpretation:
- Cost-aware adaptation primarily changes referral threshold, not model family.

### Cycle 33: Calibration regime lock (`tau=0.05`)
- Added script: `implementation/scripts/run_calibration_regime_lock.py`
- Outputs:
  - `implementation/outputs/cycle33_calibration_regime_lock/calibration_regime_policy_metrics.csv`
  - `implementation/outputs/cycle33_calibration_regime_lock/calibration_regime_lock_comparisons.csv`
- Isotonic vs raw:
  - AUC decreases with tight CIs below zero across all domains.
  - ECE and Brier improve strongly across all domains.
- Platt vs raw:
  - Near-zero AUC change and minimal calibration change.

Interpretation:
- Isotonic is reliability-first; raw/platt are ranking-first.
- Calibration regime should be selected by objective, not assumed universal.

### Cycle 34: Two-rule policy simplification
- Added script: `implementation/scripts/run_two_rule_policy_simplification.py`
- Outputs:
  - `implementation/outputs/cycle34_two_rule_policy_simplification/two_rule_policy_summary_by_lambda.csv`
  - `implementation/outputs/cycle34_two_rule_policy_simplification/two_rule_policy_domain_assignments.csv`
  - `implementation/outputs/cycle34_two_rule_policy_simplification/two_rule_policy_pattern_stability.csv`
- Result:
  - Best two-rule policy is effectively lossless vs fully domain-specific utility-optimal assignment across lambda grid.
  - Stable patterns:
    - `isotonic@0.05` (low-cost region),
    - `isotonic@0.05 + isotonic@0.10` (mid-cost),
    - `isotonic@0.10 + isotonic@0.15` (high-cost).

Interpretation:
- Deployment can be simplified to at most two rules with negligible utility loss.

### Cycle 35: Dual-mode deployment specification
- Added script: `implementation/scripts/run_dual_mode_deployment_spec.py`
- Outputs:
  - `implementation/outputs/cycle35_dual_mode_deployment_spec/dual_mode_domain_policy_table.csv`
  - `implementation/outputs/cycle35_dual_mode_deployment_spec/dual_mode_summary.csv`
  - `implementation/outputs/cycle35_dual_mode_deployment_spec/cost_aware_default_schedule.csv`
- Mode selection (`tau=0.05`):
  - ranking-first: `raw` in all domains
  - reliability-first: `isotonic` in all domains
- Aggregate tradeoff:
  - ranking-first mean AUC `0.6588`, mean ECE `0.2010`
  - reliability-first mean AUC `0.6544`, mean ECE `0.0089`
- Cost-aware default schedule from utility sweep:
  - `lambda` in `[0.000, 0.0125]`: `isotonic@0.05`
  - `lambda` in `[0.015, 0.030]`: `isotonic@0.10`

Interpretation:
- Deployment objective can be implemented as a clean mode switch (`raw` vs `isotonic`) at fixed threshold.
- Cost sensitivity can be handled via a simple two-segment isotonic threshold schedule.

### Cycle 36: Pre-implementation dry-run checklist
- Added report:
  - `reports/cycle36_preimplementation_checklist.md`
- Checklist contents:
  - required runtime inputs (`scgpt_prob`, `geneformer_prob`, `compact_prob`, domain),
  - mode-dependent calibration rule (`raw` vs `isotonic`),
  - referral threshold schedule (cost-aware `tau=0.05/0.10`),
  - serving logic and monitoring KPIs (`AUC`, `ECE`, `Brier`, `coverage_fraction`),
  - rollout gating criteria for shadow-to-live promotion.

Interpretation:
- Research outputs are now fully translated into deployable rules and monitoring requirements.

### Cycle 37: Deployment policy conformance replay
- Added script: `implementation/scripts/run_deployment_policy_conformance.py`
- Outputs:
  - `implementation/outputs/cycle37_deployment_policy_conformance/conformance_mode_metrics.csv`
  - `implementation/outputs/cycle37_deployment_policy_conformance/conformance_row_equivalence.csv`
  - `implementation/outputs/cycle37_deployment_policy_conformance/deployment_replay_predictions.tsv`
  - `implementation/outputs/cycle37_deployment_policy_conformance/cost_schedule_integrity.csv`
  - `implementation/outputs/cycle37_deployment_policy_conformance/cost_schedule_rule_replay_metrics.csv`
- Results:
  - All dual-mode table metrics were reproduced exactly (AUC/ECE/Brier/coverage diffs effectively zero).
  - Row-wise service-style execution exactly matched vectorized replay (`0` mismatches).
  - Cost schedule has no overlap but contains a small gap (`lambda 0.0125 -> 0.015`).

Interpretation:
- Policy implementation logic is now validated end-to-end on held-out rows.
- The cost schedule boundary needs explicit handling in serving config.

### Cycle 38: Monitoring backtest under pseudo-rolling windows
- Added script: `implementation/scripts/run_monitoring_backtest.py`
- Outputs:
  - `implementation/outputs/cycle38_monitoring_backtest/monitoring_backtest_window_metrics.csv`
  - `implementation/outputs/cycle38_monitoring_backtest/monitoring_backtest_summary.csv`
  - `implementation/outputs/cycle38_monitoring_backtest/monitoring_backtest_mode_aggregate_summary.csv`
  - `implementation/outputs/cycle38_monitoring_backtest/monitoring_backtest_threshold_recommendations.csv`
- Results (original checklist thresholds):
  - Coverage alerts were rare across modes/window sizes.
  - Performance alerts were highly over-triggered:
    - ranking-first: ~`0.30-0.40` any-alert rate depending on window.
    - reliability-first: ~`0.83-0.99` any-alert rate depending on window.

Interpretation:
- Original performance thresholds (`AUC drop > 0.01`, `ECE uplift > 0.01`) are too strict for stable deployment monitoring.

### Cycle 39: Monitoring threshold tuning to target false-alert rate
- Added script: `implementation/scripts/run_monitoring_threshold_tuning.py`
- Outputs:
  - `implementation/outputs/cycle39_monitoring_threshold_tuning/monitoring_threshold_tuning_grid.csv`
  - `implementation/outputs/cycle39_monitoring_threshold_tuning/monitoring_threshold_tuning_best.csv`
  - `implementation/outputs/cycle39_monitoring_threshold_tuning/monitoring_threshold_tuning_perf_only_with_current_cov.csv`
- Target:
  - `any_alert_rate <= 0.05`.
- Results:
  - Feasible tuned caps exist for all tested window sizes and modes.
  - With coverage cap fixed at `0.20`, performance caps near:
    - window 500: ranking AUC-drop `~0.046`, reliability ECE-uplift `~0.050`,
    - meet ~5% any-alert backtest rate.

Interpretation:
- Monitoring can be stabilized without changing policy logic, by widening performance caps and optionally using larger windows.

### Cycle 40: Staged rollout playbook
- Added report:
  - `reports/cycle40_rollout_playbook.md`
- Playbook includes:
  - shadow -> canary -> full stages,
  - tuned hard/soft alert thresholds,
  - explicit rollback and escalation rules,
  - lambda-gap handling requirement.

Interpretation:
- Project output moved from policy discovery to implementation-ready operational guidance.

### Cycle 41: Synthetic incident-injection monitoring evaluation
- Added script: `implementation/scripts/run_monitoring_incident_injection.py`
- Outputs:
  - `implementation/outputs/cycle41_monitoring_incident_injection/monitoring_incident_power.csv`
  - `implementation/outputs/cycle41_monitoring_incident_injection/monitoring_incident_sequence_detection.csv`
  - `implementation/outputs/cycle41_monitoring_incident_injection/monitoring_incident_performance_shift_matrix.csv`
- Compared profiles:
  - `current_strict` (coverage `0.20`, performance `0.01`)
  - `tuned_5pct` (coverage `0.20`, performance tuned by window/mode from cycle39)
- Results:
  - No-incident hard-alert baseline:
    - `current_strict`: high (ranking `~0.30-0.40`, reliability `~0.83-0.99`)
    - `tuned_5pct`: near-target (`~0.046-0.049`)
  - Under performance shifts (window 500, no added coverage shift), tuned profile hard-alert rates rise as expected:
    - ranking: `0.096` (`+0.01`) -> `0.173` (`+0.02`) -> `0.283` (`+0.03`) -> `0.563` (`+0.05`)
    - reliability: `0.156` (`+0.01`) -> `0.396` (`+0.02`) -> `0.735` (`+0.03`) -> `0.999` (`+0.05`)
  - Two-consecutive-window detection in 20-window sequences (window 500, tuned profile):
    - ranking: detection rate `0.157` (`+0.01`) to `0.996` (`+0.05`)
    - reliability: detection rate `0.341` (`+0.01`) to `1.000` (`+0.03/+0.05`)

Interpretation:
- Tuned monitoring retains practical incident sensitivity while controlling false alerts.
- Reliability-mode degradation is detected faster than ranking-mode degradation at similar shift size.

### Cycle 42: Machine-readable deployment config pack validation
- Added config:
  - `implementation/configs/deployment_policy_config_v1.json`
- Added script:
  - `implementation/scripts/run_validate_deployment_config_pack.py`
- Outputs:
  - `implementation/outputs/cycle42_config_pack_validation/config_validation_checks.csv`
  - `implementation/outputs/cycle42_config_pack_validation/config_validation_summary.csv`
  - `implementation/outputs/cycle42_config_pack_validation/config_vs_dualmode_comparison.csv`
  - `implementation/outputs/cycle42_config_pack_validation/config_vs_tuned_thresholds_comparison.csv`
- Validation result:
  - `14/14` checks passed (`all_checks_passed=True`).
  - Mode rules, monitoring caps, and cost-schedule mapping all match empirical artifacts.

Interpretation:
- The deployment policy is now frozen into a machine-readable artifact and consistency-verified.

### Current conclusion
- Single-layer scGPT geometry underestimates available regulatory signal, especially in lung-like domains.
- Multi-layer multi-seed scGPT geometry substantially improves performance and closes/reverses prior cross-model gaps across external-lung, lung, and immune.
- Deployment tradeoff is operationalized (`raw@0.05` vs `isotonic@0.05`, cost-aware move to `isotonic@0.10`), and conformance has been validated on held-out rows.
- Monitoring calibration risk has been reduced with tuned thresholds, staged rollout rules, synthetic-incident sensitivity estimates, and a validated machine-readable config pack.
