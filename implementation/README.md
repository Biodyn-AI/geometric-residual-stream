# Implementation

## Environment
- Active isolated env: `subproject38-geo-v2`
- Reproducible spec: `implementation/environment/environment.yml`
- Frozen lock (osx-arm64): `implementation/environment/conda-explicit-osx-arm64.txt`

Create from the portable spec:
```bash
conda env create -f implementation/environment/environment.yml
```

## Main scripts
- `scripts/run_layerwise_geometry_audit.py`
- `scripts/aggregate_cycle1_replicates.py`
- `scripts/label_permutation_null.py`
- `scripts/geometry_feature_shuffle_null.py`
- `scripts/evaluate_alt_geometry_features.py`
- `scripts/make_obs_subsets.py`
- `scripts/run_geneformer_embedding_geometry_audit.py`
- `scripts/run_geneformer_feature_null_controls.py`
- `scripts/run_cross_model_matched_gene_diagnostic.py`
- `scripts/run_cross_model_edge_score_concordance.py`
- `scripts/run_cross_model_feature_fusion.py`
- `scripts/run_cross_model_disagreement_stratification.py`
- `scripts/run_external_lung_source_targeted_sweep.py`
- `scripts/run_external_lung_source_calibration_diagnostics.py`
- `scripts/run_external_lung_source_ablation_gap.py`
- `scripts/run_external_lung_layer_bundle_test.py`
- `scripts/run_external_lung_seed_ensemble_bundle_test.py`
- `scripts/run_cross_domain_seed_ensemble_bundle_transfer.py`
- `scripts/run_hard_edge_profiling_after_bundle.py`
- `scripts/run_hard_edge_residual_model.py`
- `scripts/run_hard_edge_calibration_margin_sweep.py`
- `scripts/run_outer_split_compact_model.py`
- `scripts/run_outer_split_calibration_intervention.py`
- `scripts/run_outer_split_threshold_uncertainty.py`
- `scripts/run_outer_split_compact_ablation.py`
- `scripts/run_objective_conditioned_policy_selection.py`
- `scripts/run_policy_cost_utility_sweep.py`
- `scripts/run_calibration_regime_lock.py`
- `scripts/run_two_rule_policy_simplification.py`
- `scripts/run_dual_mode_deployment_spec.py`
- `scripts/run_deployment_policy_conformance.py`
- `scripts/run_monitoring_backtest.py`
- `scripts/run_monitoring_threshold_tuning.py`
- `scripts/run_monitoring_incident_injection.py`
- `scripts/run_validate_deployment_config_pack.py`

## Reproduction commands

Cycle 1 (seed 42, max_genes=512):
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_layerwise_geometry_audit.py \
  --max-cells 256 --max-genes 512 --batch-size 4 --cv-splits 5 --cv-repeats 3 \
  --bootstrap-iters 400 --seed 42 \
  --output-dir implementation/outputs/cycle1_main
```

Seeded replicates:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_layerwise_geometry_audit.py \
  --max-cells 256 --max-genes 512 --batch-size 4 --cv-splits 5 --cv-repeats 3 \
  --bootstrap-iters 400 --seed 43 \
  --output-dir implementation/outputs/cycle1_seed43

conda run -n subproject38-geo-v2 python implementation/scripts/run_layerwise_geometry_audit.py \
  --max-cells 256 --max-genes 512 --batch-size 4 --cv-splits 5 --cv-repeats 3 \
  --bootstrap-iters 400 --seed 44 \
  --output-dir implementation/outputs/cycle1_seed44
```

Aggregate replicates:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/aggregate_cycle1_replicates.py
```

Sensitivity run (max_genes=1024):
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_layerwise_geometry_audit.py \
  --max-cells 256 --max-genes 1024 --batch-size 4 --cv-splits 5 --cv-repeats 3 \
  --bootstrap-iters 400 --seed 42 \
  --output-dir implementation/outputs/cycle2_maxgenes1024
```

Permutation null:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/label_permutation_null.py \
  --run-dir implementation/outputs/cycle1_main --layer 5 --n-perm 40 \
  --output-dir implementation/outputs/cycle3_null
```

Cross-domain (immune) main run:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_layerwise_geometry_audit.py \
  --processed-h5ad /Volumes/Crucial\ X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/outputs/tabula_sapiens_immune_subset_hpn_processed.h5ad \
  --max-cells 256 --max-genes 512 --batch-size 4 --cv-splits 5 --cv-repeats 3 \
  --bootstrap-iters 400 --seed 42 \
  --output-dir implementation/outputs/cycle4_immune_main
```

Geometry-feature shuffle null (kidney):
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/geometry_feature_shuffle_null.py \
  --run-dir implementation/outputs/cycle1_main \
  --processed-h5ad /Volumes/Crucial\ X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/outputs/tabula_sapiens_processed.h5ad \
  --layer 5 --feature cosine --n-shuffles 60 \
  --output-dir implementation/outputs/cycle5_kidney_geom_shuffle_null
```

Cross-domain (lung) main run:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_layerwise_geometry_audit.py \
  --processed-h5ad /Volumes/Crucial\ X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/outputs/invariant_causal_edges/lung/processed.h5ad \
  --max-cells 256 --max-genes 512 --batch-size 4 --cv-splits 5 --cv-repeats 3 \
  --bootstrap-iters 400 --seed 42 \
  --output-dir implementation/outputs/cycle6_lung_main
```

Alternative geometry feature audit (lung layer 0):
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/evaluate_alt_geometry_features.py \
  --run-dir implementation/outputs/cycle6_lung_main \
  --processed-h5ad /Volumes/Crucial\ X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/outputs/invariant_causal_edges/lung/processed.h5ad \
  --layer 0 --tag lung --bootstrap-iters 400 --pca-dims 32,64,128
```

Permutation null for PCA-derived feature:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/label_permutation_null.py \
  --run-dir implementation/outputs/cycle6_lung_main \
  --processed-h5ad /Volumes/Crucial\ X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/outputs/invariant_causal_edges/lung/processed.h5ad \
  --layer 0 --feature pca64_centered_cosine --n-perm 40 \
  --output-dir implementation/outputs/cycle10_lung_pca64_centered_label_perm_null
```

Geneformer replication (immune):
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_geneformer_embedding_geometry_audit.py \
  --processed-h5ad /Volumes/Crucial\ X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/outputs/tabula_sapiens_immune_subset_hpn_processed.h5ad \
  --output-dir implementation/outputs/cycle12_geneformer_immune_bootstrap
```

Geneformer centered-cosine null controls:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_geneformer_feature_null_controls.py \
  --processed-h5ad /Volumes/Crucial\ X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/outputs/tabula_sapiens_immune_subset_hpn_processed.h5ad \
  --edge-dataset-tsv implementation/outputs/cycle12_geneformer_immune_bootstrap/geneformer_edge_dataset.tsv \
  --feature centered_cosine --n-shuffles 60 --n-perm 40 \
  --output-dir implementation/outputs/cycle12_geneformer_immune_centered_null
```

Cross-model matched-gene diagnostic:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_cross_model_matched_gene_diagnostic.py \
  --output-dir implementation/outputs/cycle13_cross_model_matched_gene_diagnostic
```

Cross-model edge-score concordance on shared edges:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_cross_model_edge_score_concordance.py \
  --output-dir implementation/outputs/cycle13_cross_model_edge_concordance
```

Cross-model feature fusion on shared edges:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_cross_model_feature_fusion.py \
  --output-dir implementation/outputs/cycle14_cross_model_feature_fusion
```

Cross-model disagreement stratification:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_cross_model_disagreement_stratification.py \
  --output-dir implementation/outputs/cycle15_cross_model_disagreement_stratification
```

External-lung targeted source sweep:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_external_lung_source_targeted_sweep.py \
  --top-k-sources 12 \
  --output-dir implementation/outputs/cycle16_external_lung_source_targeted_sweep
```

External-lung per-source calibration diagnostics:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_external_lung_source_calibration_diagnostics.py \
  --top-k-sources 12 --min-pairs 30 --min-positives 3 \
  --output-dir implementation/outputs/cycle17_external_lung_source_calibration
```

External-lung disagreement-source ablation:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_external_lung_source_ablation_gap.py \
  --output-dir implementation/outputs/cycle18_external_lung_source_ablation_gap
```

External-lung layer-bundle test:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_external_lung_layer_bundle_test.py \
  --output-dir implementation/outputs/cycle19_external_lung_layer_bundle_test
```

External-lung seed-ensemble bundle test:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_external_lung_seed_ensemble_bundle_test.py \
  --output-dir implementation/outputs/cycle20_external_lung_seed_ensemble_bundle_test
```

Cross-domain seed-ensemble bundle transfer:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_cross_domain_seed_ensemble_bundle_transfer.py \
  --output-dir implementation/outputs/cycle21_cross_domain_seed_ensemble_bundle_transfer
```

Hard-edge profiling after bundle improvements:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_hard_edge_profiling_after_bundle.py \
  --margin 0.2 \
  --output-dir implementation/outputs/cycle22_hard_edge_profiling_after_bundle_m02
```

Residual model on hard edges:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_hard_edge_residual_model.py \
  --margin 0.2 \
  --output-dir implementation/outputs/cycle23_hard_edge_residual_model
```

Hard-edge calibration + margin sweep:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_hard_edge_calibration_margin_sweep.py \
  --margins 0.1,0.2,0.3 \
  --output-dir implementation/outputs/cycle24_hard_edge_calibration_margin_sweep
```

Residual-model margin sensitivity:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_hard_edge_residual_model.py \
  --margin 0.1 \
  --output-dir implementation/outputs/cycle24_hard_edge_residual_model_m01

conda run -n subproject38-geo-v2 python implementation/scripts/run_hard_edge_residual_model.py \
  --margin 0.3 \
  --output-dir implementation/outputs/cycle24_hard_edge_residual_model_m03
```

Outer-split compact model (deployment-style stacking):
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_outer_split_compact_model.py \
  --outer-cv-splits 5 --outer-cv-repeats 3 --inner-cv-splits 5 \
  --thresholds 0.05,0.1,0.15,0.2,0.25,0.3 \
  --output-dir implementation/outputs/cycle25_outer_split_compact_model
```

Outer-split calibration intervention (Platt/isotonic):
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_outer_split_calibration_intervention.py \
  --oof-predictions-tsv implementation/outputs/cycle25_outer_split_compact_model/outer_split_oof_predictions.tsv \
  --output-dir implementation/outputs/cycle26_outer_split_calibration_intervention
```

Disagreement-threshold uncertainty (bootstrap CIs):
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_outer_split_threshold_uncertainty.py \
  --oof-predictions-tsv implementation/outputs/cycle25_outer_split_compact_model/outer_split_oof_predictions.tsv \
  --thresholds 0.05,0.1,0.15,0.2,0.25,0.3 \
  --n-bootstrap 500 \
  --output-dir implementation/outputs/cycle27_outer_split_threshold_uncertainty
```

Outer-split compact ablation (prob-only vs prob+baseline):
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_outer_split_compact_ablation.py \
  --outer-cv-splits 5 --outer-cv-repeats 3 --inner-cv-splits 5 \
  --output-dir implementation/outputs/cycle28_outer_split_compact_ablation
```

Objective-conditioned policy selection + joint uncertainty:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_objective_conditioned_policy_selection.py \
  --oof-predictions-tsv implementation/outputs/cycle25_outer_split_compact_model/outer_split_oof_predictions.tsv \
  --thresholds 0.05,0.1,0.15,0.2,0.25,0.3 \
  --calibration-methods raw,platt,isotonic \
  --min-coverage-fraction 0.05 \
  --n-bootstrap 500 \
  --output-dir implementation/outputs/cycle29b_objective_policy_selection_cov05
```

Coverage-floor policy sensitivity:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_objective_conditioned_policy_selection.py \
  --oof-predictions-tsv implementation/outputs/cycle25_outer_split_compact_model/outer_split_oof_predictions.tsv \
  --thresholds 0.05,0.1,0.15,0.2,0.25,0.3 \
  --calibration-methods raw,platt,isotonic \
  --min-coverage-fraction 0.10 \
  --n-bootstrap 300 \
  --output-dir implementation/outputs/cycle30_policy_selection_cov10
```

Referral-cost utility sweep:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_policy_cost_utility_sweep.py \
  --policy-candidates-csv implementation/outputs/cycle29b_objective_policy_selection_cov05/policy_candidates.csv \
  --lambda-grid 0.0,0.0025,0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.025,0.03 \
  --min-coverage-fraction 0.05 \
  --output-dir implementation/outputs/cycle32_policy_cost_utility_sweep
```

Calibration regime lock (raw vs isotonic/platt at fixed threshold):
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_calibration_regime_lock.py \
  --oof-predictions-tsv implementation/outputs/cycle25_outer_split_compact_model/outer_split_oof_predictions.tsv \
  --threshold 0.05 --methods raw,isotonic,platt \
  --n-bootstrap 500 \
  --output-dir implementation/outputs/cycle33_calibration_regime_lock
```

Two-rule policy simplification:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_two_rule_policy_simplification.py \
  --scored-grid-csv implementation/outputs/cycle32_policy_cost_utility_sweep/policy_cost_utility_scored_grid.csv \
  --domain-best-csv implementation/outputs/cycle32_policy_cost_utility_sweep/policy_cost_utility_best_by_domain_lambda.csv \
  --output-dir implementation/outputs/cycle34_two_rule_policy_simplification
```

Dual-mode deployment specification:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_dual_mode_deployment_spec.py \
  --calibration-policy-metrics-csv implementation/outputs/cycle33_calibration_regime_lock/calibration_regime_policy_metrics.csv \
  --utility-default-by-lambda-csv implementation/outputs/cycle32_policy_cost_utility_sweep/policy_cost_utility_best_default_by_lambda.csv \
  --ranking-threshold 0.05 --reliability-threshold 0.05 --auc-tolerance 0.01 \
  --output-dir implementation/outputs/cycle35_dual_mode_deployment_spec
```

Deployment conformance replay:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_deployment_policy_conformance.py \
  --oof-predictions-tsv implementation/outputs/cycle25_outer_split_compact_model/outer_split_oof_predictions.tsv \
  --dual-mode-policy-csv implementation/outputs/cycle35_dual_mode_deployment_spec/dual_mode_domain_policy_table.csv \
  --cost-schedule-csv implementation/outputs/cycle35_dual_mode_deployment_spec/cost_aware_default_schedule.csv \
  --output-dir implementation/outputs/cycle37_deployment_policy_conformance
```

Monitoring backtest (pseudo-rolling windows):
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_monitoring_backtest.py \
  --replay-predictions-tsv implementation/outputs/cycle37_deployment_policy_conformance/deployment_replay_predictions.tsv \
  --baseline-policy-csv implementation/outputs/cycle35_dual_mode_deployment_spec/dual_mode_domain_policy_table.csv \
  --window-sizes 250,500,1000 --n-permutations 250 \
  --output-dir implementation/outputs/cycle38_monitoring_backtest
```

Monitoring threshold tuning (target false-alert rate):
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_monitoring_threshold_tuning.py \
  --window-metrics-csv implementation/outputs/cycle38_monitoring_backtest/monitoring_backtest_window_metrics.csv \
  --target-any-alert-rate 0.05 \
  --output-dir implementation/outputs/cycle39_monitoring_threshold_tuning
```

Synthetic incident-injection sensitivity evaluation:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_monitoring_incident_injection.py \
  --window-metrics-csv implementation/outputs/cycle38_monitoring_backtest/monitoring_backtest_window_metrics.csv \
  --tuned-thresholds-csv implementation/outputs/cycle39_monitoring_threshold_tuning/monitoring_threshold_tuning_perf_only_with_current_cov.csv \
  --performance-shifts 0.0,0.005,0.01,0.02,0.03,0.05 \
  --coverage-shifts 0.0,0.05,0.10,0.15 \
  --sequence-length 20 --n-sequences 5000 \
  --output-dir implementation/outputs/cycle41_monitoring_incident_injection
```

Machine-readable config validation:
```bash
conda run -n subproject38-geo-v2 python implementation/scripts/run_validate_deployment_config_pack.py \
  --config-json implementation/configs/deployment_policy_config_v1.json \
  --dual-mode-policy-csv implementation/outputs/cycle35_dual_mode_deployment_spec/dual_mode_domain_policy_table.csv \
  --cost-schedule-csv implementation/outputs/cycle35_dual_mode_deployment_spec/cost_aware_default_schedule.csv \
  --tuned-thresholds-csv implementation/outputs/cycle39_monitoring_threshold_tuning/monitoring_threshold_tuning_perf_only_with_current_cov.csv \
  --validation-window-size 500 \
  --output-dir implementation/outputs/cycle42_config_pack_validation
```
