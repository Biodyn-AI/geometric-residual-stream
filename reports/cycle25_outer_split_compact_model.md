# Outer-Split Compact Model (Cycle 25)

Date: 2026-02-20

## Goal
Run a deployment-style, leakage-resistant compact combiner and quantify thresholded disagreement policy gains over single-model baselines.

## Script
- `implementation/scripts/run_outer_split_compact_model.py`

## Outputs
- `implementation/outputs/cycle25_outer_split_compact_model/outer_split_compact_model_summary.csv`
- `implementation/outputs/cycle25_outer_split_compact_model/outer_split_disagreement_policy_summary.csv`
- `implementation/outputs/cycle25_outer_split_compact_model/outer_split_disagreement_policy_best_thresholds.csv`
- `implementation/outputs/cycle25_outer_split_compact_model/outer_split_oof_predictions.tsv`

## Setup
- Outer evaluation: repeated stratified CV (`5` splits x `3` repeats).
- Stacking control: combiner trained on train-fold targets with inner-fold OOF base predictions (not in-sample train predictions).
- Domain set: immune, lung, external-lung.
- Compact combiner input: baseline confounds + scGPT base probability + Geneformer base probability.

## Global Results

| Domain | Baseline AUC | scGPT AUC | Geneformer AUC | Compact AUC | Compact - Best Single |
|---|---:|---:|---:|---:|---:|
| Immune | 0.6424 | 0.7106 | 0.6974 | 0.7276 | +0.0170 |
| Lung | 0.5711 | 0.6120 | 0.5997 | 0.6218 | +0.0098 |
| External lung | 0.5929 | 0.6188 | 0.6172 | 0.6297 | +0.0109 |

Interpretation:
- The compact combiner still improves all domains under outer-split evaluation.
- Improvement size is smaller than earlier hard-edge-only results, consistent with reduced optimism after leakage control.

## Disagreement-Threshold Policy
Thresholding by `|p_geneformer - p_scgpt| >= tau`:
- Immune:
  - Best lift at `tau=0.15`: coverage `28.7%`, compact minus best single `+0.0467`.
- External-lung:
  - Best lift at `tau=0.10`: coverage `18.1%`, lift `+0.0288`.
- Lung:
  - Stable positive lift at broad coverage (`tau=0.05`, coverage `50.1%`, lift `+0.0151`).
  - High-threshold tiny subsets become noisy and can flip sign.

Interpretation:
- Moderate disagreement thresholds provide the best tradeoff between coverage and uplift.
- Very high thresholds are sample-limited and unreliable in lung/external-lung.

## Conclusion
- Hard-edge complementarity remains real after leakage-resistant evaluation.
- A compact disagreement-aware combiner is deployment-viable and consistently better than single-model baselines at global level.
- Next step should focus on calibration-only interventions and uncertainty intervals for threshold policy selection.
