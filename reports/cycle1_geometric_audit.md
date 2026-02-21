# Geometric Residual-Stream Audit (scGPT, Kidney)

Date: 2026-02-20

## Experiment A: Initial layer-wise audit (seed 42, max_genes=512)

1. Research question
- Does layer-wise residual geometry provide TRRUST edge signal beyond gene-level confounds?

2. Method
- Extracted layer outputs from all 12 scGPT transformer layers (256 sampled cells).
- Aggregated token activations into per-gene layer embeddings.
- Built edge dataset: 288 TRRUST positives + 864 matched random negatives.
- Computed geometry score (cosine similarity source-target embeddings).
- Evaluated logistic CV model with baseline features only vs baseline+geometry.

3. Results
- Best layer: L5.
- Baseline CV AUROC: 0.5153.
- Baseline+Geometry CV AUROC: 0.5795.
- Delta CV AUROC: +0.0642.
- Bootstrap 95% CI for delta: [0.0183, 0.1108].

4. Baselines
- Gene-level baseline: mean expression, variance, detection frequency for source and target.
- Geometry-free baseline is near chance-level AUROC.

5. Interpretation
- Residual geometry contributes information not captured by univariate gene confounds.
- Peak signal occurs in early-mid layers.

6. Next step
- Run seed-level replicates and aggregate layer-stability statistics.

## Experiment B: Seed robustness (seeds 42/43/44, max_genes=512)

1. Research question
- Is the geometric gain robust across random cell sampling seeds?

2. Method
- Repeated Experiment A protocol for seeds 43 and 44.
- Aggregated per-layer metrics across three runs.

3. Results
- Per-run best deltas: +0.0642 (L5), +0.1216 (L0), +0.0925 (L5).
- Aggregate best layer by mean delta: L4.
- Mean delta CV AUROC at L4: +0.0895 (std 0.0244), range [0.0640, 0.1125].
- Fraction of replicates with bootstrap CI lower bound > 0 at L4: 1.00.

4. Baselines
- Same confound baseline protocol used in each replicate.

5. Interpretation
- Positive effect replicates and is not confined to one random sample.
- Layer peak is somewhat seed-sensitive, but high-signal region remains early-mid layers.

6. Next step
- Perform sensitivity check on gene token budget.

## Experiment C: Sensitivity to token budget (max_genes=1024)

1. Research question
- Does increasing per-cell token budget remove or preserve geometric incremental value?

2. Method
- Re-ran protocol with max_genes=1024, seed 42.

3. Results
- Best layer: L4.
- Baseline CV AUROC: 0.5077.
- Baseline+Geometry CV AUROC: 0.5565.
- Delta CV AUROC: +0.0488.
- Bootstrap 95% CI: [0.0037, 0.0910].

4. Baselines
- Same baseline feature set.

5. Interpretation
- Effect attenuates relative to max_genes=512 but remains positive and statistically non-zero.

6. Next step
- Add null controls for artifact rejection.

## Experiment D: Label-permutation null (layer 5)

1. Research question
- Can the observed delta be reproduced when labels are randomized?

2. Method
- Fixed features and CV protocol; permuted labels 40 times.
- Compared true-label delta against null distribution.

3. Results
- True delta CV AUROC: 0.0746.
- Null mean delta: 0.0078 (std 0.0201).
- Empirical right-tail p-value: 0.025.

4. Baselines
- Baseline-only vs baseline+geometry model comparison preserved under permutation protocol.

5. Interpretation
- Observed gain exceeds null expectation in this control.

6. Next step
- Extend to cross-tissue runs and cross-model (Geneformer) confirmation.
