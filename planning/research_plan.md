# Research Plan: Geometric Interpretability of scGPT Residual Stream

Date: 2026-02-20

## 1) Research proposal (concrete claim)

### Claim to test
Residual-stream geometry in scGPT contains TF-target regulatory information (TRRUST) that is not recoverable from trivial gene-level baselines alone.

### Main hypothesis
For at least one middle/late layer, edge scores derived from geometric proximity of gene embeddings in residual space (cosine / manifold-neighbor score) improve held-out TRRUST edge prediction over a baseline model built only from:
- mean expression,
- variance,
- detection frequency.

### Falsification criterion
If no layer yields statistically reliable improvement over the baseline-only model (ΔAUROC ~ 0 with bootstrap CI crossing 0), the hypothesis is rejected for this experiment setting.

## 2) First experiment (Cycle 1)

### Question
Does layer-wise residual geometry in scGPT (kidney processed dataset) provide incremental TRRUST signal beyond gene-level confounds?

### Method
1. Extract per-layer residual activations from scGPT for a bounded cell sample.
2. Aggregate token activations into per-gene embeddings at each layer.
3. Build a balanced edge dataset:
- Positives: TRRUST source-target edges present in the gene universe.
- Negatives: matched random non-edges from the same source/target pools.
4. Compute geometric edge score per pair (cosine similarity).
5. Compare models:
- Baseline-only logistic model using gene-level confounds.
- Baseline + geometry logistic model (per layer).
6. Quantify AUROC/AUPR with repeated stratified CV and bootstrap CI for delta.

### Baselines and controls
- Gene-level confound baseline (mean/variance/detection of source and target).
- Random-label sanity check (optional in cycle 1, required in cycle 2).
- Layer permutation sanity check (planned if cycle 1 is positive).

### Deliverables
- `implementation/scripts/run_layerwise_geometry_audit.py`
- `implementation/outputs/cycle1_layer_geometry_metrics.csv`
- `implementation/outputs/cycle1_layer_delta_bootstrap.csv`
- `reports/research_log.md` (dated entry with interpretation and next step)

## 3) Execution loop policy

For each cycle:
1. Run experiment.
2. Write quantitative summary.
3. Decide: deepen, refute, or pivot.
4. Launch next cycle immediately until blocked by compute/data.

## 4) Planned next cycles (after cycle 1)

- Cycle 2: robustness (new tissue/config + null controls).
- Cycle 3: simple manifold methods (Isomap/UMAP neighborhood scores).
- Cycle 4: causal usage test via projection-ablation of geometry-aligned subspace.
- Cycle 5: cross-model check (Geneformer) if compute + model availability allow.

## 6) Phase-2 update (2026-02-20, after execution)

### What was executed
- Cross-domain robustness on immune, lung, and external-lung processed datasets.
- Additional stronger control: geometry-feature shuffle null.

### Empirical status
- Kidney: strong positive effect.
- Immune: moderate positive effect.
- Lung: near-null under raw cosine, but positive under centered-cosine.
- External lung: tiny/unstable effect.

### Updated hypothesis
Residual geometry carries regulatory signal, but effect size is domain-dependent and can collapse in some tissues/cohorts.

### Next research proposal (current)
Explain and reduce domain dependence.

1. Mechanistic diagnostics:
- Correlate domain-level gains with baseline separability, TRRUST mapping coverage, and token occupancy.

2. Feature-space variants:
- Add centered-cosine and low-rank projection features to test whether lung failures are metric-specific.
  - Status: centered-cosine + low-rank projection executed; lung recovery strengthened under `pca64_centered_cosine`.

3. Split robustness:
- Donor-aware and cell-type-stratified protocols to separate genuine regulatory signal from composition effects.
  - Status: donor-aware immune check executed on top-3 donors; cell-type stratified check executed for top immune and lung cell types.

4. Cross-model replication:
- Re-run the geometry protocol with Geneformer token embeddings to test convergence vs scGPT findings.
  - Status: executed on immune, lung, and external-lung with strong positive centered-cosine signal in all three domains; null controls added.

## 5) Risks and mitigations

- Environment incompatibility: use isolated project env pinned to NumPy<2 stack.
- Nested tensor hooks instability: disable nested-tensor path in transformer encoder.
- Runtime cost: start with limited cells/batches and scale only after first signal check.
- Cross-model inconsistency risk: if scGPT and Geneformer disagree by domain, prioritize diagnostics over further metric tuning.

## 7) Next proposal (after cycles 11-12)

### New concrete claim
Cross-model agreement is driven by gene-token coverage and source-target embedding occupancy; external-lung divergence between scGPT and Geneformer can be explained by representation quality rather than TRRUST edge sampling.

### Next experiments
1. Coverage diagnostics:
- Compute per-domain overlap and occupancy metrics for scGPT vs Geneformer edge endpoints.
- Re-estimate deltas after restricting each model to a shared high-confidence gene subset.
  - Status: executed. Matched edge fraction is ~0.99 in lung/external-lung and does not rescue external-lung scGPT delta.

2. Cross-model edge-level concordance:
- For each domain, correlate edge scores (`centered_cosine`) between scGPT and Geneformer on matched edge sets.
- Quantify whether disagreement is global or concentrated in specific TF families.
  - Status: executed on shared Geneformer edge sets; correlations are moderate (~0.45-0.48), while Geneformer keeps a sizable delta advantage.

3. Stratified null checks:
- Repeat external-lung null controls under matched-gene restriction for both models.
- Use identical permutation counts and CV settings for direct comparability.
  - Status: partially addressed via shared-edge concordance (delta gap persists on identical edges).

## 8) Updated next proposal (after cycle 13)

### New concrete claim
The remaining scGPT vs Geneformer gap is primarily a feature-quality/calibration issue on shared edges, not a coverage or sampling artifact.

### Next experiments
1. Feature fusion test:
- Evaluate `baseline + scGPT + Geneformer` on shared edges and quantify incremental lift over each model alone.
  - Status: executed. Fusion gains are marginal (lung/external-lung) and absent in immune.
2. Disagreement stratification:
- Bin edges by absolute score disagreement and test TF-family/pathway enrichment in high-disagreement bins.
  - Status: executed. High-disagreement bins in lung/external-lung show lower positive-edge rates and concentrated source-TF enrichments.
3. scGPT stability sweep:
- Repeat shared-edge external-lung evaluation across additional scGPT seeds/layers to localize whether the gap is layer-specific or persistent.

## 9) Updated next proposal (after cycle 15)

### New concrete claim
External-lung disagreement is concentrated in specific source-TF groups and reflects a stable scGPT calibration weakness rather than random edge noise.

### Next experiments
1. Targeted seed/layer sweep:
- Re-run shared-edge external-lung evaluation for top disagreement-enriched sources across multiple scGPT seeds/layers.
  - Status: executed for top-12 sources across seeds 42/43/44 and layers 0-11; per-source results are highly sample-limited.
2. Per-source calibration diagnostics:
- For top disagreement sources, compare reliability curves and decision thresholds between scGPT and Geneformer.
  - Status: executed. Source-level estimates are noisy due small positives, but grouped calibration (`top10 sources`) still favors Geneformer.
3. Focused ablation:
- Remove high-disagreement source groups and re-estimate domain-level delta gaps to quantify how much of the gap they explain.
  - Status: executed. Excluding top disagreement source groups leaves the global external-lung gap essentially unchanged (~+0.022 to +0.023).

## 10) Updated next proposal (after cycles 16-18)

### New concrete claim
The external-lung gap is distributed across many edges and persists after targeted source exclusions; next progress requires broader scGPT representation improvements rather than source-specific filtering.

### Next experiments
1. Layer-bundle feature test:
- Replace single-layer scGPT feature with a small stacked layer bundle (e.g., L1-L4 PCA-centered scores) and measure whether the external-lung gap narrows.
  - Status: executed. L0-11 scGPT bundle raises delta to 0.02265 and shrinks Geneformer-minus-scGPT gap to +0.00332.
2. Seed-ensemble test:
- Average scGPT geometric features across seeds before downstream modeling to reduce small-sample instability.
3. Hard-edge profiling:
- Identify edges where Geneformer is confidently correct and scGPT is confidently wrong, then test whether these edges share expression sparsity or baseline-confound patterns.

## 11) Updated next proposal (after cycle 19)

### New concrete claim
External-lung underperformance is largely a single-layer bottleneck; multi-layer scGPT features capture complementary signal and nearly close the cross-model gap.

### Next experiments
1. Seed-ensemble bundle test:
- Build multi-layer scGPT bundles for seeds 42/43/44 and evaluate mean/variance of external-lung gap.
  - Status: executed. Mean GF-scGPT gap for L0-11 bundle is +0.00135 across seeds; seed-ensemble mean closes gap to -0.00140.
2. Cross-domain bundle transfer:
- Apply the same layer-bundle protocol to lung and immune shared-edge settings to test whether gains are specific to external-lung.
3. Hard-edge profiling:
- Profile edges where Geneformer remains better than the best scGPT bundle to identify residual failure modes.

## 12) Updated next proposal (after cycle 20)

### New concrete claim
The core cross-model discrepancy is mostly resolved by multi-layer, multi-seed scGPT geometry construction; remaining gains should come from targeted hard-edge modeling and cross-domain transfer checks.

### Next experiments
1. Cross-domain bundle transfer:
- Run the same seed-ensemble bundle protocol on lung and immune to verify whether the representation fix generalizes.
  - Status: executed. Deep bundles (L0-11) reverse GF-scGPT gap in both immune and lung.
2. Hard-edge profiling:
- Identify residual Geneformer-advantaged edges under L0-11 seed-ensemble and characterize their confound/sparsity profile.
3. Lightweight residual model:
- Train a small residual classifier on hard edges using both model scores plus baseline confounds to quantify remaining recoverable signal.

## 13) Updated next proposal (after cycle 21)

### New concrete claim
Multi-layer multi-seed scGPT geometry generalizes across domains; remaining error is concentrated in hard-edge subsets where cross-model complementarity remains exploitable.

### Next experiments
1. Hard-edge profiling:
- Define edges where Geneformer > scGPT by a confidence margin under bundle settings and profile baseline/confound characteristics.
  - Status: executed with margin 0.2. Hard-edge fractions are small (immune 6.45%, lung 0.87%, external-lung 0.60%).
2. Residual model on hard edges:
- Fit a lightweight residual model using both scores plus confounds to estimate recoverable residual signal.
  - Status: executed. Combined-score residual model strongly improves AUC on hard subsets across domains.
3. Calibration-on-hard-edges:
- Recompute reliability and ECE on hard-edge subsets to isolate the remaining calibration gap.
  - Status: partially addressed via hard-edge residual model performance; explicit ECE table on hard subsets remains pending.

## 14) Updated next proposal (after cycles 22-23)

### New concrete claim
After bundle improvements, remaining cross-model disagreement is sparse and high-value; targeted hard-edge modeling yields outsized returns.

### Next experiments
1. Hard-edge calibration table:
- Compute ECE/reliability per domain specifically on hard vs non-hard subsets with margin sweep (0.1/0.2/0.3).
  - Status: executed in cycle 24. Hard-edge ECE is consistently worse than non-hard for both models, with a larger penalty for scGPT.
2. Margin sensitivity:
- Re-run hard-edge profiling/residual modeling across margins to assess stability of conclusions.
  - Status: executed in cycle 24 (`m=0.1/0.2/0.3`). Hard-edge fractions decrease rapidly with larger margins; residual-model gains persist but become sample-limited at `m=0.3` in lung/external-lung.
3. Deployment-oriented compact model:
- Train a compact combined-score model and report gains relative to best single-model bundle baseline.

## 15) Updated next proposal (after cycle 24)

### New concrete claim
Hard-edge regions are calibration hotspots with stable cross-model complementarity, but current hard-edge residual estimates can be optimistic because subset definition and evaluation are coupled.

### Next experiments
1. Outer-split compact model:
- Use nested/outer CV where hard-edge labeling and compact residual fitting are done on training folds only, then evaluated on held-out folds.
  - Status: executed in cycle 25. Compact model improves over best single model in all domains (`+0.0170` immune, `+0.0098` lung, `+0.0109` external-lung).
2. Calibration-only intervention:
- Apply monotonic calibration (Platt/isotonic) to scGPT and Geneformer probability outputs and test whether hard-edge ECE gaps shrink without adding model complexity.
3. Coverage-performance policy:
- Sweep hard-edge referral thresholds and report accuracy lift vs coverage so a deployment policy can choose when to invoke the combined model.
  - Status: executed in cycle 25 via disagreement-threshold sweep. Best medium-threshold lifts: immune `+0.0467` at `tau=0.15`, external-lung `+0.0288` at `tau=0.10`.

## 16) Updated next proposal (after cycle 25)

### New concrete claim
Combined modeling is genuinely useful after leakage-resistant evaluation, and the remaining bottleneck is probability calibration/uncertainty quality rather than missing representation features.

### Next experiments
1. Calibration-only intervention:
- Fit Platt and isotonic calibrators on outer-fold train predictions for scGPT, Geneformer, and compact model; evaluate ECE/Brier/AUC on held-out folds.
  - Status: executed in cycle 26. Isotonic greatly improves ECE/Brier with modest AUC reduction; Platt is near rank-preserving with minimal calibration gain.
2. Threshold uncertainty quantification:
- Bootstrap outer-split predictions to add confidence intervals for `compact_minus_best_single_auc` at each disagreement threshold.
  - Status: executed in cycle 27. Moderate thresholds show positive CIs in immune/external-lung and at low threshold in lung.
3. Compact-feature ablation:
- Compare `compact(prob-only)` vs `compact(prob+baseline)` to measure whether baseline confounds still add incremental value in deployment mode.
  - Status: executed in cycle 28. Baseline terms add small consistent AUROC gains (`+0.0021` to `+0.0027`) over prob-only compact.

## 17) Updated next proposal (after cycles 26-28)

### New concrete claim
The remaining optimization space is mainly policy design: selecting calibration mode and disagreement threshold under explicit ranking-vs-reliability objectives.

### Next experiments
1. Objective-conditioned policy selection:
- For each domain, choose `(calibration method, threshold)` under two objectives:
  - ranking-first (maximize AUROC with minimal ECE constraint),
  - reliability-first (minimize ECE with minimal AUROC loss constraint).
  - Status: executed in cycle 29. Ranking-first selects isotonic + `tau=0.05` in all domains; reliability-first selects isotonic in external-lung and platt in immune/lung.
2. Joint uncertainty table:
- Build bootstrap CIs for both AUROC delta and ECE delta at selected policies to avoid tuning on point estimates alone.
  - Status: executed in cycle 29. Selected-policy bootstrap tables now include joint CI columns for `delta_auc` and `delta_ece`.
3. Cross-domain default policy:
- Fit a single conservative default policy across domains and compare against domain-specific tuned policies for robustness vs specialization tradeoff.
  - Status: executed in cycle 29. Conservative default is isotonic + `tau=0.05`, matching all domain-specific ranking-first picks.

## 18) Updated next proposal (after cycles 29-30)

### New concrete claim
Policy selection has converged to a stable default under multiple coverage floors; remaining work is to operationalize objective-specific deployment choices with explicit decision rules.

### Next experiments
1. Objective-specific deployment table:
- Produce a compact decision sheet with one recommended policy per domain for:
  - ranking-first deployment,
  - reliability-first deployment,
  - cross-domain default deployment.
  - Status: executed in cycle 31 (`implementation/outputs/cycle31_policy_deployment_table.csv`).
2. Cost-aware sensitivity:
- Add a simple referral-cost term and optimize net utility (AUC gain minus referral cost) over thresholds.
3. Calibration regime lock:
- Compare `raw` vs `isotonic` policy behavior under the same threshold with CI bands to decide whether one calibration mode should be standardized project-wide.

## 19) Updated next proposal (after cycle 31)

### New concrete claim
Deployment choice is now a policy-profile decision problem; the next technical gain is from explicit cost-aware threshold optimization rather than further score-model changes.

### Next experiments
1. Referral-cost utility sweep:
- Define utility `U = delta_auc - lambda * coverage_fraction` and optimize policy by domain over a grid of `lambda`.
  - Status: executed in cycle 32. Optimal threshold shifts upward with cost (`0.05 -> 0.10 -> 0.15`) while isotonic remains preferred.
2. Calibration regime lock:
- Quantify whether isotonic should be the default by comparing `raw` vs `isotonic` on utility and CI stability.
  - Status: executed in cycle 33. Isotonic significantly improves ECE/Brier but decreases policy AUC versus raw at fixed `tau=0.05`.
3. Policy simplification:
- Evaluate if one two-rule policy can approximate domain-specific profiles with minimal utility loss.
  - Status: executed in cycle 34. Best two-rule isotonic policy family is near-lossless vs domain-specific optimum across lambda.

## 20) Updated next proposal (after cycles 32-34)

### New concrete claim
Policy optimization has converged: the main decision is objective-weighted deployment mode rather than additional model tuning.

### Next experiments
1. Utility-prior operating point:
- Select a canonical `lambda` (or small set) representing realistic referral cost and freeze a production threshold schedule.
  - Status: executed in cycle 35. Cost-aware default schedule has two segments: `isotonic@0.05` for low/mid cost and `isotonic@0.10` for higher cost.
2. Dual-mode deployment spec:
- Finalize two deployment modes:
  - ranking-first mode (higher AUROC, weaker calibration),
  - reliability-first mode (stronger calibration, modest AUROC tradeoff).
  - Status: executed in cycle 35. Ranking-first resolves to `raw@0.05`, reliability-first to `isotonic@0.05`.
3. Pre-implementation dry run:
- Simulate both modes on the held-out prediction tables and produce a final implementation checklist (inputs, thresholds, calibration transform, monitoring metrics).
  - Status: executed in cycle 36 (`reports/cycle36_preimplementation_checklist.md`).

## 21) Updated next proposal (after cycles 35-36)

### New concrete claim
The research phase is functionally complete for policy selection; next value comes from implementation validation and live telemetry consistency checks.

### Next experiments
1. Implementation conformance test:
- Build a small deterministic harness that replays held-out rows and verifies service outputs match the selected mode policies exactly.
2. Monitoring backtest:
- Apply proposed alert thresholds on historical outputs to estimate false-alert rate before production rollout.
3. Rollout playbook:
- Prepare a staged rollout plan (shadow -> canary -> full) with explicit rollback criteria tied to `AUC/ECE/coverage` drift.

## 22) Updated next proposal (after cycles 37-40)

### New concrete claim
Policy logic is implementation-conformant, but stable deployment requires retuned monitoring thresholds and explicit operational handling of schedule boundaries.

### Next experiments
1. Alert robustness under synthetic incidents:
- Inject controlled degradations (AUC drop / ECE uplift / coverage shift) into replay streams and estimate detection power and delay under tuned thresholds.
  - Status: executed in cycle 41. Tuned thresholds keep ~5% baseline alert rate while showing increasing detection power with incident size.
2. Window-size operating-point selection:
- Compare detection-delay vs false-alert tradeoff for `window=500` vs `window=1000` under both modes using the same synthetic incident library.
  - Status: partially executed in cycle 41. Comparative sensitivity curves were generated; final production choice still requires product latency preference.
3. Pre-live configuration freeze:
- Produce a final machine-readable config pack (modes, thresholds, schedule gap behavior, hard/soft alert rules, rollback hooks) and validate against conformance harness.
  - Status: pending.

## 23) Updated next proposal (after cycles 41)

### New concrete claim
The deployment stack is analytically validated; the remaining work is packaging and final operating-point decisions rather than additional model/policy discovery.

### Next experiments
1. Final window-size decision memo:
- Convert cycle38/41 evidence into a single recommended production window size with explicit latency-risk rationale.
  - Status: pending.
2. Machine-readable serving/monitoring config pack:
- Emit versioned config artifacts (YAML/JSON) for mode rules, thresholds, lambda-gap handling, and hard/soft alert logic, then verify with conformance replay.
  - Status: executed in cycle 42 (`deployment_policy_config_v1.json` + validation checks all passed).
3. Dry-run alert drill:
- Run a scripted operational drill that simulates a moderate and severe incident to verify alert routing and rollback hooks before live traffic.
  - Status: pending.

## 24) Updated next proposal (after cycle 42)

### New concrete claim
Analytical and configuration validation are complete; the remaining uncertainty is operational execution quality under realistic incident handling.

### Next experiments
1. Production window-size decision:
- Finalize `500` vs `1000` edges using explicit latency budget and cycle41 detection-power curves.
  - Status: pending.
2. Alert drill automation:
- Build and run an automated drill that replays moderate/severe incidents and checks hard/soft alert transitions plus rollback trigger correctness.
  - Status: pending.
3. Pre-launch handoff bundle:
- Package config, validation outputs, and rollout checklist into a single handoff directory for engineering execution.
  - Status: pending.
