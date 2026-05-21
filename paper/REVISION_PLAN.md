# Revision Plan — BMC Bioinformatics, Submission 2222a497

> **STATUS (2026-05-21): EXECUTED.** All 11 work packages were run as cycles
> 43–51 (`implementation/scripts/run_wp*.py`, `revision_lib.py`). Results:
> `reports/revision_cycles43-51_new_experiments.md`. Point-by-point reviewer
> response: `paper/response_to_reviewers.md`. Manuscript revised and recompiled
> (`geometric_residual_stream_bmc.pdf`, 22 pp.). The plan below is the original
> design document.

---


**Manuscript:** *Residual-Stream Geometry of Single-Cell Foundation Models Encodes Gene Regulatory Structure Across Tissues*

**Decision:** Major revision. Two reviewers + editor.

**Guiding principle for this plan:** wherever a comment can be addressed *either* by reframing text *or* by running new/additional experiments, we choose to **run the experiments** and then reframe the text in light of the new results. Reframing-only is reserved for comments that are purely about wording (e.g., claim calibration) where no experiment can change the underlying fact.

This plan is organized in three parts:
- **Part A** — New experiments to run (the substantive work), grouped into work packages.
- **Part B** — Per-comment mapping (every editor/reviewer point → action → status).
- **Part C** — Manuscript-edit checklist and the point-by-point response-letter skeleton.

---

## Part A — New experiments (work packages)

Each work package (WP) lists: motivation, what to run, the script(s) to create/extend, outputs, and which comments it closes. New scripts live in `implementation/scripts/`; outputs in `implementation/outputs/cycle43_*` onward (continuing the existing cycle numbering — current max is cycle42).

### WP1 — Grouped / leakage-resistant cross-validation (leave-TF-out, leave-target-out, leave-both-out)
*Addresses: R1.3 (grouped CV), R1.2 (confirmatory rerun), editor "generalization".*

**Motivation.** Current evaluation uses edge-level repeated stratified 5-fold CV. A TF or target gene can appear in both train and test folds, so the classifier can memorize gene identity. The reviewer asks for grouped splits.

**What to run.**
1. New script `run_grouped_cv_generalization.py` that wraps the existing edge-classification pipeline but replaces `RepeatedStratifiedKFold` with `sklearn.model_selection.GroupKFold` / `LeaveOneGroupOut`-style splits under three grouping keys:
   - **leave-TF-out:** group = source TF; no TF shared across folds.
   - **leave-target-out:** group = target gene.
   - **leave-both-out:** iteratively hold out all edges touching a held-out TF set *and* target set (a "blocked" split; implement via a 2D block partition of the TF×target edge matrix).
2. Run for every primary domain (kidney, immune, lung, external lung) and for both the single-best-layer scGPT setting and the seed-ensemble L0–11 bundle, plus Geneformer.
3. Report ΔAUROC, absolute AUROC, and AUPRC for each grouping, with bootstrap CIs (resampling unit = group, see WP4).

**Outputs.** `cycle43_grouped_cv/{leave_tf,leave_target,leave_both}_summary.csv`; new manuscript table "Generalization under grouped cross-validation."

**Expected framing.** Signal is expected to attenuate under grouped splits; the paper will report *how much persists* and recalibrate claims to whatever survives leave-both-out (the strictest test).

**Closes:** R1.3 (fully), contributes to R1.2, editor generalization point.

### WP2 — Harder negative-edge sampling + resampling stability
*Addresses: R1.4 (negative robustness), R2.4 partially (realistic difficulty).*

**Motivation.** Negatives are currently a single random draw of (source-pool TF, target-pool gene) pairs (`_sample_negative_edges` in `run_layerwise_geometry_audit.py`). Reviewer asks for (a) explicit protocol, (b) stability across resamples, (c) harder decoys.

**What to run.** New script `run_negative_sampling_robustness.py`:
1. **Repeated random negatives:** redraw negatives with ≥20 distinct RNG seeds; report mean ± SD of ΔAUROC/AUROC/AUPRC, so a single lucky/unlucky draw cannot drive a result.
2. **Degree-matched negatives:** for each positive edge, sample a negative whose TF out-degree and target in-degree (in the TRRUST graph restricted to the domain gene universe) fall in the same quantile bin. This removes the "hub gene" shortcut.
3. **Expression-matched negatives:** sample negatives whose source and target genes match the positives on mean expression, variance, and detection frequency (the six baseline confounds) within quantile bins. This is the hardest decoy and directly tests whether geometry adds signal *after* the confound is matched at the sampling stage, not just regressed out.
4. (If feasible) **co-expression-matched negatives:** match on the pairwise expression correlation, to separate regulatory geometry from co-expression.
5. Run for all four domains, both models, primary representation settings.

**Outputs.** `cycle44_negative_robustness/negatives_{random,degree,expr,coexpr}_summary.csv`; new table "Robustness to negative-edge construction."

**Closes:** R1.4 (fully), supports R2.4.

### WP3 — AUPRC and PR-oriented metrics everywhere; absolute metrics everywhere
*Addresses: R2.4 (AUPRC), R2.1 (baseline AUROC must be reported), R2.2 (absolute usability), R1.6 (absolute metrics + PR metrics consistently).*

**Motivation.** Most tables report only ΔAUROC. With class imbalance and a ranking objective, absolute AUROC and AUPRC are needed; a ΔAUROC of +0.01 on a baseline of 0.45 is meaningless.

**What to run.** Extend *every* analysis script's reporting (not a new experiment per se, but a re-run with augmented logging):
1. Add to all summary CSVs: baseline AUROC, geometry-augmented AUROC, baseline AUPRC, geometry-augmented AUPRC, ΔAUPRC, positive-class prevalence, and the AUPRC-vs-prevalence baseline (= prevalence). `average_precision_score` is already imported in `run_layerwise_geometry_audit.py` — surface it.
2. Re-run the headline analyses (Tables 1–6) so every table carries absolute AUROC **and** AUPRC alongside any delta.
3. Add an "early-retrieval" metric relevant to prioritization: precision@k and recall@k (k = 50, 100) and partial AUROC over the top-ranked region — this directly answers R2.2 "is it usable for finding regulatory relationships."

**Outputs.** Revised versions of all six results tables; one new supplementary table with full metric panels per domain/model/layer.

**Closes:** R2.1 (fully), R2.4 (fully), R1.6 metric portion (fully), supports R2.2.

### WP4 — Uncertainty decomposition and multiplicity control
*Addresses: R1.6 (bootstrap unit, variance sources, multiplicity).*

**Motivation.** Reviewer wants clarity on the bootstrap resampling unit, which variance sources are captured, and how the many tissue/layer/metric/model comparisons are corrected.

**What to run.** New script `run_uncertainty_decomposition.py`:
1. **Variance components:** run a nested resampling that separately varies (a) CV folding, (b) cell-sampling seed, (c) negative-edge resample (from WP2), and report a variance-component breakdown (e.g., a simple ANOVA-style decomposition of ΔAUROC variance into fold / seed / negative-draw / residual). This explicitly tells the reader what each error bar contains.
2. **Bootstrap unit:** switch the bootstrap to resample **edges grouped by gene** (cluster bootstrap) wherever grouped CV is used, and document the unit explicitly in Methods. Report both edge-level and gene-cluster bootstrap CIs for the headline numbers so the reader sees the difference.
3. **Multiplicity:** assemble the full grid of tested comparisons (domains × layers × metrics × models) and apply Benjamini–Hochberg FDR control to the null-control p-values; report which results survive. Designate the confirmatory subset (WP5) as the primary family and treat the rest as exploratory.

**Outputs.** `cycle45_uncertainty/variance_components.csv`, `multiplicity_adjusted_pvalues.csv`; new Methods subsection and a supplementary table.

**Closes:** R1.6 (fully).

### WP5 — Confirmatory rerun with frozen, nested settings
*Addresses: R1.2 (exploratory vs confirmatory), editor "results accurately reported / overstated conclusions."*

**Motivation.** The narrative iteratively chose metric (centered cosine), PCA dimension, layer/bundle, and seed-aggregation by inspecting performance. Reviewer wants an explicit exploratory/confirmatory split and, if choices were not nested, a stricter rerun.

**What to run.** New script `run_confirmatory_frozen_eval.py`:
1. **Freeze the protocol.** Take the representation choices the paper currently uses (centered cosine, PCA-64, L0–11 bundle, 3-seed ensemble) and treat them as a *single pre-specified configuration*. No per-domain tuning.
2. **Nested selection where any choice remains.** Where a choice (e.g., PCA dimension, best layer) is genuinely needed, select it strictly inside the training folds via inner CV, never on full-dataset performance.
3. **Held-out confirmatory domain.** Use kidney + immune as the exploratory tissues that informed the method; designate **lung and external lung as confirmatory** — apply the frozen pipeline once and report the result as the confirmatory claim. (External lung was already described as an "independent cohort," so this is a natural split.)
4. Report the confirmatory numbers separately and prominently.

**Outputs.** `cycle46_confirmatory/confirmatory_frozen_summary.csv`; a new Results paragraph "Confirmatory evaluation" and an explicit statement in Methods of which analyses were exploratory.

**Closes:** R1.2 (fully), supports editor overstatement concern.

### WP6 — Comparable cross-model extraction for scGPT vs Geneformer
*Addresses: R1.5 (asymmetric extraction), editor "complementarity" claims.*

**Motivation.** scGPT is represented by multi-layer residual-stream features (with bundling); Geneformer by a single embedding-layer vector. The comparison is apples-to-oranges. Reviewer offers (a) make extraction comparable or (b) soften claims — we choose **(a)**.

**What to run.** New script `run_geneformer_residual_stream_extraction.py`:
1. Extract Geneformer **per-layer residual-stream activations** via forward hooks (Geneformer is a 6-layer BERT-style transformer), mirroring the scGPT hook-based extraction in `run_layerwise_geometry_audit.py`, instead of using only the static embedding layer.
2. Build the **same** geometric feature pipeline for Geneformer: centered cosine, PCA projection, multi-layer bundling, seed ensembling.
3. Re-run the cross-model comparison (Table 5/6, Fig 4) with both models now represented by matched residual-stream, multi-layer, bundled features.
4. As a secondary symmetric control, also compare both models *restricted to their embedding layer only*.
5. Recompute edge-level ρ and disagreement stratification on the matched representations.

**Outputs.** `cycle47_geneformer_residual/geneformer_layerwise_summary.csv`; revised Tables 5–6 and Fig 4; revised cross-model text.

**Note.** If the matched extraction changes the complementarity story (ρ, gap signs), the Results/Discussion and abstract are updated to whatever the matched comparison shows. Architectural-superiority language is removed regardless.

**Closes:** R1.5 (fully, via option a).

### WP7 — Apply the refined methodology to the earlier (kidney/initial) results
*Addresses: R2.3 (refined method applied to all scenarios), R1.2.*

**Motivation.** Centered cosine / PCA / bundling / seed ensembling were introduced mid-paper to rescue lung. Reviewer 2 asks: does the same refined method work for the earlier scenarios?

**What to run.** Extend `run_layerwise_geometry_audit.py` (or a thin wrapper `run_refined_method_on_kidney.py`):
1. Re-run kidney and the initial immune analysis with the **full refined pipeline** (centered cosine, PCA-64, L0–11 bundle, seed ensemble) — exactly the configuration used for lung.
2. Report the refined numbers next to the original raw-cosine numbers in Table 1 / Table 2, so the reader sees one consistent method across all domains.
3. This also feeds WP5 (the frozen pipeline is then literally identical across every domain).

**Outputs.** Updated Tables 1–2 with a unified-method column; `cycle48_refined_on_kidney/summary.csv`.

**Closes:** R2.3 (fully).

### WP8 — Directionality of the regulatory link
*Addresses: R2.6 (symmetric metrics — can GRS predict direction?).*

**Motivation.** All current geometric metrics (cosine, centered cosine, L2, dot) are symmetric in (source, target), so they cannot distinguish TF→target from target→TF. Reviewer asks whether GRS carries directional information.

**What to run.** New script `run_directionality_analysis.py`:
1. **Directed task.** Among TRRUST positives, the edge is directed (TF→target). Construct the test: given an unordered gene pair known to be a regulatory edge, predict which gene is the TF. Train a classifier on **asymmetric** features:
   - difference features: `h_s − h_t` (sign-bearing), per-layer.
   - norm asymmetry: `||h_s|| − ||h_t||`, projection of one gene onto the other's direction.
   - per-gene "TF-ness" geometric features (e.g., distance to the centroid of known TFs).
2. **TF/target role classification:** can geometry alone rank a gene's propensity to be a TF vs a target?
3. Report AUROC/AUPRC for the directionality task with null controls; if signal is weak, state plainly that current GRS is largely undirected and frame WP8 as a negative/limited result.

**Outputs.** `cycle49_directionality/directionality_summary.csv`; new Results paragraph "Directionality of the geometric signal."

**Closes:** R2.6 (fully — with either a positive directional result or an explicit, evidenced negative finding).

### WP9 — Ensemble with an existing GRN-inference method
*Addresses: R2.5 (combine GRS with established GRN inference), editor practical-utility.*

**Motivation.** Reviewer 2 wants concrete evidence (not just Discussion) that GRS *improves* an existing GRN method when ensembled.

**What to run.** New script `run_grn_method_ensemble.py`:
1. Run a standard expression-based GRN inference method on the same Tabula Sapiens domain expression matrices — **GENIE3** (tree-based, no extra deps beyond `arboreto`/`scikit-learn`) as the primary, and **Pearson/MI co-expression** as a simple secondary baseline. (GRNBoost2 via `arboreto` is the fallback if GENIE3 is too slow.)
2. Score the TRRUST positive + negative edge sets with the GRN method to get a per-edge GRN score.
3. Build three predictors and compare on AUROC/AUPRC with bootstrap CIs:
   - GRN method alone,
   - GRS (geometry) alone,
   - GRN + GRS stacked (reuse the compact stacking framework in `run_outer_split_compact_model.py`).
4. Report the incremental gain of adding GRS to the GRN method, under leakage-resistant grouped CV (WP1).

**Outputs.** `cycle50_grn_ensemble/grn_plus_grs_summary.csv`; new Results subsection "Geometric signal complements expression-based GRN inference"; new figure.

**Closes:** R2.5 (fully), strengthens editor practical-utility point.

### WP10 — "How to sift the signal" / practical-usability evidence
*Addresses: R2.2 (envision ways to use the signal), editor practical-utility, R2 major concern.*

**Motivation.** Reviewer 2's central concern: do these insights translate into practically discovering regulatory relationships? If AUROC only shows "statistically more signal," show *how* it can be used.

**What to run.** New script `run_practical_utility_eval.py`:
1. **Top-k retrieval evaluation.** Rank all candidate edges in a domain by the geometric/compact score; report precision@k, recall@k, and enrichment-fold over random for k = 25/50/100. This quantifies hypothesis-prioritization utility directly.
2. **Re-ranking benefit.** Take the GRN method's ranked candidate list (WP9) and show how many true edges move into the top-k after GRS re-ranking.
3. **Operating point.** Pick a calibrated threshold (isotonic, already implemented in `run_outer_split_calibration_intervention.py`) and report precision/recall a practitioner would obtain.
4. Tie results to the three use cases already named in Discussion (edge re-ranking, hypothesis prioritization, context-dependent assessment) with concrete numbers.

**Outputs.** `cycle51_practical_utility/retrieval_metrics.csv`; new Results paragraph + figure; rewritten "Implications for GRN inference" with quantitative backing.

**Closes:** R2.2 (fully), R2 major concern, editor practical-utility.

### WP11 — Reproducibility / operational detail audit
*Addresses: R1.7 (complete implementation detail).*

**Motivation.** Reviewer 1 wants full implementation detail for repeatability.

**What to do.** Mostly documentation, but verified against code:
1. Audit every script and extract exact values for: scGPT/Geneformer hyperparameters, logistic-regression regularization (`C`, penalty, solver), feature standardization (`StandardScaler` placement in the `Pipeline`), PCA configuration (whitening, SVD solver, k values), CV fold construction (stratification keys, repeats, RNG seeds), gene-mapping and coverage filters, cell-sampling policy (256 cells, seed list, `max_genes` budget), and per-gene / per-cell aggregation (Eq. 1).
2. Expand the Methods section and add a supplementary "Implementation details" table.
3. Add exact reproduction commands for the new cycles 43–51 to `implementation/README.md`.
4. Verify the public GitHub repo link contents match.

**Closes:** R1.7 (fully).

---

## Part B — Per-comment mapping

Legend: **[EXPERIMENT]** = new/extended runs; **[REFRAME]** = text-only; every comment that *could* be experiment-backed is.

### Editor
| Editor point | Action | WP |
|---|---|---|
| Results accurately reported | Absolute metrics everywhere + confirmatory rerun **[EXPERIMENT]** | WP3, WP5 |
| Overstated conclusions rewritten | Claim calibration **[REFRAME]** driven by new (often smaller) numbers from WP1/WP5 | WP1, WP5 + Part C |
| Limitations fully explained | Expanded Limitations **[REFRAME]**, informed by grouped-CV / directionality results | WP1, WP8 |

### Reviewer 1
| # | Comment | Action | WP |
|---|---|---|---|
| 1 | Calibrate core claims conservatively | **[REFRAME]** — but only after WP1/WP5 give us the honest effect sizes; title/abstract/discussion reworded to "incremental regulatory-relevant signal for retrospective edge prioritization" | Part C (+ WP1, WP5) |
| 2 | Exploratory vs confirmatory explicit; stricter confirmatory rerun if not nested | **[EXPERIMENT]** confirmatory frozen rerun + nested-in-fold selection | WP5, WP7 |
| 3 | Grouped robustness (leave-TF-out, leave-target-out, leave-both-out) | **[EXPERIMENT]** | WP1 |
| 4 | Negative-edge protocol, resampling stability, harder negatives | **[EXPERIMENT]** | WP2 |
| 5 | Temper scGPT vs Geneformer OR make extraction comparable | **[EXPERIMENT]** — chose "make comparable" (option a) | WP6 |
| 6 | Absolute metrics + PR metrics consistently; bootstrap unit; variance sources; multiplicity | **[EXPERIMENT]** | WP3, WP4 |
| 7 | Operational detail for reproducibility | Documentation audit against code | WP11 |
| 8 | Biological interpretation labeled as interpretation | **[REFRAME]** — CD8⁺ explanation explicitly framed as hypothesis | Part C |

### Reviewer 2
| # | Comment | Action | WP |
|---|---|---|---|
| 1 | Report actual AUROC, not just delta | **[EXPERIMENT]** | WP3 |
| 2 | Actual AUROC for usability; envision how to sift signal | **[EXPERIMENT]** retrieval/utility eval | WP3, WP10 |
| 3 | Apply refined method to earlier results | **[EXPERIMENT]** | WP7 |
| 4 | Report AUPRC (imbalance) | **[EXPERIMENT]** | WP3, WP2 |
| 5 | Ensemble GRS with existing GRN inference methods — solid evidence | **[EXPERIMENT]** | WP9 |
| 6 | Direction of regulatory link — metrics are symmetric | **[EXPERIMENT]** | WP8 |

Every reviewer comment is addressed by at least one experiment except R1.1 and R1.8, which are intrinsically wording calibrations — and even those are driven by the new experimental numbers.

---

## Part C — Manuscript edits & response letter

### C1. Claim calibration (R1.1, R1.8, editor)
- **Title:** soften "Encodes Gene Regulatory Structure" → e.g., *"Residual-Stream Geometry of Single-Cell Foundation Models Carries Incremental Gene-Regulatory Signal Across Tissues."*
- **Abstract / Discussion / Conclusions:** replace phrasings implying direct regulatory encoding with "contains incremental regulatory-relevant signal for retrospective edge prioritization." Remove "establish that residual-stream geometry carries genuine ... signal" → "provides incremental signal."
- Contribution list (Background, items 1–4): reword item 1 and item 3; item 3's "architectural superiority" framing removed pending WP6.
- CD8⁺ T-cell paragraph: explicitly label as "a hypothesis consistent with the data," not established mechanism (R1.8).
- All edits made *after* WP1/WP5/WP6 results are in, so numbers and claims match.

### C2. New / revised tables and figures
- **Table 1, 2:** add absolute AUROC + AUPRC columns; add unified-refined-method column (WP7).
- **New Table:** generalization under grouped CV (WP1).
- **New Table:** negative-sampling robustness (WP2).
- **New Table:** variance components + multiplicity-adjusted p-values (WP4) — may go to supplement.
- **Tables 5–6, Fig 4:** recomputed with matched Geneformer residual-stream extraction (WP6).
- **New figure:** GRS + GRN-method ensemble performance (WP9).
- **New figure / paragraph:** top-k retrieval / practical utility (WP10).
- **New Results paragraph:** directionality (WP8).
- **New Results paragraph:** confirmatory evaluation (WP5).
- All ΔAUROC-only tables get absolute AUROC + AUPRC (WP3).

### C3. Methods additions
- Explicit exploratory-vs-confirmatory statement (WP5).
- Negative-sampling protocol spelled out, all variants (WP2).
- Bootstrap resampling unit + variance sources + multiplicity correction (WP4).
- Grouped-CV definitions (WP1).
- Geneformer residual-stream extraction (WP6).
- Full implementation-detail subsection + supplementary table (WP11).
- Directionality task definition (WP8).
- GRN-method ensemble setup (WP9).

### C4. Limitations section additions
- Quantified residual signal under leave-both-out CV (WP1).
- Directionality limits (WP8).
- Negative-set difficulty caveats now backed by WP2.

### C5. Point-by-point response letter
Create `paper/response_to_reviewers.tex` (or `.md`): one numbered entry per editor/reviewer point, each quoting the comment, stating the action, pointing to the new table/figure/section, and quoting the key new number. Structure mirrors Part B.

### C6. Done in this revision session already
- ✅ Institutional email added to author block in `geometric_residual_stream_bmc.tex` (`ihor.kendiukhov@student.uni-tuebingen.de`).

---

## Suggested execution order
1. WP11 (doc audit — cheap, informs everything) + WP3 (metric logging — touches all scripts).
2. WP7 (refined method on kidney) → WP5 (confirmatory frozen) — establishes the single honest pipeline.
3. WP1 (grouped CV) + WP2 (negatives) + WP4 (uncertainty) — the leakage/robustness core; run on the frozen pipeline.
4. WP6 (Geneformer residual stream) — re-do cross-model.
5. WP8 (directionality), WP9 (GRN ensemble), WP10 (practical utility) — the "usefulness" trio.
6. Part C: rewrite manuscript with final numbers; write response letter.

## Risks / fallbacks
- **WP6:** if Geneformer hooks are awkward, fall back to its hidden-state outputs via `output_hidden_states=True` (HF API) — still gives per-layer residual stream.
- **WP9:** if GENIE3/arboreto runtime is prohibitive on full domains, subsample genes or use the faster GRNBoost2, or restrict to the TRRUST gene universe.
- **WP1 leave-both-out:** edge counts per block may get small in kidney (288 positives); report CIs honestly and lean on the larger lung/immune sets for the strict test.
- If grouped CV collapses signal to non-significance in a domain, **report it as such** — the editor explicitly asked for accurate reporting; a partly-negative result is acceptable and expected.
