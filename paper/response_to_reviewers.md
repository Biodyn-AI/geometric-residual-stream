# Response to Reviewers — BMC Bioinformatics, Submission 2222a497-3fe7-4af2-bb39-d283ab0df022

**Manuscript:** *Residual-Stream Geometry of Single-Cell Foundation Models Carries Incremental Gene-Regulatory Signal Across Tissues* (revised title)

We thank the editor and both reviewers for a constructive and detailed assessment.
Every substantive point has been addressed with **new experiments** rather than
text changes alone; claim language was then recalibrated in light of the new
results. All new analyses are reproducible from the analysis code added to the
public repository.

Reviewer quotations are in *italics*; our response and the resulting manuscript
changes follow each.

---

## Editor

> *Please ensure the results are accurately reported, any overstated conclusions are rewritten and the limitations of the work fully explained.*

We have (i) added absolute AUROC and AUPRC to every headline table; (ii) added a
leakage-resistant grouped-CV analysis that honestly bounds the signal (it
collapses under leave-both-out CV); (iii) frozen a confirmatory pipeline applied
once to held-out domains; (iv) recalibrated the title, abstract, contributions,
Discussion and Conclusions to "incremental signal for retrospective edge
prioritization"; and (v) expanded Limitations from five to seven points. Details
under the reviewer responses below.

---

## Reviewer 1

**1. Calibrate core claims more conservatively.**
We recalibrated throughout. The **title** now reads "Carries Incremental
Gene-Regulatory Signal" (was "Encodes Gene Regulatory Structure"). The abstract,
the five-point contribution list, Discussion and Conclusions now use "incremental
regulatory-relevant signal for retrospective edge prioritization" and explicitly
state that the signal is statistical enrichment, not a stand-alone classifier.
The recalibration is grounded in the new grouped-CV results (point 3), not only
in wording.

**2. Make exploratory vs confirmatory analyses explicit.**
New experiment. We added a Methods subsection "Exploratory versus
confirmatory analyses" stating that representation choices were selected on
kidney and immune. We then **froze** a single pre-specified pipeline (centered
cosine, bundle L0–11, stacking; no per-domain tuning) and applied it once to the
held-out confirmatory domains. Result: lung ΔAUROC +0.032±0.006, external lung
+0.027±0.005 (3/3 seeds with bootstrap CIs above zero) under edge CV; +0.0016 and
+0.0009 under leave-both-out. Reported in a new Results subsection and Table.

**3. Strengthen leakage/generalization stress tests.**
New experiments. We re-evaluated under leave-TF-out, leave-target-
out and leave-both-out grouped CV (new Methods subsection; new Table "Geometric
signal under grouped, leakage-resistant cross-validation"). The signal is fully
robust to leave-TF-out and leave-target-out (ΔAUROC comparable to or larger than
edge CV) but **collapses to near-zero under leave-both-out** (0 to +0.003);
the confound baseline also drops to AUROC≈0.50 there. This is now reported as the
honest ceiling of out-of-entity generalization, in Results, Discussion and as
Limitation 3.

**4. Expand and clarify negative-edge robustness.**
New experiment. New Methods subsection states the exact protocol
(single rejection-sampled draw for the primary set). We additionally generated
**degree-matched** and **expression-matched** harder negatives, each repeated for
8 independent draws. The signal is stable across draws (SD ≤ 0.042) and does not
weaken under harder negatives — in lung domains degree-matched negatives
*increase* ΔAUROC. New Table "Robustness to negative-edge construction."

**5. Temper the scGPT vs Geneformer comparison / make representations comparable.**
We chose to **make the extraction comparable** (option a). New experiment: we
extracted Geneformer per-layer residual-stream activations from real
forward passes over single cells (18-layer model, official rank-value
tokenization) and applied the identical geometric pipeline to both models. Under
matched extraction the apparent Geneformer advantage largely disappears (scGPT
ahead in kidney; remaining gaps +0.009–0.023). New Results subsection and Table
"Cross-model comparison under matched per-layer extraction"; the old Table caption
now flags its asymmetry; architectural-superiority language removed.

**6. Improve reporting of absolute performance, uncertainty, multiplicity.**
New experiments. Absolute AUROC and AUPRC are now in
every headline table and in a full per-layer supplementary panel. New Methods
subsection "Uncertainty decomposition and multiplicity": the bootstrap unit is
stated (edges, or gene-clusters under grouped CV); variance is decomposed into
within-run and across-seed components (across-seed dominates, 64–83%); and the
148-comparison grid is Benjamini–Hochberg corrected (42 survive q<0.05). AUPRC is
reported throughout (see also Reviewer 2.4).

**7. Add operational detail for reproducibility.**
New Methods paragraph "Implementation detail" (logistic-regression
hyperparameters, in-fold standardization, randomized-SVD PCA configuration,
cell-sampling policy, coverage filters, gene/cell aggregation) and expanded
Methods subsections for every new analysis. The repository README lists
reproduction commands for all new analyses.

**8. Keep biological interpretation clearly labeled as interpretation.**
The CD8⁺ T-cell paragraph now explicitly frames the account as "a biologically
plausible hypothesis ... offered as interpretation rather than as an established
mechanism" and lists alternative explanations (abundance, library complexity,
annotation density) that are not excluded.

---

## Reviewer 2

**1. Report actual AUROC, not just delta.**
Done. Every headline table now reports baseline-only and geometry-augmented
absolute AUROC. Baseline AUROC ranges 0.50–0.65; the geometry-augmented AUROC
(0.60–0.69) is always above both 0.5 and the baseline, so the deltas are genuine
improvements over a meaningful starting point.

**2. Actual AUROC for usability; envision how to sift the signal.**
New experiment. New Results subsection reports top-k retrieval:
the geometry-augmented model enriches true edges 1.6–2.1× over prevalence in the
top 50 candidates, and we report a calibrated operating point (precision/recall).
We state plainly that the modest absolute AUROC means the signal is an evidence
channel for re-ranking, not a stand-alone caller, and the Discussion gives a
concrete pipeline (symmetric score for presence + asymmetric score for
orientation + ensembling with GENIE3).

**3. Apply the refined method to the earlier results.**
Done. The refined pipeline (centered cosine, PCA, bundle) is now applied
uniformly to kidney and immune as well; it improves those domains too (kidney
ΔAUROC +0.076→+0.122; immune +0.024→+0.042). New unified-method Table replaces
the old two-panel table; Results states the refinement is a uniform improvement,
not a lung-specific rescue.

**4. AUROC can be misleading under class imbalance — report AUPRC.**
Done. AUPRC and ΔAUPRC are reported in all headline tables and the supplementary
panel, and precision/recall@k in the new retrieval analysis. Prevalence is
reported per domain so AUPRC can be read against its baseline.

**5. Ensemble with existing GRN inference methods — solid evidence.**
New experiment. New Results subsection and Table "Geometric signal
on top of expression-based GRN inference": we computed GENIE3-style random-forest
importance and Pearson co-expression on each domain and show that adding the
geometric signal improves AUROC by +0.023 to +0.085 under edge CV, all bootstrap
CIs excluding zero. The Discussion's GRN-implications subsection now cites this
direct evidence rather than only envisioning it.

**6. Can GRS provide directionality? The metrics appear symmetric.**
New experiment. The reviewer is correct that cosine/centered
cosine/L2/dot are symmetric and cannot encode direction. We added a directionality
analysis using **asymmetric** per-layer features (signed norm/mean differences):
under pair-grouped CV they predict edge orientation at AUROC 0.80–0.90, and a
gene-role classifier separates TFs from target-only genes at AUROC 0.74–0.78. New
Results subsection "Asymmetric geometric features carry directional information";
the symmetry of the main metrics is now stated explicitly in Methods and Results.

---

## Summary of new manuscript content

- Revised title; recalibrated abstract, contributions, Discussion, Conclusions.
- New Results subsections: leakage-resistant CV; negative robustness +
  multiplicity + confirmatory rerun; matched cross-model extraction;
  directionality; complementing expression-based GRN inference.
- New Tables: grouped CV; negative-edge robustness; matched cross-model
  comparison; GRN ensemble. Tables 1–2 rebuilt with absolute AUROC/AUPRC.
- New Methods subsections: grouped CV; negative sampling; uncertainty &
  multiplicity; exploratory vs confirmatory; Geneformer residual-stream
  extraction; directionality; GRN ensemble; plus an implementation-detail
  paragraph.
- Limitations expanded from 5 to 7 points.
- All new code and analysis outputs added to the public repository.
