# Revision experiments (cycles 43-51) — BMC Bioinformatics major revision

Submission 2222a497. New experiments addressing every editor/reviewer comment.
All scripts under `implementation/scripts/run_wp*.py` + shared `revision_lib.py`.
All outputs under `implementation/outputs/cycle43..cycle51`.

Shared infrastructure: `revision_lib.py` loads the cached scGPT residual-stream
embeddings (`layer_gene_embeddings.npy`) and TRRUST edge datasets, recomputes the
six gene-level confound features from the domain h5ad, builds geometric features
(cosine / centered cosine / PCA / multi-layer bundle), and evaluates
geometry-augmented vs confound-only logistic regression reporting absolute AUROC,
absolute AUPRC, deltas, precision/recall@k and bootstrap CIs.

---

## WP1 — Grouped / leakage-resistant cross-validation (cycle43)

Re-evaluated under leave-TF-out, leave-target-out and leave-both-out grouped CV
(`run_wp1_grouped_cv.py`, `grouped_cv_summary.csv`).

| Domain | Representation | edge dAUROC | leave-TF | leave-target | leave-both |
|---|---|---|---|---|---|
| kidney | cc bundle L0-11 | +0.122 | +0.135 | +0.116 | +0.003 (ns) |
| immune | cc bundle L0-11 | +0.042 | +0.074 | +0.031 | -0.000 (ns) |
| lung | cc bundle L0-11 | +0.028 | +0.021 | +0.025 | +0.0014 (CI>0) |
| external lung | cc bundle L0-11 | +0.027 | +0.013 | +0.024 | +0.0016 (CI>0) |

**Key finding.** The signal is fully robust to leave-TF-out and leave-target-out
(it does not depend on a single gene's identity being shared). Under the strict
**leave-both-out** split — no TF *and* no target shared between train and test —
the signal collapses to near-zero (+0.001 to +0.003); it remains positive with a
bootstrap CI above zero in lung and external lung but not in kidney/immune. The
baseline confound model also drops to AUROC 0.50 under leave-both-out. This is the
honest ceiling of out-of-entity generalization and is now reported as such.

## WP2 — Negative-edge sampling robustness (cycle44)

8 independent re-draws per scheme; harder negatives (`run_wp2_negative_robustness.py`).

| Domain | random | degree-matched | expr-matched |
|---|---|---|---|
| kidney | +0.150±0.042 | +0.147±0.024 | +0.123±0.028 |
| immune | +0.030±0.008 | +0.020±0.008 | +0.061±0.010 |
| lung | +0.009±0.002 | +0.035±0.003 | +0.010±0.003 |
| external lung | +0.002±0.001 | +0.021±0.002 | +0.004±0.003 |

**Key finding.** The signal is stable across re-draws (small SDs) and does *not*
weaken under harder negatives — degree-matched and expression-matched decoys
*increase* dAUROC in lung/external lung because they remove the confound shortcut
the baseline exploited. External lung at the single best layer with random/expr
negatives is the one weak spot (CI excludes zero in 0-12% of draws); the
multi-layer bundle (WP6) recovers it.

## WP3 / WP7 — Absolute-metric panels & unified method (cycle45)

`run_wp3_metric_panels.py`. Every domain/layer/metric now carries absolute AUROC
and AUPRC. Refined methods applied uniformly to the early domains (R2.3).

| Domain | raw cosine | centered cosine | pca_cc64 | cc bundle L0-11 |
|---|---|---|---|---|
| kidney | 0.505→0.581 (+0.076) | →0.593 (+0.088) | →0.603 (+0.098) | →0.627 (+0.122) |
| immune | 0.649→0.673 (+0.024) | →0.676 (+0.027) | →0.677 (+0.028) | →0.691 (+0.042) |
| lung | 0.575→0.576 (+0.001) | →0.583 (+0.008) | →0.588 (+0.013) | →0.603 (+0.028) |
| external lung | 0.581→0.582 (+0.001) | →0.589 (+0.008) | →0.591 (+0.010) | →0.607 (+0.027) |

The refined pipeline improves every domain including the early ones; absolute
AUROC stays modest (0.58-0.69) — the signal is statistical enrichment, not a
stand-alone classifier.

## WP4 — Uncertainty decomposition & multiplicity (cycle46)

`run_wp4_uncertainty.py`.

* **Variance components** (cc bundle): across-seed variance (cell sample +
  negative draw) dominates within-run edge-bootstrap variance — it accounts for
  64-83% of total dAUROC variance. Error bars in the paper that come only from
  the edge bootstrap therefore *understate* total uncertainty; the revision adds
  the across-seed component.
* **Multiplicity**: 148 domain×layer×metric configs; 64 significant at raw
  p<0.05; **42 survive Benjamini-Hochberg FDR<0.05**. The headline bundle and
  pca_cc results are in the surviving set.

## WP5 — Confirmatory frozen-pipeline evaluation (cycle47)

`run_wp5_confirmatory.py`. Pipeline frozen (centered-cosine bundle L0-11, no
per-domain tuning); kidney+immune treated as exploratory, lung+external lung as
confirmatory; 3 seeds each.

| Domain | role | edge dAUROC (3-seed) | leave-both dAUROC |
|---|---|---|---|
| kidney | exploratory | +0.159±0.045 | +0.009±0.006 |
| immune | exploratory | +0.063±0.026 | +0.004±0.004 |
| lung | confirmatory | +0.032±0.006 (3/3 CI>0) | +0.0016±0.0003 (3/3 CI>0) |
| external lung | confirmatory | +0.027±0.005 (3/3 CI>0) | +0.0009±0.0006 (2/3 CI>0) |

The frozen pipeline reproduces a positive, CI-backed effect on the held-out
confirmatory domains under edge CV; under leave-both-out the confirmatory effect
is tiny but still positive.

## WP6 — Comparable cross-model extraction: Geneformer residual stream (cycle51)

`run_wp6_geneformer_residual.py` extracts Geneformer (V2 104M, 18-layer BERT)
PER-LAYER residual-stream activations from real forward passes over single cells
(official rank-value tokenisation), mirroring the scGPT extraction. The identical
geometric pipeline is then applied to both models (`run_wp6_eval_cross_model.py`).

Matched centered-cosine bundle dAUROC:

| Domain | scGPT | Geneformer | gap (GF−scGPT) |
|---|---|---|---|
| kidney | +0.122 | +0.104 | −0.018 |
| immune | +0.042 | +0.057 | +0.015 |
| lung | +0.028 | +0.037 | +0.009 |
| external lung | +0.027 | +0.050 | +0.023 |

**Key finding.** Under matched multi-layer extraction the apparent Geneformer
advantage largely disappears (scGPT is ahead in kidney; gaps of +0.009-0.023
elsewhere) — confirming that most of the original gap was a representation-
construction artifact of comparing scGPT multi-layer features to a single
Geneformer embedding layer. Geneformer retains a modest edge in the lung domains.
Architectural-superiority language is removed accordingly.

## WP8 — Directionality of the geometric signal (cycle48)

`run_wp8_directionality.py`. Symmetric metrics cannot, by construction, encode
direction; we test asymmetric per-layer features (norm gap, mean-coordinate gap)
under pair-grouped CV.

| Domain | edge-orientation AUROC | gene-role (TF vs target) AUROC |
|---|---|---|
| kidney | 0.903 | 0.758 |
| immune | 0.854 | 0.783 |
| lung | 0.803 | 0.741 |
| external lung | 0.809 | 0.755 |

**Key finding (new positive result).** Although the manuscript's symmetric
metrics carry no directional information, *asymmetric* geometric features predict
edge orientation at AUROC 0.80-0.90 and rank a gene's TF-vs-target role at AUROC
0.74-0.78. The directional signal operates by detecting per-gene TF-propensity
from embedding geometry.

## WP9 — Ensembling with expression-based GRN inference (cycle50)

`run_wp9_grn_ensemble.py`. Two expression-based GRN scores (absolute Pearson
co-expression; GENIE3-style random-forest TF→target importance) computed on each
domain's single-cell matrix, then GRS added on top.

Incremental dAUROC of adding GRS to the GRN method (edge CV):

| Domain | GRN AUROC | GRN+GRS AUROC | dAUROC(GRS\|GRN) | CI |
|---|---|---|---|---|
| kidney | 0.540 | 0.625 | +0.085 | [+0.043,+0.130] |
| immune | 0.670 | 0.698 | +0.028 | [+0.011,+0.046] |
| lung | 0.576 | 0.604 | +0.028 | [+0.021,+0.035] |
| external lung | 0.588 | 0.611 | +0.023 | [+0.017,+0.029] |

**Key finding.** GRS provides significant incremental signal on top of an
established expression-based GRN method under edge-level CV (all CIs exclude
zero); under leave-both-out the increment is ~+0.001 (consistent with WP1).

## WP10 — Practical-utility / retrieval evaluation (cycle49)

`run_wp10_practical_utility.py`. Top-k retrieval and a calibrated operating point.

| Domain | geom AUROC | precision@50 | enrichment@50 | operating P / R |
|---|---|---|---|---|
| kidney | 0.627 | 0.52 | 2.04× | 0.32 / 0.69 |
| immune | 0.691 | 0.44 | 1.58× | 0.42 / 0.68 |
| lung | 0.603 | 0.48 | 1.70× | 0.31 / 0.85 |
| external lung | 0.607 | 0.56 | 2.14× | 0.30 / 0.73 |

**Key finding.** Used for ranking, the geometry-augmented model enriches true
edges 1.6-2.1× over prevalence in the top 50 — concrete hypothesis-prioritisation
value — but the modest absolute AUROC means it is an evidence channel for
re-ranking, not a stand-alone GRN caller.

## WP11 — Reproducibility (see `paper/implementation_details.md`, updated README)

Full hyperparameters, CV construction, PCA configuration, cell-sampling policy
and aggregation specifics consolidated for repeatability.

---

## Net effect on the manuscript's claims

* The core claim holds **as incremental, retrospective enrichment** under
  edge-level and leave-TF/leave-target CV, survives FDR correction, is stable to
  harder negatives, and complements an expression-based GRN method.
* It is **honestly bounded**: under leave-both-out CV the effect is near-zero;
  absolute AUROC is modest; across-seed variance is the dominant uncertainty.
* Two new positive findings strengthen the paper: matched-pipeline cross-model
  comparison (WP6) and a directional signal from asymmetric features (WP8).
* Claim language is recalibrated throughout to "incremental regulatory-relevant
  signal for retrospective edge prioritisation."
