#!/usr/bin/env python3
"""WP9 - Ensembling the geometric signal with expression-based GRN inference.

Addresses Reviewer 2 comment 5: provide solid evidence (not just Discussion) that
the geometric residual-stream signal (GRS) adds value on top of an established
expression-based GRN inference method.

Two expression-based GRN scores are computed on each domain's single-cell matrix,
restricted to the TRRUST gene universe:
  * coexpr  - absolute Pearson correlation between TF and target expression
              (the canonical co-expression signal underlying SCENIC/GENIE3)
  * genie3  - GENIE3-style tree-ensemble importance: for each target gene, a
              random-forest regressor predicts its expression from all candidate
              TFs; the TF->target importance is the edge score.

We then compare three predictors under edge-level and leave-both-out CV:
  * baseline + GRN            (expression-based method alone)
  * baseline + GRS            (geometry alone)
  * baseline + GRN + GRS      (ensemble)
The incremental AUROC/AUPRC of adding GRS to the GRN method is the headline.

Output: implementation/outputs/cycle50_grn_ensemble/grn_ensemble_summary.csv
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestRegressor

import revision_lib as rl

OUT = rl.OUTPUTS / "cycle50_grn_ensemble"
MAX_CELLS = 400
RF_TREES = 40


def dense_expr(adata, gene_idx, max_cells, seed=42):
    X = adata.X
    if X.shape[0] > max_cells:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(X.shape[0], max_cells, replace=False))
        X = X[keep]
    X = X[:, gene_idx]
    return np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for domain in rl.DOMAINS:
        dom = rl.load_domain(domain, 42)
        edges = dom.edges
        # unique genes + index mapping (idx into adata.var_names == embedding rows)
        name2idx = {}
        for col, ic in [("source", "source_idx"), ("target", "target_idx")]:
            for n, i in zip(edges[col], edges[ic]):
                name2idx[n] = int(i)
        genes = sorted(name2idx)
        gidx = np.array([name2idx[g] for g in genes])
        g2col = {g: c for c, g in enumerate(genes)}

        adata = ad.read_h5ad(rl.DOMAIN_H5AD[domain])
        expr = dense_expr(adata, gidx, MAX_CELLS)        # (cells, genes)
        del adata

        # ---- co-expression score ----
        expr_z = (expr - expr.mean(0)) / (expr.std(0) + 1e-8)
        corr = (expr_z.T @ expr_z) / expr_z.shape[0]     # gene x gene Pearson

        # ---- GENIE3-style tree importance ----
        tf_genes = sorted(set(edges["source"]))
        tf_cols = np.array([g2col[g] for g in tf_genes])
        tf_X = expr[:, tf_cols]
        target_genes = sorted(set(edges["target"]))
        genie = {}   # (tf, target) -> importance
        for tg in target_genes:
            y = expr[:, g2col[tg]]
            if y.std() < 1e-9:
                continue
            rf = RandomForestRegressor(n_estimators=RF_TREES, max_depth=8,
                                       n_jobs=-1, random_state=0)
            mask = np.array([g != tg for g in tf_genes])
            rf.fit(tf_X[:, mask], y)
            imp = np.zeros(len(tf_genes))
            imp[mask] = rf.feature_importances_
            for tf, v in zip(tf_genes, imp):
                genie[(tf, tg)] = v
        print(f"[{domain}] GRN scores computed: {len(genie)} GENIE3 edge importances")

        src = edges["source"].to_numpy()
        tgt = edges["target"].to_numpy()
        coexpr_score = np.array([abs(corr[g2col[s], g2col[t]])
                                 for s, t in zip(src, tgt)])
        genie_score = np.array([genie.get((s, t), 0.0) for s, t in zip(src, tgt)])

        # ---- geometric signal (centered-cosine bundle) ----
        geom = rl.geom_feature_matrix(dom, "centered_cosine", list(range(dom.n_layers)))
        m = rl.valid_mask(dom, list(range(dom.n_layers)), geom) & \
            np.isfinite(coexpr_score) & np.isfinite(genie_score)
        y = dom.labels[m].astype(int)
        base = dom.baseline[m]
        grn = np.column_stack([coexpr_score[m], genie_score[m]])
        geo = geom[m]
        tf_m, tg_m = src[m], tgt[m]

        for cv_mode in ("edge", "leave_both"):
            if cv_mode == "edge":
                splits = rl.edge_cv_splits(y, seed=42)
            else:
                splits = rl.leave_both_out_splits(tf_m, tg_m, seed=42)
                used = np.unique(np.concatenate(
                    [np.concatenate([tr, te]) for tr, te in splits]))
                remap = {o: i for i, o in enumerate(used)}
                yy = y[used]; bb = base[used]; gg = grn[used]; ee = geo[used]
                splits = [(np.array([remap[i] for i in tr]),
                           np.array([remap[i] for i in te])) for tr, te in splits]
            if cv_mode == "edge":
                yy, bb, gg, ee = y, base, grn, geo

            feats = {
                "baseline": bb,
                "baseline+GRN": np.column_stack([bb, gg]),
                "baseline+GRS": np.column_stack([bb, ee]),
                "baseline+GRN+GRS": np.column_stack([bb, gg, ee]),
            }
            probs = {k: rl.cv_oof_probs(v, yy, splits) for k, v in feats.items()}
            aurocs = {k: rl.safe_auroc(yy, p) for k, p in probs.items()}
            auprcs = {k: rl.safe_auprc(yy, p) for k, p in probs.items()}
            bs = rl.bootstrap_delta(yy, probs["baseline+GRN"], probs["baseline+GRN+GRS"],
                                    n_iters=1000, seed=4)
            row = dict(
                domain=domain, cv_mode=cv_mode, n_edges=int(len(yy)),
                auroc_baseline=aurocs["baseline"],
                auroc_GRN=aurocs["baseline+GRN"],
                auroc_GRS=aurocs["baseline+GRS"],
                auroc_GRN_plus_GRS=aurocs["baseline+GRN+GRS"],
                auprc_GRN=auprcs["baseline+GRN"],
                auprc_GRN_plus_GRS=auprcs["baseline+GRN+GRS"],
                delta_auroc_GRS_over_GRN=aurocs["baseline+GRN+GRS"] - aurocs["baseline+GRN"],
                delta_auprc_GRS_over_GRN=auprcs["baseline+GRN+GRS"] - auprcs["baseline+GRN"],
                boot_delta_lo=bs["delta_lo"], boot_delta_hi=bs["delta_hi"],
                ci_excludes_zero=bool(bs["delta_lo"] > 0))
            rows.append(row)
            print(f"[{domain:13s} {cv_mode:11s}] GRN={row['auroc_GRN']:.3f} "
                  f"GRS={row['auroc_GRS']:.3f} GRN+GRS={row['auroc_GRN_plus_GRS']:.3f} "
                  f"dAUROC(GRS|GRN)={row['delta_auroc_GRS_over_GRN']:+.4f} "
                  f"CI=[{bs['delta_lo']:+.4f},{bs['delta_hi']:+.4f}]")
    pd.DataFrame(rows).to_csv(OUT / "grn_ensemble_summary.csv", index=False)
    print(f"\n[done] {OUT / 'grn_ensemble_summary.csv'}")


if __name__ == "__main__":
    main()
