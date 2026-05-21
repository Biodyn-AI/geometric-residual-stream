#!/usr/bin/env python3
"""WP8 - Directionality of the geometric regulatory signal.

Addresses Reviewer 2 comment 6: the manuscript's geometric metrics (cosine,
centered cosine, L2, dot) are all symmetric in (source, target) and therefore
cannot, by construction, distinguish TF->target from target->TF. Here we test
whether asymmetric geometric features carry directional information.

Two tasks:
  1. Edge-orientation task: given a known TRRUST edge as an UNORDERED pair,
     predict which gene is the TF. Built from antisymmetric per-layer features
     (norm difference, mean-coordinate difference, TF-centroid proximity gap).
     AUROC = 0.5 means no directional information.
  2. Gene-role task: rank a gene's propensity to act as a TF (appears as a
     TRRUST source) vs a target-only gene, from its PCA-reduced embedding.

Output: implementation/outputs/cycle48_directionality/directionality_summary.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import revision_lib as rl

OUT = rl.OUTPUTS / "cycle48_directionality"


def antisym_features(dom: rl.Domain, src_idx, tgt_idx) -> np.ndarray:
    """Per-layer antisymmetric features: reversing (s,t) flips every column sign."""
    cols = []
    for li in range(dom.n_layers):
        emb = np.asarray(dom.embeddings[li], dtype=np.float32)
        centered = emb - emb.mean(axis=0, keepdims=True)
        s, t = centered[src_idx], centered[tgt_idx]
        cols.append(np.linalg.norm(s, axis=1) - np.linalg.norm(t, axis=1))   # norm gap
        cols.append(s.mean(axis=1) - t.mean(axis=1))                          # mean-coord gap
        cols.append(np.linalg.norm(s, axis=1) ** 2 - np.linalg.norm(t, axis=1) ** 2)
    return np.column_stack(cols)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for domain in rl.DOMAINS:
        dom = rl.load_domain(domain, 42)
        pos = dom.edges[dom.edges.label == 1].reset_index(drop=True)
        s_idx = pos["source_idx"].to_numpy(np.int64)
        t_idx = pos["target_idx"].to_numpy(np.int64)

        # ---- Task 1: edge orientation ----
        feat_fwd = antisym_features(dom, s_idx, t_idx)     # correct orientation
        feat_rev = antisym_features(dom, t_idx, s_idx)     # reversed
        X = np.vstack([feat_fwd, feat_rev])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(pos))]).astype(int)
        # group by unordered pair so a pair and its reverse never split across folds
        pair_id = np.concatenate([np.arange(len(pos)), np.arange(len(pos))])
        finite = np.all(np.isfinite(X), axis=1)
        X, y, pair_id = X[finite], y[finite], pair_id[finite]
        model = Pipeline([("sc", StandardScaler()),
                          ("lr", LogisticRegression(max_iter=2000, random_state=0))])
        gkf = GroupKFold(n_splits=5)
        oof = cross_val_predict(model, X, y, cv=gkf, groups=pair_id,
                                method="predict_proba")[:, 1]
        auroc_orient = roc_auc_score(y, oof)

        # ---- Task 2: gene-role (TF vs target-only) ----
        tf_genes = set(pos["source"])
        tgt_only = set(pos["target"]) - tf_genes
        name2idx = {}
        for col, ic in [("source", "source_idx"), ("target", "target_idx")]:
            for n, i in zip(dom.edges[col], dom.edges[ic]):
                name2idx[n] = int(i)
        genes = sorted(tf_genes | tgt_only)
        gidx = np.array([name2idx[g] for g in genes])
        role_y = np.array([1 if g in tf_genes else 0 for g in genes], int)
        emb_mid = np.asarray(dom.embeddings[dom.n_layers // 2], dtype=np.float32)
        centered = emb_mid - emb_mid.mean(axis=0, keepdims=True)
        k = min(32, centered.shape[1])
        proj = PCA(n_components=k, svd_solver="randomized", random_state=42).fit_transform(centered)
        Xg = proj[gidx]
        if len(np.unique(role_y)) == 2:
            oof_g = cross_val_predict(model, Xg, role_y, cv=5, method="predict_proba")[:, 1]
            auroc_role = roc_auc_score(role_y, oof_g)
        else:
            auroc_role = float("nan")

        rows.append(dict(
            domain=domain, n_positive_edges=int(len(pos)),
            orientation_auroc=float(auroc_orient),
            orientation_interpretable=("directional signal" if auroc_orient > 0.55
                                       else "near-symmetric / no clear direction"),
            n_tf=len(tf_genes), n_target_only=len(tgt_only),
            gene_role_auroc=float(auroc_role)))
        print(f"[{domain:13s}] orientation AUROC={auroc_orient:.3f}  "
              f"gene-role AUROC={auroc_role:.3f}  "
              f"(n_pos={len(pos)}, n_TF={len(tf_genes)})")
    pd.DataFrame(rows).to_csv(OUT / "directionality_summary.csv", index=False)
    print(f"\n[done] {OUT / 'directionality_summary.csv'}")


if __name__ == "__main__":
    main()
