#!/usr/bin/env python3
"""Shared infrastructure for the BMC Bioinformatics revision experiments (cycles 43+).

Operates on cached scGPT residual-stream embeddings + TRRUST edge datasets produced
by run_layerwise_geometry_audit.py. Provides domain loading, geometric feature
construction (cosine / centered cosine / PCA / multi-layer bundle), and a unified
evaluation harness that reports absolute AUROC and AUPRC plus delta metrics, with
edge-level and grouped (leave-TF / leave-target / leave-both-out) cross-validation.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
SUBPROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUTS = SUBPROJECT_ROOT / "implementation" / "outputs"
BIODYN_ROOT = SUBPROJECT_ROOT.parent
SCM = BIODYN_ROOT / "single_cell_mechinterp"

# ---------------------------------------------------------------------------
# Domain registry: domain -> seed -> cached run directory; plus h5ad path.
# ---------------------------------------------------------------------------
DOMAIN_RUNS: Dict[str, Dict[int, str]] = {
    "kidney": {42: "cycle1_main", 43: "cycle1_seed43", 44: "cycle1_seed44"},
    "immune": {42: "cycle4_immune_main", 43: "cycle4_immune_seed43", 44: "cycle4_immune_seed44"},
    "lung": {42: "cycle6_lung_main", 43: "cycle6_lung_seed43", 44: "cycle6_lung_seed44"},
    "external_lung": {
        42: "cycle7_external_lung_main",
        43: "cycle7_external_lung_seed43",
        44: "cycle7_external_lung_seed44",
    },
}

DOMAIN_H5AD: Dict[str, Path] = {
    "kidney": SCM / "outputs" / "tabula_sapiens_processed.h5ad",
    "immune": SCM / "outputs" / "tabula_sapiens_immune_subset_hpn_processed.h5ad",
    "lung": SCM / "outputs" / "invariant_causal_edges" / "lung" / "processed.h5ad",
    "external_lung": SCM / "outputs" / "invariant_causal_edges" / "external_lung" / "processed.h5ad",
}

DOMAINS = list(DOMAIN_RUNS.keys())

# Per-domain best single layer reported in the manuscript (raw cosine / pca_cc).
PAPER_BEST_LAYER = {"kidney": 4, "immune": 0, "lung": 0, "external_lung": 3}


# ---------------------------------------------------------------------------
# Gene-level confound baseline.
# ---------------------------------------------------------------------------
def compute_gene_statistics(expr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean expression, variance and detection frequency per gene."""
    if sp.issparse(expr_matrix):
        mean_expr = np.asarray(expr_matrix.mean(axis=0)).ravel().astype(np.float64)
        mean_sq = np.asarray(expr_matrix.power(2).mean(axis=0)).ravel().astype(np.float64)
        variance = np.maximum(mean_sq - mean_expr**2, 0.0)
        detection = np.asarray((expr_matrix > 0).mean(axis=0)).ravel().astype(np.float64)
    else:
        dense = np.asarray(expr_matrix, dtype=np.float64)
        mean_expr = dense.mean(axis=0)
        variance = dense.var(axis=0)
        detection = (dense > 0).mean(axis=0)
    return mean_expr, variance, detection


_GENE_STATS_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def gene_stats(domain: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cached per-gene (mean, var, detection) for a domain h5ad."""
    if domain not in _GENE_STATS_CACHE:
        adata = ad.read_h5ad(DOMAIN_H5AD[domain])
        _GENE_STATS_CACHE[domain] = compute_gene_statistics(adata.X)
        del adata
    return _GENE_STATS_CACHE[domain]


# ---------------------------------------------------------------------------
# Domain payload.
# ---------------------------------------------------------------------------
@dataclass
class Domain:
    name: str
    seed: int
    edges: pd.DataFrame          # source, target, label, source_idx, target_idx
    embeddings: np.ndarray       # (n_layers, n_genes, d) memory-mapped
    counts: np.ndarray           # (n_layers, n_genes)
    baseline: np.ndarray         # (n_edges, 6) confound features
    src_idx: np.ndarray
    tgt_idx: np.ndarray
    labels: np.ndarray

    @property
    def n_layers(self) -> int:
        return self.embeddings.shape[0]


def domain_from_edges(name: str, seed: int, edges: pd.DataFrame,
                      emb: np.ndarray, counts: np.ndarray,
                      gstats: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Domain:
    """Assemble a Domain for an arbitrary edge table (used for re-sampled negatives)."""
    mean_expr, variance, detection = gstats
    edges = edges.reset_index(drop=True)
    src_idx = edges["source_idx"].to_numpy(np.int64)
    tgt_idx = edges["target_idx"].to_numpy(np.int64)
    labels = edges["label"].to_numpy(np.int32)
    baseline = np.column_stack(
        [
            mean_expr[src_idx], mean_expr[tgt_idx],
            variance[src_idx], variance[tgt_idx],
            detection[src_idx], detection[tgt_idx],
        ]
    ).astype(np.float64)
    return Domain(name, seed, edges, emb, counts, baseline, src_idx, tgt_idx, labels)


def load_domain(domain: str, seed: int = 42) -> Domain:
    run_dir = OUTPUTS / DOMAIN_RUNS[domain][seed]
    edges = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
    emb = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
    counts = np.load(run_dir / "layer_gene_embedding_count.npy")
    return domain_from_edges(domain, seed, edges, emb, counts, gene_stats(domain))


# ---------------------------------------------------------------------------
# Geometric feature construction.
# ---------------------------------------------------------------------------
def layer_score(
    emb_layer: np.ndarray,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    metric: str,
    pca_dim: Optional[int] = None,
    seed: int = 42,
) -> np.ndarray:
    """Pairwise geometric score for one layer.

    metric in {cosine, centered_cosine, pca_cc, neg_l2}. pca_dim applies to pca_cc.
    """
    emb = np.asarray(emb_layer, dtype=np.float32)
    if metric == "cosine":
        s, t = emb[src_idx], emb[tgt_idx]
        denom = np.linalg.norm(s, axis=1) * np.linalg.norm(t, axis=1)
        return np.sum(s * t, axis=1) / np.clip(denom, 1e-8, None)
    centered = emb - emb.mean(axis=0, keepdims=True)
    if metric == "centered_cosine":
        s, t = centered[src_idx], centered[tgt_idx]
        denom = np.linalg.norm(s, axis=1) * np.linalg.norm(t, axis=1)
        return np.sum(s * t, axis=1) / np.clip(denom, 1e-8, None)
    if metric == "pca_cc":
        k = int(pca_dim or 64)
        k = min(k, centered.shape[1])
        proj = PCA(n_components=k, svd_solver="randomized", random_state=seed).fit_transform(centered)
        proj = proj.astype(np.float32)
        s, t = proj[src_idx], proj[tgt_idx]
        denom = np.linalg.norm(s, axis=1) * np.linalg.norm(t, axis=1)
        return np.sum(s * t, axis=1) / np.clip(denom, 1e-8, None)
    if metric == "neg_l2":
        s, t = emb[src_idx], emb[tgt_idx]
        return -np.linalg.norm(s - t, axis=1)
    raise ValueError(f"unknown metric {metric}")


def geom_feature_matrix(
    dom: Domain,
    metric: str,
    layers: Sequence[int],
    pca_dim: Optional[int] = None,
) -> np.ndarray:
    """(n_edges, len(layers)) geometric feature matrix (the multi-layer 'bundle')."""
    cols = [
        layer_score(dom.embeddings[li], dom.src_idx, dom.tgt_idx, metric, pca_dim, dom.seed)
        for li in layers
    ]
    return np.column_stack(cols)


def valid_mask(dom: Domain, layers: Sequence[int], geom: np.ndarray) -> np.ndarray:
    m = np.all(np.isfinite(dom.baseline), axis=1) & np.all(np.isfinite(geom), axis=1)
    for li in layers:
        m &= (dom.counts[li][dom.src_idx] > 0) & (dom.counts[li][dom.tgt_idx] > 0)
    return m


# ---------------------------------------------------------------------------
# Cross-validation splitters.
# ---------------------------------------------------------------------------
def edge_cv_splits(labels: np.ndarray, n_splits=5, n_repeats=3, seed=42):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(cv.split(np.zeros((len(labels), 1)), labels))


def grouped_cv_splits(groups: np.ndarray, n_splits=5):
    """GroupKFold splits keyed by `groups` (e.g. TF identity or target identity)."""
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros((len(groups), 1)), groups=groups))


def leave_both_out_splits(tf: np.ndarray, tgt: np.ndarray, n_blocks=4, seed=42):
    """Blocked TF x target splits: test fold k = edges whose TF AND target both fall
    in held-out block k; train = edges whose TF AND target both fall outside block k.
    Crossover edges (one side in, one side out) are excluded from that fold."""
    rng = np.random.default_rng(seed)
    uniq_tf = np.array(sorted(set(tf)))
    uniq_tg = np.array(sorted(set(tgt)))
    tf_block = {g: int(b) for g, b in zip(rng.permutation(uniq_tf),
                                          np.arange(len(uniq_tf)) % n_blocks)}
    tg_block = {g: int(b) for g, b in zip(rng.permutation(uniq_tg),
                                          np.arange(len(uniq_tg)) % n_blocks)}
    tf_b = np.array([tf_block[g] for g in tf])
    tg_b = np.array([tg_block[g] for g in tgt])
    splits = []
    for k in range(n_blocks):
        test = np.where((tf_b == k) & (tg_b == k))[0]
        train = np.where((tf_b != k) & (tg_b != k))[0]
        if len(test) > 0 and len(train) > 0:
            splits.append((train, test))
    return splits


# ---------------------------------------------------------------------------
# Evaluation harness.
# ---------------------------------------------------------------------------
def _fit_predict(x_tr, y_tr, x_te):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced",
                                      solver="lbfgs", random_state=0)),
    ])
    model.fit(x_tr, y_tr)
    return model.predict_proba(x_te)[:, 1]


def cv_oof_probs(features: np.ndarray, labels: np.ndarray, splits) -> np.ndarray:
    """Out-of-fold probabilities; averaged over repeats if an index recurs."""
    acc = np.zeros(len(labels), dtype=np.float64)
    cnt = np.zeros(len(labels), dtype=np.float64)
    for tr, te in splits:
        if len(np.unique(labels[tr])) < 2:
            continue
        acc[te] += _fit_predict(features[tr], labels[tr], features[te])
        cnt[te] += 1
    cnt = np.where(cnt > 0, cnt, 1.0)
    return acc / cnt


def safe_auroc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def safe_auprc(y, s):
    return float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def precision_recall_at_k(y, s, ks=(25, 50, 100)):
    order = np.argsort(-s)
    y_sorted = y[order]
    n_pos = int(y.sum())
    out = {}
    for k in ks:
        kk = min(k, len(y))
        hits = int(y_sorted[:kk].sum())
        out[f"precision@{k}"] = hits / kk
        out[f"recall@{k}"] = hits / n_pos if n_pos else float("nan")
        prevalence = n_pos / len(y)
        out[f"enrichment@{k}"] = (hits / kk) / prevalence if prevalence else float("nan")
    return out


def bootstrap_delta(labels, base_probs, plus_probs, n_iters=1000, seed=0,
                    groups: Optional[np.ndarray] = None):
    """Bootstrap CI of delta-AUROC. If `groups` given, resamples whole gene-groups
    (cluster bootstrap); otherwise resamples edges."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    deltas, base_aucs, plus_aucs = [], [], []
    if groups is not None:
        uniq = np.array(sorted(set(groups)))
        gidx = {g: np.where(groups == g)[0] for g in uniq}
    for _ in range(n_iters):
        if groups is not None:
            chosen = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([gidx[g] for g in chosen])
        else:
            idx = rng.integers(0, n, size=n)
        yb = labels[idx]
        if len(np.unique(yb)) < 2:
            continue
        ab = roc_auc_score(yb, base_probs[idx])
        ap = roc_auc_score(yb, plus_probs[idx])
        base_aucs.append(ab)
        plus_aucs.append(ap)
        deltas.append(ap - ab)
    if not deltas:
        return dict(delta_mean=np.nan, delta_lo=np.nan, delta_hi=np.nan,
                    base_mean=np.nan, plus_mean=np.nan)
    d = np.array(deltas)
    return dict(
        delta_mean=float(d.mean()),
        delta_lo=float(np.percentile(d, 2.5)),
        delta_hi=float(np.percentile(d, 97.5)),
        base_mean=float(np.mean(base_aucs)),
        plus_mean=float(np.mean(plus_aucs)),
    )


def evaluate(
    dom: Domain,
    metric: str,
    layers: Sequence[int],
    pca_dim: Optional[int] = None,
    cv_mode: str = "edge",
    n_boot: int = 1000,
    boot_seed: int = 0,
) -> dict:
    """Full evaluation of geometry-augmented vs baseline-only LR for one config.

    cv_mode: 'edge' | 'leave_tf' | 'leave_target' | 'leave_both'.
    Returns absolute AUROC/AUPRC for baseline and geometry-augmented models,
    deltas, bootstrap CI, and precision@k.
    """
    geom = geom_feature_matrix(dom, metric, layers, pca_dim)
    m = valid_mask(dom, layers, geom)
    y = dom.labels[m].astype(np.int32)
    base_x = dom.baseline[m]
    plus_x = np.column_stack([base_x, geom[m]])
    tf = dom.edges["source"].to_numpy()[m]
    tg = dom.edges["target"].to_numpy()[m]

    if cv_mode == "edge":
        splits = edge_cv_splits(y, seed=dom.seed)
        boot_groups = None
    elif cv_mode == "leave_tf":
        splits = grouped_cv_splits(tf)
        boot_groups = tf
    elif cv_mode == "leave_target":
        splits = grouped_cv_splits(tg)
        boot_groups = tg
    elif cv_mode == "leave_both":
        splits = leave_both_out_splits(tf, tg, seed=dom.seed)
        boot_groups = None
    else:
        raise ValueError(cv_mode)

    if cv_mode == "leave_both":
        # Restrict to the union of edges actually appearing in any fold.
        used = np.unique(np.concatenate([np.concatenate([tr, te]) for tr, te in splits]))
        remap = {orig: i for i, orig in enumerate(used)}
        y = y[used]
        base_x = base_x[used]
        plus_x = plus_x[used]
        splits = [(np.array([remap[i] for i in tr]),
                   np.array([remap[i] for i in te])) for tr, te in splits]

    base_oof = cv_oof_probs(base_x, y, splits)
    plus_oof = cv_oof_probs(plus_x, y, splits)

    res = dict(
        domain=dom.name, seed=dom.seed, metric=metric,
        layers=",".join(map(str, layers)), pca_dim=pca_dim or "",
        cv_mode=cv_mode, n_edges=int(len(y)), n_pos=int(y.sum()),
        prevalence=float(y.mean()),
        auroc_base=safe_auroc(y, base_oof),
        auroc_plus=safe_auroc(y, plus_oof),
        auprc_base=safe_auprc(y, base_oof),
        auprc_plus=safe_auprc(y, plus_oof),
    )
    res["delta_auroc"] = res["auroc_plus"] - res["auroc_base"]
    res["delta_auprc"] = res["auprc_plus"] - res["auprc_base"]
    bs = bootstrap_delta(y, base_oof, plus_oof, n_iters=n_boot, seed=boot_seed,
                         groups=boot_groups)
    res.update({f"boot_{k}": v for k, v in bs.items()})
    res["ci_excludes_zero"] = bool(bs["delta_lo"] > 0) if np.isfinite(bs["delta_lo"]) else False
    for k, v in precision_recall_at_k(y, plus_oof).items():
        res[f"geom_{k}"] = v
    for k, v in precision_recall_at_k(y, base_oof).items():
        res[f"base_{k}"] = v
    res["_y"] = y
    res["_base_oof"] = base_oof
    res["_plus_oof"] = plus_oof
    return res


if __name__ == "__main__":
    # Smoke test.
    d = load_domain("kidney", 42)
    print("loaded kidney:", d.edges.shape, d.embeddings.shape)
    r = evaluate(d, "cosine", [4], cv_mode="edge", n_boot=200)
    print({k: v for k, v in r.items() if not k.startswith("_")})
