#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent
    single_cell_root = subproject_root.parent / "single_cell_mechinterp"

    parser = argparse.ArgumentParser(
        description=(
            "Cross-model matched-gene diagnostic: recompute scGPT low-rank geometry lift on "
            "all edges vs edges restricted to Geneformer-mapped genes."
        )
    )
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle13_cross_model_matched_gene_diagnostic",
    )
    parser.add_argument(
        "--immune-processed-h5ad",
        type=Path,
        default=single_cell_root / "outputs" / "tabula_sapiens_immune_subset_hpn_processed.h5ad",
    )
    parser.add_argument(
        "--lung-processed-h5ad",
        type=Path,
        default=single_cell_root / "outputs" / "invariant_causal_edges" / "lung" / "processed.h5ad",
    )
    parser.add_argument(
        "--external-lung-processed-h5ad",
        type=Path,
        default=single_cell_root / "outputs" / "invariant_causal_edges" / "external_lung" / "processed.h5ad",
    )
    return parser.parse_args()


def _compute_gene_statistics(expr_matrix):
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


def _crossval_probs(features: np.ndarray, labels: np.ndarray, cv: RepeatedStratifiedKFold) -> np.ndarray:
    probs = np.zeros(labels.shape[0], dtype=np.float64)
    for train_idx, test_idx in cv.split(features, labels):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=0,
                    ),
                ),
            ]
        )
        model.fit(features[train_idx], labels[train_idx])
        probs[test_idx] = model.predict_proba(features[test_idx])[:, 1]
    return probs


def _bootstrap_delta_auc(
    rng: np.random.Generator,
    labels: np.ndarray,
    base_probs: np.ndarray,
    plus_probs: np.ndarray,
    n_iters: int,
) -> tuple[float, float, float]:
    deltas = []
    n = labels.shape[0]
    for _ in range(n_iters):
        idx = rng.integers(0, n, size=n)
        y_bs = labels[idx]
        if np.unique(y_bs).size < 2:
            continue
        deltas.append(float(roc_auc_score(y_bs, plus_probs[idx]) - roc_auc_score(y_bs, base_probs[idx])))
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(deltas, dtype=np.float64)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _compute_pca_centered_cosine(
    emb: np.ndarray,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    pca_dim: int,
    seed: int,
) -> np.ndarray:
    centered = emb - emb.mean(axis=0, keepdims=True)
    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=seed)
    proj = pca.fit_transform(centered).astype(np.float32, copy=False)
    src_proj = proj[src_idx]
    tgt_proj = proj[tgt_idx]
    return np.sum(src_proj * tgt_proj, axis=1) / np.clip(
        np.linalg.norm(src_proj, axis=1) * np.linalg.norm(tgt_proj, axis=1),
        1e-8,
        None,
    )


def _evaluate_subset(
    labels: np.ndarray,
    baseline: np.ndarray,
    geom: np.ndarray,
    mask: np.ndarray,
    cv: RepeatedStratifiedKFold,
    bootstrap_iters: int,
    seed: int,
) -> dict[str, float]:
    y = labels[mask]
    x_base = baseline[mask]
    x_plus = np.column_stack([x_base, geom[mask]])
    base_probs = _crossval_probs(x_base, y, cv)
    plus_probs = _crossval_probs(x_plus, y, cv)
    base_auc = float(roc_auc_score(y, base_probs))
    plus_auc = float(roc_auc_score(y, plus_probs))
    delta = plus_auc - base_auc
    bs_mean, bs_lo, bs_hi = _bootstrap_delta_auc(
        rng=np.random.default_rng(seed),
        labels=y,
        base_probs=base_probs,
        plus_probs=plus_probs,
        n_iters=bootstrap_iters,
    )
    return {
        "n_pairs": int(y.shape[0]),
        "n_positive": int(y.sum()),
        "baseline_cv_auroc": base_auc,
        "plus_cv_auroc": plus_auc,
        "delta_cv_auroc": delta,
        "delta_auc_bootstrap_mean": bs_mean,
        "delta_auc_bootstrap_ci_lo": bs_lo,
        "delta_auc_bootstrap_ci_hi": bs_hi,
    }


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent

    domain_cfg = [
        {
            "domain": "immune",
            "processed_h5ad": args.immune_processed_h5ad,
            "run_dir": subproject_root / "implementation" / "outputs" / "cycle4_immune_main",
            "layer": 0,
            "geneformer_map_csv": subproject_root
            / "implementation"
            / "outputs"
            / "cycle12_geneformer_immune_bootstrap"
            / "geneformer_gene_token_map.csv",
        },
        {
            "domain": "lung",
            "processed_h5ad": args.lung_processed_h5ad,
            "run_dir": subproject_root / "implementation" / "outputs" / "cycle6_lung_main",
            "layer": 0,
            "geneformer_map_csv": subproject_root
            / "implementation"
            / "outputs"
            / "cycle12_geneformer_lung_bootstrap"
            / "geneformer_gene_token_map.csv",
        },
        {
            "domain": "external_lung",
            "processed_h5ad": args.external_lung_processed_h5ad,
            "run_dir": subproject_root / "implementation" / "outputs" / "cycle7_external_lung_main",
            "layer": 3,
            "geneformer_map_csv": subproject_root
            / "implementation"
            / "outputs"
            / "cycle12_geneformer_external_lung_bootstrap"
            / "geneformer_gene_token_map.csv",
        },
    ]

    cv = RepeatedStratifiedKFold(
        n_splits=args.cv_splits,
        n_repeats=args.cv_repeats,
        random_state=args.seed,
    )

    rows = []
    for cfg in domain_cfg:
        domain = cfg["domain"]
        print(f"[info] evaluating {domain}")
        adata = ad.read_h5ad(cfg["processed_h5ad"])
        edge_df = pd.read_csv(cfg["run_dir"] / "cycle1_edge_dataset.tsv", sep="\t")
        layer_embeddings = np.load(cfg["run_dir"] / "layer_gene_embeddings.npy", mmap_mode="r")
        layer_counts = np.load(cfg["run_dir"] / "layer_gene_embedding_count.npy", mmap_mode="r")
        mapped_genes = set(pd.read_csv(cfg["geneformer_map_csv"])["gene"].astype(str).tolist())

        src_idx = edge_df["source_idx"].to_numpy(dtype=np.int64)
        tgt_idx = edge_df["target_idx"].to_numpy(dtype=np.int64)
        labels = edge_df["label"].to_numpy(dtype=np.int32)
        sources = edge_df["source"].astype(str).to_numpy()
        targets = edge_df["target"].astype(str).to_numpy()

        mean_expr, variance, detection = _compute_gene_statistics(adata.X)
        baseline = np.column_stack(
            [
                mean_expr[src_idx],
                mean_expr[tgt_idx],
                variance[src_idx],
                variance[tgt_idx],
                detection[src_idx],
                detection[tgt_idx],
            ]
        ).astype(np.float64)

        emb = np.asarray(layer_embeddings[cfg["layer"]], dtype=np.float32)
        counts = np.asarray(layer_counts[cfg["layer"]], dtype=np.int32)
        geom = _compute_pca_centered_cosine(
            emb=emb,
            src_idx=src_idx,
            tgt_idx=tgt_idx,
            pca_dim=args.pca_dim,
            seed=args.seed,
        )

        valid = (
            (counts[src_idx] > 0)
            & (counts[tgt_idx] > 0)
            & np.isfinite(geom)
            & np.all(np.isfinite(baseline), axis=1)
        )
        matched = np.array(
            [(s in mapped_genes) and (t in mapped_genes) for s, t in zip(sources, targets)],
            dtype=bool,
        )

        metrics_all = _evaluate_subset(
            labels=labels,
            baseline=baseline,
            geom=geom,
            mask=valid,
            cv=cv,
            bootstrap_iters=args.bootstrap_iters,
            seed=args.seed + 101,
        )
        metrics_matched = _evaluate_subset(
            labels=labels,
            baseline=baseline,
            geom=geom,
            mask=valid & matched,
            cv=cv,
            bootstrap_iters=args.bootstrap_iters,
            seed=args.seed + 202,
        )

        rows.append(
            {
                "domain": domain,
                "layer": cfg["layer"],
                "feature": f"pca{args.pca_dim}_centered_cosine",
                "n_pairs_all_valid": metrics_all["n_pairs"],
                "n_pairs_matched_valid": metrics_matched["n_pairs"],
                "matched_pair_fraction": metrics_matched["n_pairs"] / max(metrics_all["n_pairs"], 1),
                "delta_all": metrics_all["delta_cv_auroc"],
                "delta_all_ci_lo": metrics_all["delta_auc_bootstrap_ci_lo"],
                "delta_all_ci_hi": metrics_all["delta_auc_bootstrap_ci_hi"],
                "delta_matched": metrics_matched["delta_cv_auroc"],
                "delta_matched_ci_lo": metrics_matched["delta_auc_bootstrap_ci_lo"],
                "delta_matched_ci_hi": metrics_matched["delta_auc_bootstrap_ci_hi"],
                "delta_shift_matched_minus_all": metrics_matched["delta_cv_auroc"] - metrics_all["delta_cv_auroc"],
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_dir / "cross_model_matched_gene_summary.csv", index=False)
    print(out_df.to_string(index=False))
    print(f"[done] {output_dir / 'cross_model_matched_gene_summary.csv'}")


if __name__ == "__main__":
    main()
