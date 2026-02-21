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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _compute_gene_statistics(expr_matrix):
    """Compute baseline confound features used throughout this project."""
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


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent
    default_single_cell_root = subproject_root.parent / "single_cell_mechinterp"
    default_run_dir = subproject_root / "implementation" / "outputs" / "cycle1_main"

    parser = argparse.ArgumentParser(description="Evaluate alternative geometry features on fixed layer embeddings.")
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    parser.add_argument(
        "--processed-h5ad",
        type=Path,
        default=default_single_cell_root / "outputs" / "tabula_sapiens_processed.h5ad",
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-iters", type=int, default=400)
    parser.add_argument(
        "--pca-dims",
        type=str,
        default="64",
        help="Comma-separated PCA dimensions for low-rank feature variants (empty to disable).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="run")
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _bootstrap_delta_auc(
    rng: np.random.Generator,
    labels: np.ndarray,
    base_probs: np.ndarray,
    plus_probs: np.ndarray,
    n_iters: int,
):
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

    delta_arr = np.array(deltas, dtype=np.float64)
    return (
        float(delta_arr.mean()),
        float(np.percentile(delta_arr, 2.5)),
        float(np.percentile(delta_arr, 97.5)),
    )


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()

    if args.output_csv is None:
        output_csv = run_dir / f"alt_geometry_metrics_layer{args.layer}_{args.tag}.csv"
    else:
        output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    edge_df = pd.read_csv(run_dir / "cycle1_edge_dataset.tsv", sep="\t")
    layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
    layer_counts = np.load(run_dir / "layer_gene_embedding_count.npy", mmap_mode="r")
    adata = ad.read_h5ad(args.processed_h5ad)

    src_idx = edge_df["source_idx"].to_numpy(dtype=np.int64)
    tgt_idx = edge_df["target_idx"].to_numpy(dtype=np.int64)
    labels = edge_df["label"].to_numpy(dtype=np.int32)

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

    emb = np.asarray(layer_embeddings[args.layer], dtype=np.float32)
    counts = np.asarray(layer_counts[args.layer], dtype=np.int32)

    src_emb = emb[src_idx]
    tgt_emb = emb[tgt_idx]

    src_norm = np.linalg.norm(src_emb, axis=1)
    tgt_norm = np.linalg.norm(tgt_emb, axis=1)
    cosine = np.sum(src_emb * tgt_emb, axis=1) / np.clip(src_norm * tgt_norm, 1e-8, None)

    # Center embeddings globally to test whether coarse offsets dominate cosine behavior.
    emb_center = emb - emb.mean(axis=0, keepdims=True)
    src_center = emb_center[src_idx]
    tgt_center = emb_center[tgt_idx]
    center_norm = np.linalg.norm(src_center, axis=1) * np.linalg.norm(tgt_center, axis=1)
    centered_cosine = np.sum(src_center * tgt_center, axis=1) / np.clip(center_norm, 1e-8, None)

    neg_l2 = -np.linalg.norm(src_emb - tgt_emb, axis=1)
    dot = np.sum(src_emb * tgt_emb, axis=1)

    feature_map = {
        "cosine": cosine,
        "centered_cosine": centered_cosine,
        "neg_l2": neg_l2,
        "dot": dot,
    }

    pca_dims = []
    for token in args.pca_dims.split(","):
        token = token.strip()
        if not token:
            continue
        dim = int(token)
        if dim <= 0:
            continue
        if dim > emb.shape[1]:
            continue
        pca_dims.append(dim)
    pca_dims = sorted(set(pca_dims))

    for dim in pca_dims:
        # Randomized PCA keeps this pass fast while preserving the global subspace geometry.
        pca = PCA(n_components=dim, svd_solver="randomized", random_state=args.seed)
        proj = pca.fit_transform(emb_center).astype(np.float32, copy=False)
        src_proj = proj[src_idx]
        tgt_proj = proj[tgt_idx]
        proj_cos = np.sum(src_proj * tgt_proj, axis=1) / np.clip(
            np.linalg.norm(src_proj, axis=1) * np.linalg.norm(tgt_proj, axis=1), 1e-8, None
        )
        proj_neg_l2 = -np.linalg.norm(src_proj - tgt_proj, axis=1)
        feature_map[f"pca{dim}_centered_cosine"] = proj_cos
        feature_map[f"pca{dim}_neg_l2"] = proj_neg_l2

    valid_base = (
        (counts[src_idx] > 0)
        & (counts[tgt_idx] > 0)
        & np.all(np.isfinite(baseline), axis=1)
    )

    cv = RepeatedStratifiedKFold(
        n_splits=args.cv_splits,
        n_repeats=args.cv_repeats,
        random_state=args.seed,
    )

    rows = []
    for feat_name, feat_values in feature_map.items():
        valid = valid_base & np.isfinite(feat_values)
        y_feat = labels[valid]
        x_base_feat = baseline[valid]
        feat = feat_values[valid]
        x_plus = np.column_stack([x_base_feat, feat])

        probs_base_feat = _crossval_probs(x_base_feat, y_feat, cv)
        probs_plus = _crossval_probs(x_plus, y_feat, cv)
        auc_base_feat = _safe_auc(y_feat, probs_base_feat)
        aupr_base_feat = _safe_aupr(y_feat, probs_base_feat)
        auc_plus = _safe_auc(y_feat, probs_plus)
        aupr_plus = _safe_aupr(y_feat, probs_plus)
        auc_raw = _safe_auc(y_feat, feat)
        aupr_raw = _safe_aupr(y_feat, feat)
        delta_mean, delta_lo, delta_hi = _bootstrap_delta_auc(
            rng=np.random.default_rng(args.seed + args.layer * 100 + len(rows)),
            labels=y_feat,
            base_probs=probs_base_feat,
            plus_probs=probs_plus,
            n_iters=args.bootstrap_iters,
        )

        rows.append(
            {
                "tag": args.tag,
                "layer": args.layer,
                "feature": feat_name,
                "n_pairs": int(valid.sum()),
                "baseline_cv_auroc": auc_base_feat,
                "baseline_plus_feature_cv_auroc": auc_plus,
                "delta_cv_auroc": auc_plus - auc_base_feat,
                "baseline_cv_aupr": aupr_base_feat,
                "baseline_plus_feature_cv_aupr": aupr_plus,
                "delta_cv_aupr": aupr_plus - aupr_base_feat,
                "raw_feature_auroc": auc_raw,
                "raw_feature_aupr": aupr_raw,
                "delta_auc_bootstrap_mean": delta_mean,
                "delta_auc_bootstrap_ci_lo": delta_lo,
                "delta_auc_bootstrap_ci_hi": delta_hi,
            }
        )

    out_df = pd.DataFrame(rows).sort_values("delta_cv_auroc", ascending=False).reset_index(drop=True)
    out_df.to_csv(output_csv, index=False)
    print(f"[done] {output_csv}")


if __name__ == "__main__":
    main()
