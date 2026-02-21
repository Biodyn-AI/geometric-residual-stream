#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
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


def _compute_gene_statistics(expr_matrix):
    """Compute confound baseline statistics for each gene."""
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
    """Generate out-of-fold probabilities with repeated stratified CV."""
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


def _compute_feature(
    emb: np.ndarray,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    feature: str,
    seed: int,
) -> np.ndarray:
    src_emb = emb[src_idx]
    tgt_emb = emb[tgt_idx]

    if feature == "cosine":
        return np.sum(src_emb * tgt_emb, axis=1) / np.clip(
            np.linalg.norm(src_emb, axis=1) * np.linalg.norm(tgt_emb, axis=1), 1e-8, None
        )
    if feature == "centered_cosine":
        centered = emb - emb.mean(axis=0, keepdims=True)
        src_center = centered[src_idx]
        tgt_center = centered[tgt_idx]
        return np.sum(src_center * tgt_center, axis=1) / np.clip(
            np.linalg.norm(src_center, axis=1) * np.linalg.norm(tgt_center, axis=1), 1e-8, None
        )
    if feature == "neg_l2":
        return -np.linalg.norm(src_emb - tgt_emb, axis=1)
    if feature == "dot":
        return np.sum(src_emb * tgt_emb, axis=1)

    match = re.fullmatch(r"pca(\d+)_(centered_cosine|neg_l2)", feature)
    if match:
        dim = int(match.group(1))
        kind = match.group(2)
        if dim <= 0 or dim > emb.shape[1]:
            raise ValueError(f"Requested PCA dim {dim} is invalid for embedding width {emb.shape[1]}")

        centered = emb - emb.mean(axis=0, keepdims=True)
        pca = PCA(n_components=dim, svd_solver="randomized", random_state=seed)
        proj = pca.fit_transform(centered).astype(np.float32, copy=False)
        src_proj = proj[src_idx]
        tgt_proj = proj[tgt_idx]
        if kind == "centered_cosine":
            return np.sum(src_proj * tgt_proj, axis=1) / np.clip(
                np.linalg.norm(src_proj, axis=1) * np.linalg.norm(tgt_proj, axis=1), 1e-8, None
            )
        return -np.linalg.norm(src_proj - tgt_proj, axis=1)

    raise ValueError(f"Unsupported feature: {feature}")


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent
    default_single_cell_root = subproject_root.parent / "single_cell_mechinterp"
    default_run_dir = subproject_root / "implementation" / "outputs" / "cycle1_main"

    parser = argparse.ArgumentParser(
        description=(
            "Geometry-feature shuffle null: permute geometry score across edges while "
            "keeping labels and baseline features fixed."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    parser.add_argument(
        "--processed-h5ad",
        type=Path,
        default=default_single_cell_root / "outputs" / "tabula_sapiens_processed.h5ad",
    )
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument(
        "--feature",
        type=str,
        default="cosine",
    )
    parser.add_argument("--n-shuffles", type=int, default=120)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle5_geom_shuffle_null",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    geom = _compute_feature(emb=emb, src_idx=src_idx, tgt_idx=tgt_idx, feature=args.feature, seed=args.seed)

    valid = (
        (counts[src_idx] > 0)
        & (counts[tgt_idx] > 0)
        & np.isfinite(geom)
        & np.all(np.isfinite(baseline), axis=1)
    )
    y = labels[valid]
    x_base = baseline[valid]
    geom_valid = geom[valid]

    cv = RepeatedStratifiedKFold(
        n_splits=args.cv_splits,
        n_repeats=args.cv_repeats,
        random_state=args.seed,
    )

    # Reference delta with the original geometry feature.
    base_probs = _crossval_probs(x_base, y, cv)
    plus_true = np.column_stack([x_base, geom_valid])
    plus_probs_true = _crossval_probs(plus_true, y, cv)
    true_delta = float(roc_auc_score(y, plus_probs_true) - roc_auc_score(y, base_probs))

    shuffle_rows = []
    for shuffle_idx in range(args.n_shuffles):
        geom_shuf = rng.permutation(geom_valid)
        plus_shuf = np.column_stack([x_base, geom_shuf])
        plus_probs = _crossval_probs(plus_shuf, y, cv)
        delta = float(roc_auc_score(y, plus_probs) - roc_auc_score(y, base_probs))
        shuffle_rows.append({"shuffle_idx": shuffle_idx, "delta_cv_auroc": delta})

    shuffle_df = pd.DataFrame(shuffle_rows)
    shuffle_df.to_csv(output_dir / "geometry_shuffle_delta_distribution.csv", index=False)

    p_value = float((shuffle_df["delta_cv_auroc"] >= true_delta).mean()) if not shuffle_df.empty else float("nan")
    summary = pd.DataFrame(
        [
            {
                "layer": args.layer,
                "feature": args.feature,
                "n_shuffles": int(shuffle_df.shape[0]),
                "true_delta_cv_auroc": true_delta,
                "shuffle_mean_delta": float(shuffle_df["delta_cv_auroc"].mean()) if not shuffle_df.empty else float("nan"),
                "shuffle_std_delta": float(shuffle_df["delta_cv_auroc"].std()) if not shuffle_df.empty else float("nan"),
                "empirical_p_value_right_tail": p_value,
            }
        ]
    )
    summary.to_csv(output_dir / "geometry_shuffle_summary.csv", index=False)

    print(f"[done] {output_dir / 'geometry_shuffle_delta_distribution.csv'}")
    print(f"[done] {output_dir / 'geometry_shuffle_summary.csv'}")


if __name__ == "__main__":
    main()
