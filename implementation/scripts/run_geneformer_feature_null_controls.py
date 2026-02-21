#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent
    single_cell_root = subproject_root.parent / "single_cell_mechinterp"
    default_run_dir = subproject_root / "implementation" / "outputs" / "cycle12_geneformer_immune_bootstrap"

    parser = argparse.ArgumentParser(
        description=(
            "Geneformer geometry null controls: test whether centered geometry lift remains "
            "when (a) geometry values are shuffled or (b) labels are permuted."
        )
    )
    parser.add_argument(
        "--processed-h5ad",
        type=Path,
        default=single_cell_root / "outputs" / "tabula_sapiens_immune_subset_hpn_processed.h5ad",
    )
    parser.add_argument(
        "--edge-dataset-tsv",
        type=Path,
        default=default_run_dir / "geneformer_edge_dataset.tsv",
    )
    parser.add_argument("--model-id", type=str, default="ctheodoris/Geneformer")
    parser.add_argument(
        "--feature",
        type=str,
        default="centered_cosine",
        choices=["cosine", "centered_cosine", "dot", "neg_l2"],
    )
    parser.add_argument("--n-shuffles", type=int, default=60)
    parser.add_argument("--n-perm", type=int, default=40)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle12_geneformer_immune_centered_null",
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


def _compute_feature(
    emb_weight: np.ndarray,
    src_tok: np.ndarray,
    tgt_tok: np.ndarray,
    feature: str,
) -> np.ndarray:
    src_emb = emb_weight[src_tok]
    tgt_emb = emb_weight[tgt_tok]
    if feature == "cosine":
        return np.sum(src_emb * tgt_emb, axis=1) / np.clip(
            np.linalg.norm(src_emb, axis=1) * np.linalg.norm(tgt_emb, axis=1), 1e-8, None
        )
    if feature == "centered_cosine":
        centered = emb_weight - emb_weight.mean(axis=0, keepdims=True)
        src_center = centered[src_tok]
        tgt_center = centered[tgt_tok]
        return np.sum(src_center * tgt_center, axis=1) / np.clip(
            np.linalg.norm(src_center, axis=1) * np.linalg.norm(tgt_center, axis=1), 1e-8, None
        )
    if feature == "dot":
        return np.sum(src_emb * tgt_emb, axis=1)
    if feature == "neg_l2":
        return -np.linalg.norm(src_emb - tgt_emb, axis=1)
    raise ValueError(f"Unsupported feature: {feature}")


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[info] loading inputs")
    adata = ad.read_h5ad(args.processed_h5ad)
    edge_df = pd.read_csv(args.edge_dataset_tsv, sep="\t")

    required_cols = {"source_idx", "target_idx", "source_token_id", "target_token_id", "label"}
    missing = required_cols.difference(edge_df.columns)
    if missing:
        raise ValueError(
            f"Edge dataset is missing required columns: {sorted(missing)}. "
            "Expected a Geneformer edge dataset from run_geneformer_embedding_geometry_audit.py"
        )

    model = AutoModel.from_pretrained(args.model_id)
    emb_weight = model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32)

    src_idx = edge_df["source_idx"].to_numpy(dtype=np.int64)
    tgt_idx = edge_df["target_idx"].to_numpy(dtype=np.int64)
    src_tok = edge_df["source_token_id"].to_numpy(dtype=np.int64)
    tgt_tok = edge_df["target_token_id"].to_numpy(dtype=np.int64)
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

    geom = _compute_feature(emb_weight=emb_weight, src_tok=src_tok, tgt_tok=tgt_tok, feature=args.feature)
    valid = np.isfinite(geom) & np.all(np.isfinite(baseline), axis=1)
    y = labels[valid]
    x_base = baseline[valid]
    geom_valid = geom[valid]

    cv = RepeatedStratifiedKFold(
        n_splits=args.cv_splits,
        n_repeats=args.cv_repeats,
        random_state=args.seed,
    )

    base_probs = _crossval_probs(x_base, y, cv)
    plus_true = np.column_stack([x_base, geom_valid])
    plus_probs = _crossval_probs(plus_true, y, cv)
    true_base_auc = float(roc_auc_score(y, base_probs))
    true_plus_auc = float(roc_auc_score(y, plus_probs))
    true_delta = true_plus_auc - true_base_auc

    shuffle_rows = []
    for shuffle_idx in range(args.n_shuffles):
        geom_shuf = rng.permutation(geom_valid)
        plus_shuf = np.column_stack([x_base, geom_shuf])
        probs_shuf = _crossval_probs(plus_shuf, y, cv)
        delta = float(roc_auc_score(y, probs_shuf) - true_base_auc)
        shuffle_rows.append({"shuffle_idx": shuffle_idx, "delta_cv_auroc": delta})

    perm_rows = []
    for perm_idx in range(args.n_perm):
        y_perm = rng.permutation(y)
        if np.unique(y_perm).size < 2:
            continue
        probs_base_perm = _crossval_probs(x_base, y_perm, cv)
        probs_plus_perm = _crossval_probs(plus_true, y_perm, cv)
        delta = float(roc_auc_score(y_perm, probs_plus_perm) - roc_auc_score(y_perm, probs_base_perm))
        perm_rows.append({"perm_idx": perm_idx, "delta_cv_auroc": delta})

    shuffle_df = pd.DataFrame(shuffle_rows)
    perm_df = pd.DataFrame(perm_rows)
    shuffle_df.to_csv(output_dir / "geometry_shuffle_delta_distribution.csv", index=False)
    perm_df.to_csv(output_dir / "label_permutation_delta_distribution.csv", index=False)

    shuffle_p = float((shuffle_df["delta_cv_auroc"] >= true_delta).mean()) if not shuffle_df.empty else float("nan")
    perm_p = float((perm_df["delta_cv_auroc"] >= true_delta).mean()) if not perm_df.empty else float("nan")

    summary = pd.DataFrame(
        [
            {
                "feature": args.feature,
                "n_pairs": int(y.shape[0]),
                "n_positive": int(y.sum()),
                "true_baseline_cv_auroc": true_base_auc,
                "true_plus_cv_auroc": true_plus_auc,
                "true_delta_cv_auroc": float(true_delta),
                "n_shuffles": int(shuffle_df.shape[0]),
                "shuffle_mean_delta": float(shuffle_df["delta_cv_auroc"].mean()) if not shuffle_df.empty else float("nan"),
                "shuffle_std_delta": float(shuffle_df["delta_cv_auroc"].std()) if not shuffle_df.empty else float("nan"),
                "shuffle_empirical_p_value_right_tail": shuffle_p,
                "n_perm": int(perm_df.shape[0]),
                "perm_mean_delta": float(perm_df["delta_cv_auroc"].mean()) if not perm_df.empty else float("nan"),
                "perm_std_delta": float(perm_df["delta_cv_auroc"].std()) if not perm_df.empty else float("nan"),
                "perm_empirical_p_value_right_tail": perm_p,
            }
        ]
    )
    summary.to_csv(output_dir / "geneformer_null_summary.csv", index=False)

    print(f"[done] {output_dir / 'geometry_shuffle_delta_distribution.csv'}")
    print(f"[done] {output_dir / 'label_permutation_delta_distribution.csv'}")
    print(f"[done] {output_dir / 'geneformer_null_summary.csv'}")


if __name__ == "__main__":
    main()
