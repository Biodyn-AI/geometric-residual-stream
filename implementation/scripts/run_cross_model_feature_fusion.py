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
from transformers import AutoModel


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent
    single_cell_root = subproject_root.parent / "single_cell_mechinterp"

    parser = argparse.ArgumentParser(
        description=(
            "Cross-model feature fusion on shared edges: compare baseline, baseline+scGPT, "
            "baseline+Geneformer, and baseline+both."
        )
    )
    parser.add_argument("--model-id", type=str, default="ctheodoris/Geneformer")
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle14_cross_model_feature_fusion",
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


def _crossval_auc(features: np.ndarray, labels: np.ndarray, cv: RepeatedStratifiedKFold) -> float:
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
    return float(roc_auc_score(labels, probs))


def _compute_scgpt_feature(
    layer_embeddings: np.ndarray,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    pca_dim: int,
    seed: int,
) -> np.ndarray:
    centered = layer_embeddings - layer_embeddings.mean(axis=0, keepdims=True)
    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=seed)
    proj = pca.fit_transform(centered).astype(np.float32, copy=False)
    src_proj = proj[src_idx]
    tgt_proj = proj[tgt_idx]
    return np.sum(src_proj * tgt_proj, axis=1) / np.clip(
        np.linalg.norm(src_proj, axis=1) * np.linalg.norm(tgt_proj, axis=1), 1e-8, None
    )


def _compute_geneformer_feature(
    emb_weight: np.ndarray,
    src_tok: np.ndarray,
    tgt_tok: np.ndarray,
) -> np.ndarray:
    centered = emb_weight - emb_weight.mean(axis=0, keepdims=True)
    src_center = centered[src_tok]
    tgt_center = centered[tgt_tok]
    return np.sum(src_center * tgt_center, axis=1) / np.clip(
        np.linalg.norm(src_center, axis=1) * np.linalg.norm(tgt_center, axis=1), 1e-8, None
    )


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent

    cfgs = [
        {
            "domain": "immune",
            "processed_h5ad": args.immune_processed_h5ad,
            "scgpt_run_dir": subproject_root / "implementation" / "outputs" / "cycle4_immune_main",
            "layer": 0,
            "geneformer_edge_tsv": subproject_root
            / "implementation"
            / "outputs"
            / "cycle12_geneformer_immune_bootstrap"
            / "geneformer_edge_dataset.tsv",
        },
        {
            "domain": "lung",
            "processed_h5ad": args.lung_processed_h5ad,
            "scgpt_run_dir": subproject_root / "implementation" / "outputs" / "cycle6_lung_main",
            "layer": 0,
            "geneformer_edge_tsv": subproject_root
            / "implementation"
            / "outputs"
            / "cycle12_geneformer_lung_bootstrap"
            / "geneformer_edge_dataset.tsv",
        },
        {
            "domain": "external_lung",
            "processed_h5ad": args.external_lung_processed_h5ad,
            "scgpt_run_dir": subproject_root / "implementation" / "outputs" / "cycle7_external_lung_main",
            "layer": 3,
            "geneformer_edge_tsv": subproject_root
            / "implementation"
            / "outputs"
            / "cycle12_geneformer_external_lung_bootstrap"
            / "geneformer_edge_dataset.tsv",
        },
    ]

    print("[info] loading Geneformer embeddings")
    model = AutoModel.from_pretrained(args.model_id)
    emb_weight = model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32)

    cv = RepeatedStratifiedKFold(
        n_splits=args.cv_splits,
        n_repeats=args.cv_repeats,
        random_state=args.seed,
    )

    rows = []
    for cfg in cfgs:
        domain = cfg["domain"]
        print(f"[info] evaluating {domain}")
        adata = ad.read_h5ad(cfg["processed_h5ad"])
        edge_df = pd.read_csv(cfg["geneformer_edge_tsv"], sep="\t")
        layer_embeddings = np.load(cfg["scgpt_run_dir"] / "layer_gene_embeddings.npy", mmap_mode="r")
        layer_counts = np.load(cfg["scgpt_run_dir"] / "layer_gene_embedding_count.npy", mmap_mode="r")

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

        scgpt_emb = np.asarray(layer_embeddings[cfg["layer"]], dtype=np.float32)
        scgpt_counts = np.asarray(layer_counts[cfg["layer"]], dtype=np.int32)
        scgpt_feature = _compute_scgpt_feature(
            layer_embeddings=scgpt_emb,
            src_idx=src_idx,
            tgt_idx=tgt_idx,
            pca_dim=args.pca_dim,
            seed=args.seed,
        )
        geneformer_feature = _compute_geneformer_feature(
            emb_weight=emb_weight,
            src_tok=src_tok,
            tgt_tok=tgt_tok,
        )

        valid = (
            (scgpt_counts[src_idx] > 0)
            & (scgpt_counts[tgt_idx] > 0)
            & np.isfinite(scgpt_feature)
            & np.isfinite(geneformer_feature)
            & np.all(np.isfinite(baseline), axis=1)
        )
        y = labels[valid]
        x_base = baseline[valid]
        x_scgpt = np.column_stack([x_base, scgpt_feature[valid]])
        x_geneformer = np.column_stack([x_base, geneformer_feature[valid]])
        x_both = np.column_stack([x_base, scgpt_feature[valid], geneformer_feature[valid]])

        auc_base = _crossval_auc(x_base, y, cv)
        auc_scgpt = _crossval_auc(x_scgpt, y, cv)
        auc_geneformer = _crossval_auc(x_geneformer, y, cv)
        auc_both = _crossval_auc(x_both, y, cv)
        best_single_auc = max(auc_scgpt, auc_geneformer)

        rows.append(
            {
                "domain": domain,
                "layer": cfg["layer"],
                "n_pairs": int(y.shape[0]),
                "n_positive": int(y.sum()),
                "baseline_cv_auroc": auc_base,
                "baseline_plus_scgpt_cv_auroc": auc_scgpt,
                "baseline_plus_geneformer_cv_auroc": auc_geneformer,
                "baseline_plus_both_cv_auroc": auc_both,
                "delta_scgpt_over_base": auc_scgpt - auc_base,
                "delta_geneformer_over_base": auc_geneformer - auc_base,
                "delta_both_over_base": auc_both - auc_base,
                "delta_both_over_best_single": auc_both - best_single_auc,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_dir / "cross_model_feature_fusion_summary.csv", index=False)
    print(out_df.to_string(index=False))
    print(f"[done] {output_dir / 'cross_model_feature_fusion_summary.csv'}")


if __name__ == "__main__":
    main()
