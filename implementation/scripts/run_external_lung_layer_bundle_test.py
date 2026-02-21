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
            "External-lung layer-bundle test: evaluate whether stacked scGPT layer features "
            "reduce the scGPT vs Geneformer gap on shared edges."
        )
    )
    parser.add_argument(
        "--processed-h5ad",
        type=Path,
        default=single_cell_root / "outputs" / "invariant_causal_edges" / "external_lung" / "processed.h5ad",
    )
    parser.add_argument(
        "--shared-edge-tsv",
        type=Path,
        default=subproject_root
        / "implementation"
        / "outputs"
        / "cycle12_geneformer_external_lung_bootstrap"
        / "geneformer_edge_dataset.tsv",
    )
    parser.add_argument("--model-id", type=str, default="ctheodoris/Geneformer")
    parser.add_argument("--scgpt-seed", type=int, default=42)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle19_external_lung_layer_bundle_test",
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


def _crossval_auc(features: np.ndarray, labels: np.ndarray, cv_splits: int, cv_repeats: int, seed: int) -> float:
    n_pos = int(labels.sum())
    n_neg = int(labels.shape[0] - n_pos)
    n_splits = int(min(cv_splits, n_pos, n_neg))
    if labels.shape[0] < 8 or n_pos < 2 or n_neg < 2 or n_splits < 2:
        return float("nan")
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=cv_repeats, random_state=seed)
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


def _compute_scgpt_layer_feature(
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
        np.linalg.norm(src_proj, axis=1) * np.linalg.norm(tgt_proj, axis=1), 1e-8, None
    )


def _compute_geneformer_feature(emb_weight: np.ndarray, src_tok: np.ndarray, tgt_tok: np.ndarray) -> np.ndarray:
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

    adata = ad.read_h5ad(args.processed_h5ad)
    edge_df = pd.read_csv(args.shared_edge_tsv, sep="\t")

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
    base_valid = np.all(np.isfinite(baseline), axis=1)

    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent
    if args.scgpt_seed == 42:
        run_dir = subproject_root / "implementation" / "outputs" / "cycle7_external_lung_main"
    else:
        run_dir = subproject_root / "implementation" / "outputs" / f"cycle7_external_lung_seed{args.scgpt_seed}"

    layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
    layer_counts = np.load(run_dir / "layer_gene_embedding_count.npy", mmap_mode="r")

    # Precompute scGPT layer features once.
    layer_feats = {}
    layer_valids = {}
    for layer in range(layer_embeddings.shape[0]):
        emb = np.asarray(layer_embeddings[layer], dtype=np.float32)
        counts = np.asarray(layer_counts[layer], dtype=np.int32)
        feat = _compute_scgpt_layer_feature(
            emb=emb,
            src_idx=src_idx,
            tgt_idx=tgt_idx,
            pca_dim=args.pca_dim,
            seed=args.seed,
        )
        valid = base_valid & (counts[src_idx] > 0) & (counts[tgt_idx] > 0) & np.isfinite(feat)
        layer_feats[layer] = feat
        layer_valids[layer] = valid

    # Geneformer feature.
    model = AutoModel.from_pretrained(args.model_id)
    emb_weight = model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32)
    gf_feat = _compute_geneformer_feature(emb_weight=emb_weight, src_tok=src_tok, tgt_tok=tgt_tok)
    gf_valid = base_valid & np.isfinite(gf_feat)

    bundles = [
        ("L3", [3]),
        ("L1_4", [1, 2, 3, 4]),
        ("L0_5", [0, 1, 2, 3, 4, 5]),
        ("L0_11", list(range(12))),
    ]

    rows = []
    for name, layers in bundles:
        valid = gf_valid.copy()
        for layer in layers:
            valid &= layer_valids[layer]
        y = labels[valid]
        x_base = baseline[valid]
        sc_stack = np.column_stack([layer_feats[layer][valid] for layer in layers])
        gf = gf_feat[valid].reshape(-1, 1)

        auc_base = _crossval_auc(x_base, y, args.cv_splits, args.cv_repeats, args.seed)
        auc_sc = _crossval_auc(np.column_stack([x_base, sc_stack]), y, args.cv_splits, args.cv_repeats, args.seed)
        auc_gf = _crossval_auc(np.column_stack([x_base, gf]), y, args.cv_splits, args.cv_repeats, args.seed)
        auc_both = _crossval_auc(
            np.column_stack([x_base, sc_stack, gf]), y, args.cv_splits, args.cv_repeats, args.seed
        )

        rows.append(
            {
                "bundle": name,
                "layers": ",".join(str(x) for x in layers),
                "n_layers": len(layers),
                "n_pairs": int(y.shape[0]),
                "n_positive": int(y.sum()),
                "baseline_cv_auroc": auc_base,
                "baseline_plus_scgpt_cv_auroc": auc_sc,
                "baseline_plus_geneformer_cv_auroc": auc_gf,
                "baseline_plus_both_cv_auroc": auc_both,
                "scgpt_delta_over_base": auc_sc - auc_base if np.isfinite(auc_sc) and np.isfinite(auc_base) else float("nan"),
                "geneformer_delta_over_base": auc_gf - auc_base
                if np.isfinite(auc_gf) and np.isfinite(auc_base)
                else float("nan"),
                "both_delta_over_base": auc_both - auc_base if np.isfinite(auc_both) and np.isfinite(auc_base) else float("nan"),
                "gap_geneformer_minus_scgpt": auc_gf - auc_sc if np.isfinite(auc_sc) and np.isfinite(auc_gf) else float("nan"),
                "both_minus_geneformer": auc_both - auc_gf if np.isfinite(auc_both) and np.isfinite(auc_gf) else float("nan"),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_dir / "external_lung_layer_bundle_summary.csv", index=False)
    print(out_df.to_string(index=False))
    print(f"[done] {output_dir / 'external_lung_layer_bundle_summary.csv'}")


if __name__ == "__main__":
    main()
