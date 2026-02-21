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
            "External-lung source ablation: remove high-disagreement source groups and "
            "measure scGPT vs Geneformer delta gaps on shared edges."
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
    parser.add_argument(
        "--source-enrichment-csv",
        type=Path,
        default=subproject_root
        / "implementation"
        / "outputs"
        / "cycle15_cross_model_disagreement_stratification"
        / "source_enrichment_top_disagreement_bin.csv",
    )
    parser.add_argument("--model-id", type=str, default="ctheodoris/Geneformer")
    parser.add_argument("--scgpt-layer", type=int, default=3)
    parser.add_argument("--scgpt-seed", type=int, default=42)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle18_external_lung_source_ablation_gap",
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


def _eval_delta(
    y: np.ndarray,
    x_base: np.ndarray,
    feat: np.ndarray,
    cv_splits: int,
    cv_repeats: int,
    seed: int,
) -> dict[str, float]:
    n_pos = int(y.sum())
    n_neg = int(y.shape[0] - n_pos)
    min_class = min(n_pos, n_neg)
    n_splits = int(min(cv_splits, min_class))
    if y.shape[0] < 8 or n_pos < 2 or n_neg < 2 or n_splits < 2:
        return {
            "n_pairs": int(y.shape[0]),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_splits_used": n_splits,
            "baseline_cv_auroc": float("nan"),
            "plus_cv_auroc": float("nan"),
            "delta_cv_auroc": float("nan"),
        }
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=cv_repeats, random_state=seed)
    x_plus = np.column_stack([x_base, feat])
    base_probs = _crossval_probs(x_base, y, cv)
    plus_probs = _crossval_probs(x_plus, y, cv)
    base_auc = float(roc_auc_score(y, base_probs))
    plus_auc = float(roc_auc_score(y, plus_probs))
    return {
        "n_pairs": int(y.shape[0]),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_splits_used": n_splits,
        "baseline_cv_auroc": base_auc,
        "plus_cv_auroc": plus_auc,
        "delta_cv_auroc": plus_auc - base_auc,
    }


def _compute_scgpt_feature(
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

    source_enrich = pd.read_csv(args.source_enrichment_csv)
    ext = source_enrich[source_enrich["domain"] == "external_lung"].sort_values(
        "enrichment_ratio_top_vs_overall", ascending=False
    )
    ranked_sources = ext["source"].drop_duplicates().tolist()
    ratio_ge_2_sources = ext[ext["enrichment_ratio_top_vs_overall"] >= 2.0]["source"].drop_duplicates().tolist()

    adata = ad.read_h5ad(args.processed_h5ad)
    edge_df = pd.read_csv(args.shared_edge_tsv, sep="\t")
    src_idx = edge_df["source_idx"].to_numpy(dtype=np.int64)
    tgt_idx = edge_df["target_idx"].to_numpy(dtype=np.int64)
    src_tok = edge_df["source_token_id"].to_numpy(dtype=np.int64)
    tgt_tok = edge_df["target_token_id"].to_numpy(dtype=np.int64)
    labels = edge_df["label"].to_numpy(dtype=np.int32)
    sources = edge_df["source"].astype(str).to_numpy()

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

    # scGPT feature (seed/layer fixed to preserve comparability with previous domain-level analysis).
    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent
    if args.scgpt_seed == 42:
        run_dir = subproject_root / "implementation" / "outputs" / "cycle7_external_lung_main"
    else:
        run_dir = subproject_root / "implementation" / "outputs" / f"cycle7_external_lung_seed{args.scgpt_seed}"
    layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
    layer_counts = np.load(run_dir / "layer_gene_embedding_count.npy", mmap_mode="r")
    emb = np.asarray(layer_embeddings[args.scgpt_layer], dtype=np.float32)
    counts = np.asarray(layer_counts[args.scgpt_layer], dtype=np.int32)
    sc_feature = _compute_scgpt_feature(emb=emb, src_idx=src_idx, tgt_idx=tgt_idx, pca_dim=args.pca_dim, seed=args.seed)
    sc_valid = base_valid & (counts[src_idx] > 0) & (counts[tgt_idx] > 0) & np.isfinite(sc_feature)

    # Geneformer feature.
    model = AutoModel.from_pretrained(args.model_id)
    emb_weight = model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32)
    gf_feature = _compute_geneformer_feature(emb_weight=emb_weight, src_tok=src_tok, tgt_tok=tgt_tok)
    gf_valid = base_valid & np.isfinite(gf_feature)

    valid = sc_valid & gf_valid
    scenarios = [
        ("none", set()),
        ("exclude_top5", set(ranked_sources[:5])),
        ("exclude_top10", set(ranked_sources[:10])),
        ("exclude_top20", set(ranked_sources[:20])),
        ("exclude_top50", set(ranked_sources[:50])),
        ("exclude_ratio_ge_2", set(ratio_ge_2_sources)),
    ]

    rows = []
    for name, excluded in scenarios:
        mask = valid.copy()
        if excluded:
            mask &= ~np.isin(sources, list(excluded))

        y = labels[mask]
        x_base = baseline[mask]
        sc = sc_feature[mask]
        gf = gf_feature[mask]

        sc_stats = _eval_delta(
            y=y,
            x_base=x_base,
            feat=sc,
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
            seed=args.seed,
        )
        gf_stats = _eval_delta(
            y=y,
            x_base=x_base,
            feat=gf,
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
            seed=args.seed,
        )

        rows.append(
            {
                "scenario": name,
                "n_excluded_sources": len(excluded),
                "n_pairs": sc_stats["n_pairs"],
                "n_positive": sc_stats["n_positive"],
                "n_negative": sc_stats["n_negative"],
                "scgpt_delta_cv_auroc": sc_stats["delta_cv_auroc"],
                "geneformer_delta_cv_auroc": gf_stats["delta_cv_auroc"],
                "delta_gap_geneformer_minus_scgpt": gf_stats["delta_cv_auroc"] - sc_stats["delta_cv_auroc"]
                if np.isfinite(gf_stats["delta_cv_auroc"]) and np.isfinite(sc_stats["delta_cv_auroc"])
                else float("nan"),
                "scgpt_baseline_cv_auroc": sc_stats["baseline_cv_auroc"],
                "geneformer_baseline_cv_auroc": gf_stats["baseline_cv_auroc"],
                "scgpt_plus_cv_auroc": sc_stats["plus_cv_auroc"],
                "geneformer_plus_cv_auroc": gf_stats["plus_cv_auroc"],
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_dir / "external_lung_source_ablation_gap_summary.csv", index=False)
    print(out_df.to_string(index=False))
    print(f"[done] {output_dir / 'external_lung_source_ablation_gap_summary.csv'}")


if __name__ == "__main__":
    main()
