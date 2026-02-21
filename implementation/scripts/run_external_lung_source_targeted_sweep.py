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
            "Targeted external-lung source sweep: evaluate baseline+scGPT geometry across "
            "seeds/layers for top disagreement-enriched sources on the shared edge set."
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
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--layers", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--top-k-sources", type=int, default=12)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle16_external_lung_source_targeted_sweep",
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


def _evaluate_delta(
    y: np.ndarray,
    x_base: np.ndarray,
    feat: np.ndarray,
    cv_splits: int,
    cv_repeats: int,
    seed: int,
) -> dict[str, float]:
    n_pos = int(y.sum())
    n_neg = int(y.shape[0] - n_pos)
    min_class = int(min(n_pos, n_neg))
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

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent

    source_enrich = pd.read_csv(args.source_enrichment_csv)
    ext_src = (
        source_enrich[source_enrich["domain"] == "external_lung"]
        .sort_values("enrichment_ratio_top_vs_overall", ascending=False)["source"]
        .drop_duplicates()
        .head(args.top_k_sources)
        .tolist()
    )
    if not ext_src:
        raise RuntimeError("No external_lung sources found in source enrichment table")

    print(f"[info] selected sources ({len(ext_src)}): {', '.join(ext_src)}")

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
    source_masks = {s: (sources == s) for s in ext_src}

    # Geneformer reference on the same source subsets.
    print("[info] loading Geneformer embeddings for source-level reference")
    model = AutoModel.from_pretrained(args.model_id)
    emb_weight = model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32)
    gf_feature = _compute_geneformer_feature(emb_weight=emb_weight, src_tok=src_tok, tgt_tok=tgt_tok)
    gf_valid = base_valid & np.isfinite(gf_feature)

    ref_rows = []
    for source in ext_src:
        mask = gf_valid & source_masks[source]
        stats = _evaluate_delta(
            y=labels[mask],
            x_base=baseline[mask],
            feat=gf_feature[mask],
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
            seed=args.seed,
        )
        ref_rows.append(
            {
                "model": "geneformer",
                "source": source,
                "seed": -1,
                "layer": -1,
                "feature": "centered_cosine",
                **stats,
            }
        )

    # scGPT seed-layer sweep on source subsets.
    sc_rows = []
    for seed in seeds:
        run_dir = subproject_root / "implementation" / "outputs" / f"cycle7_external_lung_seed{seed}"
        if not run_dir.exists() and seed == 42:
            # Historical naming: seed 42 is the "main" run directory.
            run_dir = subproject_root / "implementation" / "outputs" / "cycle7_external_lung_main"
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing run directory for seed {seed}: {run_dir}")
        layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
        layer_counts = np.load(run_dir / "layer_gene_embedding_count.npy", mmap_mode="r")

        print(f"[info] scGPT sweep seed={seed}")
        for layer in layers:
            emb = np.asarray(layer_embeddings[layer], dtype=np.float32)
            counts = np.asarray(layer_counts[layer], dtype=np.int32)
            sc_feature = _compute_scgpt_feature(
                emb=emb,
                src_idx=src_idx,
                tgt_idx=tgt_idx,
                pca_dim=args.pca_dim,
                seed=seed,
            )
            valid = base_valid & (counts[src_idx] > 0) & (counts[tgt_idx] > 0) & np.isfinite(sc_feature)

            for source in ext_src:
                mask = valid & source_masks[source]
                stats = _evaluate_delta(
                    y=labels[mask],
                    x_base=baseline[mask],
                    feat=sc_feature[mask],
                    cv_splits=args.cv_splits,
                    cv_repeats=args.cv_repeats,
                    seed=args.seed,
                )
                sc_rows.append(
                    {
                        "model": "scgpt",
                        "source": source,
                        "seed": seed,
                        "layer": layer,
                        "feature": f"pca{args.pca_dim}_centered_cosine",
                        **stats,
                    }
                )

    detail_df = pd.concat([pd.DataFrame(sc_rows), pd.DataFrame(ref_rows)], ignore_index=True)
    detail_df.to_csv(output_dir / "source_seed_layer_detail.csv", index=False)

    sc_df = detail_df[detail_df["model"] == "scgpt"].copy()
    agg_df = (
        sc_df.groupby(["source", "layer"], as_index=False)
        .agg(
            n_runs=("delta_cv_auroc", "count"),
            mean_delta_cv_auroc=("delta_cv_auroc", "mean"),
            std_delta_cv_auroc=("delta_cv_auroc", "std"),
            mean_n_pairs=("n_pairs", "mean"),
            mean_n_positive=("n_positive", "mean"),
            mean_n_negative=("n_negative", "mean"),
        )
        .sort_values(["source", "mean_delta_cv_auroc"], ascending=[True, False])
    )
    agg_df.to_csv(output_dir / "source_layer_scgpt_aggregate.csv", index=False)

    best_sc_df = agg_df.sort_values(["source", "mean_delta_cv_auroc"], ascending=[True, False]).groupby(
        "source", as_index=False
    ).head(1)
    gf_df = detail_df[detail_df["model"] == "geneformer"][
        ["source", "delta_cv_auroc", "n_pairs", "n_positive", "n_negative"]
    ].rename(
        columns={
            "delta_cv_auroc": "geneformer_delta_cv_auroc",
            "n_pairs": "geneformer_n_pairs",
            "n_positive": "geneformer_n_positive",
            "n_negative": "geneformer_n_negative",
        }
    )
    best_compare_df = best_sc_df.merge(gf_df, on="source", how="left")
    best_compare_df["gap_geneformer_minus_best_scgpt"] = (
        best_compare_df["geneformer_delta_cv_auroc"] - best_compare_df["mean_delta_cv_auroc"]
    )
    best_compare_df.to_csv(output_dir / "source_best_scgpt_vs_geneformer.csv", index=False)

    print("[info] best per-source comparison:")
    print(best_compare_df.to_string(index=False))
    print(f"[done] {output_dir / 'source_seed_layer_detail.csv'}")
    print(f"[done] {output_dir / 'source_layer_scgpt_aggregate.csv'}")
    print(f"[done] {output_dir / 'source_best_scgpt_vs_geneformer.csv'}")


if __name__ == "__main__":
    main()
