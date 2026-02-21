#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from transformers import AutoModel


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent
    single_cell_root = subproject_root.parent / "single_cell_mechinterp"

    parser = argparse.ArgumentParser(
        description=(
            "Cross-model disagreement stratification on shared edges: quantify label rates by "
            "disagreement decile and source-TF enrichment in high-disagreement bins."
        )
    )
    parser.add_argument("--model-id", type=str, default="ctheodoris/Geneformer")
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--n-quantiles", type=int, default=10)
    parser.add_argument("--min-source-count", type=int, default=20)
    parser.add_argument("--min-source-top-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle15_cross_model_disagreement_stratification",
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


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma <= 0:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mu) / sigma


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

    quantile_rows = []
    enrich_rows = []

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

        df = pd.DataFrame(
            {
                "source": sources[valid],
                "target": targets[valid],
                "label": labels[valid],
                "scgpt_score": scgpt_feature[valid],
                "geneformer_score": geneformer_feature[valid],
            }
        )
        df["scgpt_z"] = _zscore(df["scgpt_score"].to_numpy(dtype=np.float64))
        df["geneformer_z"] = _zscore(df["geneformer_score"].to_numpy(dtype=np.float64))
        df["abs_disagreement"] = np.abs(df["scgpt_z"] - df["geneformer_z"])

        # Disagreement quantile bins and label rates.
        bins = pd.qcut(df["abs_disagreement"], q=args.n_quantiles, labels=False, duplicates="drop")
        df["disagreement_bin"] = bins.astype(int)
        grouped = (
            df.groupby("disagreement_bin", as_index=False)
            .agg(
                n_edges=("label", "size"),
                n_positive=("label", "sum"),
                positive_rate=("label", "mean"),
                mean_abs_disagreement=("abs_disagreement", "mean"),
                mean_scgpt_score=("scgpt_score", "mean"),
                mean_geneformer_score=("geneformer_score", "mean"),
            )
            .sort_values("disagreement_bin")
        )
        grouped.insert(0, "domain", domain)
        quantile_rows.append(grouped)

        # Source enrichment in top disagreement bin.
        top_bin = int(df["disagreement_bin"].max())
        top_df = df[df["disagreement_bin"] == top_bin]
        overall_source_counts = df["source"].value_counts()
        top_source_counts = top_df["source"].value_counts()
        n_overall = int(df.shape[0])
        n_top = int(top_df.shape[0])
        for source, overall_count in overall_source_counts.items():
            if int(overall_count) < args.min_source_count:
                continue
            top_count = int(top_source_counts.get(source, 0))
            if top_count < args.min_source_top_count:
                continue
            overall_share = overall_count / max(n_overall, 1)
            top_share = top_count / max(n_top, 1)
            enrich_rows.append(
                {
                    "domain": domain,
                    "source": source,
                    "overall_count": int(overall_count),
                    "top_bin_count": top_count,
                    "overall_share": overall_share,
                    "top_bin_share": top_share,
                    "enrichment_ratio_top_vs_overall": top_share / max(overall_share, 1e-12),
                }
            )

    quantile_df = pd.concat(quantile_rows, axis=0, ignore_index=True) if quantile_rows else pd.DataFrame()
    enrich_df = pd.DataFrame(enrich_rows)
    if not enrich_df.empty:
        enrich_df = enrich_df.sort_values(
            ["domain", "enrichment_ratio_top_vs_overall", "top_bin_count"],
            ascending=[True, False, False],
        )

    quantile_df.to_csv(output_dir / "disagreement_quantile_label_rates.csv", index=False)
    enrich_df.to_csv(output_dir / "source_enrichment_top_disagreement_bin.csv", index=False)

    print("[info] disagreement quantiles:")
    print(quantile_df.to_string(index=False))
    print("[info] top source enrichments (head):")
    if enrich_df.empty:
        print("empty")
    else:
        print(enrich_df.groupby("domain", group_keys=False).head(10).to_string(index=False))
    print(f"[done] {output_dir / 'disagreement_quantile_label_rates.csv'}")
    print(f"[done] {output_dir / 'source_enrichment_top_disagreement_bin.csv'}")


if __name__ == "__main__":
    main()
