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
            "Cross-domain seed-ensemble bundle transfer: apply the external-lung multi-layer "
            "scGPT bundling protocol to immune and lung shared-edge settings."
        )
    )
    parser.add_argument("--model-id", type=str, default="ctheodoris/Geneformer")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
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
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle21_cross_domain_seed_ensemble_bundle_transfer",
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


def _domain_run_dir(subproject_root: Path, domain: str, seed: int) -> Path:
    if domain == "immune":
        if seed == 42:
            return subproject_root / "implementation" / "outputs" / "cycle4_immune_main"
        return subproject_root / "implementation" / "outputs" / f"cycle4_immune_seed{seed}"
    if domain == "lung":
        if seed == 42:
            return subproject_root / "implementation" / "outputs" / "cycle6_lung_main"
        return subproject_root / "implementation" / "outputs" / f"cycle6_lung_seed{seed}"
    raise ValueError(f"Unsupported domain: {domain}")


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent

    domains = [
        {
            "domain": "immune",
            "processed_h5ad": args.immune_processed_h5ad,
            "shared_edge_tsv": subproject_root
            / "implementation"
            / "outputs"
            / "cycle12_geneformer_immune_bootstrap"
            / "geneformer_edge_dataset.tsv",
        },
        {
            "domain": "lung",
            "processed_h5ad": args.lung_processed_h5ad,
            "shared_edge_tsv": subproject_root
            / "implementation"
            / "outputs"
            / "cycle12_geneformer_lung_bootstrap"
            / "geneformer_edge_dataset.tsv",
        },
    ]

    model = AutoModel.from_pretrained(args.model_id)
    emb_weight = model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32)

    bundles = [
        ("L3", [3]),
        ("L1_4", [1, 2, 3, 4]),
        ("L0_5", [0, 1, 2, 3, 4, 5]),
        ("L0_11", list(range(12))),
    ]

    rows = []
    for cfg in domains:
        domain = cfg["domain"]
        print(f"[info] evaluating {domain}")
        adata = ad.read_h5ad(cfg["processed_h5ad"])
        edge_df = pd.read_csv(cfg["shared_edge_tsv"], sep="\t")
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

        gf_feat = _compute_geneformer_feature(emb_weight=emb_weight, src_tok=src_tok, tgt_tok=tgt_tok)
        gf_valid = base_valid & np.isfinite(gf_feat)

        # Per-seed layer features.
        seed_layer_feat: dict[int, dict[int, np.ndarray]] = {}
        seed_layer_valid: dict[int, dict[int, np.ndarray]] = {}
        for seed in seeds:
            run_dir = _domain_run_dir(subproject_root, domain, seed)
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            layer_counts = np.load(run_dir / "layer_gene_embedding_count.npy", mmap_mode="r")
            layer_feat = {}
            layer_valid = {}
            for layer in range(layer_embeddings.shape[0]):
                emb = np.asarray(layer_embeddings[layer], dtype=np.float32)
                counts = np.asarray(layer_counts[layer], dtype=np.int32)
                feat = _compute_scgpt_layer_feature(
                    emb=emb,
                    src_idx=src_idx,
                    tgt_idx=tgt_idx,
                    pca_dim=args.pca_dim,
                    seed=seed,
                )
                valid = base_valid & (counts[src_idx] > 0) & (counts[tgt_idx] > 0) & np.isfinite(feat)
                layer_feat[layer] = feat
                layer_valid[layer] = valid
            seed_layer_feat[seed] = layer_feat
            seed_layer_valid[seed] = layer_valid

        # Single-seed rows.
        for seed in seeds:
            for name, layers in bundles:
                valid = gf_valid.copy()
                for layer in layers:
                    valid &= seed_layer_valid[seed][layer]
                y = labels[valid]
                x_base = baseline[valid]
                sc_stack = np.column_stack([seed_layer_feat[seed][layer][valid] for layer in layers])
                gf = gf_feat[valid].reshape(-1, 1)

                auc_base = _crossval_auc(x_base, y, args.cv_splits, args.cv_repeats, args.seed)
                auc_sc = _crossval_auc(
                    np.column_stack([x_base, sc_stack]), y, args.cv_splits, args.cv_repeats, args.seed
                )
                auc_gf = _crossval_auc(np.column_stack([x_base, gf]), y, args.cv_splits, args.cv_repeats, args.seed)
                auc_both = _crossval_auc(
                    np.column_stack([x_base, sc_stack, gf]), y, args.cv_splits, args.cv_repeats, args.seed
                )

                rows.append(
                    {
                        "domain": domain,
                        "mode": "single_seed",
                        "seed": seed,
                        "bundle": name,
                        "layers": ",".join(str(x) for x in layers),
                        "n_layers": len(layers),
                        "n_pairs": int(y.shape[0]),
                        "n_positive": int(y.sum()),
                        "baseline_cv_auroc": auc_base,
                        "baseline_plus_scgpt_cv_auroc": auc_sc,
                        "baseline_plus_geneformer_cv_auroc": auc_gf,
                        "baseline_plus_both_cv_auroc": auc_both,
                        "scgpt_delta_over_base": auc_sc - auc_base
                        if np.isfinite(auc_sc) and np.isfinite(auc_base)
                        else float("nan"),
                        "geneformer_delta_over_base": auc_gf - auc_base
                        if np.isfinite(auc_gf) and np.isfinite(auc_base)
                        else float("nan"),
                        "both_delta_over_base": auc_both - auc_base
                        if np.isfinite(auc_both) and np.isfinite(auc_base)
                        else float("nan"),
                        "gap_geneformer_minus_scgpt": auc_gf - auc_sc
                        if np.isfinite(auc_sc) and np.isfinite(auc_gf)
                        else float("nan"),
                        "both_minus_geneformer": auc_both - auc_gf
                        if np.isfinite(auc_both) and np.isfinite(auc_gf)
                        else float("nan"),
                    }
                )

        # Seed-ensemble rows.
        for name, layers in bundles:
            valid = gf_valid.copy()
            for layer in layers:
                for seed in seeds:
                    valid &= seed_layer_valid[seed][layer]
            y = labels[valid]
            x_base = baseline[valid]
            ens_layers = []
            for layer in layers:
                stack = np.column_stack([seed_layer_feat[seed][layer][valid] for seed in seeds])
                ens_layers.append(np.mean(stack, axis=1))
            sc_ens = np.column_stack(ens_layers)
            gf = gf_feat[valid].reshape(-1, 1)

            auc_base = _crossval_auc(x_base, y, args.cv_splits, args.cv_repeats, args.seed)
            auc_sc = _crossval_auc(np.column_stack([x_base, sc_ens]), y, args.cv_splits, args.cv_repeats, args.seed)
            auc_gf = _crossval_auc(np.column_stack([x_base, gf]), y, args.cv_splits, args.cv_repeats, args.seed)
            auc_both = _crossval_auc(np.column_stack([x_base, sc_ens, gf]), y, args.cv_splits, args.cv_repeats, args.seed)

            rows.append(
                {
                    "domain": domain,
                    "mode": "seed_ensemble_mean",
                    "seed": -1,
                    "bundle": name,
                    "layers": ",".join(str(x) for x in layers),
                    "n_layers": len(layers),
                    "n_pairs": int(y.shape[0]),
                    "n_positive": int(y.sum()),
                    "baseline_cv_auroc": auc_base,
                    "baseline_plus_scgpt_cv_auroc": auc_sc,
                    "baseline_plus_geneformer_cv_auroc": auc_gf,
                    "baseline_plus_both_cv_auroc": auc_both,
                    "scgpt_delta_over_base": auc_sc - auc_base
                    if np.isfinite(auc_sc) and np.isfinite(auc_base)
                    else float("nan"),
                    "geneformer_delta_over_base": auc_gf - auc_base
                    if np.isfinite(auc_gf) and np.isfinite(auc_base)
                    else float("nan"),
                    "both_delta_over_base": auc_both - auc_base
                    if np.isfinite(auc_both) and np.isfinite(auc_base)
                    else float("nan"),
                    "gap_geneformer_minus_scgpt": auc_gf - auc_sc
                    if np.isfinite(auc_sc) and np.isfinite(auc_gf)
                    else float("nan"),
                    "both_minus_geneformer": auc_both - auc_gf
                    if np.isfinite(auc_both) and np.isfinite(auc_gf)
                    else float("nan"),
                }
            )

    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(output_dir / "cross_domain_seed_ensemble_bundle_detail.csv", index=False)

    seed_only = detail_df[detail_df["mode"] == "single_seed"]
    agg_df = (
        seed_only.groupby(["domain", "bundle"], as_index=False)
        .agg(
            n_seed_runs=("seed", "count"),
            mean_scgpt_delta=("scgpt_delta_over_base", "mean"),
            std_scgpt_delta=("scgpt_delta_over_base", "std"),
            mean_gap_geneformer_minus_scgpt=("gap_geneformer_minus_scgpt", "mean"),
            std_gap_geneformer_minus_scgpt=("gap_geneformer_minus_scgpt", "std"),
            mean_both_minus_geneformer=("both_minus_geneformer", "mean"),
            std_both_minus_geneformer=("both_minus_geneformer", "std"),
        )
        .sort_values(["domain", "bundle"])
    )
    agg_df.to_csv(output_dir / "cross_domain_seed_ensemble_bundle_seed_aggregate.csv", index=False)

    print("[info] detail:")
    print(detail_df.to_string(index=False))
    print("[info] aggregate:")
    print(agg_df.to_string(index=False))
    print(f"[done] {output_dir / 'cross_domain_seed_ensemble_bundle_detail.csv'}")
    print(f"[done] {output_dir / 'cross_domain_seed_ensemble_bundle_seed_aggregate.csv'}")


if __name__ == "__main__":
    main()
