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
            "Hard-edge profiling after bundle improvements: identify residual Geneformer-"
            "advantaged edges under seed-ensemble L0-11 scGPT setup."
        )
    )
    parser.add_argument("--model-id", type=str, default="ctheodoris/Geneformer")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=0.4)
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle22_hard_edge_profiling_after_bundle",
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


def _crossval_probs(features: np.ndarray, labels: np.ndarray, cv_splits: int, cv_repeats: int, seed: int) -> np.ndarray:
    n_pos = int(labels.sum())
    n_neg = int(labels.shape[0] - n_pos)
    n_splits = int(min(cv_splits, n_pos, n_neg))
    if labels.shape[0] < 8 or n_pos < 2 or n_neg < 2 or n_splits < 2:
        return np.full(labels.shape[0], np.nan, dtype=np.float64)

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
    return probs


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


def _run_dir(subproject_root: Path, domain: str, seed: int) -> Path:
    if domain == "immune":
        if seed == 42:
            return subproject_root / "implementation" / "outputs" / "cycle4_immune_main"
        return subproject_root / "implementation" / "outputs" / f"cycle4_immune_seed{seed}"
    if domain == "lung":
        if seed == 42:
            return subproject_root / "implementation" / "outputs" / "cycle6_lung_main"
        return subproject_root / "implementation" / "outputs" / f"cycle6_lung_seed{seed}"
    if domain == "external_lung":
        if seed == 42:
            return subproject_root / "implementation" / "outputs" / "cycle7_external_lung_main"
        return subproject_root / "implementation" / "outputs" / f"cycle7_external_lung_seed{seed}"
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
            "edge_tsv": subproject_root
            / "implementation"
            / "outputs"
            / "cycle12_geneformer_immune_bootstrap"
            / "geneformer_edge_dataset.tsv",
        },
        {
            "domain": "lung",
            "processed_h5ad": args.lung_processed_h5ad,
            "edge_tsv": subproject_root
            / "implementation"
            / "outputs"
            / "cycle12_geneformer_lung_bootstrap"
            / "geneformer_edge_dataset.tsv",
        },
        {
            "domain": "external_lung",
            "processed_h5ad": args.external_lung_processed_h5ad,
            "edge_tsv": subproject_root
            / "implementation"
            / "outputs"
            / "cycle12_geneformer_external_lung_bootstrap"
            / "geneformer_edge_dataset.tsv",
        },
    ]

    model = AutoModel.from_pretrained(args.model_id)
    emb_weight = model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32)

    summary_rows = []
    profile_rows = []
    hard_edge_tables = []

    for cfg in domains:
        domain = cfg["domain"]
        print(f"[info] profiling {domain}")
        adata = ad.read_h5ad(cfg["processed_h5ad"])
        edge_df = pd.read_csv(cfg["edge_tsv"], sep="\t")

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

        # Seed-ensemble scGPT L0-11 features.
        per_seed_layer = []
        per_seed_valid = []
        for seed in seeds:
            run_dir = _run_dir(subproject_root, domain, seed)
            layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
            layer_counts = np.load(run_dir / "layer_gene_embedding_count.npy", mmap_mode="r")
            layer_feats = []
            layer_valids = []
            for layer in range(12):
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
                layer_feats.append(feat)
                layer_valids.append(valid)
            per_seed_layer.append(layer_feats)
            per_seed_valid.append(layer_valids)

        valid = base_valid.copy()
        for seed_idx in range(len(seeds)):
            for layer in range(12):
                valid &= per_seed_valid[seed_idx][layer]

        gf_feat = _compute_geneformer_feature(emb_weight=emb_weight, src_tok=src_tok, tgt_tok=tgt_tok)
        valid &= np.isfinite(gf_feat)

        # Build seed-ensemble scGPT layer stack (mean across seeds per layer).
        sc_layers = []
        for layer in range(12):
            stack = np.column_stack([per_seed_layer[s][layer][valid] for s in range(len(seeds))])
            sc_layers.append(np.mean(stack, axis=1))
        sc_stack = np.column_stack(sc_layers)

        y = labels[valid]
        x_base = baseline[valid]
        x_sc = np.column_stack([x_base, sc_stack])
        x_gf = np.column_stack([x_base, gf_feat[valid]])

        sc_probs = _crossval_probs(x_sc, y, args.cv_splits, args.cv_repeats, args.seed)
        gf_probs = _crossval_probs(x_gf, y, args.cv_splits, args.cv_repeats, args.seed)

        # Define hard edges where Geneformer is much more confident and correct.
        pos_hard = (y == 1) & ((gf_probs - sc_probs) >= args.margin)
        neg_hard = (y == 0) & ((sc_probs - gf_probs) >= args.margin)
        hard = pos_hard | neg_hard

        domain_df = edge_df.loc[valid, ["source", "target", "label"]].copy()
        domain_df["domain"] = domain
        domain_df["scgpt_prob"] = sc_probs
        domain_df["geneformer_prob"] = gf_probs
        domain_df["prob_gap_gf_minus_scgpt"] = gf_probs - sc_probs
        domain_df["hard_edge"] = hard
        domain_df["hard_type"] = np.where(
            pos_hard, "gf_adv_pos", np.where(neg_hard, "gf_adv_neg", "not_hard")
        )
        hard_edge_tables.append(domain_df)

        n = int(domain_df.shape[0])
        n_hard = int(hard.sum())
        summary_rows.append(
            {
                "domain": domain,
                "n_edges": n,
                "n_positive": int(y.sum()),
                "n_hard_edges": n_hard,
                "hard_edge_fraction": n_hard / max(n, 1),
                "n_hard_positive_type": int(pos_hard.sum()),
                "n_hard_negative_type": int(neg_hard.sum()),
                "mean_prob_gap_hard": float(np.mean((gf_probs - sc_probs)[hard])) if n_hard > 0 else float("nan"),
                "mean_prob_gap_nonhard": float(np.mean((gf_probs - sc_probs)[~hard])) if n_hard < n else float("nan"),
            }
        )

        # Baseline/confound profile comparison: hard vs non-hard.
        prof = pd.DataFrame(
            {
                "hard": hard,
                "source_mean_expr": x_base[:, 0],
                "target_mean_expr": x_base[:, 1],
                "source_var_expr": x_base[:, 2],
                "target_var_expr": x_base[:, 3],
                "source_detection": x_base[:, 4],
                "target_detection": x_base[:, 5],
                "label": y,
            }
        )
        for grp_name, grp_mask in [("hard", prof["hard"].to_numpy()), ("non_hard", ~prof["hard"].to_numpy())]:
            if not np.any(grp_mask):
                continue
            grp = prof.loc[grp_mask]
            profile_rows.append(
                {
                    "domain": domain,
                    "group": grp_name,
                    "n_edges": int(grp.shape[0]),
                    "positive_rate": float(grp["label"].mean()),
                    "source_mean_expr_mean": float(grp["source_mean_expr"].mean()),
                    "target_mean_expr_mean": float(grp["target_mean_expr"].mean()),
                    "source_var_expr_mean": float(grp["source_var_expr"].mean()),
                    "target_var_expr_mean": float(grp["target_var_expr"].mean()),
                    "source_detection_mean": float(grp["source_detection"].mean()),
                    "target_detection_mean": float(grp["target_detection"].mean()),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    profile_df = pd.DataFrame(profile_rows)
    hard_edges_df = pd.concat(hard_edge_tables, axis=0, ignore_index=True)

    summary_df.to_csv(output_dir / "hard_edge_summary.csv", index=False)
    profile_df.to_csv(output_dir / "hard_edge_baseline_profile.csv", index=False)
    hard_edges_df.to_csv(output_dir / "hard_edges_detailed.tsv", sep="\t", index=False)

    print(summary_df.to_string(index=False))
    print(profile_df.to_string(index=False))
    print(f"[done] {output_dir / 'hard_edge_summary.csv'}")
    print(f"[done] {output_dir / 'hard_edge_baseline_profile.csv'}")
    print(f"[done] {output_dir / 'hard_edges_detailed.tsv'}")


if __name__ == "__main__":
    main()
