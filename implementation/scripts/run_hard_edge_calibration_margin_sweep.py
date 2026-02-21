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
            "Hard-edge calibration + margin sensitivity sweep after seed-ensemble "
            "bundle improvements."
        )
    )
    parser.add_argument("--model-id", type=str, default="ctheodoris/Geneformer")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margins", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--n-calibration-bins", type=int, default=10)
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
        default=subproject_root / "implementation" / "outputs" / "cycle24_hard_edge_calibration_margin_sweep",
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


def _group_masks(y: np.ndarray, sc_probs: np.ndarray, gf_probs: np.ndarray, margin: float):
    pos_hard = (y == 1) & ((gf_probs - sc_probs) >= margin)
    neg_hard = (y == 0) & ((sc_probs - gf_probs) >= margin)
    hard = pos_hard | neg_hard
    return {
        "all": np.ones_like(y, dtype=bool),
        "hard": hard,
        "non_hard": ~hard,
    }, pos_hard, neg_hard


def _ece_and_bins(y: np.ndarray, probs: np.ndarray, n_bins: int):
    bins = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    bin_idx = np.digitize(probs, bins[1:-1], right=False)
    total = max(y.shape[0], 1)
    ece = 0.0
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin_index": b,
                    "bin_left": float(bins[b]),
                    "bin_right": float(bins[b + 1]),
                    "n_edges": 0,
                    "mean_pred_prob": np.nan,
                    "empirical_positive_rate": np.nan,
                    "abs_calibration_gap": np.nan,
                }
            )
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(y[mask]))
        gap = abs(acc - conf)
        ece += (n / total) * gap
        rows.append(
            {
                "bin_index": b,
                "bin_left": float(bins[b]),
                "bin_right": float(bins[b + 1]),
                "n_edges": n,
                "mean_pred_prob": conf,
                "empirical_positive_rate": acc,
                "abs_calibration_gap": gap,
            }
        )
    return float(ece), rows


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    margins = [float(x) for x in args.margins.split(",") if x.strip()]

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

    margin_rows = []
    cal_rows = []
    reliability_rows = []

    for cfg in domains:
        domain = cfg["domain"]
        print(f"[info] domain={domain}")
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
        for s in range(len(seeds)):
            for layer in range(12):
                valid &= per_seed_valid[s][layer]

        gf_feat = _compute_geneformer_feature(emb_weight=emb_weight, src_tok=src_tok, tgt_tok=tgt_tok)
        valid &= np.isfinite(gf_feat)

        ens_layers = []
        for layer in range(12):
            stack = np.column_stack([per_seed_layer[s][layer][valid] for s in range(len(seeds))])
            ens_layers.append(np.mean(stack, axis=1))
        sc_stack = np.column_stack(ens_layers)

        y = labels[valid]
        x_base = baseline[valid]
        x_sc = np.column_stack([x_base, sc_stack])
        x_gf = np.column_stack([x_base, gf_feat[valid]])

        sc_probs = _crossval_probs(x_sc, y, args.cv_splits, args.cv_repeats, args.seed)
        gf_probs = _crossval_probs(x_gf, y, args.cv_splits, args.cv_repeats, args.seed)

        for margin in margins:
            group_masks, pos_hard, neg_hard = _group_masks(y=y, sc_probs=sc_probs, gf_probs=gf_probs, margin=margin)
            hard = group_masks["hard"]
            margin_rows.append(
                {
                    "domain": domain,
                    "margin": margin,
                    "n_edges": int(y.shape[0]),
                    "n_positive": int(y.sum()),
                    "positive_rate": float(np.mean(y)),
                    "n_hard_edges": int(hard.sum()),
                    "hard_edge_fraction": float(np.mean(hard)),
                    "n_hard_positive_type": int(pos_hard.sum()),
                    "n_hard_negative_type": int(neg_hard.sum()),
                    "mean_prob_gap_hard": float(np.mean((gf_probs - sc_probs)[hard]))
                    if np.any(hard)
                    else float("nan"),
                    "mean_prob_gap_nonhard": float(np.mean((gf_probs - sc_probs)[~hard]))
                    if np.any(~hard)
                    else float("nan"),
                }
            )

            for group_name, mask in group_masks.items():
                if not np.any(mask):
                    continue
                y_g = y[mask]
                if y_g.size == 0:
                    continue
                for model_name, probs_g in [("scgpt", sc_probs[mask]), ("geneformer", gf_probs[mask])]:
                    auc = float("nan")
                    if np.unique(y_g).size >= 2:
                        auc = float(roc_auc_score(y_g, probs_g))
                    brier = float(np.mean((probs_g - y_g) ** 2))
                    ece, bin_rows = _ece_and_bins(y=y_g, probs=probs_g, n_bins=args.n_calibration_bins)
                    cal_rows.append(
                        {
                            "domain": domain,
                            "margin": margin,
                            "group": group_name,
                            "model": model_name,
                            "n_edges": int(y_g.shape[0]),
                            "n_positive": int(y_g.sum()),
                            "positive_rate": float(np.mean(y_g)),
                            "mean_pred_prob": float(np.mean(probs_g)),
                            "auc": auc,
                            "brier_score": brier,
                            "ece": ece,
                        }
                    )
                    for b in bin_rows:
                        reliability_rows.append(
                            {
                                "domain": domain,
                                "margin": margin,
                                "group": group_name,
                                "model": model_name,
                                **b,
                            }
                        )

    margin_df = pd.DataFrame(margin_rows)
    cal_df = pd.DataFrame(cal_rows)
    reliability_df = pd.DataFrame(reliability_rows)

    margin_path = output_dir / "hard_edge_margin_summary.csv"
    cal_path = output_dir / "hard_edge_calibration_summary.csv"
    reliability_path = output_dir / "hard_edge_reliability_bins.csv"

    margin_df.to_csv(margin_path, index=False)
    cal_df.to_csv(cal_path, index=False)
    reliability_df.to_csv(reliability_path, index=False)

    print(margin_df.to_string(index=False))
    print(cal_df.to_string(index=False))
    print(f"[done] {margin_path}")
    print(f"[done] {cal_path}")
    print(f"[done] {reliability_path}")


if __name__ == "__main__":
    main()
