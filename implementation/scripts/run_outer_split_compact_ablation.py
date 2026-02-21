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
        description="Outer-split compact-model ablation: prob-only vs prob+baseline."
    )
    parser.add_argument("--model-id", type=str, default="ctheodoris/Geneformer")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--outer-cv-splits", type=int, default=5)
    parser.add_argument("--outer-cv-repeats", type=int, default=3)
    parser.add_argument("--inner-cv-splits", type=int, default=5)
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
        "--external-lung-processed-h5ad",
        type=Path,
        default=single_cell_root / "outputs" / "invariant_causal_edges" / "external_lung" / "processed.h5ad",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle28_outer_split_compact_ablation",
    )
    return parser.parse_args()


def _logreg_pipeline() -> Pipeline:
    return Pipeline(
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


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _oof_base_scores(
    x_sc_train: np.ndarray,
    x_gf_train: np.ndarray,
    y_train: np.ndarray,
    inner_cv_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_train = y_train.shape[0]
    sc_oof = np.zeros(n_train, dtype=np.float64)
    gf_oof = np.zeros(n_train, dtype=np.float64)
    cnt = np.zeros(n_train, dtype=np.int32)

    n_pos = int(y_train.sum())
    n_neg = int(n_train - n_pos)
    n_splits = int(min(inner_cv_splits, n_pos, n_neg))
    if n_train < 8 or n_pos < 2 or n_neg < 2 or n_splits < 2:
        sc_model = _logreg_pipeline()
        gf_model = _logreg_pipeline()
        sc_model.fit(x_sc_train, y_train)
        gf_model.fit(x_gf_train, y_train)
        return (
            sc_model.predict_proba(x_sc_train)[:, 1],
            gf_model.predict_proba(x_gf_train)[:, 1],
        )

    inner_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=seed)
    for tr, va in inner_cv.split(x_sc_train, y_train):
        sc_model = _logreg_pipeline()
        gf_model = _logreg_pipeline()
        sc_model.fit(x_sc_train[tr], y_train[tr])
        gf_model.fit(x_gf_train[tr], y_train[tr])
        sc_oof[va] += sc_model.predict_proba(x_sc_train[va])[:, 1]
        gf_oof[va] += gf_model.predict_proba(x_gf_train[va])[:, 1]
        cnt[va] += 1

    missing = cnt == 0
    if np.any(missing):
        sc_model = _logreg_pipeline()
        gf_model = _logreg_pipeline()
        sc_model.fit(x_sc_train, y_train)
        gf_model.fit(x_gf_train, y_train)
        sc_oof[missing] = sc_model.predict_proba(x_sc_train[missing])[:, 1]
        gf_oof[missing] = gf_model.predict_proba(x_gf_train[missing])[:, 1]
        cnt[missing] = 1

    return sc_oof / np.clip(cnt, 1, None), gf_oof / np.clip(cnt, 1, None)


def _outer_oof_ablation(
    x_base: np.ndarray,
    x_sc: np.ndarray,
    x_gf: np.ndarray,
    y: np.ndarray,
    outer_cv_splits: int,
    outer_cv_repeats: int,
    inner_cv_splits: int,
    seed: int,
) -> dict[str, np.ndarray]:
    n = y.shape[0]
    sums = {
        "scgpt": np.zeros(n, dtype=np.float64),
        "geneformer": np.zeros(n, dtype=np.float64),
        "compact_prob_only": np.zeros(n, dtype=np.float64),
        "compact_prob_plus_baseline": np.zeros(n, dtype=np.float64),
    }
    cnt = np.zeros(n, dtype=np.int32)

    n_pos = int(y.sum())
    n_neg = int(n - n_pos)
    n_splits = int(min(outer_cv_splits, n_pos, n_neg))
    if n < 8 or n_pos < 2 or n_neg < 2 or n_splits < 2:
        return {k: np.full(n, np.nan, dtype=np.float64) for k in sums}

    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=outer_cv_repeats, random_state=seed)
    for tr, te in outer_cv.split(x_base, y):
        x_base_tr, x_base_te = x_base[tr], x_base[te]
        x_sc_tr, x_sc_te = x_sc[tr], x_sc[te]
        x_gf_tr, x_gf_te = x_gf[tr], x_gf[te]
        y_tr = y[tr]

        sc_model = _logreg_pipeline()
        gf_model = _logreg_pipeline()
        sc_model.fit(x_sc_tr, y_tr)
        gf_model.fit(x_gf_tr, y_tr)
        sc_te_prob = sc_model.predict_proba(x_sc_te)[:, 1]
        gf_te_prob = gf_model.predict_proba(x_gf_te)[:, 1]

        sc_tr_oof, gf_tr_oof = _oof_base_scores(
            x_sc_train=x_sc_tr,
            x_gf_train=x_gf_tr,
            y_train=y_tr,
            inner_cv_splits=inner_cv_splits,
            seed=seed,
        )

        compact_prob_only = _logreg_pipeline()
        compact_prob_plus_base = _logreg_pipeline()
        compact_prob_only.fit(np.column_stack([sc_tr_oof, gf_tr_oof]), y_tr)
        compact_prob_plus_base.fit(np.column_stack([x_base_tr, sc_tr_oof, gf_tr_oof]), y_tr)

        cp_te = compact_prob_only.predict_proba(np.column_stack([sc_te_prob, gf_te_prob]))[:, 1]
        cpb_te = compact_prob_plus_base.predict_proba(np.column_stack([x_base_te, sc_te_prob, gf_te_prob]))[:, 1]

        sums["scgpt"][te] += sc_te_prob
        sums["geneformer"][te] += gf_te_prob
        sums["compact_prob_only"][te] += cp_te
        sums["compact_prob_plus_baseline"][te] += cpb_te
        cnt[te] += 1

    out = {}
    denom = np.clip(cnt, 1, None).astype(np.float64)
    for k, v in sums.items():
        p = v / denom
        p[cnt == 0] = np.nan
        out[k] = p
    return out


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
    oof_tables = []

    for cfg in domains:
        domain = cfg["domain"]
        print(f"[info] compact ablation on {domain}")
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
        x_gf = np.column_stack([x_base, gf_feat[valid].reshape(-1, 1)])

        oof = _outer_oof_ablation(
            x_base=x_base,
            x_sc=x_sc,
            x_gf=x_gf,
            y=y,
            outer_cv_splits=args.outer_cv_splits,
            outer_cv_repeats=args.outer_cv_repeats,
            inner_cv_splits=args.inner_cv_splits,
            seed=args.seed,
        )

        auc_sc = _safe_auc(y, oof["scgpt"])
        auc_gf = _safe_auc(y, oof["geneformer"])
        auc_compact_prob = _safe_auc(y, oof["compact_prob_only"])
        auc_compact_prob_base = _safe_auc(y, oof["compact_prob_plus_baseline"])
        best_single = float(np.nanmax([auc_sc, auc_gf]))

        summary_rows.append(
            {
                "domain": domain,
                "n_edges": int(y.shape[0]),
                "n_positive": int(y.sum()),
                "positive_rate": float(np.mean(y)),
                "outer_scgpt_auc": auc_sc,
                "outer_geneformer_auc": auc_gf,
                "outer_best_single_auc": best_single,
                "outer_compact_prob_only_auc": auc_compact_prob,
                "outer_compact_prob_plus_baseline_auc": auc_compact_prob_base,
                "compact_prob_only_minus_best_single_auc": auc_compact_prob - best_single,
                "compact_prob_plus_baseline_minus_best_single_auc": auc_compact_prob_base - best_single,
                "compact_prob_plus_baseline_minus_prob_only_auc": auc_compact_prob_base - auc_compact_prob,
            }
        )

        oof_tables.append(
            pd.DataFrame(
                {
                    "domain": domain,
                    "label": y,
                    "scgpt_prob": oof["scgpt"],
                    "geneformer_prob": oof["geneformer"],
                    "compact_prob_only": oof["compact_prob_only"],
                    "compact_prob_plus_baseline": oof["compact_prob_plus_baseline"],
                }
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    oof_df = pd.concat(oof_tables, axis=0, ignore_index=True)

    summary_path = output_dir / "outer_split_compact_ablation_summary.csv"
    oof_path = output_dir / "outer_split_compact_ablation_oof.tsv"
    summary_df.to_csv(summary_path, index=False)
    oof_df.to_csv(oof_path, sep="\t", index=False)

    print(summary_df.to_string(index=False))
    print(f"[done] {summary_path}")
    print(f"[done] {oof_path}")


if __name__ == "__main__":
    main()
