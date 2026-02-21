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
            "Deployment-style compact model with outer-split stacking and disagreement "
            "coverage/performance analysis."
        )
    )
    parser.add_argument("--model-id", type=str, default="ctheodoris/Geneformer")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--outer-cv-splits", type=int, default=5)
    parser.add_argument("--outer-cv-repeats", type=int, default=3)
    parser.add_argument("--inner-cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", type=str, default="0.05,0.1,0.15,0.2,0.25,0.3")
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
        default=subproject_root / "implementation" / "outputs" / "cycle25_outer_split_compact_model",
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


def _stacking_oof_train_probs(
    x_sc_train: np.ndarray,
    x_gf_train: np.ndarray,
    y_train: np.ndarray,
    inner_cv_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_train = y_train.shape[0]
    sc_oof = np.zeros(n_train, dtype=np.float64)
    gf_oof = np.zeros(n_train, dtype=np.float64)
    fill_count = np.zeros(n_train, dtype=np.int32)

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
    for inner_train_idx, inner_val_idx in inner_cv.split(x_sc_train, y_train):
        sc_model = _logreg_pipeline()
        gf_model = _logreg_pipeline()
        sc_model.fit(x_sc_train[inner_train_idx], y_train[inner_train_idx])
        gf_model.fit(x_gf_train[inner_train_idx], y_train[inner_train_idx])
        sc_oof[inner_val_idx] += sc_model.predict_proba(x_sc_train[inner_val_idx])[:, 1]
        gf_oof[inner_val_idx] += gf_model.predict_proba(x_gf_train[inner_val_idx])[:, 1]
        fill_count[inner_val_idx] += 1

    missing = fill_count == 0
    if np.any(missing):
        sc_model = _logreg_pipeline()
        gf_model = _logreg_pipeline()
        sc_model.fit(x_sc_train, y_train)
        gf_model.fit(x_gf_train, y_train)
        sc_oof[missing] = sc_model.predict_proba(x_sc_train[missing])[:, 1]
        gf_oof[missing] = gf_model.predict_proba(x_gf_train[missing])[:, 1]
        fill_count[missing] = 1

    sc_oof /= np.clip(fill_count, 1, None)
    gf_oof /= np.clip(fill_count, 1, None)
    return sc_oof, gf_oof


def _outer_split_stacking_probs(
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
    sum_probs = {
        "baseline": np.zeros(n, dtype=np.float64),
        "scgpt": np.zeros(n, dtype=np.float64),
        "geneformer": np.zeros(n, dtype=np.float64),
        "compact": np.zeros(n, dtype=np.float64),
    }
    counts = np.zeros(n, dtype=np.int32)

    n_pos = int(y.sum())
    n_neg = int(n - n_pos)
    n_splits = int(min(outer_cv_splits, n_pos, n_neg))
    if n < 8 or n_pos < 2 or n_neg < 2 or n_splits < 2:
        return {k: np.full(n, np.nan, dtype=np.float64) for k in sum_probs}

    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=outer_cv_repeats, random_state=seed)
    for outer_train_idx, outer_test_idx in outer_cv.split(x_base, y):
        x_base_train, x_base_test = x_base[outer_train_idx], x_base[outer_test_idx]
        x_sc_train, x_sc_test = x_sc[outer_train_idx], x_sc[outer_test_idx]
        x_gf_train, x_gf_test = x_gf[outer_train_idx], x_gf[outer_test_idx]
        y_train = y[outer_train_idx]

        baseline_model = _logreg_pipeline()
        sc_model = _logreg_pipeline()
        gf_model = _logreg_pipeline()
        baseline_model.fit(x_base_train, y_train)
        sc_model.fit(x_sc_train, y_train)
        gf_model.fit(x_gf_train, y_train)

        base_test_prob = baseline_model.predict_proba(x_base_test)[:, 1]
        sc_test_prob = sc_model.predict_proba(x_sc_test)[:, 1]
        gf_test_prob = gf_model.predict_proba(x_gf_test)[:, 1]

        sc_oof_train, gf_oof_train = _stacking_oof_train_probs(
            x_sc_train=x_sc_train,
            x_gf_train=x_gf_train,
            y_train=y_train,
            inner_cv_splits=inner_cv_splits,
            seed=seed,
        )

        compact_model = _logreg_pipeline()
        compact_train_x = np.column_stack([x_base_train, sc_oof_train, gf_oof_train])
        compact_test_x = np.column_stack([x_base_test, sc_test_prob, gf_test_prob])
        compact_model.fit(compact_train_x, y_train)
        compact_test_prob = compact_model.predict_proba(compact_test_x)[:, 1]

        sum_probs["baseline"][outer_test_idx] += base_test_prob
        sum_probs["scgpt"][outer_test_idx] += sc_test_prob
        sum_probs["geneformer"][outer_test_idx] += gf_test_prob
        sum_probs["compact"][outer_test_idx] += compact_test_prob
        counts[outer_test_idx] += 1

    out = {}
    denom = np.clip(counts, 1, None).astype(np.float64)
    for key, values in sum_probs.items():
        probs = values / denom
        probs[counts == 0] = np.nan
        out[key] = probs
    return out


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

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
    threshold_rows = []
    oof_rows = []

    for cfg in domains:
        domain = cfg["domain"]
        print(f"[info] outer-split compact model on {domain}")
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

        oof = _outer_split_stacking_probs(
            x_base=x_base,
            x_sc=x_sc,
            x_gf=x_gf,
            y=y,
            outer_cv_splits=args.outer_cv_splits,
            outer_cv_repeats=args.outer_cv_repeats,
            inner_cv_splits=args.inner_cv_splits,
            seed=args.seed,
        )

        if np.any(np.isnan(oof["compact"])):
            raise RuntimeError(f"NaN predictions encountered in {domain} outer-split evaluation.")

        auc_base = _safe_auc(y, oof["baseline"])
        auc_sc = _safe_auc(y, oof["scgpt"])
        auc_gf = _safe_auc(y, oof["geneformer"])
        auc_compact = _safe_auc(y, oof["compact"])
        best_single_auc = float(np.nanmax([auc_sc, auc_gf]))

        summary_rows.append(
            {
                "domain": domain,
                "n_edges": int(y.shape[0]),
                "n_positive": int(y.sum()),
                "positive_rate": float(np.mean(y)),
                "outer_baseline_auc": auc_base,
                "outer_scgpt_auc": auc_sc,
                "outer_geneformer_auc": auc_gf,
                "outer_compact_auc": auc_compact,
                "outer_best_single_auc": best_single_auc,
                "compact_minus_best_single_auc": auc_compact - best_single_auc,
                "compact_minus_baseline_auc": auc_compact - auc_base,
            }
        )

        disagreement = np.abs(oof["geneformer"] - oof["scgpt"])
        for tau in thresholds:
            mask = disagreement >= tau
            y_sub = y[mask]
            sc_sub = oof["scgpt"][mask]
            gf_sub = oof["geneformer"][mask]
            compact_sub = oof["compact"][mask]
            auc_sc_sub = _safe_auc(y_sub, sc_sub)
            auc_gf_sub = _safe_auc(y_sub, gf_sub)
            auc_compact_sub = _safe_auc(y_sub, compact_sub)
            best_single_sub = float(np.nanmax([auc_sc_sub, auc_gf_sub]))
            threshold_rows.append(
                {
                    "domain": domain,
                    "disagreement_threshold": tau,
                    "n_edges": int(mask.sum()),
                    "coverage_fraction": float(np.mean(mask)),
                    "positive_rate": float(np.mean(y_sub)) if y_sub.size > 0 else float("nan"),
                    "auc_scgpt": auc_sc_sub,
                    "auc_geneformer": auc_gf_sub,
                    "auc_compact": auc_compact_sub,
                    "best_single_auc": best_single_sub,
                    "compact_minus_best_single_auc": auc_compact_sub - best_single_sub,
                    "mean_abs_disagreement": float(np.mean(disagreement[mask])) if np.any(mask) else float("nan"),
                }
            )

        domain_oof = pd.DataFrame(
            {
                "domain": domain,
                "label": y,
                "baseline_prob": oof["baseline"],
                "scgpt_prob": oof["scgpt"],
                "geneformer_prob": oof["geneformer"],
                "compact_prob": oof["compact"],
                "abs_disagreement": disagreement,
            }
        )
        oof_rows.append(domain_oof)

    summary_df = pd.DataFrame(summary_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    oof_df = pd.concat(oof_rows, axis=0, ignore_index=True)

    summary_path = output_dir / "outer_split_compact_model_summary.csv"
    threshold_path = output_dir / "outer_split_disagreement_policy_summary.csv"
    oof_path = output_dir / "outer_split_oof_predictions.tsv"

    summary_df.to_csv(summary_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    oof_df.to_csv(oof_path, sep="\t", index=False)

    print(summary_df.to_string(index=False))
    print(threshold_df.to_string(index=False))
    print(f"[done] {summary_path}")
    print(f"[done] {threshold_path}")
    print(f"[done] {oof_path}")


if __name__ == "__main__":
    main()
