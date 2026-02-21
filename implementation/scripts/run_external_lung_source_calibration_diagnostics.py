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
            "Per-source calibration diagnostics in external lung for scGPT vs Geneformer "
            "on shared edges."
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
        "--sweep-best-csv",
        type=Path,
        default=subproject_root
        / "implementation"
        / "outputs"
        / "cycle16_external_lung_source_targeted_sweep"
        / "source_best_scgpt_vs_geneformer.csv",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--calib-bins", type=int, default=10)
    parser.add_argument("--top-k-sources", type=int, default=12)
    parser.add_argument("--min-pairs", type=int, default=30)
    parser.add_argument("--min-positives", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle17_external_lung_source_calibration",
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


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int) -> tuple[float, pd.DataFrame]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, edges, right=True) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bin_idx == b
        if not np.any(m):
            rows.append(
                {
                    "bin": b,
                    "count": 0,
                    "mean_pred": float("nan"),
                    "empirical_rate": float("nan"),
                    "abs_gap": float("nan"),
                }
            )
            continue
        mean_pred = float(np.mean(p[m]))
        emp_rate = float(np.mean(y[m]))
        rows.append(
            {
                "bin": b,
                "count": int(np.sum(m)),
                "mean_pred": mean_pred,
                "empirical_rate": emp_rate,
                "abs_gap": abs(mean_pred - emp_rate),
            }
        )
    calib_df = pd.DataFrame(rows)
    total = max(int(calib_df["count"].sum()), 1)
    ece = float(np.nansum((calib_df["count"] / total) * calib_df["abs_gap"]))
    return ece, calib_df


def _fit_source_oof_probs(
    y: np.ndarray,
    x_base: np.ndarray,
    feat: np.ndarray,
    cv_splits: int,
    cv_repeats: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]] | tuple[None, dict[str, float]]:
    n_pos = int(y.sum())
    n_neg = int(y.shape[0] - n_pos)
    min_class = int(min(n_pos, n_neg))
    n_splits = int(min(cv_splits, min_class))
    if y.shape[0] < 8 or n_pos < 2 or n_neg < 2 or n_splits < 2:
        return None, {
            "n_pairs": int(y.shape[0]),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_splits_used": n_splits,
            "auroc": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
        }
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=cv_repeats, random_state=seed)
    x_plus = np.column_stack([x_base, feat])
    probs = _crossval_probs(x_plus, y, cv)
    return probs, {
        "n_pairs": int(y.shape[0]),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_splits_used": n_splits,
        "auroc": float(roc_auc_score(y, probs)),
        "brier": float(np.mean((probs - y) ** 2)),
    }


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_enrich = pd.read_csv(args.source_enrichment_csv)
    top_sources = (
        source_enrich[source_enrich["domain"] == "external_lung"]
        .sort_values("enrichment_ratio_top_vs_overall", ascending=False)["source"]
        .drop_duplicates()
        .head(args.top_k_sources)
        .tolist()
    )
    if not top_sources:
        raise RuntimeError("No external_lung sources found in enrichment file")

    sweep_best = pd.read_csv(args.sweep_best_csv)
    sweep_best = sweep_best[sweep_best["source"].isin(top_sources)].copy()

    # Keep sources with enough samples for calibration diagnostics.
    eligible = sweep_best[
        (sweep_best["geneformer_n_pairs"] >= args.min_pairs)
        & (sweep_best["geneformer_n_positive"] >= args.min_positives)
    ].copy()
    if eligible.empty:
        raise RuntimeError("No eligible sources under min-pairs/min-positives thresholds")

    print("[info] eligible sources:", ", ".join(eligible["source"].tolist()))

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

    # scGPT seed42 main embeddings.
    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent
    run_dir = subproject_root / "implementation" / "outputs" / "cycle7_external_lung_main"
    layer_embeddings = np.load(run_dir / "layer_gene_embeddings.npy", mmap_mode="r")
    layer_counts = np.load(run_dir / "layer_gene_embedding_count.npy", mmap_mode="r")

    # Geneformer embeddings.
    model = AutoModel.from_pretrained(args.model_id)
    emb_weight = model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32)
    gf_feature = _compute_geneformer_feature(emb_weight=emb_weight, src_tok=src_tok, tgt_tok=tgt_tok)
    gf_valid = base_valid & np.isfinite(gf_feature)

    summary_rows = []
    rel_rows = []
    for row in eligible.itertuples(index=False):
        source = str(row.source)
        layer = int(row.layer)
        source_mask = sources == source

        emb = np.asarray(layer_embeddings[layer], dtype=np.float32)
        counts = np.asarray(layer_counts[layer], dtype=np.int32)
        sc_feature = _compute_scgpt_feature(
            emb=emb,
            src_idx=src_idx,
            tgt_idx=tgt_idx,
            pca_dim=args.pca_dim,
            seed=args.seed,
        )
        sc_valid = base_valid & (counts[src_idx] > 0) & (counts[tgt_idx] > 0) & np.isfinite(sc_feature)
        valid = source_mask & sc_valid & gf_valid

        y = labels[valid]
        x_base = baseline[valid]
        sc = sc_feature[valid]
        gf = gf_feature[valid]

        sc_probs, sc_stats = _fit_source_oof_probs(
            y=y,
            x_base=x_base,
            feat=sc,
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
            seed=args.seed,
        )
        gf_probs, gf_stats = _fit_source_oof_probs(
            y=y,
            x_base=x_base,
            feat=gf,
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
            seed=args.seed,
        )

        sc_ece = float("nan")
        gf_ece = float("nan")
        if sc_probs is not None:
            sc_ece, sc_calib = _ece(y, sc_probs, args.calib_bins)
            sc_stats["ece"] = sc_ece
            sc_calib.insert(0, "model", "scgpt")
            sc_calib.insert(0, "source", source)
            rel_rows.append(sc_calib)
        if gf_probs is not None:
            gf_ece, gf_calib = _ece(y, gf_probs, args.calib_bins)
            gf_stats["ece"] = gf_ece
            gf_calib.insert(0, "model", "geneformer")
            gf_calib.insert(0, "source", source)
            rel_rows.append(gf_calib)

        summary_rows.append(
            {
                "source": source,
                "scgpt_layer_used": layer,
                "n_pairs": sc_stats["n_pairs"],
                "n_positive": sc_stats["n_positive"],
                "n_negative": sc_stats["n_negative"],
                "positive_rate": sc_stats["n_positive"] / max(sc_stats["n_pairs"], 1),
                "scgpt_auroc": sc_stats["auroc"],
                "geneformer_auroc": gf_stats["auroc"],
                "scgpt_brier": sc_stats["brier"],
                "geneformer_brier": gf_stats["brier"],
                "scgpt_ece": sc_stats.get("ece", float("nan")),
                "geneformer_ece": gf_stats.get("ece", float("nan")),
                "delta_auroc_geneformer_minus_scgpt": gf_stats["auroc"] - sc_stats["auroc"]
                if np.isfinite(gf_stats["auroc"]) and np.isfinite(sc_stats["auroc"])
                else float("nan"),
                "delta_brier_scgpt_minus_geneformer": sc_stats["brier"] - gf_stats["brier"]
                if np.isfinite(gf_stats["brier"]) and np.isfinite(sc_stats["brier"])
                else float("nan"),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("delta_auroc_geneformer_minus_scgpt", ascending=False)
    rel_df = pd.concat(rel_rows, axis=0, ignore_index=True) if rel_rows else pd.DataFrame()

    summary_df.to_csv(output_dir / "source_calibration_summary.csv", index=False)
    rel_df.to_csv(output_dir / "source_reliability_bins.csv", index=False)

    print(summary_df.to_string(index=False))
    print(f"[done] {output_dir / 'source_calibration_summary.csv'}")
    print(f"[done] {output_dir / 'source_reliability_bins.csv'}")


if __name__ == "__main__":
    main()
