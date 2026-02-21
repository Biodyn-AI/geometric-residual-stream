#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
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
            "Geneformer bootstrap audit: use mapped Geneformer token embeddings to test "
            "TRRUST edge signal beyond gene-level confounds."
        )
    )
    parser.add_argument(
        "--processed-h5ad",
        type=Path,
        default=single_cell_root / "outputs" / "tabula_sapiens_immune_subset_hpn_processed.h5ad",
    )
    parser.add_argument(
        "--trrust-tsv",
        type=Path,
        default=single_cell_root / "external" / "networks" / "trrust_human.tsv",
    )
    parser.add_argument("--model-id", type=str, default="ctheodoris/Geneformer")
    parser.add_argument(
        "--gene-name-id-pkl",
        type=str,
        default="geneformer/gene_name_id_dict_gc104M.pkl",
    )
    parser.add_argument(
        "--token-dict-pkl",
        type=str,
        default="geneformer/token_dictionary_gc104M.pkl",
    )
    parser.add_argument("--neg-ratio", type=float, default=3.0)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle12_geneformer_immune_bootstrap",
    )
    return parser.parse_args()


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _compute_gene_statistics(expr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _sample_negative_edges(
    rng: np.random.Generator,
    sources: List[str],
    targets: List[str],
    positive_edges: set[Tuple[str, str]],
    n_negatives: int,
) -> List[Tuple[str, str]]:
    negatives: set[Tuple[str, str]] = set()
    source_arr = np.array(sources, dtype=object)
    target_arr = np.array(targets, dtype=object)

    max_attempts = max(n_negatives * 50, 10000)
    attempts = 0
    while len(negatives) < n_negatives and attempts < max_attempts:
        s = source_arr[rng.integers(0, len(source_arr))]
        t = target_arr[rng.integers(0, len(target_arr))]
        attempts += 1
        pair = (str(s), str(t))
        if pair[0] == pair[1]:
            continue
        if pair in positive_edges or pair in negatives:
            continue
        negatives.add(pair)

    return sorted(negatives)


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


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _bootstrap_delta_auc(
    rng: np.random.Generator,
    labels: np.ndarray,
    base_probs: np.ndarray,
    plus_probs: np.ndarray,
    n_iters: int,
) -> Tuple[float, float, float]:
    deltas = []
    n = labels.shape[0]
    for _ in range(n_iters):
        idx = rng.integers(0, n, size=n)
        y_bs = labels[idx]
        if np.unique(y_bs).size < 2:
            continue
        deltas.append(float(roc_auc_score(y_bs, plus_probs[idx]) - roc_auc_score(y_bs, base_probs[idx])))
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(deltas, dtype=np.float64)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main() -> None:
    args = _parse_args()
    _set_seeds(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[info] loading data")
    adata = ad.read_h5ad(args.processed_h5ad)
    gene_names = adata.var_names.astype(str).tolist()
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    print("[info] loading Geneformer resources")
    model = AutoModel.from_pretrained(args.model_id)
    emb_weight = model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32)

    gene_name_id_path = hf_hub_download(repo_id=args.model_id, filename=args.gene_name_id_pkl)
    token_dict_path = hf_hub_download(repo_id=args.model_id, filename=args.token_dict_pkl)
    with open(gene_name_id_path, "rb") as handle:
        gene_name_id: Dict[str, str] = pickle.load(handle)
    with open(token_dict_path, "rb") as handle:
        token_dict: Dict[str, int] = pickle.load(handle)

    # Map gene symbols in our matrix to Geneformer token IDs via Ensembl IDs.
    mapped_idx: Dict[str, int] = {}
    for gene in gene_names:
        ens_id = gene_name_id.get(gene)
        if ens_id is None:
            continue
        tok_id = token_dict.get(ens_id)
        if tok_id is None:
            continue
        if tok_id < 0 or tok_id >= emb_weight.shape[0]:
            continue
        mapped_idx[gene] = int(tok_id)

    map_df = pd.DataFrame(
        [{"gene": g, "token_id": tid} for g, tid in mapped_idx.items()]
    ).sort_values("gene")
    map_df.to_csv(output_dir / "geneformer_gene_token_map.csv", index=False)

    map_rate = len(mapped_idx) / max(len(gene_names), 1)
    print(f"[info] mapped genes: {len(mapped_idx)} / {len(gene_names)} ({map_rate:.2%})")

    print("[info] building TRRUST edge set")
    trrust = pd.read_csv(args.trrust_tsv, sep="\t", header=None)
    trrust = trrust.iloc[:, :2].copy()
    trrust.columns = ["source", "target"]
    trrust["source"] = trrust["source"].astype(str)
    trrust["target"] = trrust["target"].astype(str)

    positives = []
    for row in trrust.itertuples(index=False):
        s, t = row.source, row.target
        if s == t:
            continue
        if s in mapped_idx and t in mapped_idx and s in gene_to_idx and t in gene_to_idx:
            positives.append((s, t))
    positives = sorted(set(positives))
    if not positives:
        raise RuntimeError("No mapped TRRUST positives found for Geneformer bootstrap")

    source_pool = sorted({s for s, _ in positives})
    target_pool = sorted({t for _, t in positives})
    neg_rng = np.random.default_rng(args.seed + 11)
    negatives = _sample_negative_edges(
        rng=neg_rng,
        sources=source_pool,
        targets=target_pool,
        positive_edges=set(positives),
        n_negatives=int(len(positives) * args.neg_ratio),
    )

    edge_rows = []
    for s, t in positives:
        edge_rows.append((s, t, 1, gene_to_idx[s], gene_to_idx[t], mapped_idx[s], mapped_idx[t]))
    for s, t in negatives:
        edge_rows.append((s, t, 0, gene_to_idx[s], gene_to_idx[t], mapped_idx[s], mapped_idx[t]))
    edge_df = pd.DataFrame(
        edge_rows,
        columns=["source", "target", "label", "source_idx", "target_idx", "source_token_id", "target_token_id"],
    )
    edge_df.to_csv(output_dir / "geneformer_edge_dataset.tsv", sep="\t", index=False)

    print("[info] computing baseline and geometry features")
    mean_expr, variance, detection = _compute_gene_statistics(adata.X)
    src_idx = edge_df["source_idx"].to_numpy(dtype=np.int64)
    tgt_idx = edge_df["target_idx"].to_numpy(dtype=np.int64)
    src_tok = edge_df["source_token_id"].to_numpy(dtype=np.int64)
    tgt_tok = edge_df["target_token_id"].to_numpy(dtype=np.int64)
    labels = edge_df["label"].to_numpy(dtype=np.int32)

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

    src_emb = emb_weight[src_tok]
    tgt_emb = emb_weight[tgt_tok]
    cosine = np.sum(src_emb * tgt_emb, axis=1) / np.clip(
        np.linalg.norm(src_emb, axis=1) * np.linalg.norm(tgt_emb, axis=1), 1e-8, None
    )
    dot = np.sum(src_emb * tgt_emb, axis=1)
    centered = emb_weight - emb_weight.mean(axis=0, keepdims=True)
    src_center = centered[src_tok]
    tgt_center = centered[tgt_tok]
    centered_cosine = np.sum(src_center * tgt_center, axis=1) / np.clip(
        np.linalg.norm(src_center, axis=1) * np.linalg.norm(tgt_center, axis=1), 1e-8, None
    )

    feature_map = {
        "cosine": cosine,
        "centered_cosine": centered_cosine,
        "dot": dot,
    }

    cv = RepeatedStratifiedKFold(
        n_splits=args.cv_splits,
        n_repeats=args.cv_repeats,
        random_state=args.seed,
    )

    rows = []
    for feat_name, feat_values in feature_map.items():
        valid = np.isfinite(feat_values) & np.all(np.isfinite(baseline), axis=1)
        y = labels[valid]
        x_base = baseline[valid]
        x_plus = np.column_stack([x_base, feat_values[valid]])

        base_probs = _crossval_probs(x_base, y, cv)
        plus_probs = _crossval_probs(x_plus, y, cv)

        auc_base = _safe_auc(y, base_probs)
        auc_plus = _safe_auc(y, plus_probs)
        aupr_base = _safe_aupr(y, base_probs)
        aupr_plus = _safe_aupr(y, plus_probs)
        raw_auc = _safe_auc(y, feat_values[valid])
        raw_aupr = _safe_aupr(y, feat_values[valid])
        bs_mean, bs_lo, bs_hi = _bootstrap_delta_auc(
            rng=np.random.default_rng(args.seed + len(rows) + 1000),
            labels=y,
            base_probs=base_probs,
            plus_probs=plus_probs,
            n_iters=args.bootstrap_iters,
        )

        rows.append(
            {
                "feature": feat_name,
                "n_pairs": int(valid.sum()),
                "n_positive": int(y.sum()),
                "raw_feature_auroc": raw_auc,
                "raw_feature_aupr": raw_aupr,
                "baseline_cv_auroc": auc_base,
                "baseline_cv_aupr": aupr_base,
                "baseline_plus_feature_cv_auroc": auc_plus,
                "baseline_plus_feature_cv_aupr": aupr_plus,
                "delta_cv_auroc": auc_plus - auc_base,
                "delta_cv_aupr": aupr_plus - aupr_base,
                "delta_auc_bootstrap_mean": bs_mean,
                "delta_auc_bootstrap_ci_lo": bs_lo,
                "delta_auc_bootstrap_ci_hi": bs_hi,
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("delta_cv_auroc", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(output_dir / "geneformer_feature_metrics.csv", index=False)

    best = metrics_df.iloc[0]
    (output_dir / "geneformer_summary.md").write_text(
        "\n".join(
            [
                "# Geneformer Embedding Geometry Bootstrap",
                "",
                f"- Processed data: {args.processed_h5ad}",
                f"- Model: {args.model_id}",
                f"- Genes mapped to Geneformer tokens: {len(mapped_idx)} / {len(gene_names)} ({map_rate:.2%})",
                f"- Positive edges: {len(positives)}",
                f"- Negative edges: {len(negatives)}",
                "",
                "## Best Feature",
                f"- Feature: {best['feature']}",
                f"- Delta CV AUROC: {best['delta_cv_auroc']:.6f}",
                f"- Baseline CV AUROC: {best['baseline_cv_auroc']:.6f}",
                f"- Baseline+Feature CV AUROC: {best['baseline_plus_feature_cv_auroc']:.6f}",
                f"- Bootstrap 95% CI: [{best['delta_auc_bootstrap_ci_lo']:.6f}, {best['delta_auc_bootstrap_ci_hi']:.6f}]",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[done] {output_dir / 'geneformer_feature_metrics.csv'}")
    print(f"[done] {output_dir / 'geneformer_summary.md'}")


if __name__ == "__main__":
    main()
