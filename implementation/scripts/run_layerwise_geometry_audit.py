#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent
    biodyn_root = subproject_root.parent
    single_cell_root = biodyn_root / "single_cell_mechinterp"
    default_output_dir = subproject_root / "implementation" / "outputs" / "cycle1"

    parser = argparse.ArgumentParser(
        description=(
            "Cycle 1 geometric audit: extract layer-wise residual embeddings from scGPT and "
            "test TRRUST edge signal beyond gene-level baselines."
        )
    )
    parser.add_argument("--single-cell-root", type=Path, default=single_cell_root)
    parser.add_argument(
        "--processed-h5ad",
        type=Path,
        default=single_cell_root / "outputs" / "tabula_sapiens_processed.h5ad",
    )
    parser.add_argument(
        "--trrust-tsv",
        type=Path,
        default=single_cell_root / "external" / "networks" / "trrust_human.tsv",
    )
    parser.add_argument(
        "--scgpt-repo",
        type=Path,
        default=single_cell_root / "external" / "scGPT",
    )
    parser.add_argument(
        "--scgpt-checkpoint",
        type=Path,
        default=single_cell_root / "external" / "scGPT_checkpoints" / "whole-human" / "best_model.pt",
    )
    parser.add_argument(
        "--scgpt-vocab",
        type=Path,
        default=single_cell_root / "external" / "scGPT_checkpoints" / "whole-human" / "vocab.json",
    )
    parser.add_argument(
        "--scgpt-args",
        type=Path,
        default=single_cell_root / "external" / "scGPT_checkpoints" / "whole-human" / "args.json",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--max-cells", type=int, default=256)
    parser.add_argument("--max-genes", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--neg-ratio", type=float, default=3.0)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-iters", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_model_args(scgpt_args: dict, vocab_map: Dict[str, int], pad_token: str) -> dict:
    return {
        "ntoken": len(vocab_map),
        "d_model": scgpt_args["embsize"],
        "nhead": scgpt_args["nheads"],
        "d_hid": scgpt_args["d_hid"],
        "nlayers": scgpt_args["nlayers"],
        "nlayers_cls": scgpt_args.get("n_layers_cls", 3),
        "n_cls": 1,
        "vocab": vocab_map,
        "dropout": scgpt_args.get("dropout", 0.5),
        "pad_token": pad_token,
        "pad_value": scgpt_args.get("pad_value", 0),
        "do_mvc": bool(scgpt_args.get("MVC", False)),
        "do_dab": False,
        "use_batch_labels": False,
        "domain_spec_batchnorm": False,
        "input_emb_style": scgpt_args.get("input_emb_style", "continuous"),
        "n_input_bins": scgpt_args.get("n_bins"),
        "cell_emb_style": "avg-pool" if scgpt_args.get("no_cls") else "cls",
        "explicit_zero_prob": False,
        # Fast transformer/nested tensor paths complicate hook capture; keep this path stable.
        "use_fast_transformer": False,
        "fast_transformer_backend": "flash",
        "pre_norm": False,
    }


def _align_layer_output(tensor: torch.Tensor, batch_size: int, seq_len: int) -> np.ndarray:
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D layer output, got shape {tuple(tensor.shape)}")

    if tensor.shape[0] == batch_size and tensor.shape[1] == seq_len:
        aligned = tensor
    elif tensor.shape[0] == seq_len and tensor.shape[1] == batch_size:
        aligned = tensor.permute(1, 0, 2)
    else:
        raise ValueError(
            "Layer output shape does not align with batch/seq dimensions: "
            f"{tuple(tensor.shape)} vs batch={batch_size}, seq={seq_len}"
        )

    return aligned.detach().cpu().numpy().astype(np.float32, copy=False)


def _compute_gene_statistics(expr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # We explicitly compute mean, variance, and detection frequency for each gene.
    # These features form the confound baseline that geometric features must beat.
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

    # Rejection sampling avoids constructing the full source x target cartesian product.
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

    if len(negatives) < n_negatives:
        print(
            f"[warn] requested {n_negatives} negatives but sampled {len(negatives)} after {attempts} attempts"
        )

    return sorted(negatives)


def _crossval_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    cv: RepeatedStratifiedKFold,
) -> np.ndarray:
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
    deltas: List[float] = []
    n = labels.shape[0]

    for _ in range(n_iters):
        idx = rng.integers(0, n, size=n)
        y_bs = labels[idx]
        if np.unique(y_bs).size < 2:
            continue
        delta = roc_auc_score(y_bs, plus_probs[idx]) - roc_auc_score(y_bs, base_probs[idx])
        deltas.append(float(delta))

    if not deltas:
        return float("nan"), float("nan"), float("nan")

    delta_arr = np.array(deltas, dtype=np.float64)
    return (
        float(delta_arr.mean()),
        float(np.percentile(delta_arr, 2.5)),
        float(np.percentile(delta_arr, 97.5)),
    )


def main() -> None:
    args = _parse_args()
    _set_seeds(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    single_cell_root = args.single_cell_root.resolve()
    if str(single_cell_root) not in sys.path:
        sys.path.insert(0, str(single_cell_root))

    from src.data.scgpt_dataset import ScGPTDataset, ScGPTDatasetConfig, collate_scgpt
    from src.interpret.causal_intervention import capture_layer_outputs, find_transformer_layers
    from src.model.scgpt_loader import load_scgpt_model
    from src.model.vocab import load_vocab
    from src.model.wrapper import ScGPTWrapper

    print("[info] loading processed AnnData")
    adata = ad.read_h5ad(args.processed_h5ad)
    if args.max_cells and adata.n_obs > args.max_cells:
        rng_cells = np.random.default_rng(args.seed)
        keep_idx = np.sort(rng_cells.choice(adata.n_obs, size=args.max_cells, replace=False))
        adata = adata[keep_idx].copy()
    print(f"[info] adata shape after sampling: cells={adata.n_obs}, genes={adata.n_vars}")

    vocab = load_vocab(args.scgpt_vocab)
    if vocab.pad_id is None:
        raise ValueError("Pad token ID is missing in vocab")

    dataset_cfg = ScGPTDatasetConfig(
        max_genes=args.max_genes,
        include_zero=False,
        sort_by_expression=True,
        pad_token_id=vocab.pad_id,
        cls_token_id=None,
    )
    dataset = ScGPTDataset(adata, vocab.gene_to_id, dataset_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_scgpt,
    )

    scgpt_args = json.loads(args.scgpt_args.read_text(encoding="utf-8"))
    pad_token = scgpt_args.get("pad_token") or vocab.pad_token
    if pad_token not in vocab.gene_to_id:
        for token in ("<pad>", "[PAD]", "<PAD>", "PAD"):
            if token in vocab.gene_to_id:
                pad_token = token
                break
    if pad_token not in vocab.gene_to_id:
        raise ValueError("Could not resolve pad token in vocab")

    model_args = _build_model_args(scgpt_args, vocab.gene_to_id, pad_token)

    print("[info] loading scGPT model")
    model, missing, unexpected = load_scgpt_model(
        entrypoint="scgpt.model.TransformerModel",
        repo_path=args.scgpt_repo,
        checkpoint_path=args.scgpt_checkpoint,
        device=args.device,
        model_args=model_args,
        prefix_to_strip=None,
    )
    if missing or unexpected:
        print(f"[warn] model load: missing={len(missing)} unexpected={len(unexpected)}")

    # Explicitly disable nested tensor fastpath to ensure forward hooks return regular tensors.
    if hasattr(model, "transformer_encoder"):
        if hasattr(model.transformer_encoder, "enable_nested_tensor"):
            model.transformer_encoder.enable_nested_tensor = False
        if hasattr(model.transformer_encoder, "use_nested_tensor"):
            model.transformer_encoder.use_nested_tensor = False

    model.eval()
    model.to(args.device)

    wrapper = ScGPTWrapper(
        model,
        {"gene_ids": "src", "gene_values": "values", "src_key_padding_mask": "src_key_padding_mask"},
    )

    layers = find_transformer_layers(model)
    n_layers = len(layers)
    n_genes = adata.n_vars
    d_model = int(scgpt_args["embsize"])
    print(f"[info] found {n_layers} transformer layers, d_model={d_model}")

    layer_sum = np.zeros((n_layers, n_genes, d_model), dtype=np.float32)
    layer_count = np.zeros((n_layers, n_genes), dtype=np.int32)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="extract_residuals", unit="batch"):
            batch_cpu = {k: v for k, v in batch.items()}
            batch_dev = {k: v.to(args.device) for k, v in batch.items()}
            batch_size, seq_len = batch_dev["gene_ids"].shape

            layer_outputs: List[torch.Tensor | None] = [None for _ in layers]
            hooks = capture_layer_outputs(layers, layer_outputs)
            try:
                _ = wrapper.forward(batch_dev)
            finally:
                for h in hooks:
                    h.remove()

            gene_indices = batch_cpu["gene_indices"].numpy()

            for layer_idx, layer_tensor in enumerate(layer_outputs):
                if layer_tensor is None:
                    continue
                aligned = _align_layer_output(layer_tensor, batch_size=batch_size, seq_len=seq_len)

                for bi in range(batch_size):
                    idx = gene_indices[bi]
                    valid = idx >= 0
                    if not np.any(valid):
                        continue
                    valid_idx = idx[valid]
                    valid_vecs = aligned[bi, valid, :]
                    np.add.at(layer_sum[layer_idx], valid_idx, valid_vecs)
                    np.add.at(layer_count[layer_idx], valid_idx, 1)

    np.save(output_dir / "layer_gene_embedding_sum.npy", layer_sum)
    np.save(output_dir / "layer_gene_embedding_count.npy", layer_count)

    # Average token-level residual vectors into per-gene embeddings for each layer.
    layer_embeddings = np.zeros_like(layer_sum, dtype=np.float32)
    for li in range(n_layers):
        counts = layer_count[li].astype(np.float32)
        denom = np.where(counts > 0, counts, 1.0)[:, None]
        layer_embeddings[li] = layer_sum[li] / denom

    np.save(output_dir / "layer_gene_embeddings.npy", layer_embeddings)

    print("[info] building TRRUST edge dataset")
    trrust = pd.read_csv(args.trrust_tsv, sep="\t", header=None)
    if trrust.shape[1] < 2:
        raise ValueError("TRRUST file must contain at least two columns")
    trrust = trrust.iloc[:, :2].copy()
    trrust.columns = ["source", "target"]
    trrust["source"] = trrust["source"].astype(str)
    trrust["target"] = trrust["target"].astype(str)

    gene_to_idx = {gene: i for i, gene in enumerate(adata.var_names.astype(str).tolist())}

    positives = []
    for row in trrust.itertuples(index=False):
        s = row.source
        t = row.target
        if s == t:
            continue
        if s in gene_to_idx and t in gene_to_idx:
            positives.append((s, t))
    positives = sorted(set(positives))
    if not positives:
        raise ValueError("No TRRUST edges mapped into the current gene universe")

    source_pool = sorted({s for s, _ in positives})
    target_pool = sorted({t for _, t in positives})

    n_neg = int(len(positives) * args.neg_ratio)
    neg_rng = np.random.default_rng(args.seed + 11)
    negatives = _sample_negative_edges(
        rng=neg_rng,
        sources=source_pool,
        targets=target_pool,
        positive_edges=set(positives),
        n_negatives=n_neg,
    )

    edge_rows = []
    for s, t in positives:
        edge_rows.append((s, t, 1, gene_to_idx[s], gene_to_idx[t]))
    for s, t in negatives:
        edge_rows.append((s, t, 0, gene_to_idx[s], gene_to_idx[t]))

    edge_df = pd.DataFrame(
        edge_rows,
        columns=["source", "target", "label", "source_idx", "target_idx"],
    )
    edge_df.to_csv(output_dir / "cycle1_edge_dataset.tsv", sep="\t", index=False)

    print("[info] computing baseline gene-level confound features")
    mean_expr, variance, detection = _compute_gene_statistics(adata.X)
    src_idx = edge_df["source_idx"].to_numpy(dtype=np.int64)
    tgt_idx = edge_df["target_idx"].to_numpy(dtype=np.int64)
    labels = edge_df["label"].to_numpy(dtype=np.int32)

    baseline_features = np.column_stack(
        [
            mean_expr[src_idx],
            mean_expr[tgt_idx],
            variance[src_idx],
            variance[tgt_idx],
            detection[src_idx],
            detection[tgt_idx],
        ]
    ).astype(np.float64)

    metrics_rows = []
    bootstrap_rows = []

    for li in range(n_layers):
        emb = layer_embeddings[li]
        counts = layer_count[li]

        src_emb = emb[src_idx]
        tgt_emb = emb[tgt_idx]
        src_norm = np.linalg.norm(src_emb, axis=1)
        tgt_norm = np.linalg.norm(tgt_emb, axis=1)
        denom = src_norm * tgt_norm

        geom_score = np.sum(src_emb * tgt_emb, axis=1) / np.clip(denom, 1e-8, None)

        valid_pairs = (
            (counts[src_idx] > 0)
            & (counts[tgt_idx] > 0)
            & np.isfinite(geom_score)
            & np.all(np.isfinite(baseline_features), axis=1)
        )

        y = labels[valid_pairs]
        if np.unique(y).size < 2:
            continue

        base_x = baseline_features[valid_pairs]
        plus_x = np.column_stack([base_x, geom_score[valid_pairs]])

        cv = RepeatedStratifiedKFold(
            n_splits=args.cv_splits,
            n_repeats=args.cv_repeats,
            random_state=args.seed,
        )

        base_probs = _crossval_predictions(base_x, y, cv)
        plus_probs = _crossval_predictions(plus_x, y, cv)

        auc_base = _safe_auc(y, base_probs)
        auc_plus = _safe_auc(y, plus_probs)
        aupr_base = _safe_aupr(y, base_probs)
        aupr_plus = _safe_aupr(y, plus_probs)
        auc_raw_geom = _safe_auc(y, geom_score[valid_pairs])
        aupr_raw_geom = _safe_aupr(y, geom_score[valid_pairs])

        delta_auc_mean, delta_auc_lo, delta_auc_hi = _bootstrap_delta_auc(
            rng=np.random.default_rng(args.seed + li + 1000),
            labels=y,
            base_probs=base_probs,
            plus_probs=plus_probs,
            n_iters=args.bootstrap_iters,
        )

        metrics_rows.append(
            {
                "layer": li,
                "n_pairs": int(valid_pairs.sum()),
                "n_positive": int(y.sum()),
                "raw_geom_auroc": auc_raw_geom,
                "raw_geom_aupr": aupr_raw_geom,
                "baseline_cv_auroc": auc_base,
                "baseline_cv_aupr": aupr_base,
                "baseline_plus_geom_cv_auroc": auc_plus,
                "baseline_plus_geom_cv_aupr": aupr_plus,
                "delta_cv_auroc": auc_plus - auc_base,
                "delta_cv_aupr": aupr_plus - aupr_base,
                "delta_auc_bootstrap_mean": delta_auc_mean,
                "delta_auc_bootstrap_ci_lo": delta_auc_lo,
                "delta_auc_bootstrap_ci_hi": delta_auc_hi,
            }
        )

        bootstrap_rows.append(
            {
                "layer": li,
                "delta_auc_bootstrap_mean": delta_auc_mean,
                "delta_auc_bootstrap_ci_lo": delta_auc_lo,
                "delta_auc_bootstrap_ci_hi": delta_auc_hi,
            }
        )

    if not metrics_rows:
        raise RuntimeError("No valid layer metrics were computed")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("layer").reset_index(drop=True)
    metrics_df.to_csv(output_dir / "cycle1_layer_geometry_metrics.csv", index=False)

    bootstrap_df = pd.DataFrame(bootstrap_rows).sort_values("layer").reset_index(drop=True)
    bootstrap_df.to_csv(output_dir / "cycle1_layer_delta_bootstrap.csv", index=False)

    best_idx = metrics_df["delta_cv_auroc"].idxmax()
    best_row = metrics_df.loc[best_idx]

    summary_md = output_dir / "cycle1_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Cycle 1 Geometry Audit Summary",
                "",
                f"- Cells used: {adata.n_obs}",
                f"- Genes in processed matrix: {adata.n_vars}",
                f"- Layers evaluated: {n_layers}",
                f"- Positive edges: {len(positives)}",
                f"- Negative edges: {len(negatives)}",
                "",
                "## Best Layer (by delta CV AUROC)",
                f"- Layer: {int(best_row['layer'])}",
                f"- Delta CV AUROC: {best_row['delta_cv_auroc']:.6f}",
                f"- Baseline CV AUROC: {best_row['baseline_cv_auroc']:.6f}",
                f"- Baseline+Geom CV AUROC: {best_row['baseline_plus_geom_cv_auroc']:.6f}",
                f"- Bootstrap delta AUC mean: {best_row['delta_auc_bootstrap_mean']:.6f}",
                f"- Bootstrap 95% CI: [{best_row['delta_auc_bootstrap_ci_lo']:.6f}, {best_row['delta_auc_bootstrap_ci_hi']:.6f}]",
            ]
        ),
        encoding="utf-8",
    )

    print("[done] wrote outputs:")
    print(f"  - {output_dir / 'cycle1_layer_geometry_metrics.csv'}")
    print(f"  - {output_dir / 'cycle1_layer_delta_bootstrap.csv'}")
    print(f"  - {output_dir / 'cycle1_edge_dataset.tsv'}")
    print(f"  - {output_dir / 'cycle1_summary.md'}")


if __name__ == "__main__":
    main()
