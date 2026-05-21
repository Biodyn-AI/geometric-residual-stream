#!/usr/bin/env python3
"""WP6 - Comparable cross-model extraction: Geneformer residual stream.

Addresses Reviewer 1 comment 5. The manuscript represents scGPT by multi-layer
residual-stream features (with bundling) but Geneformer only by its static input
embedding layer -- an asymmetric comparison. This script removes the asymmetry by
extracting Geneformer's PER-LAYER residual-stream activations from real forward
passes over single cells, mirroring the scGPT extraction, so the identical
geometric feature pipeline (centered cosine, PCA, multi-layer bundle) can be
applied to both models.

For each domain it writes, into implementation/outputs/cycle51_geneformer_residual/<domain>/:
  * layer_gene_embeddings.npy      (n_layers, n_genes, hidden) per-gene residuals
  * layer_gene_embedding_count.npy (n_layers, n_genes)
  * cycle1_edge_dataset.tsv        (copied from the matching scGPT run; indices align)

Geneformer model: ctheodoris/Geneformer (V2, 104M, 18-layer BERT). Cells are
tokenised with the official rank-value encoding (sum-normalise to 1e4, divide by
corpus gene medians, rank, keep top MAX_GENES, wrap in <cls> ... <eos>).
"""
from __future__ import annotations

import pickle
import shutil
import sys

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel

import revision_lib as rl

OUT = rl.OUTPUTS / "cycle51_geneformer_residual"
MODEL_ID = "ctheodoris/Geneformer"
MAX_CELLS = 192
MAX_GENES = 2048
BATCH = 8
SCGPT_RUN = {"kidney": "cycle1_main", "immune": "cycle4_immune_main",
             "lung": "cycle6_lung_main", "external_lung": "cycle7_external_lung_main"}


def counts_matrix(adata):
    """Best available count-like matrix for Geneformer normalisation."""
    if "decontXcounts" in adata.layers:
        return adata.layers["decontXcounts"]
    return adata.X


def main(domains) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[info] loading Geneformer", MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, output_hidden_states=True)
    model.eval()
    hidden = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    name2id = pickle.load(open(hf_hub_download(MODEL_ID, "geneformer/gene_name_id_dict_gc104M.pkl"), "rb"))
    token_dict = pickle.load(open(hf_hub_download(MODEL_ID, "geneformer/token_dictionary_gc104M.pkl"), "rb"))
    gene_median = pickle.load(open(hf_hub_download(MODEL_ID, "geneformer/gene_median_dictionary_gc104M.pkl"), "rb"))
    cls_id, eos_id, pad_id = token_dict["<cls>"], token_dict["<eos>"], token_dict["<pad>"]

    for domain in domains:
        print(f"\n=== {domain} ===")
        adata = ad.read_h5ad(rl.DOMAIN_H5AD[domain])
        n_obs = adata.n_obs
        rng = np.random.default_rng(42)
        keep = np.sort(rng.choice(n_obs, size=min(MAX_CELLS, n_obs), replace=False))
        adata = adata[keep].copy()
        var = adata.var_names.astype(str).tolist()
        n_genes = len(var)

        # gene column -> (token_id, median); only genes mapped into Geneformer.
        col_tok = np.full(n_genes, -1, dtype=np.int64)
        col_med = np.ones(n_genes, dtype=np.float64)
        for c, g in enumerate(var):
            ens = name2id.get(g)
            if ens is None:
                continue
            tok = token_dict.get(ens)
            med = gene_median.get(ens)
            if tok is None or med is None:
                continue
            col_tok[c] = tok
            col_med[c] = med
        tok2col = {int(t): c for c, t in enumerate(col_tok) if t >= 0}
        print(f"[info] {len(tok2col)}/{n_genes} genes map to Geneformer tokens")

        counts = counts_matrix(adata)
        counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)

        layer_sum = np.zeros((n_layers, n_genes, hidden), dtype=np.float32)
        layer_cnt = np.zeros((n_layers, n_genes), dtype=np.int32)

        # tokenise every cell
        seqs = []
        for ci in range(adata.n_obs):
            row = counts[ci]
            idx = row.indices
            val = row.data.astype(np.float64)
            mapped = col_tok[idx] >= 0
            idx, val = idx[mapped], val[mapped]
            if len(idx) == 0:
                seqs.append((np.array([], int), np.array([], int)))
                continue
            total = val.sum()
            norm = (val / total * 1e4) / col_med[idx]
            order = np.argsort(-norm)[:MAX_GENES]
            cols = idx[order]
            toks = col_tok[cols]
            seqs.append((toks.astype(np.int64), cols.astype(np.int64)))

        # batched forward passes
        with torch.no_grad():
            for b0 in range(0, len(seqs), BATCH):
                batch = seqs[b0:b0 + BATCH]
                maxlen = max((len(t) for t, _ in batch), default=0) + 2
                ids = np.full((len(batch), maxlen), pad_id, dtype=np.int64)
                attn = np.zeros((len(batch), maxlen), dtype=np.int64)
                for bi, (toks, _) in enumerate(batch):
                    seq = np.concatenate([[cls_id], toks, [eos_id]])
                    ids[bi, :len(seq)] = seq
                    attn[bi, :len(seq)] = 1
                out = model(input_ids=torch.tensor(ids),
                            attention_mask=torch.tensor(attn))
                hs = out.hidden_states  # tuple len n_layers+1
                for bi, (toks, cols) in enumerate(batch):
                    if len(toks) == 0:
                        continue
                    for li in range(n_layers):
                        vecs = hs[li + 1][bi, 1:1 + len(toks), :].numpy()
                        np.add.at(layer_sum[li], cols, vecs.astype(np.float32))
                        np.add.at(layer_cnt[li], cols, 1)
                print(f"  [{domain}] cells {b0 + len(batch)}/{len(seqs)}", flush=True)

        emb = np.zeros_like(layer_sum)
        for li in range(n_layers):
            denom = np.where(layer_cnt[li] > 0, layer_cnt[li], 1).astype(np.float32)[:, None]
            emb[li] = layer_sum[li] / denom

        dom_out = OUT / domain
        dom_out.mkdir(parents=True, exist_ok=True)
        np.save(dom_out / "layer_gene_embeddings.npy", emb)
        np.save(dom_out / "layer_gene_embedding_count.npy", layer_cnt)
        shutil.copy(rl.OUTPUTS / SCGPT_RUN[domain] / "cycle1_edge_dataset.tsv",
                    dom_out / "cycle1_edge_dataset.tsv")
        print(f"[done] {dom_out}  emb {emb.shape}")


if __name__ == "__main__":
    doms = sys.argv[1:] or rl.DOMAINS
    main(doms)
