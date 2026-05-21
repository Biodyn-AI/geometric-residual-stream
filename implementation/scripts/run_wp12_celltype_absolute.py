#!/usr/bin/env python3
"""WP12 - Absolute metrics for the cell-type stratified analysis (Table 3).

Reviewer 1 comment 6 / Reviewer 2 comment 1 ask for absolute AUROC/AUPRC in all
major tables. The cell-type stratification table previously reported only
delta-AUROC. This script re-evaluates each cell-type subset and reports baseline
and geometry-augmented absolute AUROC and AUPRC, using pca64_centered_cosine at
layer 0 (the representation used for that table).

Output: implementation/outputs/cycle52_celltype_absolute/celltype_absolute_summary.csv
"""
from __future__ import annotations

import anndata as ad
import pandas as pd

import revision_lib as rl

OUT = rl.OUTPUTS / "cycle52_celltype_absolute"

# cached run dir -> (display domain, cell type, h5ad path)
CELLTYPES = [
    ("cycle11_immune_celltype_cd8_t", "Immune", "CD8+ T cell",
     "implementation/data/immune_celltype_subsets/cd8_positive_alpha_beta_t_cell.h5ad"),
    ("cycle11_immune_celltype_cd4_t", "Immune", "CD4+ T cell",
     "implementation/data/immune_celltype_subsets/cd4_positive_alpha_beta_t_cell.h5ad"),
    ("cycle11_immune_celltype_b_cell", "Immune", "B cell",
     "implementation/data/immune_celltype_subsets/b_cell.h5ad"),
    ("cycle11_lung_celltype_macrophage", "Lung", "Macrophage",
     "implementation/data/lung_celltype_subsets/macrophage.h5ad"),
    ("cycle11_lung_celltype_alv_type2", "Lung", "Alveolar type II",
     "implementation/data/lung_celltype_subsets/pulmonary_alveolar_type_2_cell.h5ad"),
    ("cycle11_lung_celltype_alv_type1", "Lung", "Alveolar type I",
     "implementation/data/lung_celltype_subsets/pulmonary_alveolar_type_1_cell.h5ad"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    import numpy as np
    rows = []
    for run_dir, domain, celltype, h5ad in CELLTYPES:
        d = rl.OUTPUTS / run_dir
        edges = pd.read_csv(d / "cycle1_edge_dataset.tsv", sep="\t")
        emb = np.load(d / "layer_gene_embeddings.npy", mmap_mode="r")
        counts = np.load(d / "layer_gene_embedding_count.npy")
        adata = ad.read_h5ad(rl.SUBPROJECT_ROOT / h5ad)
        gstats = rl.compute_gene_statistics(adata.X)
        del adata
        dom = rl.domain_from_edges(celltype, 42, edges, emb, counts, gstats)
        res = rl.evaluate(dom, "pca_cc", [0], pca_dim=64, cv_mode="edge",
                          n_boot=1000, boot_seed=8)
        rows.append(dict(
            domain=domain, cell_type=celltype, n_edges=res["n_edges"],
            n_pos=res["n_pos"],
            auroc_base=res["auroc_base"], auroc_plus=res["auroc_plus"],
            delta_auroc=res["delta_auroc"],
            auprc_base=res["auprc_base"], auprc_plus=res["auprc_plus"],
            delta_auprc=res["delta_auprc"],
            boot_delta_lo=res["boot_delta_lo"], boot_delta_hi=res["boot_delta_hi"],
            ci_excludes_zero=res["ci_excludes_zero"]))
        print(f"[{domain:7s} {celltype:18s}] AUROC {res['auroc_base']:.3f}->"
              f"{res['auroc_plus']:.3f} dAUROC={res['delta_auroc']:+.4f} "
              f"CI=[{res['boot_delta_lo']:+.4f},{res['boot_delta_hi']:+.4f}] "
              f"AUPRC {res['auprc_base']:.3f}->{res['auprc_plus']:.3f}")
    pd.DataFrame(rows).to_csv(OUT / "celltype_absolute_summary.csv", index=False)
    print(f"\n[done] {OUT / 'celltype_absolute_summary.csv'}")


if __name__ == "__main__":
    main()
