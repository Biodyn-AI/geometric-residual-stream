#!/usr/bin/env python3
"""WP6 (evaluation) - Cross-model comparison with matched extraction pipelines.

Consumes the Geneformer per-layer residual-stream embeddings produced by
run_wp6_geneformer_residual.py and evaluates scGPT and Geneformer under the
IDENTICAL geometric feature pipeline (centered cosine; single best layer vs
multi-layer bundle), removing the asymmetry flagged in Reviewer 1 comment 5.

Output: implementation/outputs/cycle51_geneformer_residual/cross_model_matched_summary.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import revision_lib as rl

GF_DIR = rl.OUTPUTS / "cycle51_geneformer_residual"


def load_geneformer_domain(domain: str) -> rl.Domain:
    d = GF_DIR / domain
    edges = pd.read_csv(d / "cycle1_edge_dataset.tsv", sep="\t")
    emb = np.load(d / "layer_gene_embeddings.npy", mmap_mode="r")
    counts = np.load(d / "layer_gene_embedding_count.npy")
    return rl.domain_from_edges(domain, 42, edges, emb, counts, rl.gene_stats(domain))


def best_single(dom: rl.Domain, n_boot=600):
    best = None
    for li in range(dom.n_layers):
        r = rl.evaluate(dom, "centered_cosine", [li], cv_mode="edge", n_boot=n_boot)
        if best is None or r["delta_auroc"] > best["delta_auroc"]:
            best = r
    return best


def main() -> None:
    rows = []
    for domain in rl.DOMAINS:
        if not (GF_DIR / domain / "layer_gene_embeddings.npy").exists():
            print(f"[skip] no Geneformer residuals for {domain}")
            continue
        scg = rl.load_domain(domain, 42)
        gf = load_geneformer_domain(domain)

        for model_name, dom in [("scGPT", scg), ("Geneformer", gf)]:
            single = best_single(dom)
            bundle = rl.evaluate(dom, "centered_cosine", list(range(dom.n_layers)),
                                 cv_mode="edge", n_boot=1000)
            l0 = rl.evaluate(dom, "centered_cosine", [0], cv_mode="edge", n_boot=600)
            for rep, r in [("centered_cosine_bestL", single),
                           ("centered_cosine_bundle", bundle),
                           ("centered_cosine_L0only", l0)]:
                rows.append(dict(
                    domain=domain, model=model_name, representation=rep,
                    n_layers=dom.n_layers, best_layers=r["layers"],
                    auroc_base=r["auroc_base"], auroc_plus=r["auroc_plus"],
                    delta_auroc=r["delta_auroc"], delta_auprc=r["delta_auprc"],
                    boot_delta_lo=r["boot_delta_lo"], boot_delta_hi=r["boot_delta_hi"],
                    ci_excludes_zero=r["ci_excludes_zero"]))
            print(f"[{domain:13s} {model_name:10s}] "
                  f"bestL dAUROC={single['delta_auroc']:+.4f}  "
                  f"bundle dAUROC={bundle['delta_auroc']:+.4f}  "
                  f"L0 dAUROC={l0['delta_auroc']:+.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(GF_DIR / "cross_model_matched_summary.csv", index=False)
    # gap table: Geneformer - scGPT under the matched bundle
    print("\nMatched-pipeline gap (Geneformer - scGPT), centered_cosine_bundle:")
    for domain in df.domain.unique():
        sub = df[(df.domain == domain) & (df.representation == "centered_cosine_bundle")]
        if len(sub) == 2:
            g = sub[sub.model == "Geneformer"].delta_auroc.iloc[0]
            s = sub[sub.model == "scGPT"].delta_auroc.iloc[0]
            print(f"  {domain:13s} scGPT={s:+.4f}  Geneformer={g:+.4f}  gap={g - s:+.4f}")
    print(f"\n[done] {GF_DIR / 'cross_model_matched_summary.csv'}")


if __name__ == "__main__":
    main()
