#!/usr/bin/env python3
"""WP2 - Negative-edge sampling robustness.

Addresses Reviewer 1 comment 4 (negative protocol, resampling stability, harder
negatives) and supports Reviewer 2 comment 4. For each domain we regenerate the
negative edge set under three protocols and report stability of the geometric
signal across repeated draws:
  * random          - uniform (TF-pool x target-pool), the manuscript's protocol
  * degree_matched  - negative TF/target matched to the positive's TRRUST degree bin
  * expr_matched    - negative TF/target matched to mean-expression + detection bins

All schemes use a balanced 1:1 negative:positive ratio for a clean comparison.

Output: implementation/outputs/cycle44_negative_robustness/negative_robustness_summary.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import revision_lib as rl

OUT = rl.OUTPUTS / "cycle44_negative_robustness"
N_RESAMPLES = 8
N_BINS = 5
METRIC, PCA_DIM = "pca_cc", 64   # refined single-layer representation


def quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    return np.clip(np.digitize(values, qs[1:-1]), 0, n_bins - 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for domain in rl.DOMAINS:
        base_dom = rl.load_domain(domain, 42)
        edges = base_dom.edges
        pos = edges[edges.label == 1]
        # gene name -> embedding index (covers all pool genes since they occur in positives)
        name2idx = {}
        for col, icol in [("source", "source_idx"), ("target", "target_idx")]:
            for n, i in zip(edges[col], edges[icol]):
                name2idx[n] = int(i)
        src_pool = sorted(pos["source"].unique())
        tgt_pool = sorted(pos["target"].unique())
        pos_set = set(zip(pos["source"], pos["target"]))
        n_pos = len(pos)

        mean_expr, variance, detection = rl.gene_stats(domain)
        gstats = (mean_expr, variance, detection)

        # TRRUST degree within domain.
        outdeg = pos["source"].value_counts().to_dict()
        indeg = pos["target"].value_counts().to_dict()
        src_outdeg = np.array([outdeg.get(g, 0) for g in src_pool], float)
        tgt_indeg = np.array([indeg.get(g, 0) for g in tgt_pool], float)
        src_deg_bin = quantile_bin(src_outdeg, N_BINS)
        tgt_deg_bin = quantile_bin(tgt_indeg, N_BINS)
        # expression bins over pools
        src_expr_bin = quantile_bin(np.array([mean_expr[name2idx[g]] for g in src_pool]), N_BINS)
        tgt_expr_bin = quantile_bin(np.array([mean_expr[name2idx[g]] for g in tgt_pool]), N_BINS)

        src_arr = np.array(src_pool, dtype=object)
        tgt_arr = np.array(tgt_pool, dtype=object)
        src_g2b_deg = dict(zip(src_pool, src_deg_bin))
        tgt_g2b_deg = dict(zip(tgt_pool, tgt_deg_bin))
        src_g2b_exp = dict(zip(src_pool, src_expr_bin))
        tgt_g2b_exp = dict(zip(tgt_pool, tgt_expr_bin))
        src_bin_members = {b: src_arr[src_deg_bin == b] for b in range(N_BINS)}
        tgt_bin_members = {b: tgt_arr[tgt_deg_bin == b] for b in range(N_BINS)}
        src_bin_members_e = {b: src_arr[src_expr_bin == b] for b in range(N_BINS)}
        tgt_bin_members_e = {b: tgt_arr[tgt_expr_bin == b] for b in range(N_BINS)}

        def sample_negatives(scheme: str, rng) -> list:
            neg = set()
            tries = 0
            pos_list = list(zip(pos["source"], pos["target"]))
            while len(neg) < n_pos and tries < n_pos * 200:
                tries += 1
                if scheme == "random":
                    s = src_arr[rng.integers(len(src_arr))]
                    t = tgt_arr[rng.integers(len(tgt_arr))]
                else:
                    ps, pt = pos_list[len(neg) % n_pos]
                    if scheme == "degree_matched":
                        sm = src_bin_members[src_g2b_deg[ps]]
                        tm = tgt_bin_members[tgt_g2b_deg[pt]]
                    else:  # expr_matched
                        sm = src_bin_members_e[src_g2b_exp[ps]]
                        tm = tgt_bin_members_e[tgt_g2b_exp[pt]]
                    if len(sm) == 0 or len(tm) == 0:
                        continue
                    s = sm[rng.integers(len(sm))]
                    t = tm[rng.integers(len(tm))]
                pair = (str(s), str(t))
                if pair[0] == pair[1] or pair in pos_set or pair in neg:
                    continue
                neg.add(pair)
            return sorted(neg)

        for scheme in ["random", "degree_matched", "expr_matched"]:
            per = []
            for r in range(N_RESAMPLES):
                rng = np.random.default_rng(1000 + r)
                neg = sample_negatives(scheme, rng)
                rows_e = [(s, t, 1, name2idx[s], name2idx[t]) for s, t in
                          zip(pos["source"], pos["target"])]
                rows_e += [(s, t, 0, name2idx[s], name2idx[t]) for s, t in neg]
                edf = pd.DataFrame(rows_e, columns=["source", "target", "label",
                                                    "source_idx", "target_idx"])
                dom = rl.domain_from_edges(domain, 42, edf, base_dom.embeddings,
                                           base_dom.counts, gstats)
                res = rl.evaluate(dom, METRIC, [rl.PAPER_BEST_LAYER[domain]],
                                  pca_dim=PCA_DIM, cv_mode="edge",
                                  n_boot=200, boot_seed=r)
                per.append(res)
            df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                               for r in per])
            agg = dict(
                domain=domain, scheme=scheme, n_resamples=N_RESAMPLES,
                n_pos=int(n_pos),
                auroc_base_mean=df.auroc_base.mean(), auroc_base_sd=df.auroc_base.std(),
                auroc_plus_mean=df.auroc_plus.mean(), auroc_plus_sd=df.auroc_plus.std(),
                delta_auroc_mean=df.delta_auroc.mean(), delta_auroc_sd=df.delta_auroc.std(),
                auprc_base_mean=df.auprc_base.mean(),
                auprc_plus_mean=df.auprc_plus.mean(),
                delta_auprc_mean=df.delta_auprc.mean(), delta_auprc_sd=df.delta_auprc.std(),
                frac_ci_excludes_zero=float(df.ci_excludes_zero.mean()),
            )
            rows.append(agg)
            print(f"[{domain:13s} {scheme:15s}] dAUROC={agg['delta_auroc_mean']:+.4f}"
                  f"+-{agg['delta_auroc_sd']:.4f}  base AUROC={agg['auroc_base_mean']:.3f}"
                  f"  plus={agg['auroc_plus_mean']:.3f}  dAUPRC={agg['delta_auprc_mean']:+.4f}"
                  f"  CI>0 frac={agg['frac_ci_excludes_zero']:.2f}")
    pd.DataFrame(rows).to_csv(OUT / "negative_robustness_summary.csv", index=False)
    print(f"\n[done] {OUT / 'negative_robustness_summary.csv'}")


if __name__ == "__main__":
    main()
