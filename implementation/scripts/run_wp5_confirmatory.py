#!/usr/bin/env python3
"""WP5 - Confirmatory evaluation with a frozen pipeline.

Addresses Reviewer 1 comment 2 and the editor's overstatement concern. The
representation choices were selected during exploratory analysis on kidney and
immune. Here we FREEZE a single pre-specified pipeline -- centered-cosine
similarity, multi-layer bundle L0-11, logistic-regression stacking -- with no
per-domain tuning, and apply it once to the confirmatory domains (lung,
external lung) across three seeds, under both edge-level and the strict
leave-both-out CV.

Output: implementation/outputs/cycle47_confirmatory/confirmatory_summary.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import revision_lib as rl

OUT = rl.OUTPUTS / "cycle47_confirmatory"
EXPLORATORY = ["kidney", "immune"]
CONFIRMATORY = ["lung", "external_lung"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for domain in rl.DOMAINS:
        role = "exploratory" if domain in EXPLORATORY else "confirmatory"
        for cv_mode in ("edge", "leave_both"):
            per_seed = []
            for seed in (42, 43, 44):
                dom = rl.load_domain(domain, seed)
                res = rl.evaluate(dom, "centered_cosine", list(range(dom.n_layers)),
                                  cv_mode=cv_mode, n_boot=1000, boot_seed=5)
                per_seed.append(res)
            d = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                              for r in per_seed])
            row = dict(
                domain=domain, role=role, frozen_pipeline="centered_cosine_bundle_L0-11",
                cv_mode=cv_mode, n_seeds=3,
                auroc_base_mean=d.auroc_base.mean(),
                auroc_plus_mean=d.auroc_plus.mean(),
                delta_auroc_mean=d.delta_auroc.mean(),
                delta_auroc_sd=d.delta_auroc.std(ddof=1),
                delta_auprc_mean=d.delta_auprc.mean(),
                delta_auprc_sd=d.delta_auprc.std(ddof=1),
                boot_delta_lo_min=d.boot_delta_lo.min(),
                boot_delta_hi_max=d.boot_delta_hi.max(),
                n_seeds_ci_excludes_zero=int(d.ci_excludes_zero.sum()))
            rows.append(row)
            print(f"[{domain:13s} {role:12s} {cv_mode:11s}] "
                  f"dAUROC={row['delta_auroc_mean']:+.4f}+-{row['delta_auroc_sd']:.4f} "
                  f"AUROC {row['auroc_base_mean']:.3f}->{row['auroc_plus_mean']:.3f} "
                  f"dAUPRC={row['delta_auprc_mean']:+.4f} "
                  f"seeds-CI>0={row['n_seeds_ci_excludes_zero']}/3")
    pd.DataFrame(rows).to_csv(OUT / "confirmatory_summary.csv", index=False)
    print(f"\n[done] {OUT / 'confirmatory_summary.csv'}")


if __name__ == "__main__":
    main()
