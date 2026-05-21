#!/usr/bin/env python3
"""WP1 - Grouped / leakage-resistant cross-validation.

Addresses Reviewer 1 comment 3: edge-level random CV permits gene-identity
carryover. Re-evaluates the geometric signal under leave-TF-out, leave-target-out
and leave-both-out splits, alongside the standard edge-level CV, and reports how
much signal persists. Also feeds the confirmatory framing (R1.2) and absolute
metric reporting (R1.6 / R2.1).

Output: implementation/outputs/cycle43_grouped_cv/grouped_cv_summary.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import revision_lib as rl

OUT = rl.OUTPUTS / "cycle43_grouped_cv"

# Representative representations spanning the manuscript's pipeline evolution.
CONFIGS = [
    ("cosine_bestL", "cosine", None),          # original raw-cosine single layer
    ("pca_cc64_bestL", "pca_cc", 64),          # refined single-layer (centered + PCA)
    ("centered_cosine_bundle", "centered_cosine", None),  # multi-layer bundle L0-11
]
CV_MODES = ["edge", "leave_tf", "leave_target", "leave_both"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for domain in rl.DOMAINS:
        dom = rl.load_domain(domain, 42)
        best_l = rl.PAPER_BEST_LAYER[domain]
        all_layers = list(range(dom.n_layers))
        for cfg_name, metric, pca in CONFIGS:
            layers = all_layers if cfg_name.endswith("bundle") else [best_l]
            for cv_mode in CV_MODES:
                res = rl.evaluate(dom, metric, layers, pca_dim=pca,
                                  cv_mode=cv_mode, n_boot=1000, boot_seed=7)
                clean = {k: v for k, v in res.items() if not k.startswith("_")}
                clean["config"] = cfg_name
                rows.append(clean)
                print(f"[{domain:13s} {cfg_name:22s} {cv_mode:13s}] "
                      f"AUROC base={res['auroc_base']:.3f} plus={res['auroc_plus']:.3f} "
                      f"dAUROC={res['delta_auroc']:+.4f} "
                      f"CI=[{res['boot_delta_lo']:+.4f},{res['boot_delta_hi']:+.4f}] "
                      f"dAUPRC={res['delta_auprc']:+.4f} n={res['n_edges']}")
    df = pd.DataFrame(rows)
    cols = ["domain", "config", "cv_mode", "n_edges", "n_pos", "prevalence",
            "auroc_base", "auroc_plus", "delta_auroc",
            "auprc_base", "auprc_plus", "delta_auprc",
            "boot_delta_mean", "boot_delta_lo", "boot_delta_hi", "ci_excludes_zero"]
    df = df[cols + [c for c in df.columns if c not in cols]]
    df.to_csv(OUT / "grouped_cv_summary.csv", index=False)
    print(f"\n[done] {OUT / 'grouped_cv_summary.csv'}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
