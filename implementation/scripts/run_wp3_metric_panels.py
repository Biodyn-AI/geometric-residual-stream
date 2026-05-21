#!/usr/bin/env python3
"""WP3 / WP7 - Absolute-metric panels and unified-method comparison.

Addresses Reviewer 2 comments 1, 2, 4 and Reviewer 1 comment 6: report absolute
AUROC and AUPRC (not only delta) for every domain/layer/metric, and apply the
refined methodology (centered cosine, PCA) uniformly to the early domains
(Reviewer 2 comment 3 / R1.2).

Produces two CSVs in implementation/outputs/cycle45_metric_panels/:
  * layerwise_panel.csv   - every domain x layer x metric, full metric panel
  * unified_method.csv    - per domain: raw cosine vs refined methods, best layer
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import revision_lib as rl

OUT = rl.OUTPUTS / "cycle45_metric_panels"
METRICS = [("cosine", None), ("centered_cosine", None), ("pca_cc", 64)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel_rows, unified_rows = [], []
    for domain in rl.DOMAINS:
        dom = rl.load_domain(domain, 42)
        per_metric_best = {}
        for metric, pca in METRICS:
            best = None
            for layer in range(dom.n_layers):
                res = rl.evaluate(dom, metric, [layer], pca_dim=pca,
                                  cv_mode="edge", n_boot=400, boot_seed=3)
                row = {k: v for k, v in res.items() if not k.startswith("_")}
                panel_rows.append(row)
                if best is None or res["delta_auroc"] > best["delta_auroc"]:
                    best = row
            per_metric_best[metric] = best
            print(f"[{domain:13s} {metric:16s}] best L{best['layers']:>2s} "
                  f"AUROC {best['auroc_base']:.3f}->{best['auroc_plus']:.3f} "
                  f"dAUROC={best['delta_auroc']:+.4f} AUPRC {best['auprc_base']:.3f}"
                  f"->{best['auprc_plus']:.3f} dAUPRC={best['delta_auprc']:+.4f}")
        # multi-layer bundle (refined headline representation)
        bundle = rl.evaluate(dom, "centered_cosine", list(range(dom.n_layers)),
                             cv_mode="edge", n_boot=400, boot_seed=3)
        b = {k: v for k, v in bundle.items() if not k.startswith("_")}
        panel_rows.append(b)
        print(f"[{domain:13s} {'cc_bundle_L0-11':16s}] "
              f"AUROC {b['auroc_base']:.3f}->{b['auroc_plus']:.3f} "
              f"dAUROC={b['delta_auroc']:+.4f} dAUPRC={b['delta_auprc']:+.4f}")
        for label, row in [("raw_cosine_bestL", per_metric_best["cosine"]),
                           ("centered_cosine_bestL", per_metric_best["centered_cosine"]),
                           ("pca_cc64_bestL", per_metric_best["pca_cc"]),
                           ("centered_cosine_bundle_L0-11", b)]:
            unified_rows.append(dict(
                domain=domain, method=label, best_layers=row["layers"],
                auroc_base=row["auroc_base"], auroc_plus=row["auroc_plus"],
                delta_auroc=row["delta_auroc"],
                auprc_base=row["auprc_base"], auprc_plus=row["auprc_plus"],
                delta_auprc=row["delta_auprc"],
                boot_delta_lo=row["boot_delta_lo"], boot_delta_hi=row["boot_delta_hi"],
                ci_excludes_zero=row["ci_excludes_zero"]))
    pd.DataFrame(panel_rows).to_csv(OUT / "layerwise_panel.csv", index=False)
    pd.DataFrame(unified_rows).to_csv(OUT / "unified_method.csv", index=False)
    print(f"\n[done] {OUT}")


if __name__ == "__main__":
    main()
