#!/usr/bin/env python3
"""WP4 - Uncertainty decomposition and multiplicity control.

Addresses Reviewer 1 comment 6: clarify the bootstrap resampling unit, decompose
which variance sources each error bar captures, and correct for the many
domain/layer/metric comparisons.

  * variance_components.csv  - delta-AUROC variance split into within-run (edge
                               bootstrap) vs across-seed (cell-sample + negative
                               draw) components, for the headline bundle.
  * multiplicity.csv         - every domain x layer x metric config from the WP3
                               panel, with a one-sided p-value and Benjamini-
                               Hochberg FDR-adjusted q-value.

Run after WP3 (reads cycle45_metric_panels/layerwise_panel.csv).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import revision_lib as rl

OUT = rl.OUTPUTS / "cycle46_uncertainty"
PANEL = rl.OUTPUTS / "cycle45_metric_panels" / "layerwise_panel.csv"


def variance_components() -> pd.DataFrame:
    rows = []
    for domain in rl.DOMAINS:
        layers = list(range(12))
        per_seed = []
        boot_sds = []
        for seed in (42, 43, 44):
            dom = rl.load_domain(domain, seed)
            res = rl.evaluate(dom, "centered_cosine", layers, cv_mode="edge",
                              n_boot=1000, boot_seed=11)
            per_seed.append(res["delta_auroc"])
            # within-run SD from bootstrap CI (95% -> /3.92)
            boot_sds.append((res["boot_delta_hi"] - res["boot_delta_lo"]) / 3.92)
        per_seed = np.array(per_seed)
        within = float(np.mean(boot_sds))           # edge-resampling component
        across = float(per_seed.std(ddof=1))        # seed = cell-sample + neg-draw
        rows.append(dict(
            domain=domain, representation="centered_cosine_bundle_L0-11",
            delta_auroc_mean=float(per_seed.mean()),
            delta_auroc_seed42=per_seed[0], delta_auroc_seed43=per_seed[1],
            delta_auroc_seed44=per_seed[2],
            sd_within_run_edgeboot=within,
            sd_across_seed=across,
            total_sd=float(np.sqrt(within**2 + across**2)),
            frac_var_within=within**2 / (within**2 + across**2 + 1e-12),
            frac_var_across=across**2 / (within**2 + across**2 + 1e-12)))
        print(f"[varcomp {domain:13s}] dAUROC={per_seed.mean():+.4f} "
              f"sd_within={within:.4f} sd_across_seed={across:.4f}")
    return pd.DataFrame(rows)


def multiplicity() -> pd.DataFrame:
    if not PANEL.exists():
        print(f"[skip] {PANEL} not found - run WP3 first")
        return pd.DataFrame()
    df = pd.read_csv(PANEL).copy()
    # one-sided p-value from the bootstrap CI via a normal approximation.
    se = (df["boot_delta_hi"] - df["boot_delta_lo"]) / 3.92
    se = se.replace(0, np.nan)
    z = df["boot_delta_mean"] / se
    df["p_one_sided"] = 1.0 - stats.norm.cdf(z)
    df["p_one_sided"] = df["p_one_sided"].clip(1e-12, 1.0)
    # Benjamini-Hochberg across the whole grid.
    m = len(df)
    order = np.argsort(df["p_one_sided"].to_numpy())
    p_sorted = df["p_one_sided"].to_numpy()[order]
    q = p_sorted * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    qvals = np.empty(m)
    qvals[order] = np.clip(q, 0, 1)
    df["q_value_BH"] = qvals
    df["significant_FDR05"] = df["q_value_BH"] < 0.05
    keep = ["domain", "metric", "layers", "pca_dim", "n_edges",
            "auroc_base", "auroc_plus", "delta_auroc", "delta_auprc",
            "boot_delta_mean", "boot_delta_lo", "boot_delta_hi",
            "p_one_sided", "q_value_BH", "significant_FDR05"]
    out = df[keep].sort_values("q_value_BH").reset_index(drop=True)
    n_raw = int((out["p_one_sided"] < 0.05).sum())
    n_fdr = int(out["significant_FDR05"].sum())
    print(f"[multiplicity] {m} configs tested; {n_raw} significant at raw p<0.05; "
          f"{n_fdr} survive BH FDR<0.05")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    variance_components().to_csv(OUT / "variance_components.csv", index=False)
    mult = multiplicity()
    if not mult.empty:
        mult.to_csv(OUT / "multiplicity.csv", index=False)
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
