#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent
    parser = argparse.ArgumentParser(
        description="Bootstrap CIs for disagreement-threshold policy gains on outer-split predictions."
    )
    parser.add_argument(
        "--oof-predictions-tsv",
        type=Path,
        default=subproject_root
        / "implementation"
        / "outputs"
        / "cycle25_outer_split_compact_model"
        / "outer_split_oof_predictions.tsv",
    )
    parser.add_argument("--thresholds", type=str, default="0.05,0.1,0.15,0.2,0.25,0.3")
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle27_outer_split_threshold_uncertainty",
    )
    return parser.parse_args()


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _delta_auc(y: np.ndarray, sc: np.ndarray, gf: np.ndarray, compact: np.ndarray) -> float:
    auc_sc = _safe_auc(y, sc)
    auc_gf = _safe_auc(y, gf)
    auc_compact = _safe_auc(y, compact)
    best_single = float(np.nanmax([auc_sc, auc_gf]))
    return auc_compact - best_single


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    oof = pd.read_csv(args.oof_predictions_tsv, sep="\t")
    rows = []
    dist_rows = []

    for domain, df in oof.groupby("domain"):
        y_full = df["label"].to_numpy(dtype=np.int32)
        sc_full = df["scgpt_prob"].to_numpy(dtype=np.float64)
        gf_full = df["geneformer_prob"].to_numpy(dtype=np.float64)
        compact_full = df["compact_prob"].to_numpy(dtype=np.float64)
        abs_dis = np.abs(gf_full - sc_full)

        for tau in thresholds:
            mask = abs_dis >= tau
            y = y_full[mask]
            sc = sc_full[mask]
            gf = gf_full[mask]
            compact = compact_full[mask]
            n = int(y.shape[0])
            pos = int(y.sum())
            if n < 8 or pos < 2 or (n - pos) < 2:
                rows.append(
                    {
                        "domain": domain,
                        "disagreement_threshold": tau,
                        "n_edges": n,
                        "n_positive": pos,
                        "coverage_fraction": float(np.mean(mask)),
                        "delta_auc_point": float("nan"),
                        "delta_auc_ci_lower": float("nan"),
                        "delta_auc_ci_upper": float("nan"),
                        "delta_auc_boot_mean": float("nan"),
                        "delta_auc_boot_std": float("nan"),
                    }
                )
                continue

            point = _delta_auc(y=y, sc=sc, gf=gf, compact=compact)
            boot = np.zeros(args.n_bootstrap, dtype=np.float64)
            for i in range(args.n_bootstrap):
                sample_idx = rng.integers(0, n, size=n)
                y_b = y[sample_idx]
                sc_b = sc[sample_idx]
                gf_b = gf[sample_idx]
                compact_b = compact[sample_idx]
                if np.unique(y_b).size < 2:
                    boot[i] = np.nan
                    continue
                boot[i] = _delta_auc(y=y_b, sc=sc_b, gf=gf_b, compact=compact_b)

            boot = boot[np.isfinite(boot)]
            if boot.size == 0:
                ci_lo = float("nan")
                ci_hi = float("nan")
                boot_mean = float("nan")
                boot_std = float("nan")
            else:
                ci_lo = float(np.percentile(boot, 2.5))
                ci_hi = float(np.percentile(boot, 97.5))
                boot_mean = float(np.mean(boot))
                boot_std = float(np.std(boot))

            rows.append(
                {
                    "domain": domain,
                    "disagreement_threshold": tau,
                    "n_edges": n,
                    "n_positive": pos,
                    "coverage_fraction": float(np.mean(mask)),
                    "delta_auc_point": point,
                    "delta_auc_ci_lower": ci_lo,
                    "delta_auc_ci_upper": ci_hi,
                    "delta_auc_boot_mean": boot_mean,
                    "delta_auc_boot_std": boot_std,
                }
            )
            dist_rows.append(
                pd.DataFrame(
                    {
                        "domain": domain,
                        "disagreement_threshold": tau,
                        "delta_auc_boot": boot,
                    }
                )
            )

    summary_df = pd.DataFrame(rows)
    dist_df = pd.concat(dist_rows, axis=0, ignore_index=True) if dist_rows else pd.DataFrame()

    summary_path = output_dir / "outer_split_disagreement_policy_bootstrap_ci.csv"
    dist_path = output_dir / "outer_split_disagreement_policy_bootstrap_distribution.tsv"
    summary_df.to_csv(summary_path, index=False)
    dist_df.to_csv(dist_path, sep="\t", index=False)

    print(summary_df.to_string(index=False))
    print(f"[done] {summary_path}")
    print(f"[done] {dist_path}")


if __name__ == "__main__":
    main()
