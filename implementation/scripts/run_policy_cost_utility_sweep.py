#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Referral-cost utility sweep for policy candidates. "
            "Utility definition: delta_auc_vs_best_single - lambda * coverage_fraction."
        )
    )
    parser.add_argument(
        "--policy-candidates-csv",
        type=Path,
        default=subproject_root
        / "implementation"
        / "outputs"
        / "cycle29b_objective_policy_selection_cov05"
        / "policy_candidates.csv",
    )
    parser.add_argument(
        "--lambda-grid",
        type=str,
        default="0.0,0.0025,0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.025,0.03",
    )
    parser.add_argument("--min-coverage-fraction", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle32_policy_cost_utility_sweep",
    )
    return parser.parse_args()


def _best_rows(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    # Tie-break: higher utility, then higher AUC delta, then lower coverage.
    order_cols = ["utility", "delta_auc_vs_best_single", "coverage_fraction"]
    ascending = [False, False, True]
    rows = []
    for _, g in df.groupby(group_cols):
        rows.append(g.sort_values(order_cols, ascending=ascending).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lambdas = [float(x) for x in args.lambda_grid.split(",") if x.strip()]
    if not lambdas:
        raise ValueError("Lambda grid is empty.")

    candidates = pd.read_csv(args.policy_candidates_csv)
    req_cols = {
        "domain",
        "calibration_method",
        "disagreement_threshold",
        "coverage_fraction",
        "delta_auc_vs_best_single",
        "delta_ece_vs_best_single",
    }
    missing = req_cols - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing columns in policy candidates: {sorted(missing)}")

    # Keep only operationally meaningful candidates.
    candidates = candidates[candidates["coverage_fraction"] >= args.min_coverage_fraction].copy()
    if candidates.empty:
        raise ValueError(
            "No candidates left after min-coverage filtering. "
            "Lower --min-coverage-fraction or regenerate policy candidates."
        )
    n_total_domains = int(candidates["domain"].nunique())

    scored_tables = []
    for lam in lambdas:
        d = candidates.copy()
        d["lambda"] = lam
        d["utility"] = d["delta_auc_vs_best_single"] - lam * d["coverage_fraction"]
        scored_tables.append(d)
    scored = pd.concat(scored_tables, axis=0, ignore_index=True)

    best_domain = _best_rows(scored, ["lambda", "domain"])

    # Select one cross-domain default policy per lambda.
    default_rows = []
    for lam, g in scored.groupby("lambda"):
        grouped_all = (
            g.groupby(["calibration_method", "disagreement_threshold"], as_index=False)
            .agg(
                n_domains=("domain", "nunique"),
                mean_utility=("utility", "mean"),
                min_utility=("utility", "min"),
                std_utility=("utility", "std"),
                mean_delta_auc=("delta_auc_vs_best_single", "mean"),
                min_delta_auc=("delta_auc_vs_best_single", "min"),
                mean_delta_ece=("delta_ece_vs_best_single", "mean"),
                max_delta_ece=("delta_ece_vs_best_single", "max"),
                mean_coverage=("coverage_fraction", "mean"),
            )
            .sort_values(
                ["mean_utility", "min_utility", "mean_delta_auc", "mean_coverage"],
                ascending=[False, False, False, True],
            )
        )
        # Cross-domain default must exist for every domain.
        grouped = grouped_all[grouped_all["n_domains"] == n_total_domains]
        if grouped.empty:
            grouped = grouped_all
        top = grouped.iloc[0].to_dict()
        top["lambda"] = lam
        default_rows.append(top)
    best_default = pd.DataFrame(default_rows)

    # Compare default utility against domain-specific best utility.
    gap_rows = []
    for lam in sorted(scored["lambda"].unique()):
        default_row = best_default[best_default["lambda"] == lam].iloc[0]
        method = default_row["calibration_method"]
        tau = float(default_row["disagreement_threshold"])
        for domain in sorted(scored["domain"].unique()):
            dbest = best_domain[(best_domain["lambda"] == lam) & (best_domain["domain"] == domain)].iloc[0]
            ddef = scored[
                (scored["lambda"] == lam)
                & (scored["domain"] == domain)
                & (scored["calibration_method"] == method)
                & (scored["disagreement_threshold"] == tau)
            ]
            if ddef.empty:
                continue
            ddef_row = ddef.iloc[0]
            gap_rows.append(
                {
                    "lambda": lam,
                    "domain": domain,
                    "domain_best_calibration_method": dbest["calibration_method"],
                    "domain_best_disagreement_threshold": dbest["disagreement_threshold"],
                    "domain_best_utility": dbest["utility"],
                    "default_calibration_method": method,
                    "default_disagreement_threshold": tau,
                    "default_utility": ddef_row["utility"],
                    "utility_gap_default_minus_domain_best": ddef_row["utility"] - dbest["utility"],
                    "default_delta_auc": ddef_row["delta_auc_vs_best_single"],
                    "default_delta_ece": ddef_row["delta_ece_vs_best_single"],
                    "default_coverage": ddef_row["coverage_fraction"],
                }
            )
    gap_df = pd.DataFrame(gap_rows)

    # Stability summary for quick interpretation.
    stability = (
        best_domain.groupby(["domain", "calibration_method", "disagreement_threshold"], as_index=False)
        .agg(
            n_lambda_selected=("lambda", "count"),
            lambda_min=("lambda", "min"),
            lambda_max=("lambda", "max"),
        )
        .sort_values(["domain", "n_lambda_selected"], ascending=[True, False])
    )

    scored_path = output_dir / "policy_cost_utility_scored_grid.csv"
    best_domain_path = output_dir / "policy_cost_utility_best_by_domain_lambda.csv"
    best_default_path = output_dir / "policy_cost_utility_best_default_by_lambda.csv"
    gap_path = output_dir / "policy_cost_utility_default_vs_domain_gap.csv"
    stability_path = output_dir / "policy_cost_utility_selection_stability.csv"

    scored.to_csv(scored_path, index=False)
    best_domain.to_csv(best_domain_path, index=False)
    best_default.to_csv(best_default_path, index=False)
    gap_df.to_csv(gap_path, index=False)
    stability.to_csv(stability_path, index=False)

    print(best_domain.to_string(index=False))
    print(best_default.to_string(index=False))
    print(gap_df.to_string(index=False))
    print(stability.to_string(index=False))
    print(f"[done] {scored_path}")
    print(f"[done] {best_domain_path}")
    print(f"[done] {best_default_path}")
    print(f"[done] {gap_path}")
    print(f"[done] {stability_path}")


if __name__ == "__main__":
    main()
