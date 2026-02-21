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
            "Build dual-mode deployment specification from calibration-lock and utility-sweep outputs."
        )
    )
    parser.add_argument(
        "--calibration-policy-metrics-csv",
        type=Path,
        default=subproject_root
        / "implementation"
        / "outputs"
        / "cycle33_calibration_regime_lock"
        / "calibration_regime_policy_metrics.csv",
    )
    parser.add_argument(
        "--utility-default-by-lambda-csv",
        type=Path,
        default=subproject_root
        / "implementation"
        / "outputs"
        / "cycle32_policy_cost_utility_sweep"
        / "policy_cost_utility_best_default_by_lambda.csv",
    )
    parser.add_argument("--ranking-threshold", type=float, default=0.05)
    parser.add_argument("--reliability-threshold", type=float, default=0.05)
    parser.add_argument("--auc-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle35_dual_mode_deployment_spec",
    )
    return parser.parse_args()


def _select_ranking_mode(df: pd.DataFrame, threshold: float) -> pd.Series:
    cand = df[np.isclose(df["threshold"], threshold)].copy()
    if cand.empty:
        cand = df.copy()
    return cand.sort_values(["policy_auc", "policy_ece"], ascending=[False, True]).iloc[0]


def _select_reliability_mode(df: pd.DataFrame, threshold: float, auc_tolerance: float) -> tuple[pd.Series, bool]:
    cand = df[np.isclose(df["threshold"], threshold)].copy()
    if cand.empty:
        cand = df.copy()
    max_auc = float(cand["policy_auc"].max())
    feasible = cand[cand["policy_auc"] >= (max_auc - auc_tolerance)]
    if feasible.empty:
        return cand.sort_values(["policy_ece", "policy_auc"], ascending=[True, False]).iloc[0], True
    return feasible.sort_values(["policy_ece", "policy_auc"], ascending=[True, False]).iloc[0], False


def _build_schedule(default_df: pd.DataFrame) -> pd.DataFrame:
    d = default_df.sort_values("lambda").reset_index(drop=True)
    d["rule"] = d["calibration_method"].astype(str) + "@" + d["disagreement_threshold"].map(lambda x: f"{x:.2f}")
    rows = []
    start_idx = 0
    for i in range(1, d.shape[0] + 1):
        end_segment = i == d.shape[0] or d.loc[i, "rule"] != d.loc[start_idx, "rule"]
        if not end_segment:
            continue
        segment = d.iloc[start_idx:i]
        first = segment.iloc[0]
        rows.append(
            {
                "lambda_start": float(segment["lambda"].min()),
                "lambda_end": float(segment["lambda"].max()),
                "calibration_method": first["calibration_method"],
                "disagreement_threshold": float(first["disagreement_threshold"]),
                "rule": first["rule"],
                "mean_utility_in_segment": float(segment["mean_utility"].mean()),
                "mean_coverage_in_segment": float(segment["mean_coverage"].mean()),
            }
        )
        start_idx = i
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(args.calibration_policy_metrics_csv)
    default_by_lambda = pd.read_csv(args.utility_default_by_lambda_csv)

    required_metrics = {"domain", "method", "threshold", "coverage_fraction", "policy_auc", "policy_ece", "policy_brier"}
    if not required_metrics.issubset(set(metrics.columns)):
        missing = sorted(required_metrics - set(metrics.columns))
        raise ValueError(f"Missing columns in metrics file: {missing}")

    mode_rows = []
    for domain, ddf in metrics.groupby("domain"):
        rank = _select_ranking_mode(ddf, args.ranking_threshold)
        reli, reli_fallback = _select_reliability_mode(ddf, args.reliability_threshold, args.auc_tolerance)

        mode_rows.append(
            {
                "domain": domain,
                "mode": "ranking_first",
                "fallback_used": False,
                "method": rank["method"],
                "threshold": float(rank["threshold"]),
                "coverage_fraction": float(rank["coverage_fraction"]),
                "policy_auc": float(rank["policy_auc"]),
                "policy_ece": float(rank["policy_ece"]),
                "policy_brier": float(rank["policy_brier"]),
            }
        )
        mode_rows.append(
            {
                "domain": domain,
                "mode": "reliability_first",
                "fallback_used": bool(reli_fallback),
                "method": reli["method"],
                "threshold": float(reli["threshold"]),
                "coverage_fraction": float(reli["coverage_fraction"]),
                "policy_auc": float(reli["policy_auc"]),
                "policy_ece": float(reli["policy_ece"]),
                "policy_brier": float(reli["policy_brier"]),
            }
        )

    mode_df = pd.DataFrame(mode_rows)
    summary_df = (
        mode_df.groupby("mode", as_index=False)
        .agg(
            n_domains=("domain", "count"),
            mean_policy_auc=("policy_auc", "mean"),
            min_policy_auc=("policy_auc", "min"),
            mean_policy_ece=("policy_ece", "mean"),
            max_policy_ece=("policy_ece", "max"),
            mean_coverage=("coverage_fraction", "mean"),
            n_fallback=("fallback_used", "sum"),
        )
        .sort_values("mode")
    )

    schedule_df = _build_schedule(default_by_lambda)

    mode_path = output_dir / "dual_mode_domain_policy_table.csv"
    summary_path = output_dir / "dual_mode_summary.csv"
    schedule_path = output_dir / "cost_aware_default_schedule.csv"

    mode_df.to_csv(mode_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    schedule_df.to_csv(schedule_path, index=False)

    print(mode_df.to_string(index=False))
    print(summary_df.to_string(index=False))
    print(schedule_df.to_string(index=False))
    print(f"[done] {mode_path}")
    print(f"[done] {summary_path}")
    print(f"[done] {schedule_path}")


if __name__ == "__main__":
    main()
