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
            "Tune monitoring alert caps from backtest windows to meet a target "
            "false-alert rate."
        )
    )
    parser.add_argument(
        "--window-metrics-csv",
        type=Path,
        default=subproject_root
        / "implementation"
        / "outputs"
        / "cycle38_monitoring_backtest"
        / "monitoring_backtest_window_metrics.csv",
    )
    parser.add_argument("--target-any-alert-rate", type=float, default=0.05)
    parser.add_argument("--quantile-min", type=float, default=0.70)
    parser.add_argument("--quantile-max", type=float, default=0.999)
    parser.add_argument("--quantile-steps", type=int, default=80)
    parser.add_argument("--current-coverage-cap", type=float, default=0.20)
    parser.add_argument("--current-ranking-auc-drop-cap", type=float, default=0.01)
    parser.add_argument("--current-reliability-ece-cap", type=float, default=0.01)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle39_monitoring_threshold_tuning",
    )
    return parser.parse_args()


def _mode_metric_col(mode: str) -> str:
    if mode == "ranking_first":
        return "auc_drop"
    if mode == "reliability_first":
        return "ece_uplift"
    raise ValueError(f"Unsupported mode: {mode}")


def _evaluate_rates(cov_abs: np.ndarray, perf: np.ndarray, cov_cap: float, perf_cap: float) -> tuple[float, float, float]:
    cov_alert = cov_abs > cov_cap
    perf_alert = perf > perf_cap
    any_alert = cov_alert | perf_alert
    return float(np.mean(cov_alert)), float(np.mean(perf_alert)), float(np.mean(any_alert))


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.window_metrics_csv)
    required = {"window_size", "domain", "mode", "coverage_rel_drift", "auc_drop", "ece_uplift"}
    if not required.issubset(set(df.columns)):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Missing required columns in window metrics file: {missing}")

    q_values = np.linspace(args.quantile_min, args.quantile_max, args.quantile_steps)

    grid_rows = []
    best_rows = []
    current_cap_rows = []

    for (window_size, mode), g in df.groupby(["window_size", "mode"]):
        metric_col = _mode_metric_col(str(mode))
        cov_abs = np.abs(g["coverage_rel_drift"].to_numpy(dtype=np.float64))
        perf = g[metric_col].to_numpy(dtype=np.float64)
        perf = perf[np.isfinite(perf)]
        if perf.size == 0:
            continue

        cov_scale = float(np.nanpercentile(cov_abs, 95))
        perf_scale = float(np.nanpercentile(perf, 95))
        cov_scale = max(cov_scale, 1e-12)
        perf_scale = max(perf_scale, 1e-12)

        cov_caps = np.quantile(cov_abs, q_values)
        perf_caps = np.quantile(perf, q_values)

        best_feasible = None
        best_overall = None
        for i, cov_cap in enumerate(cov_caps):
            for j, perf_cap in enumerate(perf_caps):
                cov_rate, perf_rate, any_rate = _evaluate_rates(cov_abs, perf, float(cov_cap), float(perf_cap))
                score = float((cov_cap / cov_scale) + (perf_cap / perf_scale))
                row = {
                    "window_size": int(window_size),
                    "mode": str(mode),
                    "coverage_quantile": float(q_values[i]),
                    "performance_quantile": float(q_values[j]),
                    "coverage_cap": float(cov_cap),
                    "performance_cap": float(perf_cap),
                    "coverage_alert_rate": cov_rate,
                    "performance_alert_rate": perf_rate,
                    "any_alert_rate": any_rate,
                    "objective_score": score,
                    "target_any_alert_rate": float(args.target_any_alert_rate),
                    "feasible": bool(any_rate <= args.target_any_alert_rate),
                }
                grid_rows.append(row)

                if best_overall is None or any_rate < best_overall["any_alert_rate"] - 1e-12 or (
                    abs(any_rate - best_overall["any_alert_rate"]) <= 1e-12 and score < best_overall["objective_score"]
                ):
                    best_overall = row

                if row["feasible"]:
                    if best_feasible is None or score < best_feasible["objective_score"] - 1e-12 or (
                        abs(score - best_feasible["objective_score"]) <= 1e-12
                        and any_rate < best_feasible["any_alert_rate"]
                    ):
                        best_feasible = row

        selected = best_feasible if best_feasible is not None else best_overall
        selected = dict(selected)
        selected["fallback_used"] = bool(best_feasible is None)
        selected["metric_column"] = metric_col
        best_rows.append(selected)

        # Also quantify what happens if coverage cap is left unchanged and only
        # the performance cap is retuned by quantile.
        if mode == "ranking_first":
            fixed_cov_cap = float(args.current_coverage_cap)
            current_perf_cap = float(args.current_ranking_auc_drop_cap)
        else:
            fixed_cov_cap = float(args.current_coverage_cap)
            current_perf_cap = float(args.current_reliability_ece_cap)

        best_perf_only = None
        for j, perf_cap in enumerate(perf_caps):
            cov_rate, perf_rate, any_rate = _evaluate_rates(cov_abs, perf, fixed_cov_cap, float(perf_cap))
            row = {
                "window_size": int(window_size),
                "mode": str(mode),
                "fixed_coverage_cap": fixed_cov_cap,
                "performance_cap": float(perf_cap),
                "performance_quantile": float(q_values[j]),
                "coverage_alert_rate": cov_rate,
                "performance_alert_rate": perf_rate,
                "any_alert_rate": any_rate,
                "target_any_alert_rate": float(args.target_any_alert_rate),
                "feasible": bool(any_rate <= args.target_any_alert_rate),
                "current_performance_cap": current_perf_cap,
            }
            if best_perf_only is None or (
                row["feasible"] and not best_perf_only["feasible"]
            ) or (
                row["feasible"] == best_perf_only["feasible"]
                and (
                    row["performance_cap"] < best_perf_only["performance_cap"] - 1e-12
                    or (
                        abs(row["performance_cap"] - best_perf_only["performance_cap"]) <= 1e-12
                        and row["any_alert_rate"] < best_perf_only["any_alert_rate"]
                    )
                )
            ):
                best_perf_only = row
        current_cap_rows.append(best_perf_only)

    grid_df = pd.DataFrame(grid_rows)
    best_df = pd.DataFrame(best_rows).sort_values(["window_size", "mode"]).reset_index(drop=True)
    current_cap_df = pd.DataFrame(current_cap_rows).sort_values(["window_size", "mode"]).reset_index(drop=True)

    grid_path = output_dir / "monitoring_threshold_tuning_grid.csv"
    best_path = output_dir / "monitoring_threshold_tuning_best.csv"
    current_cap_path = output_dir / "monitoring_threshold_tuning_perf_only_with_current_cov.csv"

    grid_df.to_csv(grid_path, index=False)
    best_df.to_csv(best_path, index=False)
    current_cap_df.to_csv(current_cap_path, index=False)

    print(best_df.to_string(index=False))
    print(current_cap_df.to_string(index=False))
    print(f"[done] {grid_path}")
    print(f"[done] {best_path}")
    print(f"[done] {current_cap_path}")


if __name__ == "__main__":
    main()
