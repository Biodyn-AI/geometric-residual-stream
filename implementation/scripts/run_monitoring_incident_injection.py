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
            "Evaluate monitoring alert sensitivity by injecting synthetic degradation "
            "into backtested window metrics."
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
    parser.add_argument(
        "--tuned-thresholds-csv",
        type=Path,
        default=subproject_root
        / "implementation"
        / "outputs"
        / "cycle39_monitoring_threshold_tuning"
        / "monitoring_threshold_tuning_perf_only_with_current_cov.csv",
    )
    parser.add_argument("--performance-shifts", type=str, default="0.0,0.005,0.01,0.02,0.03,0.05")
    parser.add_argument("--coverage-shifts", type=str, default="0.0,0.05,0.10,0.15")
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--n-sequences", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--current-coverage-cap", type=float, default=0.20)
    parser.add_argument("--current-performance-cap", type=float, default=0.01)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle41_monitoring_incident_injection",
    )
    return parser.parse_args()


def _to_float_list(value: str) -> list[float]:
    out = []
    for token in value.split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    if not out:
        raise ValueError("Expected at least one numeric value.")
    return out


def _metric_column_for_mode(mode: str) -> str:
    if mode == "ranking_first":
        return "auc_drop"
    if mode == "reliability_first":
        return "ece_uplift"
    raise ValueError(f"Unsupported mode: {mode}")


def _apply_incident(
    coverage_rel_drift: np.ndarray,
    performance_metric: np.ndarray,
    coverage_shift: float,
    performance_shift: float,
) -> tuple[np.ndarray, np.ndarray]:
    drift_abs = np.abs(coverage_rel_drift) + coverage_shift
    drift_sign = np.sign(coverage_rel_drift)
    drift_sign[drift_sign == 0.0] = 1.0
    drift_incident = drift_sign * drift_abs
    perf_incident = performance_metric + performance_shift
    return drift_incident, perf_incident


def _any_alert_flags(
    coverage_rel_drift: np.ndarray,
    performance_metric: np.ndarray,
    coverage_cap: float,
    performance_cap: float,
) -> np.ndarray:
    cov_alert = np.abs(coverage_rel_drift) > coverage_cap
    perf_alert = performance_metric > performance_cap
    return cov_alert | perf_alert


def _sequence_detection_stats(alert_flags: np.ndarray, n_sequences: int, sequence_length: int, rng: np.random.Generator) -> tuple[float, float]:
    # We evaluate the staged hard-alert criterion from cycle40:
    # at least two consecutive hard-alert windows.
    n = alert_flags.shape[0]
    detected = np.zeros(n_sequences, dtype=bool)
    first_detection = np.full(n_sequences, fill_value=np.nan, dtype=np.float64)
    for i in range(n_sequences):
        seq = alert_flags[rng.integers(0, n, size=sequence_length)]
        consec = seq[:-1] & seq[1:]
        if np.any(consec):
            detected[i] = True
            first_idx = int(np.argmax(consec))
            # detection window index is 1-based and points to the second alert in the first pair
            first_detection[i] = float(first_idx + 2)
    detection_rate = float(np.mean(detected))
    median_first_detection = float(np.nanmedian(first_detection)) if np.any(detected) else float("nan")
    return detection_rate, median_first_detection


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    performance_shifts = _to_float_list(args.performance_shifts)
    coverage_shifts = _to_float_list(args.coverage_shifts)

    windows_df = pd.read_csv(args.window_metrics_csv)
    tuned_df = pd.read_csv(args.tuned_thresholds_csv)

    required_window_cols = {"window_size", "mode", "coverage_rel_drift", "auc_drop", "ece_uplift"}
    if not required_window_cols.issubset(set(windows_df.columns)):
        missing = sorted(required_window_cols - set(windows_df.columns))
        raise ValueError(f"Missing columns in window metrics file: {missing}")
    required_tuned_cols = {"window_size", "mode", "fixed_coverage_cap", "performance_cap"}
    if not required_tuned_cols.issubset(set(tuned_df.columns)):
        missing = sorted(required_tuned_cols - set(tuned_df.columns))
        raise ValueError(f"Missing columns in tuned thresholds file: {missing}")

    tuned_map = {
        (int(r["window_size"]), str(r["mode"])): {
            "coverage_cap": float(r["fixed_coverage_cap"]),
            "performance_cap": float(r["performance_cap"]),
        }
        for _, r in tuned_df.iterrows()
    }

    profiles = [
        {
            "profile": "current_strict",
            "coverage_cap": float(args.current_coverage_cap),
            "performance_cap": float(args.current_performance_cap),
            "per_group_override": False,
        },
        {
            "profile": "tuned_5pct",
            "coverage_cap": np.nan,
            "performance_cap": np.nan,
            "per_group_override": True,
        },
    ]

    power_rows = []
    sequence_rows = []

    for (window_size, mode), g in windows_df.groupby(["window_size", "mode"]):
        metric_col = _metric_column_for_mode(str(mode))
        coverage_base = g["coverage_rel_drift"].to_numpy(dtype=np.float64)
        performance_base = g[metric_col].to_numpy(dtype=np.float64)

        if not np.all(np.isfinite(performance_base)):
            performance_base = performance_base[np.isfinite(performance_base)]
        if performance_base.size == 0:
            continue

        for profile in profiles:
            if profile["per_group_override"]:
                key = (int(window_size), str(mode))
                if key not in tuned_map:
                    continue
                coverage_cap = tuned_map[key]["coverage_cap"]
                performance_cap = tuned_map[key]["performance_cap"]
            else:
                coverage_cap = float(profile["coverage_cap"])
                performance_cap = float(profile["performance_cap"])

            for coverage_shift in coverage_shifts:
                for performance_shift in performance_shifts:
                    coverage_incident, performance_incident = _apply_incident(
                        coverage_rel_drift=coverage_base,
                        performance_metric=performance_base,
                        coverage_shift=float(coverage_shift),
                        performance_shift=float(performance_shift),
                    )
                    alert_flags = _any_alert_flags(
                        coverage_rel_drift=coverage_incident,
                        performance_metric=performance_incident,
                        coverage_cap=coverage_cap,
                        performance_cap=performance_cap,
                    )
                    alert_rate = float(np.mean(alert_flags))
                    power_rows.append(
                        {
                            "window_size": int(window_size),
                            "mode": str(mode),
                            "profile": str(profile["profile"]),
                            "metric_column": metric_col,
                            "coverage_cap": coverage_cap,
                            "performance_cap": performance_cap,
                            "coverage_shift": float(coverage_shift),
                            "performance_shift": float(performance_shift),
                            "n_windows": int(alert_flags.shape[0]),
                            "hard_alert_rate": alert_rate,
                        }
                    )

                    detection_rate, median_detection_window = _sequence_detection_stats(
                        alert_flags=alert_flags,
                        n_sequences=args.n_sequences,
                        sequence_length=args.sequence_length,
                        rng=rng,
                    )
                    sequence_rows.append(
                        {
                            "window_size": int(window_size),
                            "mode": str(mode),
                            "profile": str(profile["profile"]),
                            "coverage_cap": coverage_cap,
                            "performance_cap": performance_cap,
                            "coverage_shift": float(coverage_shift),
                            "performance_shift": float(performance_shift),
                            "sequence_length": int(args.sequence_length),
                            "n_sequences": int(args.n_sequences),
                            "detected_two_consecutive_rate": detection_rate,
                            "median_first_detection_window": median_detection_window,
                        }
                    )

    power_df = pd.DataFrame(power_rows).sort_values(
        ["window_size", "mode", "profile", "coverage_shift", "performance_shift"]
    )
    sequence_df = pd.DataFrame(sequence_rows).sort_values(
        ["window_size", "mode", "profile", "coverage_shift", "performance_shift"]
    )

    # Small summary slice: pure performance incidents with no coverage shift.
    perf_slice = power_df[np.isclose(power_df["coverage_shift"], 0.0)].copy()
    perf_summary = (
        perf_slice.pivot_table(
            index=["window_size", "mode", "profile"],
            columns="performance_shift",
            values="hard_alert_rate",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values(["window_size", "mode", "profile"])
    )

    power_path = output_dir / "monitoring_incident_power.csv"
    sequence_path = output_dir / "monitoring_incident_sequence_detection.csv"
    perf_summary_path = output_dir / "monitoring_incident_performance_shift_matrix.csv"

    power_df.to_csv(power_path, index=False)
    sequence_df.to_csv(sequence_path, index=False)
    perf_summary.to_csv(perf_summary_path, index=False)

    print(perf_summary.to_string(index=False))
    print(sequence_df.head(24).to_string(index=False))
    print(f"[done] {power_path}")
    print(f"[done] {sequence_path}")
    print(f"[done] {perf_summary_path}")


if __name__ == "__main__":
    main()
