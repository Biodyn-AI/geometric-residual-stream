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
        description=(
            "Backtest deployment monitoring thresholds on replayed policy outputs "
            "using permutation-based pseudo-rolling windows."
        )
    )
    parser.add_argument(
        "--replay-predictions-tsv",
        type=Path,
        default=subproject_root
        / "implementation"
        / "outputs"
        / "cycle37_deployment_policy_conformance"
        / "deployment_replay_predictions.tsv",
    )
    parser.add_argument(
        "--baseline-policy-csv",
        type=Path,
        default=subproject_root
        / "implementation"
        / "outputs"
        / "cycle35_dual_mode_deployment_spec"
        / "dual_mode_domain_policy_table.csv",
    )
    parser.add_argument("--window-sizes", type=str, default="250,500,1000")
    parser.add_argument("--n-permutations", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-calibration-bins", type=int, default=10)
    parser.add_argument("--coverage-drift-cap", type=float, default=0.20)
    parser.add_argument("--reliability-ece-cap", type=float, default=0.01)
    parser.add_argument("--ranking-auc-drop-cap", type=float, default=0.01)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle38_monitoring_backtest",
    )
    return parser.parse_args()


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    idx = np.digitize(p, bins[1:-1], right=False)
    total = max(y.shape[0], 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += (n / total) * abs(acc - conf)
    return float(ece)


def _to_int_list(values: str) -> list[int]:
    out = []
    for token in values.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("At least one window size is required.")
    return out


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    window_sizes = _to_int_list(args.window_sizes)
    rng = np.random.default_rng(args.seed)

    replay_df = pd.read_csv(args.replay_predictions_tsv, sep="\t")
    baseline_df = pd.read_csv(args.baseline_policy_csv)

    required_replay_cols = {"domain", "mode", "label", "policy_prob", "referral_flag"}
    required_baseline_cols = {"domain", "mode", "coverage_fraction", "policy_auc", "policy_ece"}
    if not required_replay_cols.issubset(set(replay_df.columns)):
        missing = sorted(required_replay_cols - set(replay_df.columns))
        raise ValueError(f"Missing columns in replay file: {missing}")
    if not required_baseline_cols.issubset(set(baseline_df.columns)):
        missing = sorted(required_baseline_cols - set(baseline_df.columns))
        raise ValueError(f"Missing columns in baseline file: {missing}")

    baseline_map = {
        (str(row["domain"]), str(row["mode"])): {
            "coverage_fraction": float(row["coverage_fraction"]),
            "policy_auc": float(row["policy_auc"]),
            "policy_ece": float(row["policy_ece"]),
        }
        for _, row in baseline_df.iterrows()
    }

    window_rows = []
    for window_size in window_sizes:
        for (domain, mode), gdf in replay_df.groupby(["domain", "mode"]):
            y_all = gdf["label"].to_numpy(dtype=np.int32)
            p_all = gdf["policy_prob"].to_numpy(dtype=np.float64)
            r_all = gdf["referral_flag"].to_numpy(dtype=np.int8)
            n_edges = y_all.shape[0]
            n_windows = n_edges // window_size
            if n_windows < 1:
                continue

            baseline = baseline_map[(str(domain), str(mode))]
            base_cov = baseline["coverage_fraction"]
            base_auc = baseline["policy_auc"]
            base_ece = baseline["policy_ece"]

            for perm_id in range(args.n_permutations):
                perm = rng.permutation(n_edges)
                for win_id in range(n_windows):
                    lo = win_id * window_size
                    hi = lo + window_size
                    idx = perm[lo:hi]
                    y = y_all[idx]
                    p = p_all[idx]
                    r = r_all[idx]

                    auc = _safe_auc(y, p)
                    ece = _ece(y, p, args.n_calibration_bins)
                    cov = float(np.mean(r))

                    if base_cov > 0:
                        coverage_rel_drift = (cov - base_cov) / base_cov
                    else:
                        coverage_rel_drift = np.nan

                    auc_drop = float(base_auc - auc) if np.isfinite(auc) else np.nan
                    ece_uplift = float(ece - base_ece)

                    coverage_alert = bool(
                        np.isfinite(coverage_rel_drift) and abs(coverage_rel_drift) > args.coverage_drift_cap
                    )
                    ranking_alert = bool(np.isfinite(auc_drop) and auc_drop > args.ranking_auc_drop_cap)
                    reliability_alert = bool(np.isfinite(ece_uplift) and ece_uplift > args.reliability_ece_cap)

                    if mode == "ranking_first":
                        performance_alert = ranking_alert
                        performance_signal = "auc_drop"
                    elif mode == "reliability_first":
                        performance_alert = reliability_alert
                        performance_signal = "ece_uplift"
                    else:
                        performance_alert = bool(ranking_alert or reliability_alert)
                        performance_signal = "unknown_mode"

                    any_alert = bool(coverage_alert or performance_alert)
                    window_rows.append(
                        {
                            "window_size": int(window_size),
                            "domain": str(domain),
                            "mode": str(mode),
                            "n_edges_total": int(n_edges),
                            "window_n_edges": int(window_size),
                            "permutation_id": int(perm_id),
                            "window_id": int(win_id),
                            "window_auc": auc,
                            "window_ece": ece,
                            "window_coverage": cov,
                            "baseline_auc": base_auc,
                            "baseline_ece": base_ece,
                            "baseline_coverage": base_cov,
                            "auc_drop": auc_drop,
                            "ece_uplift": ece_uplift,
                            "coverage_rel_drift": coverage_rel_drift,
                            "coverage_alert": int(coverage_alert),
                            "ranking_alert": int(ranking_alert),
                            "reliability_alert": int(reliability_alert),
                            "performance_alert": int(performance_alert),
                            "performance_signal": performance_signal,
                            "any_alert": int(any_alert),
                        }
                    )

    window_df = pd.DataFrame(window_rows)
    if window_df.empty:
        raise RuntimeError("No windows produced. Reduce --window-sizes or check replay file.")

    summary_rows = []
    rec_rows = []
    group_cols = ["window_size", "domain", "mode"]
    for keys, g in window_df.groupby(group_cols):
        window_size, domain, mode = keys
        auc_drop = g["auc_drop"].to_numpy(dtype=np.float64)
        ece_uplift = g["ece_uplift"].to_numpy(dtype=np.float64)
        cov_abs_drift = np.abs(g["coverage_rel_drift"].to_numpy(dtype=np.float64))

        summary_rows.append(
            {
                "window_size": int(window_size),
                "domain": str(domain),
                "mode": str(mode),
                "n_windows": int(g.shape[0]),
                "coverage_alert_rate": float(g["coverage_alert"].mean()),
                "performance_alert_rate": float(g["performance_alert"].mean()),
                "any_alert_rate": float(g["any_alert"].mean()),
                "auc_nan_fraction": float(np.mean(~np.isfinite(auc_drop))),
                "mean_auc_drop": float(np.nanmean(auc_drop)),
                "mean_ece_uplift": float(np.nanmean(ece_uplift)),
                "mean_abs_coverage_drift": float(np.nanmean(cov_abs_drift)),
                "p95_auc_drop": float(np.nanpercentile(auc_drop, 95)),
                "p95_ece_uplift": float(np.nanpercentile(ece_uplift, 95)),
                "p95_abs_coverage_drift": float(np.nanpercentile(cov_abs_drift, 95)),
            }
        )

        rec_rows.append(
            {
                "window_size": int(window_size),
                "domain": str(domain),
                "mode": str(mode),
                "current_coverage_drift_cap": float(args.coverage_drift_cap),
                "current_ranking_auc_drop_cap": float(args.ranking_auc_drop_cap),
                "current_reliability_ece_cap": float(args.reliability_ece_cap),
                "suggested_coverage_drift_cap_p95": float(np.nanpercentile(cov_abs_drift, 95)),
                "suggested_ranking_auc_drop_cap_p95": float(np.nanpercentile(auc_drop, 95)),
                "suggested_reliability_ece_cap_p95": float(np.nanpercentile(ece_uplift, 95)),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(group_cols).reset_index(drop=True)
    recommendations_df = pd.DataFrame(rec_rows).sort_values(group_cols).reset_index(drop=True)
    aggregate_df = (
        window_df.groupby(["window_size", "mode"], as_index=False)
        .agg(
            n_windows=("any_alert", "size"),
            coverage_alert_rate=("coverage_alert", "mean"),
            performance_alert_rate=("performance_alert", "mean"),
            any_alert_rate=("any_alert", "mean"),
            mean_auc_drop=("auc_drop", "mean"),
            mean_ece_uplift=("ece_uplift", "mean"),
            mean_abs_coverage_drift=("coverage_rel_drift", lambda x: float(np.mean(np.abs(x)))),
        )
        .sort_values(["window_size", "mode"])
        .reset_index(drop=True)
    )

    window_path = output_dir / "monitoring_backtest_window_metrics.csv"
    summary_path = output_dir / "monitoring_backtest_summary.csv"
    aggregate_path = output_dir / "monitoring_backtest_mode_aggregate_summary.csv"
    recommendations_path = output_dir / "monitoring_backtest_threshold_recommendations.csv"

    window_df.to_csv(window_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)
    recommendations_df.to_csv(recommendations_path, index=False)

    print(summary_df.to_string(index=False))
    print(aggregate_df.to_string(index=False))
    print(recommendations_df.to_string(index=False))
    print(f"[done] {window_path}")
    print(f"[done] {summary_path}")
    print(f"[done] {aggregate_path}")
    print(f"[done] {recommendations_path}")


if __name__ == "__main__":
    main()
