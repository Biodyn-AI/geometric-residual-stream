#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent
    parser = argparse.ArgumentParser(
        description="Validate machine-readable deployment config against empirical policy artifacts."
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=subproject_root / "implementation" / "configs" / "deployment_policy_config_v1.json",
    )
    parser.add_argument(
        "--dual-mode-policy-csv",
        type=Path,
        default=subproject_root
        / "implementation"
        / "outputs"
        / "cycle35_dual_mode_deployment_spec"
        / "dual_mode_domain_policy_table.csv",
    )
    parser.add_argument(
        "--cost-schedule-csv",
        type=Path,
        default=subproject_root
        / "implementation"
        / "outputs"
        / "cycle35_dual_mode_deployment_spec"
        / "cost_aware_default_schedule.csv",
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
    parser.add_argument("--validation-window-size", type=int, default=500)
    parser.add_argument("--float-tol", type=float, default=1e-6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle42_config_pack_validation",
    )
    return parser.parse_args()


def _status_row(check_name: str, passed: bool, detail: str) -> dict:
    return {
        "check_name": check_name,
        "passed": bool(passed),
        "detail": detail,
    }


def _nearly_equal(a: float, b: float, tol: float) -> bool:
    return bool(abs(float(a) - float(b)) <= tol)


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config_json, "r", encoding="utf-8") as f:
        config = json.load(f)

    dual_mode_df = pd.read_csv(args.dual_mode_policy_csv)
    cost_schedule_df = pd.read_csv(args.cost_schedule_csv)
    tuned_df = pd.read_csv(args.tuned_thresholds_csv)

    checks = []
    mode_compare_rows = []
    threshold_compare_rows = []

    modes_cfg = config["policy"]["modes"]
    for mode_name in ["ranking_first", "reliability_first"]:
        if mode_name not in modes_cfg:
            checks.append(_status_row(f"config_has_mode_{mode_name}", False, "Mode missing in config."))
            continue

        cfg_method = str(modes_cfg[mode_name]["calibration_method"])
        cfg_tau = float(modes_cfg[mode_name]["disagreement_threshold"])
        sub = dual_mode_df[dual_mode_df["mode"] == mode_name].copy()
        if sub.empty:
            checks.append(_status_row(f"dual_mode_rows_for_{mode_name}", False, "No rows found in dual-mode table."))
            continue

        method_match = bool(np.all(sub["method"].astype(str) == cfg_method))
        tau_match = bool(np.all(np.isclose(sub["threshold"].astype(float), cfg_tau)))
        checks.append(
            _status_row(
                f"{mode_name}_method_match",
                method_match,
                f"Config={cfg_method}, dual_mode_unique={sorted(sub['method'].astype(str).unique().tolist())}",
            )
        )
        checks.append(
            _status_row(
                f"{mode_name}_threshold_match",
                tau_match,
                f"Config={cfg_tau:.6f}, dual_mode_unique={sorted(sub['threshold'].astype(float).unique().tolist())}",
            )
        )
        for _, row in sub.iterrows():
            mode_compare_rows.append(
                {
                    "domain": str(row["domain"]),
                    "mode": mode_name,
                    "config_method": cfg_method,
                    "table_method": str(row["method"]),
                    "config_threshold": cfg_tau,
                    "table_threshold": float(row["threshold"]),
                    "method_match": bool(str(row["method"]) == cfg_method),
                    "threshold_match": bool(np.isclose(float(row["threshold"]), cfg_tau)),
                }
            )

    tuned_sub = tuned_df[tuned_df["window_size"] == args.validation_window_size].copy()
    if tuned_sub.empty:
        checks.append(
            _status_row(
                "tuned_thresholds_window_present",
                False,
                f"No tuned-threshold rows found for window_size={args.validation_window_size}.",
            )
        )
    else:
        checks.append(
            _status_row(
                "tuned_thresholds_window_present",
                True,
                f"Found {tuned_sub.shape[0]} rows for window_size={args.validation_window_size}.",
            )
        )
        hard_cfg = config["monitoring"]["hard_alert_thresholds"]
        cov_cfg = float(hard_cfg["coverage_rel_drift_abs"])
        ranking_cfg = float(hard_cfg["ranking_auc_drop"])
        reli_cfg = float(hard_cfg["reliability_ece_uplift"])

        for mode_name, cfg_perf in [("ranking_first", ranking_cfg), ("reliability_first", reli_cfg)]:
            row = tuned_sub[tuned_sub["mode"] == mode_name]
            if row.empty:
                checks.append(_status_row(f"tuned_row_for_{mode_name}", False, "Missing tuned threshold row."))
                continue
            row = row.iloc[0]
            tuned_cov = float(row["fixed_coverage_cap"])
            tuned_perf = float(row["performance_cap"])
            cov_match = _nearly_equal(cov_cfg, tuned_cov, args.float_tol)
            perf_match = _nearly_equal(cfg_perf, tuned_perf, args.float_tol)
            checks.append(
                _status_row(
                    f"{mode_name}_coverage_cap_match",
                    cov_match,
                    f"Config={cov_cfg:.6f}, tuned={tuned_cov:.6f}",
                )
            )
            checks.append(
                _status_row(
                    f"{mode_name}_performance_cap_match",
                    perf_match,
                    f"Config={cfg_perf:.6f}, tuned={tuned_perf:.6f}",
                )
            )
            threshold_compare_rows.append(
                {
                    "window_size": int(args.validation_window_size),
                    "mode": mode_name,
                    "config_coverage_cap": cov_cfg,
                    "tuned_coverage_cap": tuned_cov,
                    "coverage_cap_match": cov_match,
                    "config_performance_cap": cfg_perf,
                    "tuned_performance_cap": tuned_perf,
                    "performance_cap_match": perf_match,
                }
            )

    schedule_cfg = config["policy"]["cost_aware_default"]
    boundary_cfg = float(schedule_cfg["lambda_boundary"])
    rules_cfg = schedule_cfg["rules"]
    if len(rules_cfg) < 2:
        checks.append(_status_row("cost_schedule_rules_count", False, "Expected at least two rules in config."))
    else:
        checks.append(_status_row("cost_schedule_rules_count", True, f"Found {len(rules_cfg)} rules in config."))

    sched = cost_schedule_df.sort_values("lambda_start").reset_index(drop=True)
    if sched.shape[0] < 2:
        checks.append(_status_row("empirical_schedule_segments", False, "Need at least two empirical schedule segments."))
    else:
        low_emp = sched.iloc[0]
        high_emp = sched.iloc[-1]
        low_cfg = rules_cfg[0]
        high_cfg = rules_cfg[1]

        low_rule_match = (
            str(low_cfg["calibration_method"]) == str(low_emp["calibration_method"])
            and np.isclose(float(low_cfg["disagreement_threshold"]), float(low_emp["disagreement_threshold"]))
        )
        high_rule_match = (
            str(high_cfg["calibration_method"]) == str(high_emp["calibration_method"])
            and np.isclose(float(high_cfg["disagreement_threshold"]), float(high_emp["disagreement_threshold"]))
        )
        checks.append(
            _status_row(
                "cost_schedule_low_rule_match",
                bool(low_rule_match),
                (
                    f"Config={low_cfg['calibration_method']}@{float(low_cfg['disagreement_threshold']):.3f}, "
                    f"empirical={low_emp['calibration_method']}@{float(low_emp['disagreement_threshold']):.3f}"
                ),
            )
        )
        checks.append(
            _status_row(
                "cost_schedule_high_rule_match",
                bool(high_rule_match),
                (
                    f"Config={high_cfg['calibration_method']}@{float(high_cfg['disagreement_threshold']):.3f}, "
                    f"empirical={high_emp['calibration_method']}@{float(high_emp['disagreement_threshold']):.3f}"
                ),
            )
        )

        empirical_high_start = float(high_emp["lambda_start"])
        boundary_match = _nearly_equal(boundary_cfg, empirical_high_start, args.float_tol)
        checks.append(
            _status_row(
                "cost_schedule_boundary_match",
                boundary_match,
                f"Config boundary={boundary_cfg:.6f}, empirical_high_start={empirical_high_start:.6f}",
            )
        )

        # Config should explicitly close empirical gap by covering [0, 0.03] continuously.
        low_max = float(low_cfg["lambda_max_exclusive"])
        high_min = float(high_cfg["lambda_min_inclusive"])
        contiguous_cover = _nearly_equal(low_max, high_min, args.float_tol)
        checks.append(
            _status_row(
                "config_cost_schedule_contiguous",
                contiguous_cover,
                f"low_max_exclusive={low_max:.6f}, high_min_inclusive={high_min:.6f}",
            )
        )

    checks_df = pd.DataFrame(checks)
    passed_all = bool(checks_df["passed"].all()) if not checks_df.empty else False
    summary_df = pd.DataFrame(
        [
            {
                "config_id": str(config.get("config_id", "unknown")),
                "n_checks": int(checks_df.shape[0]),
                "n_passed": int(checks_df["passed"].sum()),
                "n_failed": int((~checks_df["passed"]).sum()),
                "all_checks_passed": passed_all,
            }
        ]
    )

    checks_path = output_dir / "config_validation_checks.csv"
    summary_path = output_dir / "config_validation_summary.csv"
    mode_compare_path = output_dir / "config_vs_dualmode_comparison.csv"
    threshold_compare_path = output_dir / "config_vs_tuned_thresholds_comparison.csv"

    checks_df.to_csv(checks_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame(mode_compare_rows).to_csv(mode_compare_path, index=False)
    pd.DataFrame(threshold_compare_rows).to_csv(threshold_compare_path, index=False)

    print(summary_df.to_string(index=False))
    print(checks_df.to_string(index=False))
    print(f"[done] {checks_path}")
    print(f"[done] {summary_path}")
    print(f"[done] {mode_compare_path}")
    print(f"[done] {threshold_compare_path}")


if __name__ == "__main__":
    main()
