#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Find the best two-rule policy (<=2 unique policy rules across domains) "
            "and compare utility loss against domain-specific best policies."
        )
    )
    parser.add_argument(
        "--scored-grid-csv",
        type=Path,
        default=subproject_root
        / "implementation"
        / "outputs"
        / "cycle32_policy_cost_utility_sweep"
        / "policy_cost_utility_scored_grid.csv",
    )
    parser.add_argument(
        "--domain-best-csv",
        type=Path,
        default=subproject_root
        / "implementation"
        / "outputs"
        / "cycle32_policy_cost_utility_sweep"
        / "policy_cost_utility_best_by_domain_lambda.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle34_two_rule_policy_simplification",
    )
    return parser.parse_args()


def _policy_id(row: pd.Series) -> tuple[str, float]:
    return (str(row["calibration_method"]), float(row["disagreement_threshold"]))


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scored = pd.read_csv(args.scored_grid_csv)
    domain_best = pd.read_csv(args.domain_best_csv)

    domains = sorted(scored["domain"].unique())
    if len(domains) < 2:
        raise ValueError("Need at least 2 domains for two-rule simplification analysis.")

    rows = []
    per_domain_rows = []

    for lam in sorted(scored["lambda"].unique()):
        s = scored[scored["lambda"] == lam].copy()
        db = domain_best[domain_best["lambda"] == lam].copy()
        domain_best_map = {str(r["domain"]): float(r["utility"]) for _, r in db.iterrows()}
        domain_best_mean = float(np.mean(list(domain_best_map.values())))
        domain_best_min = float(np.min(list(domain_best_map.values())))

        # Build candidate set per domain.
        cand_by_domain: dict[str, list[pd.Series]] = {}
        for d in domains:
            cand_by_domain[d] = [row for _, row in s[s["domain"] == d].iterrows()]
            if not cand_by_domain[d]:
                raise RuntimeError(f"No candidates for domain={d}, lambda={lam}")

        # Exhaustive search over domain assignments; small search space.
        best_combo = None
        best_score = None
        for combo in itertools.product(*[cand_by_domain[d] for d in domains]):
            rule_set = {_policy_id(r) for r in combo}
            if len(rule_set) > 2:
                continue
            util = [float(r["utility"]) for r in combo]
            mean_u = float(np.mean(util))
            min_u = float(np.min(util))
            # Tie-break with higher mean, higher min, and lower avg coverage.
            mean_cov = float(np.mean([float(r["coverage_fraction"]) for r in combo]))
            score = (mean_u, min_u, -mean_cov)
            if best_score is None or score > best_score:
                best_score = score
                best_combo = combo

        if best_combo is None:
            raise RuntimeError(f"No feasible two-rule assignment for lambda={lam}")

        rule_set = sorted({_policy_id(r) for r in best_combo})
        mean_u = float(np.mean([float(r["utility"]) for r in best_combo]))
        min_u = float(np.min([float(r["utility"]) for r in best_combo]))

        rows.append(
            {
                "lambda": lam,
                "n_rules": len(rule_set),
                "rule_1": f"{rule_set[0][0]}@{rule_set[0][1]:.2f}" if len(rule_set) >= 1 else "",
                "rule_2": f"{rule_set[1][0]}@{rule_set[1][1]:.2f}" if len(rule_set) >= 2 else "",
                "two_rule_mean_utility": mean_u,
                "two_rule_min_utility": min_u,
                "domain_best_mean_utility": domain_best_mean,
                "domain_best_min_utility": domain_best_min,
                "mean_utility_gap_two_rule_minus_domain_best": mean_u - domain_best_mean,
                "min_utility_gap_two_rule_minus_domain_best": min_u - domain_best_min,
            }
        )

        for row in best_combo:
            d = str(row["domain"])
            per_domain_rows.append(
                {
                    "lambda": lam,
                    "domain": d,
                    "assigned_calibration_method": row["calibration_method"],
                    "assigned_disagreement_threshold": row["disagreement_threshold"],
                    "assigned_utility": row["utility"],
                    "domain_best_utility": domain_best_map[d],
                    "utility_gap_assigned_minus_domain_best": row["utility"] - domain_best_map[d],
                }
            )

    summary_df = pd.DataFrame(rows).sort_values("lambda")
    assignment_df = pd.DataFrame(per_domain_rows).sort_values(["lambda", "domain"])
    pattern_df = (
        summary_df.groupby(["rule_1", "rule_2"], as_index=False)
        .agg(
            n_lambda=("lambda", "count"),
            lambda_min=("lambda", "min"),
            lambda_max=("lambda", "max"),
            mean_gap=("mean_utility_gap_two_rule_minus_domain_best", "mean"),
            min_gap=("mean_utility_gap_two_rule_minus_domain_best", "min"),
        )
        .sort_values(["n_lambda", "mean_gap"], ascending=[False, False])
    )

    summary_path = output_dir / "two_rule_policy_summary_by_lambda.csv"
    assignment_path = output_dir / "two_rule_policy_domain_assignments.csv"
    pattern_path = output_dir / "two_rule_policy_pattern_stability.csv"

    summary_df.to_csv(summary_path, index=False)
    assignment_df.to_csv(assignment_path, index=False)
    pattern_df.to_csv(pattern_path, index=False)

    print(summary_df.to_string(index=False))
    print(assignment_df.to_string(index=False))
    print(pattern_df.to_string(index=False))
    print(f"[done] {summary_path}")
    print(f"[done] {assignment_path}")
    print(f"[done] {pattern_path}")


if __name__ == "__main__":
    main()
