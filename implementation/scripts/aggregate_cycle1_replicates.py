#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent
    default_outputs = subproject_root / "implementation" / "outputs"

    parser = argparse.ArgumentParser(description="Aggregate cycle1 replicate metrics across seeds")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            str(default_outputs / "cycle1_main" / "cycle1_layer_geometry_metrics.csv"),
            str(default_outputs / "cycle1_seed43" / "cycle1_layer_geometry_metrics.csv"),
            str(default_outputs / "cycle1_seed44" / "cycle1_layer_geometry_metrics.csv"),
        ],
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=default_outputs / "cycle1_aggregate" / "cycle1_layer_aggregate.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=default_outputs / "cycle1_aggregate" / "cycle1_aggregate_summary.md",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    frames = []
    for idx, path_str in enumerate(args.inputs):
        path = Path(path_str)
        df = pd.read_csv(path)
        df["replicate"] = idx
        df["source_file"] = str(path)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    grouped = (
        all_df.groupby("layer", as_index=False)
        .agg(
            n_replicates=("replicate", "nunique"),
            delta_cv_auroc_mean=("delta_cv_auroc", "mean"),
            delta_cv_auroc_std=("delta_cv_auroc", "std"),
            delta_cv_auroc_min=("delta_cv_auroc", "min"),
            delta_cv_auroc_max=("delta_cv_auroc", "max"),
            baseline_cv_auroc_mean=("baseline_cv_auroc", "mean"),
            baseline_plus_geom_cv_auroc_mean=("baseline_plus_geom_cv_auroc", "mean"),
            raw_geom_auroc_mean=("raw_geom_auroc", "mean"),
            n_pairs_mean=("n_pairs", "mean"),
            n_positive_mean=("n_positive", "mean"),
        )
        .sort_values("layer")
        .reset_index(drop=True)
    )

    # Robustness proxy: how often does bootstrap lower bound stay above 0 across replicates?
    ci_robust = (
        all_df.assign(ci_positive=lambda d: d["delta_auc_bootstrap_ci_lo"] > 0)
        .groupby("layer", as_index=False)["ci_positive"]
        .mean()
        .rename(columns={"ci_positive": "frac_replicates_ci_lower_gt_zero"})
    )

    out_df = grouped.merge(ci_robust, on="layer", how="left")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    best_row = out_df.loc[out_df["delta_cv_auroc_mean"].idxmax()]

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        "\n".join(
            [
                "# Cycle 1 Replicate Aggregate",
                "",
                f"- Replicates aggregated: {all_df['replicate'].nunique()}",
                f"- Layers aggregated: {out_df.shape[0]}",
                "",
                "## Best Layer by Mean Delta CV AUROC",
                f"- Layer: {int(best_row['layer'])}",
                f"- Mean delta CV AUROC: {best_row['delta_cv_auroc_mean']:.6f}",
                f"- Delta CV AUROC std: {best_row['delta_cv_auroc_std']:.6f}",
                f"- Replicate range: [{best_row['delta_cv_auroc_min']:.6f}, {best_row['delta_cv_auroc_max']:.6f}]",
                f"- Fraction replicates with bootstrap lower CI > 0: {best_row['frac_replicates_ci_lower_gt_zero']:.2f}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[done] {args.output_csv}")
    print(f"[done] {args.output_md}")


if __name__ == "__main__":
    main()
