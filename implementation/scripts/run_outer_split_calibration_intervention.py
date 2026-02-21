#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    subproject_root = script_dir.parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Calibration-only intervention on outer-split predictions using "
            "Platt and isotonic calibrators."
        )
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
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-calibration-bins", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle26_outer_split_calibration_intervention",
    )
    return parser.parse_args()


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int) -> tuple[float, list[dict]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    idx = np.digitize(p, bins[1:-1], right=False)
    total = max(y.shape[0], 1)
    ece = 0.0
    rows = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin_index": b,
                    "bin_left": float(bins[b]),
                    "bin_right": float(bins[b + 1]),
                    "n_edges": 0,
                    "mean_pred_prob": np.nan,
                    "empirical_positive_rate": np.nan,
                    "abs_calibration_gap": np.nan,
                }
            )
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        gap = abs(acc - conf)
        ece += (n / total) * gap
        rows.append(
            {
                "bin_index": b,
                "bin_left": float(bins[b]),
                "bin_right": float(bins[b + 1]),
                "n_edges": n,
                "mean_pred_prob": conf,
                "empirical_positive_rate": acc,
                "abs_calibration_gap": gap,
            }
        )
    return float(ece), rows


def _crossfit_platt(scores: np.ndarray, y: np.ndarray, cv_splits: int, cv_repeats: int, seed: int) -> np.ndarray:
    n = y.shape[0]
    out = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int32)
    n_pos = int(y.sum())
    n_neg = int(n - n_pos)
    n_splits = int(min(cv_splits, n_pos, n_neg))
    if n < 8 or n_pos < 2 or n_neg < 2 or n_splits < 2:
        model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=0)
        model.fit(scores.reshape(-1, 1), y)
        return model.predict_proba(scores.reshape(-1, 1))[:, 1]
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=cv_repeats, random_state=seed)
    for train_idx, test_idx in cv.split(scores.reshape(-1, 1), y):
        model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=0)
        model.fit(scores[train_idx].reshape(-1, 1), y[train_idx])
        out[test_idx] += model.predict_proba(scores[test_idx].reshape(-1, 1))[:, 1]
        counts[test_idx] += 1
    return out / np.clip(counts, 1, None)


def _crossfit_isotonic(scores: np.ndarray, y: np.ndarray, cv_splits: int, cv_repeats: int, seed: int) -> np.ndarray:
    n = y.shape[0]
    out = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int32)
    n_pos = int(y.sum())
    n_neg = int(n - n_pos)
    n_splits = int(min(cv_splits, n_pos, n_neg))
    if n < 8 or n_pos < 2 or n_neg < 2 or n_splits < 2:
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(scores, y)
        return model.transform(scores)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=cv_repeats, random_state=seed)
    for train_idx, test_idx in cv.split(scores.reshape(-1, 1), y):
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(scores[train_idx], y[train_idx])
        out[test_idx] += model.transform(scores[test_idx])
        counts[test_idx] += 1
    return out / np.clip(counts, 1, None)


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_df = pd.read_csv(args.oof_predictions_tsv, sep="\t")
    prob_cols = ["baseline_prob", "scgpt_prob", "geneformer_prob", "compact_prob"]

    summary_rows = []
    reliability_rows = []

    for domain, ddf in oof_df.groupby("domain"):
        y = ddf["label"].to_numpy(dtype=np.int32)
        for col in prob_cols:
            raw = ddf[col].to_numpy(dtype=np.float64)
            calibrated = {
                "raw": raw,
                "platt": _crossfit_platt(
                    scores=raw,
                    y=y,
                    cv_splits=args.cv_splits,
                    cv_repeats=args.cv_repeats,
                    seed=args.seed,
                ),
                "isotonic": _crossfit_isotonic(
                    scores=raw,
                    y=y,
                    cv_splits=args.cv_splits,
                    cv_repeats=args.cv_repeats,
                    seed=args.seed,
                ),
            }
            raw_auc = _safe_auc(y, calibrated["raw"])
            raw_brier = float(np.mean((calibrated["raw"] - y) ** 2))
            raw_ece, raw_bins = _ece(y=y, p=calibrated["raw"], n_bins=args.n_calibration_bins)
            summary_rows.append(
                {
                    "domain": domain,
                    "model": col.replace("_prob", ""),
                    "calibration": "raw",
                    "n_edges": int(y.shape[0]),
                    "n_positive": int(y.sum()),
                    "positive_rate": float(np.mean(y)),
                    "auc": raw_auc,
                    "brier_score": raw_brier,
                    "ece": raw_ece,
                    "delta_auc_vs_raw": 0.0,
                    "delta_brier_vs_raw": 0.0,
                    "delta_ece_vs_raw": 0.0,
                }
            )
            for b in raw_bins:
                reliability_rows.append(
                    {
                        "domain": domain,
                        "model": col.replace("_prob", ""),
                        "calibration": "raw",
                        **b,
                    }
                )

            for method in ["platt", "isotonic"]:
                p = calibrated[method]
                auc = _safe_auc(y, p)
                brier = float(np.mean((p - y) ** 2))
                ece, bins = _ece(y=y, p=p, n_bins=args.n_calibration_bins)
                summary_rows.append(
                    {
                        "domain": domain,
                        "model": col.replace("_prob", ""),
                        "calibration": method,
                        "n_edges": int(y.shape[0]),
                        "n_positive": int(y.sum()),
                        "positive_rate": float(np.mean(y)),
                        "auc": auc,
                        "brier_score": brier,
                        "ece": ece,
                        "delta_auc_vs_raw": auc - raw_auc,
                        "delta_brier_vs_raw": brier - raw_brier,
                        "delta_ece_vs_raw": ece - raw_ece,
                    }
                )
                for b in bins:
                    reliability_rows.append(
                        {
                            "domain": domain,
                            "model": col.replace("_prob", ""),
                            "calibration": method,
                            **b,
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)
    reliability_df = pd.DataFrame(reliability_rows)

    summary_path = output_dir / "outer_split_calibration_summary.csv"
    reliability_path = output_dir / "outer_split_calibration_reliability_bins.csv"
    summary_df.to_csv(summary_path, index=False)
    reliability_df.to_csv(reliability_path, index=False)

    print(summary_df.to_string(index=False))
    print(f"[done] {summary_path}")
    print(f"[done] {reliability_path}")


if __name__ == "__main__":
    main()
