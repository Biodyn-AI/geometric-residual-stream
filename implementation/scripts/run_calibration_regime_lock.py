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
            "Calibration regime lock analysis: compare raw vs isotonic (and optional platt) "
            "under a fixed referral threshold with bootstrap uncertainty."
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
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--methods", type=str, default="raw,isotonic,platt")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-calibration-bins", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle33_calibration_regime_lock",
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
    for tr, te in cv.split(scores.reshape(-1, 1), y):
        model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=0)
        model.fit(scores[tr].reshape(-1, 1), y[tr])
        out[te] += model.predict_proba(scores[te].reshape(-1, 1))[:, 1]
        counts[te] += 1
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
    for tr, te in cv.split(scores.reshape(-1, 1), y):
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(scores[tr], y[tr])
        out[te] += model.transform(scores[te])
        counts[te] += 1
    return out / np.clip(counts, 1, None)


def _apply_method(
    scores: np.ndarray, y: np.ndarray, method: str, cv_splits: int, cv_repeats: int, seed: int
) -> np.ndarray:
    if method == "raw":
        return scores
    if method == "platt":
        return _crossfit_platt(scores, y, cv_splits, cv_repeats, seed)
    if method == "isotonic":
        return _crossfit_isotonic(scores, y, cv_splits, cv_repeats, seed)
    raise ValueError(f"Unsupported method: {method}")


def _bootstrap_diff(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_bins: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float, float]:
    n = y.shape[0]
    d_auc = np.zeros(n_bootstrap, dtype=np.float64)
    d_ece = np.zeros(n_bootstrap, dtype=np.float64)
    d_brier = np.zeros(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        pa = p_a[idx]
        pb = p_b[idx]
        if np.unique(yb).size < 2:
            d_auc[i] = np.nan
        else:
            d_auc[i] = _safe_auc(yb, pa) - _safe_auc(yb, pb)
        d_ece[i] = _ece(yb, pa, n_bins) - _ece(yb, pb, n_bins)
        d_brier[i] = float(np.mean((pa - yb) ** 2) - np.mean((pb - yb) ** 2))

    out = {}
    for name, arr in [("auc", d_auc), ("ece", d_ece), ("brier", d_brier)]:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            out[f"{name}_ci_lower"] = float("nan")
            out[f"{name}_ci_upper"] = float("nan")
            out[f"{name}_boot_std"] = float("nan")
        else:
            out[f"{name}_ci_lower"] = float(np.percentile(arr, 2.5))
            out[f"{name}_ci_upper"] = float(np.percentile(arr, 97.5))
            out[f"{name}_boot_std"] = float(np.std(arr))
    return (
        out["auc_ci_lower"],
        out["auc_ci_upper"],
        out["auc_boot_std"],
        out["ece_ci_lower"],
        out["ece_ci_upper"],
        out["ece_boot_std"],
        out["brier_ci_lower"],
        out["brier_ci_upper"],
        out["brier_boot_std"],
    )


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if "raw" not in methods:
        raise ValueError("Method list must include 'raw' for regime lock comparison.")
    rng = np.random.default_rng(args.seed)

    oof = pd.read_csv(args.oof_predictions_tsv, sep="\t")
    metrics_rows = []
    policy_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    for domain, df in oof.groupby("domain"):
        y = df["label"].to_numpy(dtype=np.int32)
        raw_sc = df["scgpt_prob"].to_numpy(dtype=np.float64)
        raw_gf = df["geneformer_prob"].to_numpy(dtype=np.float64)
        raw_compact = df["compact_prob"].to_numpy(dtype=np.float64)

        for method in methods:
            sc = _apply_method(raw_sc, y, method, args.cv_splits, args.cv_repeats, args.seed)
            gf = _apply_method(raw_gf, y, method, args.cv_splits, args.cv_repeats, args.seed)
            compact = _apply_method(raw_compact, y, method, args.cv_splits, args.cv_repeats, args.seed)

            sc_auc = _safe_auc(y, sc)
            gf_auc = _safe_auc(y, gf)
            if sc_auc >= gf_auc:
                best_name = "scgpt"
                best = sc
            else:
                best_name = "geneformer"
                best = gf

            referral = np.abs(gf - sc) >= args.threshold
            policy = np.where(referral, compact, best)

            policy_cache[(domain, method)] = (y, policy)
            metrics_rows.append(
                {
                    "domain": domain,
                    "method": method,
                    "threshold": args.threshold,
                    "coverage_fraction": float(np.mean(referral)),
                    "best_single_model": best_name,
                    "best_single_auc": _safe_auc(y, best),
                    "best_single_ece": _ece(y, best, args.n_calibration_bins),
                    "best_single_brier": float(np.mean((best - y) ** 2)),
                    "policy_auc": _safe_auc(y, policy),
                    "policy_ece": _ece(y, policy, args.n_calibration_bins),
                    "policy_brier": float(np.mean((policy - y) ** 2)),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)

    compare_rows = []
    pair_targets = [("isotonic", "raw")]
    if "platt" in methods:
        pair_targets.append(("platt", "raw"))
    for domain in sorted(oof["domain"].unique()):
        for a, b in pair_targets:
            if (domain, a) not in policy_cache or (domain, b) not in policy_cache:
                continue
            y_a, p_a = policy_cache[(domain, a)]
            y_b, p_b = policy_cache[(domain, b)]
            if not np.array_equal(y_a, y_b):
                raise RuntimeError("Label arrays do not align for regime comparison.")
            y = y_a
            d_auc = _safe_auc(y, p_a) - _safe_auc(y, p_b)
            d_ece = _ece(y, p_a, args.n_calibration_bins) - _ece(y, p_b, args.n_calibration_bins)
            d_brier = float(np.mean((p_a - y) ** 2) - np.mean((p_b - y) ** 2))
            (
                auc_lo,
                auc_hi,
                auc_std,
                ece_lo,
                ece_hi,
                ece_std,
                brier_lo,
                brier_hi,
                brier_std,
            ) = _bootstrap_diff(
                y=y,
                p_a=p_a,
                p_b=p_b,
                n_bins=args.n_calibration_bins,
                n_bootstrap=args.n_bootstrap,
                rng=rng,
            )
            compare_rows.append(
                {
                    "domain": domain,
                    "comparison": f"{a}_minus_{b}",
                    "threshold": args.threshold,
                    "delta_policy_auc": d_auc,
                    "delta_policy_ece": d_ece,
                    "delta_policy_brier": d_brier,
                    "delta_policy_auc_ci_lower": auc_lo,
                    "delta_policy_auc_ci_upper": auc_hi,
                    "delta_policy_auc_boot_std": auc_std,
                    "delta_policy_ece_ci_lower": ece_lo,
                    "delta_policy_ece_ci_upper": ece_hi,
                    "delta_policy_ece_boot_std": ece_std,
                    "delta_policy_brier_ci_lower": brier_lo,
                    "delta_policy_brier_ci_upper": brier_hi,
                    "delta_policy_brier_boot_std": brier_std,
                }
            )

    compare_df = pd.DataFrame(compare_rows)

    metrics_path = output_dir / "calibration_regime_policy_metrics.csv"
    compare_path = output_dir / "calibration_regime_lock_comparisons.csv"
    metrics_df.to_csv(metrics_path, index=False)
    compare_df.to_csv(compare_path, index=False)

    print(metrics_df.to_string(index=False))
    print(compare_df.to_string(index=False))
    print(f"[done] {metrics_path}")
    print(f"[done] {compare_path}")


if __name__ == "__main__":
    main()
