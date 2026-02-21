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
            "Objective-conditioned policy selection over (calibration, disagreement threshold), "
            "with bootstrap uncertainty and cross-domain default policy."
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
    parser.add_argument("--thresholds", type=str, default="0.05,0.1,0.15,0.2,0.25,0.3")
    parser.add_argument("--calibration-methods", type=str, default="raw,platt,isotonic")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-calibration-bins", type=int, default=10)
    parser.add_argument("--ranking-ece-increase-cap", type=float, default=0.01)
    parser.add_argument("--reliability-auc-loss-cap", type=float, default=0.01)
    parser.add_argument("--min-coverage-fraction", type=float, default=0.05)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle29_objective_conditioned_policy_selection",
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


def _apply_calibration(
    raw: np.ndarray,
    y: np.ndarray,
    method: str,
    cv_splits: int,
    cv_repeats: int,
    seed: int,
) -> np.ndarray:
    if method == "raw":
        return raw
    if method == "platt":
        return _crossfit_platt(raw, y, cv_splits, cv_repeats, seed)
    if method == "isotonic":
        return _crossfit_isotonic(raw, y, cv_splits, cv_repeats, seed)
    raise ValueError(f"Unsupported calibration method: {method}")


def _bootstrap_policy_deltas(
    y: np.ndarray,
    policy_probs: np.ndarray,
    ref_probs: np.ndarray,
    n_bins: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float, float]:
    n = y.shape[0]
    deltas_auc = np.zeros(n_bootstrap, dtype=np.float64)
    deltas_ece = np.zeros(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        p_pol = policy_probs[idx]
        p_ref = ref_probs[idx]
        if np.unique(y_b).size < 2:
            deltas_auc[i] = np.nan
        else:
            deltas_auc[i] = _safe_auc(y_b, p_pol) - _safe_auc(y_b, p_ref)
        deltas_ece[i] = _ece(y_b, p_pol, n_bins) - _ece(y_b, p_ref, n_bins)

    deltas_auc = deltas_auc[np.isfinite(deltas_auc)]
    deltas_ece = deltas_ece[np.isfinite(deltas_ece)]
    if deltas_auc.size == 0:
        auc_lo = float("nan")
        auc_hi = float("nan")
        auc_std = float("nan")
    else:
        auc_lo = float(np.percentile(deltas_auc, 2.5))
        auc_hi = float(np.percentile(deltas_auc, 97.5))
        auc_std = float(np.std(deltas_auc))

    if deltas_ece.size == 0:
        ece_lo = float("nan")
        ece_hi = float("nan")
        ece_std = float("nan")
    else:
        ece_lo = float(np.percentile(deltas_ece, 2.5))
        ece_hi = float(np.percentile(deltas_ece, 97.5))
        ece_std = float(np.std(deltas_ece))

    return auc_lo, auc_hi, auc_std, ece_lo, ece_hi, ece_std


def _select_ranking_first(
    candidates: pd.DataFrame,
    ece_increase_cap: float,
    min_coverage_fraction: float,
) -> tuple[pd.Series, bool]:
    coverage_ok = candidates[candidates["coverage_fraction"] >= min_coverage_fraction]
    search = coverage_ok if not coverage_ok.empty else candidates
    feasible = search[search["delta_ece_vs_best_single"] <= ece_increase_cap]
    if feasible.empty:
        selected = search.sort_values(
            ["delta_auc_vs_best_single", "delta_ece_vs_best_single", "coverage_fraction"],
            ascending=[False, True, False],
        ).iloc[0]
        return selected, True
    selected = feasible.sort_values(
        ["delta_auc_vs_best_single", "delta_ece_vs_best_single", "coverage_fraction"],
        ascending=[False, True, False],
    ).iloc[0]
    return selected, False


def _select_reliability_first(
    candidates: pd.DataFrame,
    auc_loss_cap: float,
    min_coverage_fraction: float,
) -> tuple[pd.Series, bool]:
    coverage_ok = candidates[candidates["coverage_fraction"] >= min_coverage_fraction]
    search = coverage_ok if not coverage_ok.empty else candidates
    feasible = search[search["delta_auc_vs_best_single"] >= -auc_loss_cap]
    if feasible.empty:
        selected = search.sort_values(
            ["delta_ece_vs_best_single", "delta_auc_vs_best_single", "coverage_fraction"],
            ascending=[True, False, False],
        ).iloc[0]
        return selected, True
    selected = feasible.sort_values(
        ["delta_ece_vs_best_single", "delta_auc_vs_best_single", "coverage_fraction"],
        ascending=[True, False, False],
    ).iloc[0]
    return selected, False


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    calib_methods = [x.strip() for x in args.calibration_methods.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    oof = pd.read_csv(args.oof_predictions_tsv, sep="\t")

    candidate_rows = []
    selected_rows = []
    selected_boot_rows = []
    cached_arrays: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for domain, df in oof.groupby("domain"):
        y = df["label"].to_numpy(dtype=np.int32)
        raw_sc = df["scgpt_prob"].to_numpy(dtype=np.float64)
        raw_gf = df["geneformer_prob"].to_numpy(dtype=np.float64)
        raw_compact = df["compact_prob"].to_numpy(dtype=np.float64)

        for cal in calib_methods:
            sc = _apply_calibration(raw_sc, y, cal, args.cv_splits, args.cv_repeats, args.seed)
            gf = _apply_calibration(raw_gf, y, cal, args.cv_splits, args.cv_repeats, args.seed)
            compact = _apply_calibration(raw_compact, y, cal, args.cv_splits, args.cv_repeats, args.seed)
            cached_arrays[(domain, cal)] = {"y": y, "sc": sc, "gf": gf, "compact": compact}

            sc_auc = _safe_auc(y, sc)
            gf_auc = _safe_auc(y, gf)
            if sc_auc >= gf_auc:
                best_single_name = "scgpt"
                best_single_probs = sc
                best_single_auc = sc_auc
            else:
                best_single_name = "geneformer"
                best_single_probs = gf
                best_single_auc = gf_auc
            best_single_ece = _ece(y, best_single_probs, args.n_calibration_bins)
            best_single_brier = float(np.mean((best_single_probs - y) ** 2))

            abs_disagreement = np.abs(gf - sc)
            for tau in thresholds:
                referral = abs_disagreement >= tau
                policy_probs = np.where(referral, compact, best_single_probs)
                policy_auc = _safe_auc(y, policy_probs)
                policy_ece = _ece(y, policy_probs, args.n_calibration_bins)
                policy_brier = float(np.mean((policy_probs - y) ** 2))
                candidate_rows.append(
                    {
                        "domain": domain,
                        "calibration_method": cal,
                        "disagreement_threshold": tau,
                        "n_edges": int(y.shape[0]),
                        "n_positive": int(y.sum()),
                        "coverage_fraction": float(np.mean(referral)),
                        "n_referred_edges": int(referral.sum()),
                        "best_single_model": best_single_name,
                        "best_single_auc": best_single_auc,
                        "best_single_ece": best_single_ece,
                        "best_single_brier": best_single_brier,
                        "policy_auc": policy_auc,
                        "policy_ece": policy_ece,
                        "policy_brier": policy_brier,
                        "delta_auc_vs_best_single": policy_auc - best_single_auc,
                        "delta_ece_vs_best_single": policy_ece - best_single_ece,
                        "delta_brier_vs_best_single": policy_brier - best_single_brier,
                    }
                )

    candidates_df = pd.DataFrame(candidate_rows).sort_values(
        ["domain", "calibration_method", "disagreement_threshold"]
    )

    # Per-domain objective-conditioned policy selection.
    for domain, ddf in candidates_df.groupby("domain"):
        ranking_sel, ranking_fallback = _select_ranking_first(
            ddf, args.ranking_ece_increase_cap, args.min_coverage_fraction
        )
        reliability_sel, reliability_fallback = _select_reliability_first(
            ddf, args.reliability_auc_loss_cap, args.min_coverage_fraction
        )

        selected_rows.append(
            {
                "domain": domain,
                "objective": "ranking_first",
                "fallback_used": bool(ranking_fallback),
                "selection_constraint": (
                    f"delta_ece <= {args.ranking_ece_increase_cap}, coverage >= {args.min_coverage_fraction}"
                ),
                **ranking_sel.to_dict(),
            }
        )
        selected_rows.append(
            {
                "domain": domain,
                "objective": "reliability_first",
                "fallback_used": bool(reliability_fallback),
                "selection_constraint": (
                    f"delta_auc >= -{args.reliability_auc_loss_cap}, coverage >= {args.min_coverage_fraction}"
                ),
                **reliability_sel.to_dict(),
            }
        )

    selected_df = pd.DataFrame(selected_rows)

    # Bootstrap joint uncertainty for selected policies.
    for _, row in selected_df.iterrows():
        domain = str(row["domain"])
        cal = str(row["calibration_method"])
        tau = float(row["disagreement_threshold"])
        arr = cached_arrays[(domain, cal)]
        y = arr["y"]
        sc = arr["sc"]
        gf = arr["gf"]
        compact = arr["compact"]

        if str(row["best_single_model"]) == "scgpt":
            ref = sc
        else:
            ref = gf

        referral = np.abs(gf - sc) >= tau
        policy_probs = np.where(referral, compact, ref)
        auc_lo, auc_hi, auc_std, ece_lo, ece_hi, ece_std = _bootstrap_policy_deltas(
            y=y,
            policy_probs=policy_probs,
            ref_probs=ref,
            n_bins=args.n_calibration_bins,
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        )
        selected_boot_rows.append(
            {
                **row.to_dict(),
                "delta_auc_ci_lower": auc_lo,
                "delta_auc_ci_upper": auc_hi,
                "delta_auc_boot_std": auc_std,
                "delta_ece_ci_lower": ece_lo,
                "delta_ece_ci_upper": ece_hi,
                "delta_ece_boot_std": ece_std,
            }
        )

    selected_boot_df = pd.DataFrame(selected_boot_rows)

    # Cross-domain default policy candidate sweep.
    default_rows = []
    for cal in calib_methods:
        for tau in thresholds:
            cand = candidates_df[
                (candidates_df["calibration_method"] == cal)
                & (candidates_df["disagreement_threshold"] == tau)
            ]
            if cand.empty:
                continue
            default_rows.append(
                {
                    "calibration_method": cal,
                    "disagreement_threshold": tau,
                    "n_domains": int(cand["domain"].nunique()),
                    "mean_delta_auc": float(cand["delta_auc_vs_best_single"].mean()),
                    "min_delta_auc": float(cand["delta_auc_vs_best_single"].min()),
                    "mean_delta_ece": float(cand["delta_ece_vs_best_single"].mean()),
                    "max_delta_ece": float(cand["delta_ece_vs_best_single"].max()),
                    "mean_coverage_fraction": float(cand["coverage_fraction"].mean()),
                }
            )
    default_df = pd.DataFrame(default_rows).sort_values(
        ["mean_delta_auc", "min_delta_auc", "mean_delta_ece"],
        ascending=[False, False, True],
    )

    feasible_default = default_df[
        (default_df["min_delta_auc"] >= 0.0)
        & (default_df["max_delta_ece"] <= args.ranking_ece_increase_cap)
    ]
    if feasible_default.empty:
        default_selected = default_df.sort_values(
            ["min_delta_auc", "mean_delta_auc", "max_delta_ece"],
            ascending=[False, False, True],
        ).iloc[0]
        default_fallback = True
    else:
        default_selected = feasible_default.sort_values(
            ["mean_delta_auc", "max_delta_ece", "mean_coverage_fraction"],
            ascending=[False, True, False],
        ).iloc[0]
        default_fallback = False

    default_selected_df = pd.DataFrame(
        [
            {
                **default_selected.to_dict(),
                "default_fallback_used": bool(default_fallback),
                "default_constraint": (
                    f"min_delta_auc >= 0 and max_delta_ece <= {args.ranking_ece_increase_cap}"
                ),
            }
        ]
    )

    # Compare cross-domain default vs domain-specific ranking-first selection.
    compare_rows = []
    default_cal = str(default_selected["calibration_method"])
    default_tau = float(default_selected["disagreement_threshold"])
    for domain, ddf in candidates_df.groupby("domain"):
        default_row = ddf[
            (ddf["calibration_method"] == default_cal) & (ddf["disagreement_threshold"] == default_tau)
        ].iloc[0]
        ranking_row = selected_df[
            (selected_df["domain"] == domain) & (selected_df["objective"] == "ranking_first")
        ].iloc[0]
        compare_rows.append(
            {
                "domain": domain,
                "default_calibration_method": default_cal,
                "default_disagreement_threshold": default_tau,
                "default_delta_auc": float(default_row["delta_auc_vs_best_single"]),
                "default_delta_ece": float(default_row["delta_ece_vs_best_single"]),
                "default_coverage_fraction": float(default_row["coverage_fraction"]),
                "domain_specific_calibration_method": str(ranking_row["calibration_method"]),
                "domain_specific_disagreement_threshold": float(ranking_row["disagreement_threshold"]),
                "domain_specific_delta_auc": float(ranking_row["delta_auc_vs_best_single"]),
                "domain_specific_delta_ece": float(ranking_row["delta_ece_vs_best_single"]),
                "domain_specific_coverage_fraction": float(ranking_row["coverage_fraction"]),
                "delta_auc_gap_default_minus_domain_specific": float(
                    default_row["delta_auc_vs_best_single"] - ranking_row["delta_auc_vs_best_single"]
                ),
            }
        )
    compare_df = pd.DataFrame(compare_rows)

    candidates_path = output_dir / "policy_candidates.csv"
    selected_path = output_dir / "policy_selected_by_objective.csv"
    selected_boot_path = output_dir / "policy_selected_with_bootstrap.csv"
    default_grid_path = output_dir / "cross_domain_default_policy_grid.csv"
    default_selected_path = output_dir / "cross_domain_default_policy_selected.csv"
    compare_path = output_dir / "cross_domain_default_vs_domain_specific.csv"

    candidates_df.to_csv(candidates_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    selected_boot_df.to_csv(selected_boot_path, index=False)
    default_df.to_csv(default_grid_path, index=False)
    default_selected_df.to_csv(default_selected_path, index=False)
    compare_df.to_csv(compare_path, index=False)

    print(selected_df.to_string(index=False))
    print(selected_boot_df.to_string(index=False))
    print(default_selected_df.to_string(index=False))
    print(compare_df.to_string(index=False))
    print(f"[done] {candidates_path}")
    print(f"[done] {selected_path}")
    print(f"[done] {selected_boot_path}")
    print(f"[done] {default_grid_path}")
    print(f"[done] {default_selected_path}")
    print(f"[done] {compare_path}")


if __name__ == "__main__":
    main()
