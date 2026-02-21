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
            "Replay selected deployment policies on held-out rows and verify deterministic "
            "conformance against the published dual-mode policy table."
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
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-calibration-bins", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=subproject_root / "implementation" / "outputs" / "cycle37_deployment_policy_conformance",
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
    raw_scores: np.ndarray,
    y: np.ndarray,
    method: str,
    cv_splits: int,
    cv_repeats: int,
    seed: int,
) -> np.ndarray:
    if method == "raw":
        return raw_scores
    if method == "platt":
        return _crossfit_platt(raw_scores, y, cv_splits, cv_repeats, seed)
    if method == "isotonic":
        return _crossfit_isotonic(raw_scores, y, cv_splits, cv_repeats, seed)
    raise ValueError(f"Unsupported calibration method: {method}")


def _run_policy_vectorized(
    sc: np.ndarray,
    gf: np.ndarray,
    compact: np.ndarray,
    best_single: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    referral = np.abs(gf - sc) >= threshold
    policy = np.where(referral, compact, best_single)
    return policy, referral


def _run_policy_rowwise(
    sc: np.ndarray,
    gf: np.ndarray,
    compact: np.ndarray,
    best_single: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    # This mirrors production serving logic in an explicit per-row loop so we
    # can check that vectorized analysis code and row-wise service logic match.
    n = sc.shape[0]
    policy = np.zeros(n, dtype=np.float64)
    referral = np.zeros(n, dtype=bool)
    for i in range(n):
        if abs(gf[i] - sc[i]) >= threshold:
            policy[i] = compact[i]
            referral[i] = True
        else:
            policy[i] = best_single[i]
    return policy, referral


def _build_schedule_integrity(schedule_df: pd.DataFrame) -> pd.DataFrame:
    s = schedule_df.sort_values(["lambda_start", "lambda_end"]).reset_index(drop=True).copy()
    rows = []
    prev_end = None
    eps = 1e-12
    for i, row in s.iterrows():
        start = float(row["lambda_start"])
        end = float(row["lambda_end"])
        if end < start:
            raise ValueError(f"Invalid schedule segment {i}: lambda_end < lambda_start.")
        gap = 0.0 if prev_end is None else max(0.0, start - prev_end)
        overlap = 0.0 if prev_end is None else max(0.0, prev_end - start)
        rows.append(
            {
                "segment_index": int(i),
                "lambda_start": start,
                "lambda_end": end,
                "rule": str(row["rule"]),
                "calibration_method": str(row["calibration_method"]),
                "disagreement_threshold": float(row["disagreement_threshold"]),
                "gap_from_previous": float(gap),
                "overlap_with_previous": float(overlap),
                "has_gap": bool(gap > eps),
                "has_overlap": bool(overlap > eps),
            }
        )
        prev_end = end
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_df = pd.read_csv(args.oof_predictions_tsv, sep="\t")
    dual_mode_df = pd.read_csv(args.dual_mode_policy_csv)
    schedule_df = pd.read_csv(args.cost_schedule_csv)

    required_mode_cols = {
        "domain",
        "mode",
        "method",
        "threshold",
        "coverage_fraction",
        "policy_auc",
        "policy_ece",
        "policy_brier",
    }
    if not required_mode_cols.issubset(set(dual_mode_df.columns)):
        missing = sorted(required_mode_cols - set(dual_mode_df.columns))
        raise ValueError(f"Missing required columns in dual-mode file: {missing}")

    methods_needed = sorted(set(dual_mode_df["method"].astype(str)).union(set(schedule_df["calibration_method"])))

    # Cache calibrated score arrays per (domain, method) once, then reuse.
    cache: dict[tuple[str, str], dict[str, np.ndarray | str]] = {}
    for domain, df_domain in oof_df.groupby("domain"):
        y = df_domain["label"].to_numpy(dtype=np.int32)
        raw_sc = df_domain["scgpt_prob"].to_numpy(dtype=np.float64)
        raw_gf = df_domain["geneformer_prob"].to_numpy(dtype=np.float64)
        raw_compact = df_domain["compact_prob"].to_numpy(dtype=np.float64)

        for method in methods_needed:
            sc = _apply_method(raw_sc, y, str(method), args.cv_splits, args.cv_repeats, args.seed)
            gf = _apply_method(raw_gf, y, str(method), args.cv_splits, args.cv_repeats, args.seed)
            compact = _apply_method(raw_compact, y, str(method), args.cv_splits, args.cv_repeats, args.seed)

            sc_auc = _safe_auc(y, sc)
            gf_auc = _safe_auc(y, gf)
            if sc_auc >= gf_auc:
                best_single_model = "scgpt"
                best_single = sc
            else:
                best_single_model = "geneformer"
                best_single = gf

            cache[(str(domain), str(method))] = {
                "y": y,
                "sc": sc,
                "gf": gf,
                "compact": compact,
                "best_single": best_single,
                "best_single_model": best_single_model,
            }

    mode_rows = []
    equiv_rows = []
    replay_rows = []

    for _, mode_row in dual_mode_df.iterrows():
        domain = str(mode_row["domain"])
        mode = str(mode_row["mode"])
        method = str(mode_row["method"])
        threshold = float(mode_row["threshold"])

        arrays = cache[(domain, method)]
        y = arrays["y"]
        sc = arrays["sc"]
        gf = arrays["gf"]
        compact = arrays["compact"]
        best_single = arrays["best_single"]
        best_single_model = str(arrays["best_single_model"])

        policy_vec, referral_vec = _run_policy_vectorized(sc, gf, compact, best_single, threshold)
        policy_row, referral_row = _run_policy_rowwise(sc, gf, compact, best_single, threshold)

        prob_abs_diff = np.abs(policy_vec - policy_row)
        prob_mismatch = int(np.sum(prob_abs_diff > 1e-12))
        referral_mismatch = int(np.sum(referral_vec != referral_row))

        coverage = float(np.mean(referral_vec))
        auc = _safe_auc(y, policy_vec)
        ece = _ece(y, policy_vec, args.n_calibration_bins)
        brier = float(np.mean((policy_vec - y) ** 2))

        mode_rows.append(
            {
                "domain": domain,
                "mode": mode,
                "method": method,
                "threshold": threshold,
                "best_single_model": best_single_model,
                "n_edges": int(y.shape[0]),
                "coverage_fraction_observed": coverage,
                "coverage_fraction_expected": float(mode_row["coverage_fraction"]),
                "coverage_fraction_diff": coverage - float(mode_row["coverage_fraction"]),
                "policy_auc_observed": auc,
                "policy_auc_expected": float(mode_row["policy_auc"]),
                "policy_auc_diff": auc - float(mode_row["policy_auc"]),
                "policy_ece_observed": ece,
                "policy_ece_expected": float(mode_row["policy_ece"]),
                "policy_ece_diff": ece - float(mode_row["policy_ece"]),
                "policy_brier_observed": brier,
                "policy_brier_expected": float(mode_row["policy_brier"]),
                "policy_brier_diff": brier - float(mode_row["policy_brier"]),
            }
        )

        equiv_rows.append(
            {
                "domain": domain,
                "mode": mode,
                "method": method,
                "threshold": threshold,
                "n_edges": int(y.shape[0]),
                "prob_max_abs_diff": float(np.max(prob_abs_diff) if prob_abs_diff.size else 0.0),
                "prob_mean_abs_diff": float(np.mean(prob_abs_diff) if prob_abs_diff.size else 0.0),
                "prob_mismatch_count": prob_mismatch,
                "referral_mismatch_count": referral_mismatch,
                "exact_match": bool(prob_mismatch == 0 and referral_mismatch == 0),
            }
        )

        replay_rows.append(
            pd.DataFrame(
                {
                    "domain": domain,
                    "mode": mode,
                    "method": method,
                    "threshold": threshold,
                    "label": y.astype(np.int32),
                    "policy_prob": policy_vec.astype(np.float64),
                    "referral_flag": referral_vec.astype(np.int8),
                    "best_single_model": best_single_model,
                }
            )
        )

    mode_metrics_df = pd.DataFrame(mode_rows).sort_values(["mode", "domain"]).reset_index(drop=True)
    row_equiv_df = pd.DataFrame(equiv_rows).sort_values(["mode", "domain"]).reset_index(drop=True)
    replay_df = pd.concat(replay_rows, axis=0, ignore_index=True)

    schedule_integrity_df = _build_schedule_integrity(schedule_df)

    rule_rows = []
    seen_rules: set[tuple[str, float]] = set()
    for _, sched_row in schedule_df.iterrows():
        rule_key = (str(sched_row["calibration_method"]), float(sched_row["disagreement_threshold"]))
        if rule_key in seen_rules:
            continue
        seen_rules.add(rule_key)

        method = rule_key[0]
        threshold = rule_key[1]
        per_domain = []
        for domain in sorted(oof_df["domain"].unique()):
            arrays = cache[(str(domain), method)]
            y = arrays["y"]
            sc = arrays["sc"]
            gf = arrays["gf"]
            compact = arrays["compact"]
            best_single = arrays["best_single"]
            policy, referral = _run_policy_vectorized(sc, gf, compact, best_single, threshold)
            per_domain.append(
                {
                    "domain": str(domain),
                    "policy_auc": _safe_auc(y, policy),
                    "policy_ece": _ece(y, policy, args.n_calibration_bins),
                    "policy_brier": float(np.mean((policy - y) ** 2)),
                    "coverage_fraction": float(np.mean(referral)),
                }
            )
        d = pd.DataFrame(per_domain)
        rule_rows.append(
            {
                "rule": f"{method}@{threshold:.2f}",
                "calibration_method": method,
                "disagreement_threshold": threshold,
                "n_domains": int(d.shape[0]),
                "mean_policy_auc": float(d["policy_auc"].mean()),
                "mean_policy_ece": float(d["policy_ece"].mean()),
                "mean_policy_brier": float(d["policy_brier"].mean()),
                "mean_coverage_fraction": float(d["coverage_fraction"].mean()),
                "min_policy_auc": float(d["policy_auc"].min()),
                "max_policy_ece": float(d["policy_ece"].max()),
            }
        )
    schedule_rule_df = pd.DataFrame(rule_rows).sort_values("rule").reset_index(drop=True)

    mode_metrics_path = output_dir / "conformance_mode_metrics.csv"
    row_equiv_path = output_dir / "conformance_row_equivalence.csv"
    replay_path = output_dir / "deployment_replay_predictions.tsv"
    schedule_integrity_path = output_dir / "cost_schedule_integrity.csv"
    schedule_rule_path = output_dir / "cost_schedule_rule_replay_metrics.csv"

    mode_metrics_df.to_csv(mode_metrics_path, index=False)
    row_equiv_df.to_csv(row_equiv_path, index=False)
    replay_df.to_csv(replay_path, sep="\t", index=False)
    schedule_integrity_df.to_csv(schedule_integrity_path, index=False)
    schedule_rule_df.to_csv(schedule_rule_path, index=False)

    print(mode_metrics_df.to_string(index=False))
    print(row_equiv_df.to_string(index=False))
    print(schedule_integrity_df.to_string(index=False))
    print(schedule_rule_df.to_string(index=False))
    print(f"[done] {mode_metrics_path}")
    print(f"[done] {row_equiv_path}")
    print(f"[done] {replay_path}")
    print(f"[done] {schedule_integrity_path}")
    print(f"[done] {schedule_rule_path}")


if __name__ == "__main__":
    main()
