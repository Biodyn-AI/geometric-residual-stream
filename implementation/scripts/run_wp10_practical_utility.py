#!/usr/bin/env python3
"""WP10 - Practical-utility / retrieval evaluation.

Addresses Reviewer 2 comment 2 and the editor's practical-utility concern: show
HOW the geometric signal can be used, not only that it is statistically present.
For each domain we rank candidate edges by the geometry-augmented model score
and report top-k retrieval quality (precision@k, recall@k, enrichment over the
prevalence baseline), compared against the confound-only baseline ranking, plus
a calibrated operating point a practitioner would obtain.

Output: implementation/outputs/cycle49_practical_utility/retrieval_summary.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

import revision_lib as rl

OUT = rl.OUTPUTS / "cycle49_practical_utility"
KS = (25, 50, 100, 200)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for domain in rl.DOMAINS:
        dom = rl.load_domain(domain, 42)
        res = rl.evaluate(dom, "centered_cosine", list(range(dom.n_layers)),
                          cv_mode="edge", n_boot=200, boot_seed=9)
        y = res["_y"]
        base_p = res["_base_oof"]
        geom_p = res["_plus_oof"]
        prevalence = float(y.mean())

        def topk(scores, k):
            order = np.argsort(-scores)
            kk = min(k, len(y))
            hits = int(y[order][:kk].sum())
            return hits / kk, hits / int(y.sum()), (hits / kk) / prevalence

        rec = dict(domain=domain, n_edges=int(len(y)), n_pos=int(y.sum()),
                   prevalence=prevalence,
                   auroc_base=res["auroc_base"], auroc_geom=res["auroc_plus"],
                   auprc_base=res["auprc_base"], auprc_geom=res["auprc_plus"])
        for k in KS:
            pb, rb, eb = topk(base_p, k)
            pg, rg, eg = topk(geom_p, k)
            rec[f"base_precision@{k}"] = pb
            rec[f"geom_precision@{k}"] = pg
            rec[f"geom_recall@{k}"] = rg
            rec[f"geom_enrichment@{k}"] = eg
            rec[f"precision_gain@{k}"] = pg - pb

        # calibrated operating point: threshold maximising F1 on OOF geometry scores
        thr_grid = np.quantile(geom_p, np.linspace(0.05, 0.95, 37))
        best = max(thr_grid, key=lambda thr: f1_score(y, (geom_p >= thr).astype(int),
                                                      zero_division=0))
        pred = (geom_p >= best).astype(int)
        rec["operating_threshold"] = float(best)
        rec["operating_precision"] = float(precision_score(y, pred, zero_division=0))
        rec["operating_recall"] = float(recall_score(y, pred, zero_division=0))
        rec["operating_f1"] = float(f1_score(y, pred, zero_division=0))
        rows.append(rec)
        print(f"[{domain:13s}] AUROC base={rec['auroc_base']:.3f} geom={rec['auroc_geom']:.3f} "
              f"| prec@50 base={rec['base_precision@50']:.2f}->geom={rec['geom_precision@50']:.2f} "
              f"(enrich {rec['geom_enrichment@50']:.2f}x) "
              f"| op P/R={rec['operating_precision']:.2f}/{rec['operating_recall']:.2f}")
    pd.DataFrame(rows).to_csv(OUT / "retrieval_summary.csv", index=False)
    print(f"\n[done] {OUT / 'retrieval_summary.csv'}")


if __name__ == "__main__":
    main()
