# Cross-Domain Seed-Ensemble Bundle Transfer (Cycle 21)

Date: 2026-02-20

## Goal
Test whether the multi-layer multi-seed scGPT bundling improvement (seen in external lung) transfers to immune and lung.

## Script and outputs
- Script: `implementation/scripts/run_cross_domain_seed_ensemble_bundle_transfer.py`
- Outputs:
  - `implementation/outputs/cycle21_cross_domain_seed_ensemble_bundle_transfer/cross_domain_seed_ensemble_bundle_detail.csv`
  - `implementation/outputs/cycle21_cross_domain_seed_ensemble_bundle_transfer/cross_domain_seed_ensemble_bundle_seed_aggregate.csv`

## Mean single-seed results (42/43/44)

### Immune

| Bundle | Mean scGPT delta | Mean GF-scGPT gap | Mean (Both - GF) |
|---|---:|---:|---:|
| L3 | 0.01774 | +0.03098 | +0.00003 |
| L1-4 | 0.03081 | +0.01791 | +0.00660 |
| L0-5 | 0.03262 | +0.01610 | +0.00613 |
| L0-11 | 0.05263 | **-0.00391** | +0.02610 |

### Lung

| Bundle | Mean scGPT delta | Mean GF-scGPT gap | Mean (Both - GF) |
|---|---:|---:|---:|
| L3 | 0.00326 | +0.02409 | -0.00033 |
| L1-4 | 0.01460 | +0.01275 | +0.00504 |
| L0-5 | 0.03054 | **-0.00319** | +0.01615 |
| L0-11 | 0.03521 | **-0.00787** | +0.02045 |

## Seed-ensemble highlights
- Immune (L0-11 ensemble):
  - scGPT delta: 0.06634
  - Geneformer delta: 0.05361
  - gap (GF - scGPT): -0.01273
  - combined delta: 0.08847
- Lung (L0-11 ensemble):
  - scGPT delta: 0.03937
  - Geneformer delta: 0.02813
  - gap (GF - scGPT): -0.01125
  - combined delta: 0.05276

## Interpretation
- The representation fix transfers: multi-layer bundling is not external-lung-specific.
- In both immune and lung, deep bundles (L0-11) close and reverse the Geneformer-minus-scGPT gap.
- Combined scGPT+Geneformer remains strongest, indicating cross-model complementarity persists after transfer.

## Conclusion
- The cross-model discrepancy is largely resolved by multi-layer, multi-seed scGPT geometry construction across domains, not just in external lung.
