# External-Lung Seed-Ensemble Bundle Test (Cycle 20)

Date: 2026-02-20

## Goal
Test whether multi-seed averaging further stabilizes and improves the external-lung multi-layer scGPT bundle result.

## Script and outputs
- Script: `implementation/scripts/run_external_lung_seed_ensemble_bundle_test.py`
- Outputs:
  - `implementation/outputs/cycle20_external_lung_seed_ensemble_bundle_test/external_lung_seed_ensemble_bundle_detail.csv`
  - `implementation/outputs/cycle20_external_lung_seed_ensemble_bundle_test/external_lung_seed_ensemble_bundle_seed_aggregate.csv`

## Key results

### Mean across single-seed runs (42/43/44)

| Bundle | Mean scGPT delta | Mean GF-scGPT gap | Mean (Both - GF) |
|---|---:|---:|---:|
| L3 | 0.00305 | +0.02464 | +0.00070 |
| L1-4 | 0.01585 | +0.01184 | +0.00643 |
| L0-5 | 0.02268 | +0.00501 | +0.01170 |
| L0-11 | 0.02633 | +0.00135 | +0.01461 |

### Seed-ensemble (mean feature across seeds)
- L0-11 bundle:
  - scGPT delta: 0.02578
  - Geneformer delta: 0.02438
  - gap (GF - scGPT): **-0.00140** (scGPT slightly ahead)
  - combined model (`scGPT + Geneformer`) delta: 0.03958

## Interpretation
- Multi-layer scGPT gains are robust across seeds.
- The single-layer external-lung gap is mostly removed by layer bundling and effectively closed under seed-ensemble bundling.
- Combined scGPT+Geneformer remains strongest, indicating persistent complementarity even after scGPT improves.

## Conclusion
- External-lung divergence is largely a representation-construction issue (single layer vs multi-layer/ensemble), not an intrinsic inability of scGPT geometry to capture the signal.
