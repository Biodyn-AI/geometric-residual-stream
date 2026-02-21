# Cycle 2 Sensitivity: Max Genes 512 vs 1024

Date: 2026-02-20

Comparison setup:
- Same protocol and seed (42).
- Only changed `max_genes`: 512 -> 1024.

Top-layer outcomes:
- `max_genes=512`: best L5, delta CV AUROC +0.0642, 95% CI [0.0183, 0.1108].
- `max_genes=1024`: best L4, delta CV AUROC +0.0488, 95% CI [0.0037, 0.0910].

Interpretation:
- Effect size attenuates when allowing longer per-cell token sequences.
- Positive incremental value remains, but with weaker margin.
- This suggests geometric signal is not purely a truncation artifact, while also indicating sensitivity to token budget and/or representation dilution.
