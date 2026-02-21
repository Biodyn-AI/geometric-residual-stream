# Residual-Stream Geometry of Single-Cell Foundation Models Encodes Gene Regulatory Structure Across Tissues

This repository contains code, analysis reports, and manuscript source for the paper:

> **Residual-Stream Geometry of Single-Cell Foundation Models Encodes Gene Regulatory Structure Across Tissues**
> Ihor Kendiukhov (University of Tubingen)
> *Submitted to BMC Bioinformatics*

## Overview

We systematically investigate whether the geometric arrangement of gene vectors in the residual stream of single-cell foundation models (scGPT, Geneformer) predicts transcription factor--target regulatory edges from the TRRUST database, beyond expression-level confounds.

**Key findings:**
- Residual-stream geometry provides statistically significant incremental signal for regulatory edge classification across four Tabula Sapiens tissue contexts
- Centered-cosine similarity and PCA projection recover signal in initially weak domains
- scGPT and Geneformer encode partially complementary regulatory information (edge-level rho ~ 0.47)
- Multi-layer representation bundling eliminates an apparent scGPT disadvantage, showing regulatory information is distributed across transformer depth
- A compact stacking model yields consistent improvements under leakage-resistant nested cross-validation

## Repository Structure

```
.
├── paper/                      # Manuscript source
│   ├── geometric_residual_stream_bmc.tex   # BMC Bioinformatics formatted manuscript
│   ├── cover_letter.tex                     # Cover letter
│   └── figures/                             # Publication figures (PDF)
├── implementation/
│   ├── scripts/                # All analysis scripts (Python)
│   ├── configs/                # Deployment configuration
│   └── environment/            # Conda environment specifications
├── reports/                    # Detailed per-cycle analysis reports
└── planning/                   # Research plan
```

## Reproduction

### Environment setup

```bash
conda env create -f implementation/environment/environment.yml
conda activate subproject38-geo-v2
```

### Data requirements

The following external datasets are required (not included due to size):

- **Tabula Sapiens**: https://tabula-sapiens-portal.ds.czbiohub.org/
- **TRRUST v2**: https://www.grnpedia.org/trrust/
- **scGPT**: https://github.com/bowang-lab/scGPT (pre-trained whole-human model)
- **Geneformer**: https://huggingface.co/ctheodoris/Geneformer

### Running experiments

See `implementation/README.md` for full reproduction commands covering all 42 experimental cycles. Example for the initial geometric signal detection:

```bash
python implementation/scripts/run_layerwise_geometry_audit.py \
  --max-cells 256 --max-genes 512 --batch-size 4 \
  --cv-splits 5 --cv-repeats 3 --bootstrap-iters 400 \
  --seed 42 --output-dir implementation/outputs/cycle1_main
```

## Analysis Reports

The `reports/` directory contains detailed analysis reports for each experimental cycle, documenting methodology, results, and interpretations. These reports provide full transparency into the iterative research process.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code or find the results useful, please cite:

```bibtex
@article{kendiukhov2025residual,
  title={Residual-Stream Geometry of Single-Cell Foundation Models Encodes Gene Regulatory Structure Across Tissues},
  author={Kendiukhov, Ihor},
  journal={BMC Bioinformatics},
  year={2025},
  note={Under review}
}
```
