#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create AnnData subsets by an obs column.")
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--obs-column", type=str, required=True)
    parser.add_argument("--values", nargs="+", required=True, help="Exact values from obs column to subset.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    adata = ad.read_h5ad(args.input_h5ad)
    if args.obs_column not in adata.obs.columns:
        raise KeyError(f"obs column '{args.obs_column}' not found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    obs_series = adata.obs[args.obs_column].astype(str)

    for value in args.values:
        mask = obs_series == value
        subset = adata[mask].copy()
        slug = _slugify(value)
        out_path = args.output_dir / f"{slug}.h5ad"
        subset.write_h5ad(out_path)
        print(f"[done] value='{value}' cells={subset.n_obs} genes={subset.n_vars} -> {out_path}")


if __name__ == "__main__":
    main()
