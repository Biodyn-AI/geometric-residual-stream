# Environment Manifests

- Primary spec (portable): `environment.yml`
- Exact lock for this machine class (osx-arm64): `conda-explicit-osx-arm64.txt`
- Historical requested package set: `environment.from-history.yml`

Create/update from the portable spec:

```bash
conda env create -f implementation/environment/environment.yml
```

Recreate the exact osx-arm64 environment used in this run:

```bash
conda create -n subproject38-geo-v2 --file implementation/environment/conda-explicit-osx-arm64.txt
```
