# Results data dictionary

This directory preserves the machine-readable tables and figures included in the DOI-backed `v0.0.0` snapshot. Historical files are retained without rewriting their values.

## Tables

| Path | Role | Notes |
| --- | --- | --- |
| `tables/*_numpy.metrics.json` | NumPy per-system measurement | One stored run per workload |
| `tables/*_torch.metrics.json` | PyTorch per-system measurement | One stored run per workload |
| `tables/combined_from_metrics.csv` | Derived per-system table | Generated from the ten metric JSON files |
| `tables/timings.csv` | Legacy compact export | Headerless, four columns: run identifier, XYZ name, backend, elapsed seconds |
| `tables/timings_local.csv` | Local raw microbenchmarks | Three repetitions for NumPy-only, mixed, and full PyTorch paths |
| `tables/summary_local.csv` | Local grouped summary | Means and standard deviations derived from `timings_local.csv` |
| `tables/env_report.json` | DOI-era environment fragment | Python, PyTorch, CUDA, cuDNN, and CUDA-device fields only |

The per-system and local tables represent different experiment scopes and should not be merged into a single performance ranking.

## Figures

| Path | Role | Regeneration status |
| --- | --- | --- |
| `figures/wall_time_by_system.png` | DOI-era backend comparison | Regenerable in concept from metric JSON files, but preserved unchanged |
| `figures/wall_time_vs_species.png` | DOI-era species plot | Preserved unchanged; its scaling interpretation is not repeated as a current conclusion |
| `figures/prof_benzenes_callgraph.png` | DOI-era cProfile call graph | Underlying `.prof` file is not in the DOI snapshot |
| `figures/prof_ellipsoids_callgraph.png` | DOI-era cProfile call graph | Underlying `.prof` file is not in the DOI snapshot |

Current figure generation writes to `results/generated/` and does not overwrite these DOI-era artifacts.

## Validation

Run:

```bash
python scripts/validate_results.py
```

The validator checks JSON schemas, return codes, positive timings, consistency between metric JSON files and derived tables, and recomputes the local grouped summary.

See [`docs/METHODOLOGY.md`](../docs/METHODOLOGY.md) for interpretation boundaries and the exact reproducibility scope.
