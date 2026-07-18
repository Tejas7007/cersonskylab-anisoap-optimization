# Methodology and interpretation

## Study status

This repository records a Fall 2025 profiling and optimization study of AniSOAP execution paths. The DOI-backed release is preserved by the `v0.0.0` tag and Zenodo DOI `10.5281/zenodo.17519888`.

The repository is a historical evidence package. It is not the maintained AniSOAP acceleration implementation. Current batching, PyTorch, Rust, numerical-parity, and benchmark work lives in [`AniSOAP-Torch`](https://github.com/Tejas7007/AniSOAP-Torch).

## Evidence families

The committed artifacts contain two experiment families with different scopes.

### 1. Per-system backend metrics

The ten `results/tables/*.metrics.json` files store one elapsed-time measurement for NumPy and PyTorch paths across five workloads:

- `one_species`
- `benzenes`
- `three_species`
- `four_species`
- `ellipsoids`

Each record includes:

- workload filename;
- backend mode;
- return code;
- elapsed wall time;
- maximum resident-set size;
- input and output block counts.

The derived `combined_from_metrics.csv` and legacy `timings.csv` summarize the same values. In these committed measurements, PyTorch is faster for four workloads and slightly slower for ellipsoids. The corresponding NumPy-to-PyTorch time ratios are approximately 3.60, 1.85, 2.15, 1.13, and 0.97.

These records do not preserve the complete execution environment, input structures, benchmark runner, warm-up policy, or repeated-trial distribution. They support a narrow comparison of the stored runs, not a universal backend recommendation.

### 2. Local operation microbenchmarks

`timings_local.csv` stores three repetitions for `numpy_only`, `torch_mixed`, and `torch_full` paths on five small workloads. `summary_local.csv` stores grouped means and standard deviations.

These measurements isolate small operations rather than the full AniSOAP descriptor pipeline. NumPy is faster than `torch_full` across all five stored local workloads, by approximately 4.42 to 14.14 times.

This result is not contradictory to the per-system metrics. Small-kernel launch, conversion, dispatch, allocation, and framework overhead can dominate isolated microbenchmarks while larger end-to-end workloads can favor a different execution path.

## DOI-era reported results

The DOI-era README reported additional CHTC 50-frame measurements, including 12–25% PyTorch improvements for benzenes and ellipsoids, plus Apple MPS observations.

The raw CHTC logs, input XYZ files, benchmark runner, and machine-readable 50-frame table are not present in the `v0.0.0` DOI snapshot. Those values should therefore be treated as reported historical observations rather than independently regenerable measurements from this repository.

The current README does not use those values as its primary evidence.

## Profiling call graphs

The repository preserves two DOI-era cProfile call-graph images:

- `results/figures/prof_benzenes_callgraph.png`
- `results/figures/prof_ellipsoids_callgraph.png`

They document the original hotspot investigation and show repeated contraction-heavy call paths. The underlying `.prof` files were removed before the DOI snapshot, so the images can be inspected but not regenerated from the current repository.

## Environment metadata

`results/tables/env_report.json` records only:

- Python version;
- PyTorch version;
- CUDA availability and version;
- cuDNN version;
- detected CUDA devices.

It does not record the CPU model, operating-system build, BLAS implementation, thread variables, Git revision, timestamp, or MPS availability. The improved `scripts/export_env.py` captures these fields for future reruns, but it cannot reconstruct metadata that was not preserved in 2025.

## Reproducibility scope

The current repository can:

1. validate the consistency of committed CSV and JSON artifacts;
2. recompute summary ratios from the stored measurements;
3. regenerate comparison plots from committed tables;
4. capture a more complete environment report for future experiments.

The current repository cannot reproduce the original AniSOAP measurements from scratch because the benchmark runner and input structures are absent from the DOI snapshot.

Run the supported checks with:

```bash
python -m pip install -r requirements.txt
python scripts/validate_results.py
python scripts/make_plots.py
```

Generated plots are written to `results/generated/` so the DOI-era figures remain unchanged.

## Interpretation boundaries

Do not infer from this repository that:

- PyTorch is always faster than NumPy;
- performance scales linearly with species count;
- the stored results generalize to CUDA, MPS, or other CPUs;
- float32 and float64 are universally equivalent;
- the historical prototype is production-ready;
- the stored measurements describe the current AniSOAP implementation.

The strongest supported conclusion is narrower: backend performance depends on workload scale and execution scope, and the profiling study identified repeated tensor contractions and Python-level execution structure as important optimization targets.

## Relationship to AniSOAP-Torch

The later [`AniSOAP-Torch`](https://github.com/Tejas7007/AniSOAP-Torch) repository follows from these observations with a different implementation and benchmark design. Its batching and Rust path should not be compared directly with the exploratory NumPy/PyTorch measurements here unless the benchmark environment, workload, and implementation revision are matched.
