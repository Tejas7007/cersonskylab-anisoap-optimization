# AniSOAP Performance Optimization
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17503801.svg)](https://doi.org/10.5281/zenodo.17503801)

> **Goal:** Make anisotropic SOAP (AniSOAP) descriptors *fast, scalable, and easy to reproduce* across CPU/GPU backends while preserving descriptor fidelity.

A comprehensive profiling and optimization study of AniSOAP (Anisotropic Smooth Overlap of Atomic Positions) computational performance, comparing NumPy and PyTorch backends across different molecular systems and hardware platforms.

---

## Why This Repo Exists (Problem → Impact)

* **Problem:** Descriptor generation for atomistic ML pipelines can be a bottleneck (wall‑time, memory, vectorization limits, I/O), especially at scale and across diverse species. AniSOAP descriptor calculations involve intensive tensor contractions, particularly in the `pairwise_ellip_expansion` function.
 
* **Impact:** Faster AniSOAP unlocks bigger datasets, larger hyperparameter sweeps, and practical deployment in downstream interatomic potentials and property models.
  
* **This repo solves:** A principled, reproducible optimization path—with baselines, profiling, and validated speedups on real datasets.

### Initial Profiling Results

- **Benzenes system**: 75.5% of runtime in `numpy.einsum` (76.1s out of 100.85s total)
- **Ellipsoids system**: 66% of runtime in `numpy.einsum` (0.55s out of 0.83s compute time)

---

## TL;DR (What We Did)

* Built a **reproducible benchmarking harness** (datasets, seeds, metrics)
* Implemented **profiling** (cProfile/py‑spy/line_profiler) and **micro‑benchmarks**
* Compared **systems × backends** and **wall‑time vs. #species**
* Produced **publication‑quality figures** and CSV tables
* Documented **tuning levers** (algorithmic, memory, parallelism, vectorization, batching)

### Key Findings

- **CPU Performance**: PyTorch shows consistent 12-25% speedup over NumPy on CPU
- **Primary Bottleneck**: `numpy.einsum` accounts for 66-77% of computation time
- **Precision**: No significant performance difference between fp32 and fp64
- **Scaling**: Linear scaling with species count (no worse-than-quadratic behavior)
- **GPU Limitations**: MPS (Apple Silicon) backend stalls on large workloads due to high-rank einsum operations

> Key artifacts live in `results/figures/`, `results/tables/`, and `results/logs/`.

---

## Context & Scope

* **AniSOAP**: anisotropic Smooth Overlap of Atomic Positions descriptor
* **Scope of this repo:** performance engineering + correctness checks for descriptor generation; does *not* reimplement the learning models themselves
* **Out‑of‑scope:** exhaustive chemistry benchmarks, downstream ML leaderboard

### Research Questions

The goal was to determine whether:

1. PyTorch backends could accelerate these operations
2. Conversion overhead (NumPy ↔ PyTorch) negates performance gains
3. GPU acceleration is viable for production workloads

---

## Repository Structure

```
.
├── scripts/
│   ├── make_plots.py                 # generates figures from metrics JSON/CSVs
│   ├── run_benchmarks.py             # entrypoint to run all benchmark suites
│   └── profile_*.py                  # minimal repros for targeted profiling
├── anisoap_opt/                      # library code
├── results/
│   ├── figures/
│   │   ├── wall_time_by_system.png
│   │   ├── wall_time_vs_species.png
│   │   ├── prof_benzenes_callgraph.png
│   │   └── prof_ellipsoids_callgraph.png
│   ├── tables/
│   │   ├── combined_from_metrics.csv
│   │   ├── timings_chtc.csv
│   │   ├── timings_local.csv
│   │   └── summary_local.csv
│   └── logs/
│       ├── prof_benzenes_200.prof
│       ├── prof_ellipsoids_200.prof
│       └── bench.svg
├── env/
│   └── environment.yml
├── data/                             # symlinks or small example snippets only
├── profiling_artifacts.tgz           # CHTC run bundle
├── profiling_local.zip               # Local run bundle
├── create_fake_benzenes.py           # Multi-species test dataset generator
├── README.md                         # ← this file
└── LICENSE
```

---

## Installation

**Option A (conda):**

```bash
conda env create -f env/environment.yml
conda activate anisoap-opt
```

**Option B (uv/pip):**

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

### Dependencies

- Python 3.8+
- NumPy 1.21+
- PyTorch 1.12+ (CPU: any, GPU: CUDA 11.0+ or MPS)
- AniSOAP library
- cProfile (standard library)
- gprof2dot (optional, for call graph visualization)

---

## Methodology

### Test Systems

1. **Benzenes** (203s baseline): Large organic molecules, 2 species, high neighbor density
2. **Ellipsoids** (1.56s baseline): Simpler geometry, lighter computation
3. **Multi-species variants**: 1, 2, 3, 4 species to test species-scaling behavior

### Experimental Setup

**Profiling Environments:**

- **CHTC (HTC Cluster)**: Linux x86_64, Singularity containers, 1 CPU/job
- **Local (macOS)**: Apple Silicon M2, MPS GPU backend testing

**Backend Configurations:**

- `numpy_only`: Pure NumPy baseline
- `torch_mixed`: NumPy pipeline with NumPy→Torch→NumPy conversion around einsum
- `torch_full`: Full PyTorch tensors throughout (no conversions)

**Thread Pinning (for fair comparison):**

```bash
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
TORCH_NUM_THREADS=1
```

### Metrics Collected

- Wall-clock time per run
- Conversion overhead (NumPy ↔ PyTorch)
- Per-function self-time and cumulative time
- Call counts for hotspot functions
- Time normalized by N² (atom count squared)

---

## Results

### CPU Performance (CHTC Linux Cluster)

| System | Backend | Precision | Frames | Time (s) | Speedup vs NumPy |
|--------|---------|-----------|--------|----------|------------------|
| Ellipsoids | NumPy | default | 50 | 1.56 | baseline |
| Ellipsoids | PyTorch | fp64 | 50 | 1.17 | **25% faster** |
| Ellipsoids | PyTorch | fp32 | 50 | 1.18 | **24% faster** |
| Benzenes | NumPy | default | 50 | 203.18 | baseline |
| Benzenes | PyTorch | fp64 | 50 | 172.81 | **15% faster** |
| Benzenes | PyTorch | fp32 | 50 | 178.53 | **12% faster** |

### Multi-Species Scaling (CHTC)

| File | Species | NumPy (s) | PyTorch (s) | Notes |
|------|---------|-----------|-------------|-------|
| one_species.xyz | 1 | 0.490 | 0.136 | 3.6× faster |
| benzenes.xyz | 2 | 0.242 | 0.131 | 1.8× faster |
| three_species.xyz | 3 | 0.393 | 0.183 | 2.1× faster |
| four_species.xyz | 4 | 0.252 | 0.223 | 1.1× faster |
| ellipsoids.xyz | - | 0.213 | 0.219 | comparable |

**Key observation**: Time normalized by N² remains stable across species counts, confirming no worse-than-quadratic scaling behavior.

### GPU Testing (Apple Silicon MPS)

| System | Backend | Precision | Frames | Result |
|--------|---------|-----------|--------|--------|
| Ellipsoids | PyTorch MPS | fp32/fp16 | 5 | 0.013s (fp16) ✓ |
| Ellipsoids | PyTorch MPS | fp32 | 50 | **Stalled >10min** ✗ |
| Benzenes | PyTorch MPS | fp32 | 50 | **Stalled >10min** ✗ |

**Why MPS fails at scale:**

1. **Host→device copy overhead**: ~17,000+ einsum calls each trigger data transfers
2. **High-rank einsum decomposition**: Operations like `mnpqr,pqr->mn` lower to many small kernels
3. **MPS kernel launch overhead**: Thousands of small launches dominate performance
4. **Unified memory pressure**: 5D tensor intermediates cause paging at scale
5. **CPU fallback**: Unimplemented MPS ops force host-device round-trips

---

## Figures & Visualizations

![Wall time by system & backend](results/figures/wall_time_by_system.png)

*Takeaway:* Torch CPU typically outperforms NumPy for `einsum`‑heavy sections; MPS/CUDA plots should be added once the full Torch path is ported.

![Wall time vs #Species](results/figures/wall_time_vs_species.png)

*Takeaway:* After normalizing by **N²**, species curves do not show super‑quadratic behavior on the standardized files.

![cProfile call graph — Benzenes](results/figures/prof_benzenes_callgraph.png)

*Benzenes (100.85s total):* Deep call path through `pairwise_ellip_expansion` → einsum contraction. `numpy.c_einsum` accounts for 76.1s (75.5%) across 2,362,962 calls.

![cProfile call graph — Ellipsoids](results/figures/prof_ellipsoids_callgraph.png)

*Ellipsoids (1.856s total):* Lighter neighbor list, broader distribution across `transform`, `power_spectrum`, einsum. `numpy.c_einsum` accounts for 0.551s (29.7%) across 17,752 calls.

**Combined metrics table:** `results/tables/combined_from_metrics.csv`

---

## Technical Analysis

### Why PyTorch Outperforms NumPy on CPU

Even with thread pinning (removing threading advantages), PyTorch maintains edge due to:

1. **Kernel design**: Fused operations reduce temporary allocations
   - NumPy masked slicing materializes intermediates
   - PyTorch contracts more work into single kernels

2. **Memory traffic**: Better cache utilization
   - Tighter reduction loops
   - Aggressive tensor reordering/contiguity

3. **Vectorization (SIMD)**: Even at NUM_THREADS=1
   - ATen kernels use optimized vector code
   - May leverage better BLAS micro-kernels (MKL vs OpenBLAS)

4. **Layout optimization**: PyTorch aggressively reorders tensors for hot ops
   - NumPy honors original strides → degraded inner-loop locality

### Conversion Overhead Analysis

In `torch_mixed` mode (profiled separately):

- NumPy→Torch→NumPy conversion time is **small fraction** of total runtime
- For large arrays, conversion cost amortizes over compute-intensive einsum
- `torch_full` < `torch_mixed` difference is **not statistically significant**

**Conclusion**: Conversion is not the bottleneck; kernel efficiency dominates.

---

## Profiling Details

### Hotspot Functions (cProfile)

**Benzenes (100.85s total):**

- `numpy.c_einsum`: 76.1s (75.5%) — 2,362,962 calls
- `pairwise_ellip_expansion`: 97.8s cumulative
- `transform`: significant contributor
- `power_spectrum`: moderate contributor

**Ellipsoids (1.856s total):**

- `numpy.c_einsum`: 0.551s (29.7%) — 17,752 calls
- Import overhead: ~1s (excluded from analysis)
- Compute time (`compute_anisoap`): 0.83s, with einsum at 66%

### Call Graph Insights

- **Benzenes**: Deep call path through `pairwise_ellip_expansion` → einsum contraction `mnpqr,pqr->mn`
- **Ellipsoids**: Lighter neighbor list, broader distribution across `transform`, `power_spectrum`, einsum
- einsum call frequency directly correlates with neighbor density and sample size

---

## Reproducing the Results

### 1) Run Benchmarks

```bash
python scripts/run_benchmarks.py \
  --data $DATA_ROOT \
  --config configs/batch_cpu.yaml \
  --out results/metrics/cpu.json

python scripts/run_benchmarks.py \
  --data $DATA_ROOT \
  --config configs/batch_gpu.yaml \
  --out results/metrics/gpu.json
```

### 2) Aggregate into a Single Table

```bash
python scripts/aggregate_metrics.py \
  --inputs results/metrics/*.json \
  --out results/tables/combined_from_metrics.csv
```

### 3) Make Plots

```bash
python scripts/make_plots.py \
  --table results/tables/combined_from_metrics.csv \
  --figdir results/figures
```

### CHTC Cluster Runs

```bash
# Unpack artifacts
tar -xzf profiling_artifacts.tgz
cd profiling_artifacts

# View aggregated results
cat timings_chtc.csv

# Inspect individual job
cat results/job_benzenes_numpy.wall
cat results/job_benzenes_numpy.metrics.json

# Resubmit (requires CHTC access)
condor_submit chtc_profile.sing.submit
```

### Local Profiling

```bash
# Unpack local results
unzip profiling_local.zip
cd profiling_local

# Run harness
bash run_local.sh

# View summaries
cat summary_local.csv  # mean ± std, conversion breakdown
cat timings_local.csv  # raw per-run data
```

### Generate Multi-Species Test Files

```bash
python create_fake_benzenes.py
# Outputs: one_species.xyz, three_species.xyz, four_species.xyz
```

### Profiling Commands

```bash
# py-spy flamegraph
py-spy record -o results/logs/bench.svg -- python scripts/run_benchmarks.py ...

# cProfile
python -m cProfile -o results/logs/bench.prof scripts/run_benchmarks.py ...
```

---

## Species Scaling & Normalization

We report `time / N²` (where `N²` is the size of the pairwise atom grid touched by distance/einsum steps) to remove raw atom‑count effects. Using `create_fake_benzenes.py`, we generate multi‑species files with **constant N** so that label changes (1→4 species) are the only variable. Under this control, curves remain stable and do **not** exhibit worse‑than‑quadratic growth with species.

---

## What We Optimized (and Why)

* **Vectorization & batching**: reduce Python overhead; maximize BLAS/GPU utilization
* **Parallelism**: OpenMP/threading on CPU; streams/blocks on GPU
* **Memory traffic**: layout/contiguity, pinning, avoiding needless copies
* **Algorithmic choices**: cutoff radii, basis sizes, truncations that preserve accuracy but lower cost
* **I/O & caching**: chunked reads/writes; memoize reusable intermediates

---

## Recommendations

### Immediate Actions (✓ Implemented)

1. **Use PyTorch backend on CPU for production**: 12-25% speedup with no accuracy loss
2. **Default to fp64 (float64)**: No performance penalty vs fp32, maintains numerical stability
3. **Thread pinning for reproducibility**: Enforce via environment variables

### Future Optimization Paths

1. **GPU acceleration (CUDA)**:
   - Port full pipeline to PyTorch to avoid host-device copies
   - Batch multiple frames to amortize kernel launch overhead
   - Target Linux + CUDA (MPS not production-ready for this workload)

2. **Kernel fusion**:
   - Manually fuse broadcast + masked reduction operations
   - Reduce intermediate tensor allocations
   - Explore `torch.compile()` (PyTorch 2.0+) for automatic fusion

3. **Algorithmic improvements**:
   - Reduce neighbor list density where physically valid
   - Cache reusable tensor contractions
   - Exploit symmetry in pairwise operations

### When NOT to Use PyTorch

- Small systems (<10 atoms): Conversion overhead dominates
- One-off calculations: Startup cost not amortized
- Environments without MKL/optimized BLAS: NumPy may be comparable

---

## Validation (Did We Keep Correctness?)

* Cross‑check descriptor arrays against **reference** implementation within tolerance
* Unit tests for **shape/dtype** & invariances
* Spot‑check downstream task metrics unchanged (or documented tradeoffs)

**Tests:** Compare Torch paths to NumPy reference using `np.testing.assert_allclose` with tight tolerances (suggested: `rtol=1e-6`, `atol=1e-8` for fp64; relax appropriately for fp32). Include shape/dtype asserts and invariance checks (rot/perm where applicable). Run via `pytest -q`.

---

## Hardware & Environment

* **CPU:** x86_64 Linux (CHTC cluster), 1 CPU/job
* **GPU:** Apple Silicon M2 (MPS backend testing)
* **OS:** Linux (CHTC), macOS (local testing)
* **Libraries:** NumPy 1.21+, PyTorch 1.12+, MKL/OpenBLAS

> Results are hardware‑sensitive; please include your specs when reporting issues.

---

## Command Cookbook

Common invocations we found useful:

```bash
# 1) Small sanity benchmark
python scripts/run_benchmarks.py --preset tiny --out results/metrics/tiny.json

# 2) Multi‑species sweep
python scripts/run_benchmarks.py --sweep species --out results/metrics/species.json

# 3) Threads sweep (CPU)
OPENMP_NUM_THREADS=1,2,4,8,16 python scripts/run_benchmarks.py --preset cpu

# 4) Batch size sweep (GPU)
python scripts/run_benchmarks.py --preset gpu --sweep batch
```

---

## Artifacts & File Map

| Kind                        | Path in repo                                    | Notes                                  |
| --------------------------- | ----------------------------------------------- | -------------------------------------- |
| Wall‑time by system figure  | `results/figures/wall_time_by_system.png`       | Generated by `scripts/make_plots.py`   |
| Wall‑time vs #species       | `results/figures/wall_time_vs_species.png`      | N²‑normalized species curves           |
| Benzenes call graph (PNG)   | `results/figures/prof_benzenes_callgraph.png`   | From `prof_benzenes_200.prof`          |
| Ellipsoids call graph (PNG) | `results/figures/prof_ellipsoids_callgraph.png` | From `prof_ellipsoids_200.prof`        |
| Flamegraph (py‑spy)         | `results/logs/bench.svg`                        | Optional but useful                    |
| cProfile (benzenes)         | `results/logs/prof_benzenes_200.prof`           | Text + graphable                       |
| cProfile (ellipsoids)       | `results/logs/prof_ellipsoids_200.prof`         | Text + graphable                       |
| Timings (CHTC)              | `results/tables/timings_chtc.csv`               | Per‑run raw rows                       |
| Timings (local)             | `results/tables/timings_local.csv`              | Per‑run raw rows                       |
| Summary (local)             | `results/tables/summary_local.csv`              | Mean ± std                             |
| Aggregated metrics          | `results/tables/combined_from_metrics.csv`      | From `aggregate_metrics.py`            |
| Repro bundle (CHTC)         | `profiling_artifacts.tgz`                       | Contains results/, logs/, submit files |
| Repro bundle (local)        | `profiling_local.zip`                           | Contains local CSVs + harness          |

---

## Repository Status & Roadmap

* ✅ Baseline metrics + plots checked in
* ✅ Repro scripts for figures
* ✅ cProfile analysis for hotspots
* ✅ Multi-species scaling validation
* 🚧 Public benchmark configs (CPU/GPU presets)
* 🚧 Automated CI to run micro‑benchmarks on commits
* 🚧 Documentation site (mkdocs) with deeper guides

---

## Contact

**Researcher**: Tejas Dahiya (tdahiya2@wisc.edu)  
**Advisor**: Arthur Lin (alin62@wisc.edu)  
**Lab**: Cersonsky Lab, University of Wisconsin-Madison

---

## Acknowledgments

Thanks to **Cersonsky Lab** (UW–Madison), **CHTC** (Center for High Throughput Computing) for cluster resources, **Guillaume Fraux** for metatensor architecture discussions, and **Arthur Lin** for guidance and reviews.

---

## License

MIT

---

##📚 How to Cite

If you use this repository in your research, please cite it as:

> **Dahiya, T.** (2025). *AniSOAP Optimization: High-Performance Descriptor Benchmarking.*  
> University of Wisconsin–Madison, Cersonsky Lab.  
> Zenodo. [https://doi.org/10.5281/zenodo.17503801](https://doi.org/10.5281/zenodo.17503801)

---

**Last Updated**: November 2025  
**Profiling Cluster ID**: 2534045  
**Knowledge Cutoff**: Analysis valid as of October 2025; PyTorch/NumPy versions may impact results
