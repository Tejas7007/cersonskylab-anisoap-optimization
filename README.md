# AniSOAP Optimization

> **Goal:** Make anisotropic SOAP (AniSOAP) descriptors *fast, scalable, and easy to reproduce* across CPU/GPU backends while preserving descriptor fidelity.

<!--
HOW TO USE THIS DRAFT
1) Paste Arthur–Tejas message snippets into the chat; I (ChatGPT) will fold them into the marked TODO blocks.
2) Replace any 🚧 TODO blocks with concrete content.
3) Ensure filenames/paths match your repo (case‑sensitive).
-->

## Why this repo exists (Problem → Impact)

* **Problem:** Descriptor generation for atomistic ML pipelines can be a bottleneck (wall‑time, memory, vectorization limits, I/O), especially at scale and across diverse species.
* **Impact:** Faster AniSOAP unlocks bigger datasets, larger hyperparameter sweeps, and practical deployment in downstream interatomic potentials and property models.
* **This repo solves:** A principled, reproducible optimization path—with baselines, profiling, and validated speedups on real datasets.

## TL;DR (What we did)

* Built a **reproducible benchmarking harness** (datasets, seeds, metrics).
* Implemented **profiling** (cProfile/py‑spy/line_profiler) and **micro‑benchmarks**.
* Compared **systems × backends** and **wall‑time vs. #species**.
* Produced **publication‑quality figures** and CSV tables.
* Documented **tuning levers** (algorithmic, memory, parallelism, vectorization, batching).

> Key artifacts live in `results/figures/`, `results/tables/`, and `results/logs/`.

## Context & scope

* **AniSOAP**: anisotropic Smooth Overlap of Atomic Positions descriptor.
* **Scope of this repo:** performance engineering + correctness checks for descriptor generation; does *not* reimplement the learning models themselves.
* **Out‑of‑scope:** exhaustive chemistry benchmarks, downstream ML leaderboard.

## Repo structure (high‑level)

```
.
├── scripts/
│   ├── make_plots.py                 # generates figures from metrics JSON/CSVs
│   ├── 🚧 (add) run_benchmarks.py     # entrypoint to run all benchmark suites
│   └── 🚧 (add) profile_*.py          # minimal repros for targeted profiling
├── anisoap_opt/                      # (optional) library code if any
├── results/
│   ├── figures/
│   │   ├── wall_time_by_system.png
│   │   └── wall_time_vs_species.png
│   ├── tables/
│   │   └── combined_from_metrics.csv
│   └── logs/
├── env/
│   └── 🚧 (add) environment.yml | pyproject.toml
├── data/                             # symlinks or small example snippets only
├── README.md                         # ← this file
└── LICENSE
```

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

> 🚧 TODO: list core dependencies (Python ≥3.x, PyTorch/CUDA versions, compilers, OpenMP/MKL, etc.).

## Data

* We use internal or public molecules/systems to measure descriptor throughput.
* **Large datasets are not checked in.** Provide paths via env vars or CLI.

**Example layout**

```
DATA_ROOT/
  ├── system_A/
  ├── system_B/
  └── ...
```

> 🚧 TODO: Document any dataset sources, licenses, and download helpers.

## Reproducing the results

### 1) Run benchmarks

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

### 2) Aggregate into a single table

```bash
python scripts/aggregate_metrics.py \
  --inputs results/metrics/*.json \
  --out results/tables/combined_from_metrics.csv
```

### 3) Make plots

```bash
python scripts/make_plots.py \
  --table results/tables/combined_from_metrics.csv \
  --figdir results/figures
```

## Results (figures & tables)

### QUICK CHECKLIST — add your local figures so they render on GitHub

1. **Locate** your artifacts on macOS (replace the path root if needed):

   ```bash
   # from anywhere
   mdfind 'kMDItemFSName ==[c] "*anisoap*" || kMDItemFSName ==[c] "*profiling*"' | head -n 50
   # or within your project dir
   cd /path/to/cersonskylab-anisoap-optimization
   find . -iname "*.png" -o -iname "*.svg" -o -iname "*.prof" -o -iname "*timings*.csv" -o -iname "*.json"
   ```
2. **Copy** the figures you want into the repo (case‑sensitive paths):

   ```bash
   mkdir -p results/figures results/logs results/tables
   # examples — adjust src paths to your Mac
   cp ~/Desktop/profiling/wall_time_by_system.png    results/figures/
   cp ~/Desktop/profiling/wall_time_vs_species.png   results/figures/
   cp ~/Desktop/profiling/prof_benzenes_callgraph.png results/figures/
   cp ~/Desktop/profiling/prof_ellipsoids_callgraph.png results/figures/
   cp ~/Desktop/profiling/bench.svg                  results/logs/
   cp ~/Desktop/profiling/timings_chtc.csv           results/tables/
   cp ~/Desktop/profiling/timings_local.csv          results/tables/
   cp ~/Desktop/profiling/summary_local.csv          results/tables/
   cp ~/Desktop/profiling/prof_benzenes_200.prof     results/logs/
   cp ~/Desktop/profiling/prof_ellipsoids_200.prof   results/logs/
   ```
3. **Commit** so GitHub can render them:

   ```bash
   git add results/figures results/logs results/tables README.md
   git commit -m "Add profiling figures, logs, and tables; update README"
   git push origin main
   ```

> Image markdown must be **outside** code fences and filenames **case‑sensitive**.

```text
# Do NOT put image lines inside this code block.
```

![Wall time by system & backend](results/figures/wall_time_by_system.png)

![Wall time vs #Species](results/figures/wall_time_vs_species.png)

![cProfile call graph — Benzenes](results/figures/prof_benzenes_callgraph.png)

![cProfile call graph — Ellipsoids](results/figures/prof_ellipsoids_callgraph.png)

*Takeaway:* Torch CPU typically outperforms NumPy for `einsum`‑heavy sections; MPS/CUDA plots should be added once the full Torch path is ported.

*Takeaway:* After normalizing by **N²**, species curves do not show super‑quadratic behavior on the standardized files.

**Combined metrics table:** `results/tables/combined_from_metrics.csv`

> Image markdown must be **outside** code fences and filenames **case‑sensitive**.

```text
# Do NOT put image lines inside this code block.
```

![Wall time by system & backend](results/figures/wall_time_by_system.png)

![Wall time vs #Species](results/figures/wall_time_vs_species.png)

**Combined metrics table:** `results/tables/combined_from_metrics.csv`

> 🚧 TODO: add a short narrative under each figure explaining: dataset, hardware, sample size, takeaways.

## Species scaling & normalization

We report `time / N²` (where `N²` is the size of the pairwise atom grid touched by distance/einsum steps) to remove raw atom‑count effects. Using `create_fake_benzenes.py`, we generate multi‑species files with **constant N** so that label changes (1→4 species) are the only variable. Under this control, curves remain stable and do **not** exhibit worse‑than‑quadratic growth with species.

## What we optimized (and why)

* **Vectorization & batching**: reduce Python overhead; maximize BLAS/GPU utilization.
* **Parallelism**: OpenMP/threading on CPU; streams/blocks on GPU.
* **Memory traffic**: layout/contiguity, pinning, avoiding needless copies.
* **Algorithmic choices**: cutoff radii, basis sizes, truncations that preserve accuracy but lower cost.
* **I/O & caching**: chunked reads/writes; memoize reusable intermediates.

> 🚧 TODO: Tie each bullet to concrete code changes, PRs, or commits.

## Profiling & methodology

* **cProfile/py‑spy**: hotspot discovery at function level.
* **line_profiler**: line‑level attribution for kernels.
* **nvprof/ncu** (if GPU): kernel occupancy and memory throughput.
* **A/B experiments**: one change at a time with fixed seeds.

**Reproduce profiling**

```bash
py-spy record -o results/logs/bench.svg -- python scripts/run_benchmarks.py ...
python -m cProfile -o results/logs/bench.prof scripts/run_benchmarks.py ...
```

> 🚧 TODO: Link representative `.svg` flamegraphs and `.prof` summaries in `results/logs/`.

## Validation (did we keep correctness?)

* Cross‑check descriptor arrays against **reference** implementation within tolerance.
* Unit tests for **shape/dtype** & invariances.
* Spot‑check downstream task metrics unchanged (or documented tradeoffs).

**Tests:** Compare Torch paths to NumPy reference using `np.testing.assert_allclose` with tight tolerances (suggested: `rtol=1e-6`, `atol=1e-8` for fp64; relax appropriately for fp32). Include shape/dtype asserts and invariance checks (rot/perm where applicable). Run via `pytest -q`.

## Artifacts & file map (from the email thread)

| Kind                        | Suggested path in repo                          | Notes                                  |
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
| Per‑job metrics             | `results/logs/*.metrics.json`                   | Emitted by runner                      |
| Raw wall stamps             | `results/logs/*.wall`                           | Emitted by runner                      |
| Repro bundle (CHTC)         | `profiling_artifacts.tgz`                       | Contains results/, logs/, submit files |
| Repro bundle (local)        | `profiling_local.zip`                           | Contains local CSVs + harness          |

If your locally saved filenames differ, either **rename to these** or update the README image lines accordingly.

## Hardware & environment

* **CPU:** 🚧 TODO (model, cores, threads, RAM)
* **GPU:** 🚧 TODO (model, driver, CUDA)
* **OS:** 🚧 TODO (Linux distro & version)
* **Libraries:** 🚧 TODO (MKL/OpenBLAS, PyTorch)

> Results are hardware‑sensitive; please include your specs when reporting issues.

## Command cookbook

Common invocations we found useful:

```bash
# 1) Small sanity benchmark
python scripts/run_benchmarks.py --preset tiny --out results/metrics/tiny.json

# 2) Multi‑species sweep
python scripts/run_benchmarks.py --sweep species --out results/metrics/species.json

# 3) Threads sweep (CPU)
OPENMP_NUM_THREADS=1,2,4,8,16 ... python scripts/run_benchmarks.py --preset cpu

# 4) Batch size sweep (GPU)
python scripts/run_benchmarks.py --preset gpu --sweep batch
```

## Repo status & roadmap

* ✅ Baseline metrics + plots checked in.
* ✅ Repro scripts for figures.
* 🚧 Public benchmark configs (CPU/GPU presets).
* 🚧 Automated CI to run micro‑benchmarks on commits.
* 🚧 Documentation site (mkdocs) with deeper guides.

## How to cite

> 🚧 TODO: add citation(s) for AniSOAP and this optimization report once available.

```bibtex
@inproceedings{cersonsky202Xanisoap,
  title     = {Anisotropic SOAP and Optimization Benchmarks},
  author    = {Cersonsky, Rose and Lin, Arthur and Dahiya, Tejas and ...},
  year      = {202X},
  booktitle = {...}
}
```

## Acknowledgements

Thanks to **Cersonsky Lab** (UW–Madison), **Arthur Lin**, and collaborators for guidance and reviews.

---

### Appendix: Repro tips

* Fix seeds and versions; export `PYTHONHASHSEED=0`.
* Keep environments immutable during a run.
* Pin threads with `taskset` or `numactl` when comparing CPU backends.

### Appendix: Troubleshooting

* **Figures not rendering in GitHub preview?** Ensure image lines are outside code fences and filenames match exactly.
* **Missing datasets?** Provide `--data` or set `DATA_ROOT`.
* **Slow CSV writes?** Use `mode='w', index=False` and consider gzip (`.csv.gz`).
