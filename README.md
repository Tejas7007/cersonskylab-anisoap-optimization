AniSOAP Optimization

Goal: Make anisotropic SOAP (AniSOAP) descriptor computation fast, scalable, and reproducible across CPU/GPU backends while preserving descriptor fidelity.

Overview


Problem. Descriptor generation can dominate runtime (wall-time, memory traffic, vectorization limits, I/O), especially across multi-species systems.


Impact. Faster AniSOAP enables larger datasets, broader hyper-param sweeps, and smoother deployment in downstream potentials/property models.


This repo provides. A principled, reproducible optimization path with baselines, profiling, species-scaling analysis, and validated speedups.


Artifacts live in results/figures/, results/tables/, and results/logs/.

Repo Structure
.
├── scripts/
│   ├── make_plots.py
│   ├── run_benchmarks.py
│   └── aggregate_metrics.py
├── results/
│   ├── figures/
│   │   ├── wall_time_by_system.png
│   │   ├── wall_time_vs_species.png
│   │   ├── prof_benzenes_callgraph.png
│   │   └── prof_ellipsoids_callgraph.png
│   ├── tables/
│   │   ├── timings_chtc.csv
│   │   ├── timings_local.csv
│   │   ├── summary_local.csv
│   │   └── combined_from_metrics.csv
│   └── logs/
│       ├── prof_benzenes_200.prof
│       ├── prof_ellipsoids_200.prof
│       └── bench.svg
├── submit/
├── data/
└── README.md


Installation
Conda
conda env create -f env/environment.yml
conda activate anisoap-opt

uv/pip
uv venv && source .venv/bin/activate
uv pip install -e .

Depends on (minimums & notes):


Python ≥ 3.10


NumPy ≥ 1.26, SciPy ≥ 1.11


PyTorch ≥ 2.2  (CPU OK; MPS optional on Apple Silicon; CUDA optional on Linux)


BLAS/LAPACK: MKL or OpenBLAS


pandas ≥ 2.0, matplotlib ≥ 3.7


Dev profiling: py-spy ≥ 0.3, gprof2dot ≥ 2024.6.6, graphviz (dot)


Tests: pytest ≥ 7.0


Notes
# macOS (Apple Silicon)
brew install graphviz   # or: conda install -c conda-forge graphviz

# Linux (CUDA example)
pip install "torch==2.*+cu121" --index-url https://download.pytorch.org/whl/cu121


Data


Inputs: one_species.xyz, benzenes.xyz (2 spp.), three_species.xyz, four_species.xyz, ellipsoids.xyz.


For constant-N species sweeps: create_fake_benzenes.py (changes labels only).


Place large/raw inputs under $DATA_ROOT (not tracked).


Example
DATA_ROOT/
  ├── one_species.xyz
  ├── benzenes.xyz
  ├── three_species.xyz
  ├── four_species.xyz
  ├── ellipsoids.xyz
  └── create_fake_benzenes.py


Methodology
Modes (backends)
ModeDescriptionnumpy_onlyPure NumPy baseline.torch_mixedNumPy pipeline with a NumPy→Torch→NumPy wrapper around the hotspot einsum.torch_fullKeeps tensors in Torch through the hotspot to avoid conversion overhead.
CPU fairness (thread pinning)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1

Species scaling & normalization
Report time / N², where N² is the pairwise atom grid touched by distance/einsum. With constant-N files (via create_fake_benzenes.py), label changes (1→4 species) are the only variable; curves remain stable and show no worse-than-quadratic growth.

Reproducing Results
1) Run benchmarks (constant N; vary species)
ANISOAP_BACKEND=numpy_only \
python scripts/run_benchmarks.py --data $DATA_ROOT \
  --files one_species.xyz benzenes.xyz three_species.xyz four_species.xyz \
  --out results/metrics/numpy_species.json

ANISOAP_BACKEND=torch_mixed \
python scripts/run_benchmarks.py --data $DATA_ROOT \
  --files one_species.xyz benzenes.xyz three_species.xyz four_species.xyz \
  --out results/metrics/torch_mixed_species.json

ANISOAP_BACKEND=torch_full \
python scripts/run_benchmarks.py --data $DATA_ROOT \
  --files one_species.xyz benzenes.xyz three_species.xyz four_species.xyz \
  --out results/metrics/torch_full_species.json

2) Aggregate
python scripts/aggregate_metrics.py \
  --inputs results/metrics/*.json \
  --out results/tables/combined_from_metrics.csv

3) Plot
python scripts/make_plots.py \
  --table results/tables/combined_from_metrics.csv \
  --figdir results/figures


Results

Images must be outside code fences and paths are case-sensitive.


Torch CPU often outperforms NumPy for einsum-heavy sections (~10–25% faster). Frequently torch_full ≤ torch_mixed ≤ numpy_only, implying conversion overhead is small but non-zero and per-core kernel efficiency dominates.

After N² normalization and constant-N control, species curves do not show super-quadratic growth.


Tables: results/tables/combined_from_metrics.csv, timings_chtc.csv, timings_local.csv, summary_local.csv

Profiling & Notes


Hotspot: numpy.einsum along pairwise_ellip_expansion → transform → power_spectrum.


Tools: cProfile, py-spy; Torch profiler for backend traces.


Facts (from runs):


Benzenes baseline: total 100.85 s; numpy.c_einsum 76.1 s (~75.5%); 2,362,962 calls.


Ellipsoids baseline: total 1.856 s; numpy.c_einsum 0.551 s (~29.7%). Excluding ~1 s import overhead, compute section ≈0.83 s with einsum ≈ 66%.




Backend behavior: Torch’s ATen kernels can fuse broadcast+masked reductions and use tuned vector micro-kernels; with threads pinned, Torch can still win per-core. On Apple MPS, small runs are very fast but large high-rank einsum workloads can stall due to many tiny kernel launches + host↔device copies.


Reproduce profiling
py-spy record -o results/logs/bench.svg -- python scripts/run_benchmarks.py --files benzenes.xyz
python -m cProfile -o results/logs/prof_benzenes_200.prof   scripts/run_benchmarks.py --files benzenes.xyz
python -m cProfile -o results/logs/prof_ellipsoids_200.prof scripts/run_benchmarks.py --files ellipsoids.xyz


Validation


Check descriptor shape/dtype and invariances.


fp32 vs fp64: no meaningful differences observed on profiled paths.


Tests: np.testing.assert_allclose (suggested rtol=1e-6, atol=1e-8 for fp64; relax for fp32). Run via pytest -q.



Hardware & Environment


CHTC: x86_64 Linux, 1 CPU/job (thread-pinned).


Local: Apple M2 (macOS), Torch CPU + MPS.


GPU: CUDA (Linux) to be evaluated with full Torch path; MPS explored locally.



Command Cookbook
# species sweep across three backends
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1
for mode in numpy_only torch_mixed torch_full; do
  ANISOAP_BACKEND=$mode \
  python scripts/run_benchmarks.py --data $DATA_ROOT \
    --files one_species.xyz benzenes.xyz three_species.xyz four_species.xyz \
    --out results/metrics/${mode}_species.json
done

# aggregate + plot
python scripts/aggregate_metrics.py --inputs results/metrics/*.json \
  --out results/tables/combined_from_metrics.csv
python scripts/make_plots.py --table results/tables/combined_from_metrics.csv \
  --figdir results/figures


Acknowledgements
Cersonsky Lab (UW–Madison), Arthur Lin, and collaborators.
