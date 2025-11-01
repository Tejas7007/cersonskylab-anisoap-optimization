Cersonsky Lab — AniSOAP Optimization

Reproducible profiling & optimization of AniSOAP (Cersonsky Lab, UW–Madison): scripts, figures, traces, and PyTorch/HPC configs.

Owner: Tejas Dahiya (@Tejas7007)
Context: Internship work with Arthur Lin and Dr. Rose Cersonsky focused on profiling bottlenecks (e.g., einsum) and accelerating CPU/GPU pipelines (PyTorch / Torch MPS / HPC).
Repository focus: Show exactly what was done, including artifacts, scripts, results, plots, and the reasoning behind decisions.

Repository Contents
scripts/
  organize_artifacts.py   # Ingest .tgz/.zip bundles into results/
  plot_results.py         # Produce figures from timings_* CSVs
  export_env.py           # Dump Python/PyTorch/CUDA env details
results/
  tables/                 # timings_chtc.csv, timings_local.csv, summary_local.csv, env_report.json
  figures/                # scaling_species.png, util_before_after.png, cProfile images (optional)
  profiler_traces/        # profiler traces (if included)
  logs/{chtc,local}/      # raw logs from cluster / local runs
examples/                 # (optional) configs or small stubs
README.md

Artifacts Intake (from emails to Arthur)

Use the exact bundles you sent to Arthur:

profiling_artifacts.tgz — CHTC run (cluster 2534045) with results/, logs/, timings_chtc.csv, submit/wrapper files.

profiling_local.zip — macOS runs with timings_local.csv, summary_local.csv, and run_local.sh.

Ingest + plot:

python scripts/organize_artifacts.py --chtc profiling_artifacts.tgz --local profiling_local.zip
python scripts/plot_results.py
python scripts/export_env.py > results/tables/env_report.json


Artifacts will land under results/ (CSV tables, figures, traces, logs). Figures include:

results/figures/scaling_species.png (time ÷ N² vs #species)

results/figures/util_before_after.png (if summary_local.csv has tag,util)

If you only have the CHTC bundle, omit --local.

Quick Results (Overview)

Species scaling (controlled): Holding N constant and varying species, runtime normalized by N² is stable → no worse-than-quadratic behavior observed.

CPU performance: PyTorch on CPU is typically ~10–25% faster than NumPy for the profiled workloads. Conversion overhead (NumPy↔Torch) is not the primary cost; kernel/threading and masked-reduction implementations matter more.

Threading parity matters: For fair CPU comparisons, pin threads for both stacks (see “Thread Parity” below).

Apple Silicon (Torch MPS): Tiny runs are extremely fast (e.g., ~0.013 s for 5 frames, fp16), but larger runs stall due to many small kernel launches, device copies around each einsum, dtype promotions, unified memory pressure, and occasional CPU fallbacks.

Example Table (50 frames; from your emails)
System	Backend	Precision	Device	Frames	Time (s)	Notes
Ellipsoids	NumPy	default	CPU	50	1.56	Baseline
Ellipsoids	Torch	fp64	CPU	50	1.17	~25% faster
Ellipsoids	Torch	fp32	CPU	50	~1.18	Similar to fp64
Benzenes	NumPy	default	CPU	50	203.18	Baseline
Benzenes	Torch	fp64	CPU	50	172.81	~15% faster
Benzenes	Torch	fp32	CPU	50	178.53	~12% faster
Ellipsoids	Torch	fp32	MPS	50	stalled	Torch MPS limitation
Benzenes	Torch	fp32	MPS	50	stalled	Torch MPS limitation
Figures

If present, these render directly in the repo:

Species scaling (normalized):
results/figures/scaling_species.png

GPU utilization before/after (if available):
results/figures/util_before_after.png

cProfile call graphs (optional, if you add them):
results/figures/cprofile/prof_benzenes_200.png
results/figures/cprofile/prof_ellipsoids_200.png

Detailed Worklog (summarized from emails & runs)
1) cProfile hotspot analysis (Sept 25, 2025)

Inputs: benzenes.xyz, ellipsoids.xyz.

Hotspot: numpy.einsum dominates; for benzenes ~75.5% of total runtime (76.1 s of 100.85 s), for ellipsoids ~29.7% (0.551 s of 1.856 s).

Call counts: benzenes: ~2,362,962 einsum calls; ellipsoids: ~17,752 calls.

Interpretation: Benzenes has a denser neighbor list and far more contractions (mnpqr,pqr->mn), so a greater share falls into einsum. Ellipsoids is tiny; import/setup and other functions (transform, power_spectrum) take a larger relative slice.

2) Apples-to-apples comparison (Sept 26, 2025)

Arthur asked to exclude ~1 s import overhead for fair comparison.

On that basis: ellipsoids → einsum ≈ 66% of compute; benzenes → einsum ≈ 77%.

Takeaway: the hotspot conclusion is consistent across inputs once startup overhead is removed.

3) CPU: NumPy vs Torch (Oct 2, 2025)

Platform: CHTC (x86_64 Linux), 1 CPU per job, Singularity container.

Result: Torch on CPU is consistently faster than NumPy (~12–25% depending on system) with no major precision sensitivity (fp32 ≈ fp64).

Interpretation: Differences likely from BLAS/thread pools and kernel implementations for broadcast + masked reductions (Torch is often linked against MKL and manages threads differently). Conversions NumPy↔Torch are a small fraction of runtime at scale.

4) Apple Silicon Torch MPS (Oct 2, 2025)

Tiny runs (5 frames): ~0.013 s for fp16 (orders-of-magnitude faster than CPU).

Larger runs (50 frames): stalls indefinitely. Likely causes:

Host→device copies around each einsum (patch wraps NumPy arrays every call).

High-rank einsum lowers into many small kernels (MPS launch overhead dominates).

Dtype promotions (e.g., fp16→fp32) introduce sync and slow paths.

Unified memory pressure with large 5D intermediates (paging/pressure).

CPU fallbacks for unsupported MPS paths → extra device↔host transfers.

5) Species scaling experiment (multi-species XYZ)

Files (from Arthur): one_species.xyz, benzenes.xyz (2 species), three_species.xyz, four_species.xyz, plus ellipsoids.xyz.

Method: hold N constant, vary species, normalize wall-time by N² (or optionally by ∑ₐ,ᵦ NₐNᵦ).

Observation: time/N² stable across species → no worse-than-quadratic scaling observed for this workload.

Controlled Experiment Plan (proposed & executed parts)

To disentangle conversion costs vs kernel/threading effects on CPU:

Fairness (thread parity):

Submit-time:

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1


In Torch:

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


Compare three modes end-to-end:

numpy_only — pure NumPy pipeline.

torch_mixed — NumPy pipeline with a JIT NumPy→Torch→NumPy only around einsum; also time conversions separately.

torch_full — keep tensors in Torch throughout (no in/out conversions).

Replicates & reporting: 3× per (file, mode) → report mean ± std. Normalize times by N² to show species scaling cleanly.

Decision logic (“what would convince us”):

If conversion is meaningful → torch_full < torch_mixed beyond noise, and measured conversion share is large.

If kernels/threading dominate → Torch beats NumPy similarly in both Torch modes, and conversion share is small.

Results to date support the kernel/threading story.

Reproduce Locally

Place your bundles in the repo root:

profiling_artifacts.tgz

profiling_local.zip (optional)

Ingest & plot:

python scripts/organize_artifacts.py --chtc profiling_artifacts.tgz --local profiling_local.zip
python scripts/plot_results.py
python scripts/export_env.py > results/tables/env_report.json


(Optional) Add cProfile images into results/figures/cprofile/ and link them in this README.

Recommendations / Next Steps

CPU path (current): Prefer Torch-full pipelines for contraction-heavy codepaths. Keep thread parity when comparing to NumPy.

Apple Silicon (MPS): To scale beyond tiny runs, reduce per-call overhead:

batch contractions (fewer launches),

avoid per-step host↔device copies (keep data resident),

ensure no silent dtype promotions,

check for MPS-unsupported ops and avoid CPU fallbacks (or gate to CPU cleanly).

GPU on CUDA: For steady loops, try torch.compile(mode="reduce-overhead") and CUDA Graphs; consider Triton for hot micro-kernels if shape-stable.

Provenance: Keep posting bundles (timings_*.csv, logs, traces) and push updated figures/tables after each sweep.

Acknowledgements

Thanks to Arthur Lin and Dr. Rose Cersonsky for guidance and collaboration.
This repository exists to make the internship work transparent and reproducible for the Cersonsky Lab and the broader community.

Citation

If this repo helps, please cite:

@software{Dahiya_CersonskyLab_AniSOAP_Optimization_2025,
  author = {Dahiya, Tejas and Lin, Arthur and Cersonsky, Rose},
  title  = {Cersonsky Lab — AniSOAP Optimization: Profiling & Acceleration Workflows},
  year   = {2025},
  url    = {https://github.com/Tejas7007/cersonskylab-anisoap-optimization}
}

Save this into your repo

From your repo root:

printf "%s\n" '<PASTE ALL OF THE README CONTENT ABOVE HERE>' > README.md
git add README.md
git commit -m "docs: add full README with history, artifacts, and results"
git push


If you want, I can also generate a short RESULTS.md (provenance timeline + command snippets) and a CITATION.cff.
