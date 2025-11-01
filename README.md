# Cersonsky Lab — AniSOAP Optimization

_Reproducible profiling & optimization of **AniSOAP** (Cersonsky Lab, UW–Madison): scripts, figures, traces, and PyTorch/HPC configs._

**Owner:** Tejas Dahiya ([@Tejas7007](https://github.com/Tejas7007))  
**Mentors:** Arthur Lin, Dr. Rose Cersonsky

---

## Repository Contents

```text
scripts/
├─ organize_artifacts.py   # Ingest .tgz/.zip bundles into results/
├─ plot_results.py         # Produce figures from timings_* CSVs
└─ export_env.py           # Dump Python/PyTorch/CUDA env details

results/
├─ tables/
│  ├─ timings_chtc.csv
│  ├─ timings_local.csv
│  ├─ summary_local.csv
│  └─ env_report.json
├─ figures/
│  ├─ scaling_species.png
│  └─ util_before_after.png
├─ profiler_traces/
└─ logs/
   ├─ chtc/
   └─ local/

examples/
README.md
