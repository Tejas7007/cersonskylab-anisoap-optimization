# Cersonsky Lab — AniSOAP Optimization

_Reproducible profiling & optimization of **AniSOAP** (Cersonsky Lab, UW–Madison): scripts, figures, traces, and PyTorch/HPC configs._

**Owner:** Tejas Dahiya ([@Tejas7007](https://github.com/Tejas7007))  
**Mentors:** Arthur Lin, Dr. Rose Cersonsky

---

## Table of Contents
- [Repository Contents](#repository-contents)
- [Artifacts Intake](#artifacts-intake)
- [TL;DR Results](#tldr-results)
- [Figures](#figures)
- [Detailed Worklog](#detailed-worklog)
- [Thread Parity (fair CPU tests)](#thread-parity-fair-cpu-tests)
- [Reproduce Locally](#reproduce-locally)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## Repository Contents

```text
scripts/
├─ organize_artifacts.py
├─ plot_results.py
├─ export_env.py
└─ make_plots.py

results/
├─ tables/
│  ├─ timings.csv
│  ├─ timings_chtc.csv
│  ├─ timings_local.csv
│  ├─ summary_local.csv
│  ├─ env_report.json
│  ├─ combined_from_metrics.csv
│  ├─ ellipsoids_numpy.metrics.json
│  ├─ ellipsoids_torch.metrics.json
│  ├─ one_species_numpy.metrics.json
│  ├─ one_species_torch.metrics.json
│  ├─ three_species_numpy.metrics.json
│  ├─ three_species_torch.metrics.json
│  ├─ four_species_numpy.metrics.json
│  └─ four_species_torch.metrics.json
├─ figures/
│  ├─ wall_time_by_system.png
│  └─ wall_time_vs_species.png
├─ profiler_traces/
└─ logs/
   ├─ chtc/
   └─ local/

examples/
README.md
Figures

Wall time by system & backend


Wall time vs #Species
