AniSOAP Optimization

Goal: Make anisotropic SOAP (AniSOAP) descriptor computation fast, scalable, and reproducible across CPU/GPU backends while preserving descriptor fidelity.

Overview

Problem. Descriptor generation can dominate runtime (wall-time, memory traffic, vectorization limits, I/O), especially across multi-species systems.

Impact. Faster AniSOAP unlocks larger datasets, broader hyper-param sweeps, and smoother deployment in downstream potentials/property models.

This repo provides. A principled, reproducible optimization path with baselines, profiling, species-scaling analysis, and validated speedups.

Artifacts live in results/figures/, results/tables/, and results/logs/.
