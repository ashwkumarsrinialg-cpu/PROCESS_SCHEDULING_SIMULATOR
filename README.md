.
├── paper.pdf
├── scheduler_sim.py
├── final_mobile.zip
├── plots/
│   ├── stat_comparison.png
│   ├── hw_fidelity_delta.png
│   ├── sensitivity_alpha.png
│   ├── sensitivity_temperature.png
│   ├── linux_eevdf_params.png
│   ├── per_class_metrics.png
│   ├── scalability.png
│   ├── sim_robustness.png
│   └── *_allocation_tl.png
└── README.md

# Hybrid EEVDF+MLFQ: A Thermally-Aware, Hardware-Validated CPU Scheduler for Mobile Linux Platforms

**Paper**: [paper.pdf](paper.pdf)  
**Date**: June 28, 2026

## Overview

This repository contains all artefacts for the research paper on a hybrid EEVDF+MLFQ scheduler designed for mobile Linux/Android platforms.

## File Structure


## Components

### 1. `scheduler_sim.py` — Hardware-Informed Simulator

The main discrete-event simulator used to generate all results and plots in the paper.  
It supports an optional **hardware fidelity mode** calibrated from real Pixel 7 measurements.

```bash
# Standard Monte Carlo runs
python3 scheduler_sim.py --n-runs 200 --n-procs 120

# With hardware fidelity (recommended for paper results)
python3 scheduler_sim.py --n-runs 200 --n-procs 120 --hw-fidelity

2. final_mobile.zip — Additional Components
Contains:

Kernel prototype patch (0001-sched-hybrid.patch)
Android benchmark harness (android_bench/)
Sim-to-real validation scripts (validation/)


3. plots/ Directory
All figures included in paper.pdf:

Statistical comparison
Hardware fidelity impact
Sensitivity studies (α and temperature)
Parameter sweeps and scalability analysis
CPU allocation timelines (Gantt charts)

