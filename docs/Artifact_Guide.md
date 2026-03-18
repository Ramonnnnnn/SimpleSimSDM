# Artifact Evaluation Guide

This document provides guidance for artifact evaluation of SimpleSimSDM,
aligned with the SBRC Tool Track artifact assessment criteria.

---

## 1. Overview

SimpleSimSDM is an open-source, Python-based discrete-event simulator for
benchmarking RMSSA strategies in SDM-EON networks. The repository includes:

- Full source code
- Documentation (installation, usage, reproducibility)
- Example configurations
- Docker-based execution environment
- Automated statistical reporting with confidence intervals

Repository URL:
https://github.com/Ramonnnnnn/SimpleSim

License:
MIT License

---

## 2. Artifact Availability (SeloD)

The artifact satisfies availability requirements:

- Public GitHub repository
- Open-source license (MIT)
- Permanent URL
- Complete source code included
- Documentation provided in the `docs/` directory

---

## 3. Artifact Functional (SeloF)

The artifact is functional and can be executed without modifying the code.

To validate functionality:

1. Install dependencies (see Installation.md), or
2. Build and run using Docker

Example execution:

    python main.py --config config/nsf_experiment.yaml

Expected outputs:

- CSV files in `results/`
- Plots in `plots/`
- Automatic 95% confidence interval computation

---

## 4. Artifact Sustainable (SeloS)

The artifact promotes sustainability through:

- Modular architecture
- Clear separation between simulation core and allocation logic
- External configuration files (YAML/XML)
- Docker-based reproducible environment
- Structured documentation

This design allows long-term maintenance and extension of the simulator.

---

## 5. Reproducible Experiments (SeloR)

The artifact supports reproducible experimentation:

- Fixed random seed support via YAML configuration
- Externalized topology definitions
- Automated statistical aggregation
- Structured CSV export
- Deterministic Docker environment

To reproduce paper results:

    python main.py --config config/nsf_experiment.yaml

Or via Docker:

    docker compose up --build

Full instructions are provided in `Reproducibility.md`.

---

## 6. Reviewer Checklist

Reviewers can verify:

- Successful execution without code modification
- Correct CSV generation
- Presence of 95% confidence intervals
- Plot generation
- Reproducibility with fixed seed
- Successful Docker execution

If all steps succeed, the artifact meets the criteria for
Availability, Functionality, Sustainability, and Reproducibility seals.
