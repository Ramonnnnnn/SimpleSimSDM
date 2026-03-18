# Reproducibility Guide

This document describes how to reproduce the experimental results
presented in the SimpleSim paper and how to validate the
functionality of the simulator.

------------------------------------------------------------------------

## 1. System Requirements

-   Python 3.10 or higher
-   pip package manager
-   Operating Systems: Linux, macOS, or Windows
-   Minimum 8 GB RAM recommended.

All dependencies are listed in `pyproject.toml`.

------------------------------------------------------------------------

## 2. Installation/Running (Docker)

Clone the repository:

    git clone https://github.com/Ramonnnnnn/SimpleSim.git
Access repository:

    cd SimpleSimSDM
Build and run docker container with demonstration parameters:

    docker compose up --build



------------------------------------------------------------------------

## 3. Reproducing the Experiments from the Paper

The benchmark scenario used in the paper can be reproduced using the
predefined configuration files available in the `config/` directory. To run this scenario,
follow the procedures described above.



This configuration includes:

-   NSF topology
-   Multi-core fiber modeling
-   Multiple traffic load levels
-   5 simulation rounds per load
-   95% confidence interval computation

------------------------------------------------------------------------

## 4. Expected Outputs

After execution, the following artifacts will be generated:

-   `results/*.csv` --- structured performance metrics
-   `plots/*.png` --- comparative performance curves

The CSV files contain:

-   Mean metric values per load level
-   95% confidence intervals
-   Blocking statistics
-   Fragmentation
-   Crosstalk metrics

Generated plots should match the trends reported in the published
article.

------------------------------------------------------------------------

## 5. Deterministic Execution

For fully reproducible results, define a random seed in the YAML
configuration:

    random_seed: 42

Using a fixed seed ensures consistent traffic generation and allocation
behavior across executions.



------------------------------------------------------------------------

## 7. Artifact Validation Checklist

Reviewers can verify:

-   Successful execution without code modification
-   Automatic CSV generation
-   95% confidence interval computation
-   Plot generation
-   Reproducibility using fixed random seed

If all steps above succeed, the artifact satisfies availability,
functionality, and reproducibility requirements.
