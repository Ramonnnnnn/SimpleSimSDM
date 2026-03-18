# README

# SimpleSim

## An Open-Source Python Simulator for SDM-EON RMSSA Benchmarking

SimpleSimSDM is a modular discrete-event simulator designed for
benchmarking Routing, Modulation, Spectrum, and Space Allocation (RMSSA)
strategies in Space-Division Multiplexing Elastic Optical Networks
(SDM-EONs).

The tool emphasizes reproducibility, extensibility, and seamless
integration with Python-based scientific and machine learning workflows.

------------------------------------------------------------------------

## Key Features

-   Modular architecture separating topology, simulation engine,
    allocation logic, and statistics.
-   XML-based topology definition.
-   YAML-based experiment configuration.
-   Native support for multi-core fiber modeling.
-   Crosstalk-aware allocation support.
-   Automated 95% confidence interval computation.
-   Built-in performance visualization (Matplotlib).
-   Extensible RMSSA interface (heuristic and ML-ready).
-   Structured CSV output for reproducible benchmarking.

------------------------------------------------------------------------

## Installation

### Requirements

-   Python 3.11 or higher
-   pip
-   docker

### Instructions

``` bash
git clone  https://github.com/Ramonnnnnn/SimpleSim
cd  https://github.com/Ramonnnnnn/SimpleSim
docker compose up
```


## Supported Performance Metrics

Primary metrics:

-   Bandwidth Blocking Ratio (BBR)
-   Fragmentation
-   Inter-core Crosstalk (CpS)



## Architecture Overview

SimpleSimSDM consists of five logical layers:

1.  Configuration Layer (YAML parameters)
2.  Topology Builder (XML + NetworkX abstraction)
3.  Event Engine (SimPy)
4.  RMSSA Module Interface
5.  Statistics and Visualization

The allocation module is fully decoupled from the event engine, enabling
rapid experimentation with new strategies.


## Reproducibility

SimpleSimSDM promotes reproducible research by:

-   Externalizing experimental parameters
-   Using structured CSV outputs
-   Automatically computing confidence intervals
-   Decoupling allocation logic from simulation core

Experiments can be replicated by sharing configuration and topology
files.

## License

This project is released under the MIT License.

------------------------------------------------------------------------

## Citation

If you use SimpleSimSDM in your research, please cite:

@inproceedings{oliveira2026simplesim, title={SimpleSim: An Open-Source
Python Simulator for SDM-EON Resource Allocation}, author={Oliveira,
Ramon A.}, booktitle={Simpósio Brasileiro de Redes de Computadores
(SBRC)}, year={2026} }

------------------------------------------------------------------------
## Documentation

Detailed documentation is available at:
https://github.com/Ramonnnnnn/SimpleSim/docs

## Contact

Ramon A. Oliveira\
Federal University of Pará (UFPA)\
ramon.oliveira@itec.ufpa.br

## Personalizing Simulation Parameters

To set personalized simulation parameters you must first create a YAML configuration file containing the required simulation parameters.

### Step 1: Create a Configuration File

* Create a `.yaml` file with your desired configuration settings.
* It is recommended to place this file inside the `configuration_files` directory for better organization.

### Step 2: Set the Configuration Path

* Open the `main.py` script.
* Locate the `load_config` variable where the configuration file path is defined.
* Update this path so that it points to your YAML file.

### Step 3: Run the Program

* Execute the main script:

```bash
python main.py
```

The program will load the configuration from the specified YAML file and start execution.

---

## Notes

* Ensure that the YAML file is correctly formatted to avoid runtime errors (look at the provided example).
* You may create multiple configuration files for different experiments and switch between them by updating the path in `main.py`.
* It is good practice to create one configuration file per experiment for better record-keeping.
