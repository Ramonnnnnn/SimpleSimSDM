# Usage Examples

This document provides practical examples for running simulations
with SimpleSimSDM using different RMSSA strategies and configurations.

---

## 1. Basic Execution Example

Run the default benchmark configuration:

```
python main.py --config config/nsf_experiment.yaml
```

This will:

- Load the NSF topology
- Execute multiple traffic load levels
- Perform multiple simulation rounds
- Generate CSV files with 95% confidence intervals
- Produce performance plots

Output files are stored in:

- `csv_metric_files/`
- `plots/`

---

## 2. Running Different RMSSA Strategies

To evaluate different allocation strategies, modify the `region_finding_algorithm` field
in the YAML configuration file. We provide a few baselines, such as the ubiquitous First-Fit
and Best-Fit approaches.

The former baseline will attempt allocation at the first continuous and contiguous 
free-slot region capable of satisfying the spectrum requirements for a given call while remaining
under strict crosstalk thresholds, and the latter will attempt allocation at the shortest continuous
and contiguous free-slot region capable of providing enough spectral resources.

Custom RMSSA approaches can be developed according to one's research parameters and design policies,
as long as the final candidate allocation regions are formatted as a python dictionary `{key:[(1,2),(1,3),(1,4)], key2: [(2,40),(2,41),(2,42),...]}: `.


## 3. Custom Traffic Load Levels

You can define custom traffic loads intervals in your YAML file by setting `starting_load` and `final_load` to the
desired values. The `step` parameter defines how many Erlangs will be increased at every simulation round until
the current simulation load is equal to `final_load`:

```
starting_load = 600
final_load = 1000
step = 25
```



---
## 4. Confidence Interval

By setting the `rounds_per_load` parameter you can choose how many simulation rounds (how many samples) will be 
used to compute the 95% confidence interval:


---
## 5. Enabling Crosstalk Constraints

To enforce strict crosstalk thresholds to light-path allocation, in accordance with the worst-case
scenario for RMSSA in SDM-EONs, set the `consider_crosstalk_threshold` variable to `true`. This should guarantee a
stricter, more precise representation of resource allocation in SDM-EONs



If set to `false`, allocation ignores crosstalk constraints.

---

## 6. Setting a Fixed Random Seed

For deterministic experiments:

```
random_seed: 42
```

This ensures consistent traffic generation across runs.

---

## 7. Running Inside Docker

Build the container:

```
docker build -t simplesim .
```

Run the default experiment:

```
docker run --rm simplesim
```

To run a custom configuration:

```
docker run --rm simplesim python main.py --config config/custom.yaml
```

---

## 8. Interpreting Output

Each CSV file contains:

- Load level
- Mean performance metrics
- 95% confidence intervals

Plots include:

- Bandwidth Blocking Ratio (BBR)
- Fragmentation
- Inter-core Crosstalk (CpS)

These results can be directly used for benchmarking and comparison studies.
