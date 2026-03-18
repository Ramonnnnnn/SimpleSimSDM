# Architecture Overview

SimpleSimSDM is a modular discrete-event simulator designed for
benchmarking Routing, Modulation, Spectrum, and Space Allocation (RMSSA)
strategies in Space-Division Multiplexing Elastic Optical Networks
(SDM-EONs).

The architecture follows a layered and decoupled design to promote
extensibility, reproducibility, and seamless integration with external
allocation strategies, including machine learning-based approaches.

------------------------------------------------------------------------

## Architectural Layers

The simulator is organized into five main components:

### 1. Configuration Layer

Simulation parameters are externalized using:

-   YAML files for experiment configuration
-   XML files for topology definitions

This separation ensures that experimental setups can be modified or
replicated without changing the simulator source code.

Key configurable parameters include:

-   Number of cores and spectrum slots
-   Traffic load levels
-   Number of simulation rounds
-   Routing policies
-   Crosstalk enforcement
-   Output directory structure

------------------------------------------------------------------------

### 2. Topology Builder

Network topologies defined in XML format are parsed and converted into
directed graph representations using the NetworkX library.

-   Nodes represent optical switches.
-   Edges represent bidirectional fiber links.
-   Edge attributes store link length, number of cores, and spectrum
    slots.

This abstraction enables routing algorithms and allocation strategies to
operate independently of physical-layer implementation details.

------------------------------------------------------------------------

### 3. Event Engine

The event engine is implemented using the SimPy discrete-event
simulation framework.

Responsibilities:

-   Traffic generation
-   Request scheduling
-   Resource allocation triggering
-   Connection teardown and resource release

The engine maintains the global spectrum occupancy state across all
cores and links, ensuring consistent allocation and deallocation
operations.

------------------------------------------------------------------------

### 4. RMSSA Module Interface

The RMSSA module constitutes the decision layer of the simulator.

For each incoming connection request, the module receives:

-   The selected routing path
-   Spectrum occupancy state along the path
-   Traffic demand parameters (e.g., bandwidth, holding time)

It returns either:

-   A feasible allocation (core + contiguous slot range), or
-   A blocking decision

The interface is intentionally lightweight and decoupled from the event
engine, allowing new strategies to be implemented as independent classes
without modifying other components.

This design supports heuristic, metaheuristic, and machine
learning-based approaches, including reinforcement learning agents.

------------------------------------------------------------------------

### 5. Statistics and Visualization

During simulation execution, performance metrics are collected across
multiple rounds.

Primary metrics include:

-   Bandwidth Blocking Ratio (BBR)
-   Fragmentation
-   Inter-core Crosstalk (CpS)

For each traffic load level, the simulator automatically computes:

-   Mean performance values
-   95% confidence intervals

Results are exported as structured CSV files, and comparative
performance plots are generated using Matplotlib.

------------------------------------------------------------------------

## Data Flow

The simulation workflow follows these steps:

1.  Load configuration (YAML) and topology (XML).
2.  Instantiate the network graph.
3.  Generate traffic events using SimPy.
4.  Trigger RMSSA allocation decisions.
5.  Update spectrum occupancy state.
6.  Collect and aggregate performance metrics.
7.  Export results and generate plots.

------------------------------------------------------------------------

## Design Principles

The architecture is guided by the following principles:

-   **Modularity:** Clear separation between simulation core and
    allocation logic.
-   **Reproducibility:** Externalized configurations and automated
    statistical reporting.
-   **Extensibility:** Easy integration of new RMSSA strategies.
-   **ML Compatibility:** Seamless interoperability with Python-based
    scientific and machine learning ecosystems.

This modular and layered design enables SimpleSimSDM to serve as a
flexible platform for benchmarking both traditional and learning-based
SDM-EON resource allocation strategies.
