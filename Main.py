# BEFORE SIMULATION:
# CHECK NAME OF TOPOLOGY + CSVs (MAIN)
# CHECK IF YOU WANT CROSSTALK (MAIN)
# CHECK IF YOU WANT MULTI-CRITERIA (MAIN) -> SEE HOW IT IS CALCULATED (TOPOLOGY > Set_edge_Custom_weight)
# CHECK ALLOCATION ALGORITHM (MAIN) -> to add, go to allocator
# name convention: metric_algorithm.csv
# metric name MUST be same as method name for final calculations
# metrics available: "BBR", "fragmentation" & "CpS"
#
# IMPLEMENTATION IDEAS
#   BETTER SEED IMPLEMENTATION
#   Log: include hops, ee.
import os
import simpy
import time
import Allocator
import TopologyBuilder
import TrafficGenerator
import Metrics
import Logger
import Parser
import concurrent.futures
import csv_metric_files.SortResults
import FinalStatisticsPlotter

#Get configurations.
#A .yaml file can be used to better map the configurations used for every run
# Assure load_config loads the right configuration file


from ConfigLoader import load_config
import os
config_path = os.getenv("CONFIG_PATH", "configuration_files/config.yaml")
cfg = load_config(config_path)

# Simulation parameters
matrix_rows = cfg["matrix_rows"] #
matrix_cols = cfg["matrix_cols"]
max_attempts = cfg["max_attempts"]
rounds_per_load = cfg["rounds_per_load"]
verbose = cfg["verbose"]
seed = cfg["seed"]
starting_load = cfg["starting_load"]
final_load = cfg["final_load"]
step = cfg["step"]
use_multi_criteria = cfg["use_multi_criteria"]
consider_crosstalk_threshold = cfg["consider_crosstalk_threshold"]

# Decides core and spectrum assignment strategy
region_finding_algorithm = cfg["region_finding_algorithm"]

# RL parameters (only necessary if using RL, otherwise NONE)
rl_environment = cfg["rl_environment"]
max_episode_length = cfg["max_episode_length"]
total_timesteps = cfg["total_timesteps"]
trained_model_path = cfg["trained_model_path"]

# Logging
log_name = cfg["log_name"]
csv_files = cfg["csv_files"]
csv_save_folder = cfg["csv_save_folder"]
XML_path = cfg["xml_path"]
logger = Logger.Logger(csv_save_folder, csv_files)

if verbose:
    print(f"SDM-EON with: {matrix_rows} cores and {matrix_cols} slots")
    print(f"Rounds per load: {rounds_per_load} (for confidence interval calculation)")
    print(f"Starts at: {starting_load}E, up to {final_load}, with a {step}E step")
    if use_multi_criteria:
        print("Uses multi-criteria routing.")
    else:
        print("Uses minimum distance routing.")

def run_simulation_for_load(load):

    parser_object = Parser.XmlParser(XML_path)
    rates = [entry['rate'] for entry in parser_object.get_calls_info()]
    slot_capacity = parser_object.get_slots_bandwidth()
    metrics = Metrics.Metrics(rates, slot_capacity)

    for interval in range(rounds_per_load):
        imposed_load = load
        env = simpy.Environment()
        topology = TopologyBuilder.NetworkXGraphBuilder(XML_path, matrix_rows, matrix_cols)
        if verbose and imposed_load == starting_load and interval == 0 :
            topology.draw_graph()

        traffic_generator_object = TrafficGenerator.TrafficGenerator(XML_path, seed[interval])
        mean_holding_time = traffic_generator_object.get_mean_holding_time()
        _, _, inter_arrival_times, _ = traffic_generator_object.generate_poisson_events(imposed_load, mean_holding_time, max_attempts)
        max_attempts2 = len(inter_arrival_times)  #
        call_duration_distribution, _ = traffic_generator_object.generate_call_durations(max_attempts2+1, mean_holding_time)
        generated_pairs = traffic_generator_object.generate_pairs(max_attempts2)
        call_types_dist = traffic_generator_object.generate_normal_distribution_call_types(max_attempts2)
        allocator = Allocator.Allocator(env, max_attempts2, traffic_generator_object, topology, generated_pairs, call_types_dist, inter_arrival_times, call_duration_distribution, verbose, imposed_load, metrics, seed[interval], interval, use_multi_criteria, consider_crosstalk_threshold, region_finding_algorithm, rl_environment, trained_model_path, max_episode_length, total_timesteps)
        env.process(allocator.allocation_process())
        start_time = time.time()
        env.run()
        end_time = time.time()
        execution_time = end_time - start_time
        elapsed_simulation_time = env.now
        if verbose:
            print(f"Execution Time: {execution_time} (load: {load}, round: {interval}/{rounds_per_load})")
        metrics.add_execution_time_of_round(execution_time)
        metrics.add_simulation_time_of_round(elapsed_simulation_time)
        metrics.calculate_end_of_simulation_round_fragmentation()
        metrics.calculate_end_of_simulation_round_bcr(allocator.get_blocked_attempts(), allocator.get_max_attempts())
        metrics.calculate_end_of_simulation_round_CpS()
        metrics.calculate_end_of_simulation_round_crosstalk()
        metrics.calculate_end_of_simulation_round_bbr(allocator.get_blocked_bandwidth(), allocator.get_successful_bandwidth())
        metrics.reset_roundwise_stats()

    results = {}
    for csv_file in csv_files:
        metric_name, algorithm = csv_file.split('_')
        method_name = f"calculate_final_{metric_name}_and_confidence_interval"
        method = getattr(metrics, method_name)
        final_metric, confidence = method()
        results[csv_file] = (final_metric, confidence)

    return load, results


def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_simulation_for_load, load) for load in range(starting_load, final_load + 1, step)]
        for future in concurrent.futures.as_completed(futures):
            load, results = future.result()
            for csv_file, (final_metric, confidence) in results.items():
                logger.add_datapoint(csv_file, load, final_metric, confidence)
            print(f"Completed simulation for load {load}")
            sort_results = csv_metric_files.SortResults.sort_csv_by_first_column(csv_save_folder)
            grapher = FinalStatisticsPlotter.MultiCurvePlotter(csv_save_folder, "plots") #updates plot after every simulation round is complete

            # Plot metrics
            grapher.parse_csv_files()
            grapher.plot_curves()


if __name__ == "__main__":
    main()
