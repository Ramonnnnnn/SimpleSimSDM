# BEFORE SIMULATION:
# CHECK NAME OF TOPOLOGY + CSVs (MAIN)
# CHECK IF YOU WANT CROSSTALK (MAIN)
# CEHCK IF YOU WANT MULTI-CRITERIA (MAIN) -> SEE HOW IT IS CALCULATED (TOPOLOGY > Set_edge_Custom_weight)
# CHECK ALLOCATION ALGORITHM (MAIN) -> to add, go to allocator
# name convention: metric_algorithm.csv
# metric name MUST be same as method name for final calculations
# metrics available: "BBR", "fragmentation" & "CpS"


#IMPLEMENTATION IDEAS
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
import math

# Simulation parameters
matrix_rows = 7
matrix_cols = 320
max_attempts = 10000
verbose = False
seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
starting_load = 50
final_load = 600
step = 25
use_multi_criteria = True
consider_crosstalk_threshold = True
region_finding_algorithm = "MMM" # FF BF MMM MF   ###

base_dir = os.path.dirname(__file__)
csv_files = ['BBR_ONE5050(NSF.newest).csv', 'fragmentation_ONE5050(NSF.newest).csv', 'CpS_ONE5050(NSF.newest).csv', 'BCR_ONE5050(NSF.newest).csv', 'crosstalk_ONE5050(NSF.newest).csv']
csv_save_folder = os.path.join(base_dir, "CVSs")
logger = Logger.Logger(csv_save_folder, csv_files)
XML_path = os.path.join(base_dir, "xml/Image-nsf.xml")


def run_simulation_for_load(load):

    parser_object = Parser.XmlParser(XML_path)
    rates = [entry['rate'] for entry in parser_object.get_calls_info()]
    slot_capacity = parser_object.get_slots_bandwidth()
    metrics = Metrics.Metrics(rates, slot_capacity, max_attempts)

    for interval in range(5):
        imposed_load = load
        env = simpy.Environment()
        topology = TopologyBuilder.NetworkXGraphBuilder(XML_path, matrix_rows, matrix_cols)
        traffic_generator_object = TrafficGenerator.TrafficGenerator(XML_path, max_attempts, seed[interval])
        generated_pairs = traffic_generator_object.generate_pairs(max_attempts)
        call_types_dist = traffic_generator_object.generate_normal_distribution_call_types(max_attempts)
        allocator = Allocator.Allocator(env, max_attempts, traffic_generator_object, topology, generated_pairs, call_types_dist, verbose, imposed_load, metrics, seed[interval], interval, use_multi_criteria, consider_crosstalk_threshold, region_finding_algorithm)
        env.process(allocator.allocation_process())
        start_time = time.time()
        env.run()
        end_time = time.time()
        execution_time = end_time - start_time
        elapsed_simulation_time = env.now

        metrics.add_execution_time_of_round(execution_time)
        metrics.add_simulation_time_of_round(elapsed_simulation_time)
        metrics.calculate_end_of_simulation_round_fragmentation()
        metrics.calculate_end_of_simulation_round_bcr(allocator.get_blocked_attempts())
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


if __name__ == "__main__":
    main()
