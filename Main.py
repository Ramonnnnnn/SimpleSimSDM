import os
import simpy
import time
import Allocator
import TopologyBuilder
import TrafficGenerator
import Metrics
import Logger
import Parser
import InterfaceTerminal
import FinalStatisticsPlotter

# FIX SEEDS!!!!!!!!

# Simulation parameters
matrix_rows = 7
matrix_cols = 320
max_attempts = 10000
#Provide Simulation Details?
verbose = False
# Load Parameters with interval
seed = [1,2,3,4,5,6,7,8,9,10]
starting_load = 500
final_load = 1000
step = 50

# Log: include hops, ee, utilization.
# name convention: metric_algorithm.csv
# metric name MUST be same as method name for final calculations
# metrics available: "BBR", "fragmentation" & "CpS"

base_dir = os.path.dirname(__file__)

csv_files = ['BBR_MMM.csv', 'fragmentation_MMM.csv', 'CpS_MMM.csv']
csv_save_folder = os.path.join(base_dir, "CVSs")
logger = Logger.Logger(csv_save_folder, csv_files)

# Topology Blueprint
XML_path = os.path.join(base_dir, "xml/Image-nsf.xml" )

for load in range(starting_load, final_load + 1, step):
    #Parse XML
    parser_object = Parser.XmlParser(XML_path)
    # Start Metrics Measurement and Calculation
    # All rates practiced (will be useful for fragmentation calculation)
    rates = [entry['rate'] for entry in parser_object.get_calls_info()]
    slot_capacity = parser_object.get_slots_bandwidth()

    # Instantiate metrics (will all be generated according to a 95% confidence interval)
    metrics = Metrics.Metrics(rates, slot_capacity, max_attempts)


    for interval in range(5):
        #Load for this simulation round
        imposed_load = load

        # Create the simulation environment
        env = simpy.Environment()


        # Create Topology
        topology = TopologyBuilder.NetworkXGraphBuilder(XML_path, matrix_rows, matrix_cols)


        # Create source -> Destination pairs according to normal distribution
        traffic_generator_object = TrafficGenerator.TrafficGenerator(XML_path, max_attempts, seed[interval])
        num_pairs_to_generate = max_attempts
        generated_pairs = traffic_generator_object.generate_pairs(num_pairs_to_generate)


        #Create accompanying call-type normal distribution
        num_call_types_dist = num_pairs_to_generate
        call_types_dist = traffic_generator_object.generate_normal_distribution_call_types(num_call_types_dist)

        # Create the Allocator process
        allocator = Allocator.Allocator(env, max_attempts, traffic_generator_object, topology, generated_pairs, call_types_dist, verbose, imposed_load, metrics, seed[interval], interval) #Inicializa
        env.process(allocator.allocation_process()) # Calls

        # Run the simulation
        start_time = time.time()
        env.run()
        end_time = time.time()

        # Calculate the total execution time
        execution_time = end_time - start_time

        # Calculate the elapsed simulation time
        elapsed_simulation_time = env.now


        #End of round calculations
        metrics.add_execution_time_of_round(execution_time)
        metrics.add_simulation_time_of_round(elapsed_simulation_time)
        metrics.calculate_end_of_simulation_round_fragmentation()
        metrics.calculate_end_of_simulation_round_bbr(allocator.get_blocked_attempts())
        metrics.calculate_end_of_simulation_round_CpS()
        #Reset round_instance_specific data
        metrics.reset_roundwise_stats()

    print(f"Mean execution time: {metrics.calculate_mean_execution_time_of_round():.2f} seconds")
    print(f"Elapsed simulation time: {metrics.calculate_mean_simulation_time_of_round():.2f} seconds")

    # final_frag, confidence = metrics.calculate_final_fragmentation_and_confidence_interval()
    # logger.add_datapoint('fragmentation_FF.csv', load, final_frag, confidence)
    # final_BBR, confidence = metrics.calculate_final_BBR_and_confidence()
    # logger.add_datapoint('BBR_FF.csv', load, final_BBR, confidence)

    for csv_file in csv_files:
        metric_name, algorithm = csv_file.split('_')
        method_name = f"calculate_final_{metric_name}_and_confidence_interval"
        method = getattr(metrics, method_name)
        final_metric, confidence = method()
        logger.add_datapoint(csv_file, load, final_metric, confidence)


