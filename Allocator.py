import random
import numpy as np
import TopologyBuilder
import Parser
from Modulation import Modulation
import math
import Metrics
import InterfaceTerminal

#RESOURCE ALLOCATION ALGOs
import FirstFit
import SmallestFit
import MMM

class Allocator:
    def __init__(self, env, max_attempts, traffic_generator_obj, topology, s_d_distribution, call_type_distribution, verbose, imposed_load, metrics, seed, interval):
        #Env
        self.env = env
        self.verbose = verbose
        self.seed = seed

        #Number of attempts to update progress bar:
        self.univ_attempt = 0
        self.interval = interval # we simulate n simulations in the same load to obtain error margin. This point to which step we are right now

        #Number of simulations per_round
        self.max_attempts = max_attempts #usually 10.000
        self.imposed_load = imposed_load #set parameter

        #Statistics
        self.calls_made = 0
        self.blocked_attempts = 0


        #Traffic
        self.traffic_generator_obj = traffic_generator_obj #instantiate
        self.node_list = traffic_generator_obj.get_node_list() # Nodes: ['1', '2', ...]
        self.call_info = traffic_generator_obj.get_call_info()
        self.traffic_info = traffic_generator_obj.get_traffic_info()

        #Topology
        self.topology = topology
        self.graph = topology.get_graph()
        if verbose == True:
            topology.draw_graph()

        if self.seed is not None:
            np.random.seed(self.seed)

        #Distributions
        self.generated_pairs = s_d_distribution
        self.num_call_types_dist = call_type_distribution
        #scale: average time between calls || size: number of samples
        self.time_between_calls = np.random.default_rng().exponential(scale=4, size=self.max_attempts)

        #Calculations
        load = imposed_load
        mean_holding_time = traffic_generator_obj.get_mean_holding_time() # Sum(holding_time * (weight/total_weight)) p/ for every call type.
        mean_rate = traffic_generator_obj.get_mean_rate() # Sum(rate * (weight/total_weight)) '' ''
        max_rate = traffic_generator_obj.get_max_rate() # Max_rate parameter of the XML

        self.mean_arrival_time = (mean_holding_time*(mean_rate/max_rate))/load

        self.metrics = metrics
        

    #Does the heavy lifting: Source, destination,
    def allocate_slots(self, attempt):

        #Should calculate periodic statistics
        if attempt % 100 == 0:
            self.metrics.mean_frag_topology(self.topology.get_all_edge_matrices())
            self.metrics.mean_CpS_topology(self.topology.get_all_edge_matrices())
            # single load progress bar:
            InterfaceTerminal.InterfaceTerminal.print_progress_bar(attempt, self.max_attempts, prefix=f'Simulation Progress: Load={self.imposed_load}, Interval={self.interval}', suffix='', length=50)

        if self.verbose:
            print(f"Call-Number: {attempt}")
            print(f"(Source/Destination): {self.generated_pairs[attempt][0]} -> {self.generated_pairs[attempt][1]}")
            print(f"rate: {self.call_info[self.num_call_types_dist[attempt]]['rate']}")

        #SHORTEST PATH TO THE SRC-DST
        shortest_paths = self.topology.find_n_shortest_paths(self.graph, self.generated_pairs[attempt][0], self.generated_pairs[attempt][1], 5)
        rate = self.call_info[self.num_call_types_dist[attempt]]['rate'] #25, 50... 1000


        self.calls_made += 1
        for i in range(len(shortest_paths)):
            #Modulation and demand
            distance = self.topology.get_physical_distance(shortest_paths[i])
            modulation_instance = Modulation()
            modulation = modulation_instance.get_modulation_by_distance(distance)
            bandwidth = modulation_instance.get_bandwidth(modulation) #The shorter the distance, the larger is the bandwidth (up to 75.0)
            demanded_slots = int(rate/bandwidth)


            #Spectrum
            available_spectrum = self.topology.or_matrices_along_path(shortest_paths[i])
            # if self.verbose:
            #     print(f"Spectrum before allocation: {available_spectrum}")


            #Region-Finding-Algorithm
            first_fit = FirstFit.FirstFit(available_spectrum) #instantiate class
            first_fit.find_connected_components() #call region finding method
            list_of_allocable_regions = first_fit.get_connected_components() #store

            #BEST FIT - FIX NAMING CONVENTIONS
            # first_fit = SmallestFit.SmallestFit(available_spectrum)
            # first_fit.find_connected_components()
            # list_of_allocable_regions = first_fit.get_connected_components()

            #MMM ALGORITHM
            mmm = MMM.MeenyMinyMo(available_spectrum)
            mmm.find_connected_components()
            list_of_allocable_regions = mmm.get_connected_components()



            #Slots to allocate
            slots_to_allocate = demanded_slots # How many
            if self.verbose:
                print(f"Attempt {attempt}: Demand of {slots_to_allocate} slots")

            #Plant Seed
            if self.seed is not None:
                np.random.seed(self.seed)

            #Allocate
            successful = False
            for id, region in list_of_allocable_regions.items():
                if len(region) >= demanded_slots:
                    #Allocation
                    self.allocate_along_path(shortest_paths[i], region, demanded_slots)
                    #self.calls_made += 1
                    successful = True
                    available_spectrum = self.topology.or_matrices_along_path(shortest_paths[i])
                    # if self.verbose:
                    #     print("Success")
                    #     print(f"Spectrum after allocation: {available_spectrum}")
                    #Deallocation
                    deallocate_time = abs(random.normalvariate(mu=1, sigma=1)) #Time to remain allocated -> change later
                    self.env.process(self.deallocate_slots(shortest_paths[i], region, demanded_slots, deallocate_time))
                    break
            if successful:
                break

        if successful == False:
            #print(f"Blocked attempt at allocating: {demanded_slots} slots")
            self.blocked_attempts += 1
            #self.calls_made += 1


    def deallocate_slots(self, shortest_path_i, region, demanded_slots, deallocate_time):
        yield self.env.timeout(deallocate_time)
        #print(f"Deallocating slots: {demanded_slots} slots")
        self.deallocate_along_path(shortest_path_i, region, demanded_slots)

    def allocation_process(self):
        attempt = 0
        if self.seed is not None:
            np.random.seed(self.seed)

        while attempt < self.max_attempts:
            #Time distribution to initiate calls
            yield self.env.timeout(-1 * (self.mean_arrival_time * math.log(random.uniform(0.0, 1.0))))

            self.allocate_slots(attempt)
            attempt += 1



    def allocate_along_path(self, shortest_path_i, region, demanded_slots):

        # Iterate over pairs of nodes in the path
        for i in range(len(shortest_path_i) - 1):
            src, dst = shortest_path_i[i], shortest_path_i[i + 1]

            # Get the matrix associated with the edge
            edge_matrix_path = self.topology.get_edge_matrix(src, dst)

            # Perform allocation operation
            for count in range(demanded_slots):
                row, col = region[count]
                edge_matrix_path[row][col] = 1
            self.topology.update_matrix_in_topology(src, dst, edge_matrix_path)






    def deallocate_along_path(self, shortest_path, region, demanded_slots):

        # Iterate over pairs of nodes in the path
        for i in range(len(shortest_path) - 1):
            src, dst = shortest_path[i], shortest_path[i + 1]
            # Get the matrix associated with the edge
            edge_matrix_path = self.topology.get_edge_matrix(src, dst)

            # Perform allocation operation
            for count in range(demanded_slots):
                row, col = region[count]
                edge_matrix_path[row][col] = 0
            self.topology.update_matrix_in_topology(src, dst, edge_matrix_path)




    def print_statistics(self):
        proportion_blocked = self.blocked_attempts / self.calls_made * 100
        print(f"\nCalls Made: {self.calls_made}")
        print(f"Blocked calls: {self.blocked_attempts}")
        print(f"Proportion of blocked calls: {proportion_blocked:.2f}%")

    def get_blocked_attempts(self):
        return self.blocked_attempts

    def get_univ_attempt(self):
        return self.univ_attempt