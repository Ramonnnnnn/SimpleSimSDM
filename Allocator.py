import numpy as np
from Modulation import Modulation
import InterfaceTerminal
import LightPath

# RESOURCE ALLOCATION ALGOs
import FirstFit
import SmallestFit
import MMM
import MF
import FcaRcsa


# Reinforcement Learning Resources
# from stable_baselines3 import PPO


class Allocator:
    def __init__(self, env, max_attempts, traffic_generator_obj, topology, s_d_distribution, call_type_distribution,
                 inter_arrival_time_distribution, call_duration_distribution,
                 verbose, imposed_load, metrics, seed, interval, use_multi_criteria, consider_crosstalk_threshold,
                 region_finding_algorithm, rl_environment, trained_model_path, max_episode_length, total_timesteps):
        # Env
        self.env = env
        self.verbose = verbose
        self.seed = seed

        # Number of attempts to update progress bar:
        self.univ_attempt = 0
        self.interval = interval  # we simulate n simulations in the same load to obtain error margin. This point to which step we are right now

        # Number of simulations per_round
        self.max_attempts = max_attempts  # usually 10.000
        # Load in erlangs
        self.imposed_load = imposed_load  # set parameter

        # Statistics
        self.calls_made = 0
        self.blocked_attempts = 0
        self.blocked_bandwidth = 0
        self.successful_bandwidth = 0

        # Traffic
        self.traffic_generator_obj = traffic_generator_obj  # instantiate
        self.node_list = traffic_generator_obj.get_node_list()  # Nodes: ['1', '2', ...]
        self.call_info = traffic_generator_obj.get_call_info()
        self.traffic_info = traffic_generator_obj.get_traffic_info()

        # Topology
        self.topology = topology
        self.graph = topology.get_graph()
        if verbose:
            topology.draw_graph()

        if self.seed is not None:
            np.random.seed(self.seed)

        # Distributions
        self.generated_pairs = s_d_distribution
        self.num_call_types_dist = call_type_distribution
        self.inter_arrival_time_distribution = inter_arrival_time_distribution
        self.call_duration_distribution = call_duration_distribution
        # scale: average time between calls || size: number of samples
        self.time_between_calls = np.random.default_rng().exponential(scale=4,
                                                                      size=self.max_attempts)  #Soon to deprecated

        # Calculations
        load = imposed_load
        mean_holding_time = traffic_generator_obj.get_mean_holding_time()  # Sum(holding_time * (weight/total_weight)) p/ for every call type.
        mean_rate = traffic_generator_obj.get_mean_rate()  # Sum(rate * (weight/total_weight)) '' ''
        max_rate = traffic_generator_obj.get_max_rate()  # Max_rate parameter of the XML

        #self.mean_arrival_time = (mean_holding_time * (mean_rate / max_rate)) / load #Soon to be deprecated

        # METRICS
        self.metrics = metrics

        # Light paths - info on every allocated request
        self.lightpath_manager = LightPath.Lightpath()

        # MULTI-CRITERIA
        self.use_multi_criteria = use_multi_criteria

        # Crosstalk
        self.consider_crosstalk_threshold = consider_crosstalk_threshold

        # Region Finding Algorithm
        self.region_finding_algorithm = region_finding_algorithm

        # Modulation
        self.modulation_histogram = []
        self.path_length_histogram = []
        self.denied_modulation_histogram = []
        self.denied_path_length_histogram = []
        self.served_nodes = []
        self.denied_nodes = []



        if self.region_finding_algorithm == "reinforcement_agent_test":
            from stable_baselines3 import PPO
            # RL Testing variable
            self.spectrum_current_attempt = None
            self.demanded_slots_current_attempt = None
            self.rl_list_of_regions = {} # Empty dict - Legacy format
            # Check environment
            if rl_environment == "Tetris1":
                from rl_environments.TetrisResourceAllocation import TetrisResourceAllocation  # Change training to testing approach
                self.rl_env = TetrisResourceAllocation(cores=7, slots=320, max_episode_length=max_episode_length, total_timesteps=total_timesteps, reference_to_allocator=self)
                self.model = PPO.load(trained_model_path, env=self.rl_env, deterministic=True)  # calls step n_steps times
            elif rl_environment == "Tetris3":
                from rl_environments.TetrisResourceAllocation3 import TetrisResourceAllocation  # Change training to testing approach
                self.rl_env = TetrisResourceAllocation(cores=7, slots=320, max_episode_length=max_episode_length, total_timesteps=total_timesteps, reference_to_allocator=self)
                self.model = PPO.load(trained_model_path, env=self.rl_env, deterministic=True)  # calls step n_steps times
            elif rl_environment == "Tetris4_v4":
                from rl_environments.TetrisResourceAllocation4 import TetrisResourceAllocation  # Change training to testing approach
                self.rl_env = TetrisResourceAllocation(cores=7, slots=320, max_episode_length=max_episode_length, total_timesteps=total_timesteps, reference_to_allocator=self)
                self.model = PPO.load(trained_model_path, env=self.rl_env, deterministic=True)  # calls step n_steps times
            else:
                raise KeyError("Environment not found!")

    # Does the heavy lifting: Source, destination,
    def allocate_slots(self, attempt):
        # Should calculate periodic statistics
        if attempt % 1 == 0:
            self.metrics.mean_frag_topology(self.topology.get_all_edge_matrices())
            self.metrics.mean_CpS_topology(self.topology.get_all_edge_matrices())
            # single load progress bar:
            InterfaceTerminal.InterfaceTerminal.print_progress_bar(attempt, self.max_attempts,
                                                                   prefix=f'Simulation Progress: Load={self.imposed_load}, Interval={self.interval}',
                                                                   suffix='', length=50)
        # Measure crosstalk
        self.metrics.mean_crosstalk_topology(self.topology.get_all_crosstalk_edge_matrices())

        if self.verbose:
            print(f"Call-Number: {attempt}")
            print(f"(Source/Destination): {self.generated_pairs[attempt][0]} -> {self.generated_pairs[attempt][1]}")
            print(f"rate: {self.call_info[self.num_call_types_dist[attempt]]['rate']}")

        if self.use_multi_criteria == False:
            # SHORTEST PATHS TO THE SRC-DST
            shortest_paths = self.topology.find_n_shortest_paths(self.graph, self.generated_pairs[attempt][0],
                                                                 self.generated_pairs[attempt][1], 5)
            rate = self.call_info[self.num_call_types_dist[attempt]]['rate']  # 25, 50... 1000
        if self.use_multi_criteria == True:
            # SHORTEST MULTI-CRITERIA PATH
            shortest_paths = self.topology.find_n_shortest_weighted_paths(self.graph, self.generated_pairs[attempt][0],
                                                                          self.generated_pairs[attempt][1], 5)
            rate = self.call_info[self.num_call_types_dist[attempt]]['rate']  # 25, 50... 1000

        if self.region_finding_algorithm == "fca_rcsa":
            # Sort paths according to their fragmentation
            shortest_paths = self.rank_paths_by_fragmentation(shortest_paths)

        self.calls_made += 1
        for i in range(len(shortest_paths)):
            # Modulation and demand
            distance = self.topology.get_physical_distance(shortest_paths[i])
            modulation_instance = Modulation()
            modulation = modulation_instance.get_modulation_by_distance(distance)
            # Allocate
            successful = False
            if modulation == -1:
                continue  # Skip to the next path if modulation is -1

            bandwidth = modulation_instance.get_bandwidth(
                modulation)  # The shorter the distance, the larger is the bandwidth (up to 75.0)
            demanded_slots = int(rate / bandwidth)
            if demanded_slots == 0:
                demanded_slots = 1

            # Spectrum

            if self.consider_crosstalk_threshold:
                available_spectrum = self.topology.precise_slot_xt_available_spectrum_seven_core_MCF(shortest_paths[i],
                                                                                                     modulation)
            else:
                available_spectrum = self.topology.or_matrices_along_path(shortest_paths[i])

            # Store spectrum to be retrieved by RL model
            self.spectrum_current_attempt = available_spectrum
            self.demanded_slots_current_attempt = demanded_slots

            if self.region_finding_algorithm == 'FF':
                # Region-Finding-Algorithm
                first_fit = FirstFit.FirstFit(available_spectrum)  # instantiate class
                first_fit.find_connected_components()  # call region finding method
                list_of_allocable_regions = first_fit.get_connected_components()  # store
            elif self.region_finding_algorithm == 'BF':
                best_fit = SmallestFit.SmallestFit(available_spectrum)
                best_fit.find_connected_components()
                list_of_allocable_regions = best_fit.get_connected_components()
            elif self.region_finding_algorithm == 'MMM':
                mmm = MMM.MeenyMinyMo(available_spectrum)
                mmm.find_connected_components()
                list_of_allocable_regions = mmm.get_connected_components()
            elif self.region_finding_algorithm == 'MF':
                mf = MF.MeenyFirst(available_spectrum)
                mf.find_connected_components()
                list_of_allocable_regions = mf.get_connected_components()
            elif self.region_finding_algorithm == "fca_rcsa":
                # shortest paths were ranked above
                fca_rcsa = FcaRcsa.FcaRcsa(available_spectrum, demanded_slots)
                fca_rcsa.frag_coef_rcsa()
                list_of_allocable_regions = fca_rcsa.output_dict()
            elif self.region_finding_algorithm == "reinforcement_agent_test":
                state, info = self.rl_env.reset()  # Reset once at the start

                done = False
                while not done:  # Keep predicting until allocation is complete
                    action = self.model.predict(state)
                    _, _, done, done, _ = self.rl_env.step(action)  # Step until done
                # Retrieve the updated dictionary of allocable regions
                list_of_allocable_regions = self.rl_list_of_regions
                #print("breakpoint")

            else:
                raise KeyError(f"Region finding algorithm:  {self.region_finding_algorithm} not recognized.")

            # Slots to allocate
            slots_to_allocate = demanded_slots  # How many
            if self.verbose:
                print(f"Attempt {attempt}: Demand of {slots_to_allocate} slots")

            # Plant Seed
            if self.seed is not None:
                np.random.seed(self.seed)

            # Allocate
            successful = False
            for id, region in list_of_allocable_regions.items():
                if len(region) >= demanded_slots:
                    # Allocation
                    self.lightpath_manager.add_lightpath(attempt, shortest_paths[i], region, modulation, demanded_slots)
                    self.allocate_along_path(shortest_paths[i], region, demanded_slots, modulation, attempt)
                    successful = True
                    self.successful_bandwidth += rate
                    self.env.process(self.deallocate_slots(shortest_paths[i], region, demanded_slots,
                                                           self.call_duration_distribution[attempt], attempt,
                                                           modulation))
                    break
            if successful:
                break

        if not successful:
            self.blocked_attempts += 1
            self.blocked_bandwidth += rate

    def deallocate_slots(self, shortest_path_i, region, demanded_slots, deallocate_time, attempt, modulation):
        yield self.env.timeout(deallocate_time)
        # print(f"Deallocating slots: {demanded_slots} slots")
        self.deallocate_along_path(shortest_path_i, region, demanded_slots, modulation, attempt)

    def allocation_process(self):
        attempt = 0
        if self.seed is not None:
            np.random.seed(self.seed)

        while attempt < self.max_attempts:
            # Time distribution to initiate calls
            yield self.env.timeout(self.inter_arrival_time_distribution[attempt])
            self.allocate_slots(attempt)
            attempt += 1

    def allocate_along_path(self, shortest_path_i, region, demanded_slots, modulation, lightpath_ID):

        #print(f"Time as allocation is made: {self.env.now}, ID: {lightpath_ID}, env.now: {self.env.now}")
        # Iterate over pairs of nodes in the path
        affected_lightpath_ids = []
        for i in range(len(shortest_path_i) - 1):
            src, dst = shortest_path_i[i], shortest_path_i[i + 1]

            # Get the spectrum matrix associated with the edge
            edge_matrix_path = self.topology.get_edge_matrix(src, dst)
            if self.consider_crosstalk_threshold:
                # Get modulation matrix for the edge:
                modulation_matrix = self.topology.get_modulation_edge_matrix(src, dst)
                # Get light-path ID matrix
                lightpath_matrix = self.topology.get_lightpath_matrix(src, dst)

            # Perform allocation operation
            for count in range(demanded_slots):
                row, col = region[count]
                edge_matrix_path[row][col] = 1
                if self.consider_crosstalk_threshold:
                    modulation_matrix[row][col] = modulation
                    lightpath_matrix[row][col] = lightpath_ID
                    # Include affected light paths for future update
                    neighbors = self.figure_neighbors_seven_core_MCF(row)
                    for nei in neighbors:
                        if lightpath_matrix[nei][col] != -1 and lightpath_matrix[nei][
                            col] not in affected_lightpath_ids:
                            affected_lightpath_ids.append(lightpath_matrix[nei][col])

            # Update in the topology
            self.topology.update_matrix_in_topology(src, dst, edge_matrix_path)
            if self.consider_crosstalk_threshold:
                self.topology.update_modulation_matrix_in_topology(src, dst, modulation_matrix)
                self.topology.update_lightpath_matrix_in_topology(src, dst, lightpath_matrix)

        if self.consider_crosstalk_threshold:
            # Write new XT values for allocated path.
            self.topology.calculate_and_set_max_XT_for_lightpath_seven_core_MCF(
                self.lightpath_manager.get_attribute(lightpath_ID, "path"),
                self.lightpath_manager.get_attribute(lightpath_ID, "occupied_slot_list"))
            # Update affected light paths
            for lp_id in affected_lightpath_ids:
                self.topology.calculate_and_set_max_XT_for_lightpath_seven_core_MCF(
                    self.lightpath_manager.get_attribute(lp_id, "path"),
                    self.lightpath_manager.get_attribute(lp_id, "occupied_slot_list"))

        if self.use_multi_criteria == True:
            for i in range(len(shortest_path_i) - 1):
                src, dst = shortest_path_i[i], shortest_path_i[i + 1]

                # update occupancy
                self.topology.set_edge_occupancy(src, dst)
                # update fragmentation
                self.topology.set_edge_fragmentation(src, dst)
                # update_weight
                self.topology.set_edge_custom_weight(src, dst)

    @staticmethod
    def count_occ_neighbors(matrix, neighbor_core1, neighbor_core2, slot):
        aux = 0
        if matrix[neighbor_core1][slot] == 1:
            aux += 1
        if matrix[neighbor_core2][slot] == 1:
            aux += 1
        return aux

    def deallocate_along_path(self, shortest_path_i, region, demanded_slots, modulation, lightpath_ID):
        # print(f"Time as deallocation is made: {self.env.now}, ID: {lightpath_ID}, env.now: {self.env.now}")
        # Iterate over pairs of nodes in the path
        affected_lightpath_ids = []
        for i in range(len(shortest_path_i) - 1):
            src, dst = shortest_path_i[i], shortest_path_i[i + 1]

            # Get the spectrum matrix associated with the edge
            edge_matrix_path = self.topology.get_edge_matrix(src, dst)
            if self.consider_crosstalk_threshold:
                # Get modulation matrix for the edge:
                modulation_matrix = self.topology.get_modulation_edge_matrix(src, dst)
                # Get light path ID matrix
                lightpath_matrix = self.topology.get_lightpath_matrix(src, dst)
                # Get noise matrix
                noise_matrix = self.topology.get_noise_edge_matrix(src, dst)

            # Perform de allocation operation
            for count in range(demanded_slots):
                row, col = region[count]
                edge_matrix_path[row][col] = 0
                if self.consider_crosstalk_threshold:
                    modulation_matrix[row][col] = -1
                    lightpath_matrix[row][col] = -1
                    noise_matrix[row][col] = -61.93  #Write base values directly
                    # Include affected lightpaths for future update
                    neighbors = self.figure_neighbors_seven_core_MCF(row)
                    for nei in neighbors:
                        if lightpath_matrix[nei][col] != -1 and lightpath_matrix[nei][
                            col] not in affected_lightpath_ids:
                            affected_lightpath_ids.append(lightpath_matrix[nei][col])

            # Update in the topology (before moving to next link)
            self.topology.update_matrix_in_topology(src, dst, edge_matrix_path)
            if self.consider_crosstalk_threshold:
                self.topology.update_modulation_matrix_in_topology(src, dst, modulation_matrix)
                self.topology.update_lightpath_matrix_in_topology(src, dst, lightpath_matrix)

        if self.consider_crosstalk_threshold:
            #Remove de-allocated light-path (after all links are looked)
            self.lightpath_manager.remove_lightpath(lightpath_ID)
            # Update affected light paths
            for lp_id in affected_lightpath_ids:
                self.topology.calculate_and_set_max_XT_for_lightpath_seven_core_MCF(
                    self.lightpath_manager.get_attribute(lp_id, "path"),
                    self.lightpath_manager.get_attribute(lp_id, "occupied_slot_list"))

        if self.use_multi_criteria:
            for i in range(len(shortest_path_i) - 1):
                src, dst = shortest_path_i[i], shortest_path_i[i + 1]

                # update occupancy
                self.topology.set_edge_occupancy(src, dst)
                # update fragmentation
                self.topology.set_edge_fragmentation(src, dst)
                # update_weight
                self.topology.set_edge_custom_weight(src, dst)

    def rank_paths_by_fragmentation(self, shortest_paths):
        fragmentation_scores = []

        # Calculate the fragmentation for each path
        for i, path in enumerate(shortest_paths):
            spectrum = self.topology.or_matrices_along_path(path)
            fragmentation = self.calculate_fragmentation_fca_rcsa(spectrum)
            fragmentation_scores.append((fragmentation, path))

        # Sort paths based on the fragmentation score in ascending order
        fragmentation_scores.sort(key=lambda x: x[0])

        # Reorder shortest_paths based on sorted fragmentation
        sorted_paths = [path for _, path in fragmentation_scores]

        return sorted_paths

    def calculate_fragmentation_fca_rcsa(self, matrix):
        total_free_slots = 0
        largest_frag = 0
        cores = len(matrix)
        slots = len(matrix[0])

        for core in range(cores):
            size_frag = 0
            for slot in range(slots):
                while not matrix[core][slot] and slot <= slots - 2:
                    size_frag += 1
                    total_free_slots += 1
                    slot += 1

                if size_frag >= largest_frag:
                    largest_frag = size_frag
                size_frag = 0  # Reset fragment size counter

        # Calculate the fragmentation coefficient
        if largest_frag <= 1 or total_free_slots == 0:
            return 1
        else:
            fragmentation = largest_frag / total_free_slots
            return 1 - fragmentation

    def print_statistics(self):
        proportion_blocked = self.blocked_attempts / self.calls_made * 100
        print(f"\nCalls Made: {self.calls_made}")
        print(f"Blocked calls: {self.blocked_attempts}")
        print(f"Proportion of blocked calls: {proportion_blocked:.2f}%")

    def get_blocked_attempts(self):
        return self.blocked_attempts

    def get_univ_attempt(self):
        return self.univ_attempt

    def get_blocked_bandwidth(self):
        return self.blocked_bandwidth

    def get_successful_bandwidth(self):
        return self.successful_bandwidth

    @staticmethod
    def figure_neighbors(core, total_cores):
        if core == 0:
            neighbor1 = total_cores - 1
            neighbor2 = 1
        elif core == total_cores - 1:
            neighbor1 = total_cores - 2
            neighbor2 = 0
        else:
            neighbor1 = core + 1
            neighbor2 = core - 1
        return neighbor1, neighbor2

    @staticmethod
    def figure_neighbors_seven_core_MCF(core):
        neighbors_map = {
            0: [1, 5, 6],
            1: [0, 2, 6],
            2: [1, 3, 6],
            3: [2, 4, 6],
            4: [3, 5, 6],
            5: [0, 4, 6],
            6: [0, 1, 2, 3, 4, 5]
        }
        return neighbors_map.get(core, [])

    def get_max_attempts(self):
        return self.max_attempts
