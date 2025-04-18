import networkx as nx
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import Parser
import math
import Modulation
import LinkMemory

def linear_to_decibel(value):
    return 10 * math.log10(value)


def add_decibel(v1, v2):
    return 10 * math.log10(math.pow(10, (v1 / 10)) + math.pow(10, (v2 / 10)))


def subtract_decibel(v1, v2):
    return 10 * math.log10(math.pow(10, (v1 / 10)) - math.pow(10, (v2 / 10)))

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


class NetworkXGraphBuilder:

    def __init__(self, xml_file, matrix_rows, matrix_columns):
        self.print_once = False
        self.graph = nx.DiGraph()
        self.matrix_rows = matrix_rows
        self.matrix_columns = matrix_columns
        self.edge_matrices = {}
        self.noise_edge_matrices = {}
        self.modulation_edge_matrices = {}
        self.lightpath_edge_matrices = {}
        self.link_manager = LinkMemory.LinkManager()

        # Parse the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract nodes
        nodes = root.find(".//nodes")
        for node in nodes.iter("node"):
            node_id = node.attrib["id"]
            self.graph.add_node(node_id)

        # Extract links and create edges
        links = root.find(".//links")
        for link in links.iter("link"):
            link_id = link.attrib["id"]
            source = link.attrib["source"]
            destination = link.attrib["destination"]
            delay = float(link.attrib["delay"])
            bandwidth = float(link.attrib["bandwidth"])
            weight = float(link.attrib["weight"])
            distance = float(link.attrib["distance"])

            # Create edge
            self.graph.add_edge(source, destination, id=link_id, delay=delay, bandwidth=bandwidth, weight=weight,
                                distance=distance, occupancy=0.0, fragmentation=0.0, multi_parameter_weight=0.0,
                                cps=0.0, crosstalk=-60)

            # Create spectrum matrix for the edge
            matrix = np.zeros((matrix_rows, matrix_columns))
            noise_matrix = np.full((matrix_rows, matrix_columns), -61.93)
            modulation_matrix = np.full((matrix_rows, matrix_columns), -1)
            lightpath_matrix = np.full((matrix_rows, matrix_columns), -1)
            self.edge_matrices[(source, destination)] = matrix
            self.noise_edge_matrices[(source, destination)] = noise_matrix
            self.modulation_edge_matrices[(source, destination)] = modulation_matrix
            self.lightpath_edge_matrices[(source, destination)] = lightpath_matrix

        # EXTRACT DISTANCE INFO FOR NORMALISATION
        self.distances = [self.graph[src][dst]['distance'] for src, dst in self.graph.edges]
        self.min_distance = min(self.distances)
        self.max_distance = max(self.distances)

        # EXTRACT CALLS INFO
        # Parse XML
        parser_object = Parser.XmlParser(xml_file)
        # Start Metrics Measurement and Calculation
        # All rates practiced (will be useful for fragmentation calculation)
        self.rates = [entry['rate'] for entry in parser_object.get_calls_info()]
        self.slot_capacity = parser_object.get_slots_bandwidth()


    def get_edge_id(self, source, destination):
        return self.graph[source][destination]['id']
    def find_and_print_shortest_paths(self):
        for source in self.graph.nodes:
            for target in self.graph.nodes:
                if source != target:
                    shortest_paths = self.find_n_shortest_paths(self.graph, source, target, 1)
                    if shortest_paths:
                        shortest_path = shortest_paths[0]
                        distance = self.get_physical_distance(shortest_path)
                        if distance > 6300:
                            print(f"({source}->{target}) {distance}km")

    def find_n_shortest_paths(self, graph, source, target, n):
        # Use Jin Y Yen's algorithm to find the n shortest paths O(KNˆ3)
        return list(
            islice(nx.shortest_simple_paths(graph, source, target, weight='weight'), n)
        )

    def find_n_shortest_weighted_paths(self, graph, source, target, n):
        try:
            # Use Jin Y Yen's algorithm to find the n shortest paths O(KNˆ3)
            return list(
                islice(nx.shortest_simple_paths(graph, source, target, weight='multi_parameter_weight'), n)
            )
        except nx.NetworkXNoPath:
            # Return an empty list if no path is found between source and target
            return []

    def get_graph(self):
        return self.graph

    def get_edge_matrix(self, source, destination):
        return self.edge_matrices.get((source, destination))

    def get_modulation_edge_matrix(self, source, destination):
        return self.modulation_edge_matrices.get((source, destination))

    def get_noise_edge_matrix(self, source, destination):
        return self.noise_edge_matrices.get((source, destination))

    def get_lightpath_matrix(self, source, destination):
        return self.lightpath_edge_matrices.get((source, destination))

    def get_all_edge_matrices(self):
        all_matrices = []

        for key, value in self.edge_matrices.items():
            if isinstance(value, np.ndarray):
                all_matrices.append(
                    value.tolist())  # Here I change back to regular arrays. try and keep numpy DSs along the way
        return all_matrices

    def get_all_crosstalk_edge_matrices(self):
        all_matrices = []

        for key, value in self.noise_edge_matrices.items():
            if isinstance(value, np.ndarray):
                all_matrices.append(value.tolist())

        return all_matrices

    def or_matrices_along_path(self, shortest_path):
        result_matrix = np.zeros((self.matrix_rows, self.matrix_columns))  # Has to start with zeros.

        # Iterate over pairs of nodes in the path
        for i in range(len(shortest_path) - 1):
            src, dst = shortest_path[i], shortest_path[i + 1]

            # Get the matrix associated with the edge
            edge_matrix_path = self.get_edge_matrix(src, dst)

            # Perform logical OR operation
            result_matrix = np.logical_or(result_matrix, edge_matrix_path)

        return result_matrix

    def update_matrix_in_topology(self, source, destination, edge_matrix):
        # Update the graph with the modified matrix
        self.graph[source][destination]['matrix'] = edge_matrix

    def update_modulation_matrix_in_topology(self, source, destination, modulation_matrix):
        self.graph[source][destination]['modulation_matrix'] = modulation_matrix

    def update_noise_matrix_in_topology(self, source, destination, noise_matrix):
        self.graph[source][destination]['noise_matrix'] = noise_matrix

    def update_lightpath_matrix_in_topology(self, source, destination, lightpath_matrix):
        self.graph[source][destination]['lightpath_matrix'] = lightpath_matrix

    def get_physical_distance(self, path_i):
        distance = 0
        # Iterate over pairs of nodes in the path
        for i in range(len(path_i) - 1):
            src, dst = path_i[i], path_i[i + 1]

            # Check pair for distance
            distance += int(self.graph[src][dst]["distance"])

        return distance

    def get_single_link_distance_meters(self, src, dst):
        return int(self.graph[src][dst]['distance'] * 1000)

    def set_edge_occupancy(self, source, destination):
        if self.graph.has_edge(source, destination):
            edge_matrix = self.get_edge_matrix(source, destination)
            total_elements = edge_matrix.size
            num_ones = np.count_nonzero(edge_matrix)
            occupancy = (num_ones / total_elements) * 100  # In percent
            self.graph[source][destination]['occupancy'] = occupancy
        else:
            raise ValueError(f"Edge from {source} to {destination} does not exist.")

    def get_edge_occupancy(self, source, destination):
        if self.graph.has_edge(source, destination):
            return self.graph[source][destination]['occupancy']
        else:
            raise ValueError(f"Edge from {source} to {destination} does not exist.")

    def set_edge_cps(self, source, destination):
        if self.graph.has_edge(source, destination):
            edge_matrix = self.get_edge_matrix(source, destination)
            # CPS CALCULATION
            if len(edge_matrix) == 1:
                return -1

            counter = 0
            used_slots = 0
            rows = len(edge_matrix)
            columns = len(edge_matrix[0])
            for row in range(rows):
                for column in range(columns):
                    if edge_matrix[row][column] == 1:
                        used_slots += 1
                        if row == 0:
                            if edge_matrix[rows - 1][column] == 1:
                                counter += 1
                            if edge_matrix[row + 1][column] == 1:
                                counter += 1
                        elif row == rows - 1:
                            if edge_matrix[0][column] == 1:
                                counter += 1
                            if edge_matrix[row - 1][column] == 1:
                                counter += 1
                        else:
                            if edge_matrix[row - 1][column] == 1:
                                counter += 1
                            if edge_matrix[row + 1][column] == 1:
                                counter += 1

            if used_slots > 0:
                self.graph[source][destination]['cps'] = counter / used_slots
            elif used_slots == 0:
                self.graph[source][destination]['cps'] = 0
        else:
            raise ValueError(f"Edge from {source} to {destination} does not exist.")

    def get_edge_cps(self, source, destination):
        if self.graph.has_edge(source, destination):
            return self.graph[source][destination]['cps']
        else:
            raise ValueError(f"Edge from {source} to {destination} does not exist.")

    def set_occupancy_of_all_edge_matrices(self):
        for src, dst in self.graph.edges:
            edge_matrix = self.get_edge_matrix(src, dst)
            if edge_matrix is not None:
                self.set_edge_occupancy(src, dst)

    def set_edge_fragmentation(self, source, destination):

        if self.graph.has_edge(source, destination):
            edge_matrix = self.get_edge_matrix(source, destination)
            if edge_matrix is not None:
                frag_lengths = []
                frag_potential = []
                for row in range(len(edge_matrix)):
                    temp_frag_size = 0
                    col = 0
                    while col < len(edge_matrix[0]):
                        temp_frag_size = 0
                        if edge_matrix[row][col] == 0:
                            temp_frag_size += 1
                            col += 1
                            while col < len(edge_matrix[0]) and edge_matrix[row][col] == 0:
                                temp_frag_size += 1
                                col += 1
                            frag_lengths.append(temp_frag_size)
                        elif edge_matrix[row][col] == 1:
                            col += 1
                for frag in frag_lengths:
                    count = 0
                    for rate in self.rates:
                        if (rate / self.slot_capacity) >= frag:
                            count += 1
                    frag_potential.append(count / len(self.rates))
                self.graph[source][destination]['fragmentation'] = np.mean(frag_potential)
        else:
            raise ValueError(f"Edge from {source} to {destination} does not exist.")

    def get_edge_fragmentation(self, source, destination):
        if self.graph.has_edge(source, destination):
            return self.graph[source][destination]['fragmentation']
        else:
            raise ValueError(f"Edge from {source} to {destination} does not exist.")

    def set_fragmentation_of_all_edge_matrices(self):
        for src, dst in self.graph.edges:
            edge_matrix = self.get_edge_matrix(src, dst)
            if edge_matrix is not None:
                self.set_edge_fragmentation(src, dst)

    # CAN BE MODIFIED TO REFLECT NEW WAYS OF DEFINING EDGE WEIGHT

    def set_edge_custom_weight(self, source, destination):
        if self.graph.has_edge(source, destination):
            normalized_distance = (self.graph[source][destination]['distance'] - self.min_distance) / (self.max_distance - self.min_distance)
            normalized_occupancy = (self.graph[source][destination]['occupancy'] - 0) / (100 - 0)
            normalized_fragmentation = (self.graph[source][destination]['fragmentation'] - 0) / (1 - 0)
            self.graph[source][destination]['multi_parameter_weight'] = (normalized_occupancy * 0.5) + (normalized_distance * 0.5)   #formula for weighted distance
            #self.graph[source][destination]['multi_parameter_weight'] = normalized_occupancy #For Zhang algo
        else:
            raise ValueError(f"Edge from {source} to {destination} does not exist.")

    def get_edge_custom_weight(self, source, destination):
        if self.graph.has_edge(source, destination):
            return self.graph[source][destination]['multi_parameter_weight']
        else:
            raise ValueError(f"Edge from {source} to {destination} does not exist.")

    def set_custom_weight_of_all_edge_matrices(self):
        for src, dst in self.graph.edges:
            edge_matrix = self.get_edge_matrix(src, dst)
            if edge_matrix is not None:
                self.set_edge_custom_weight(src, dst)

    # ###########################################################################
    ############################################################################
    # ######################FOR CROSSTALK CALCULATION ###########################



    def calculate_and_set_max_XT_for_lightpath(self, path, occupied_slots):

        coefficient = 10 ** -9
        every_slot_xt = []
        for slot in occupied_slots:
            row, col = slot
            sum_of_every_link_xt_component = 0
            for i in range(len(path) - 1):
                src, dst = path[i], path[i + 1]
                link_length = self.get_single_link_distance_meters(src, dst)
                # Get the spectrum matrix associated with the edge
                edge_matrix_path = self.get_edge_matrix(src, dst)
                # Get noise matrix for the edge:
                noise_matrix = self.get_noise_edge_matrix(src, dst)
                aux = 0
                nei1, nei2 = figure_neighbors(row, len(edge_matrix_path))
                if edge_matrix_path[nei1][col]:
                    aux += 1
                if edge_matrix_path[nei2][col]:
                    aux += 1
                sum_of_every_link_xt_component += aux * coefficient * link_length
            if sum_of_every_link_xt_component == 0:
                every_slot_xt.append(-61.93)
            else:
                every_slot_xt.append(add_decibel(noise_matrix[row][col], linear_to_decibel(sum_of_every_link_xt_component)))
        maximum_xt = np.max(every_slot_xt)
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            # Get noise matrix for the edge:
            noise_matrix = self.get_noise_edge_matrix(src, dst)
            for slot in occupied_slots:
                row, col = slot
                noise_matrix[row][col] = maximum_xt
            # Update Noise
            self.update_noise_matrix_in_topology(src, dst, noise_matrix)

    def calculate_and_set_max_XT_for_lightpath_seven_core_MCF(self, path, occupied_slots):

        coefficient = 10 ** -9
        every_slot_xt = []
        for slot in occupied_slots:
            row, col = slot
            sum_of_every_link_xt_component = 0
            for i in range(len(path) - 1):
                src, dst = path[i], path[i + 1]
                link_length = self.get_single_link_distance_meters(src, dst)
                # Get the spectrum matrix associated with the edge
                edge_matrix_path = self.get_edge_matrix(src, dst)
                # Get noise matrix for the edge:
                noise_matrix = self.get_noise_edge_matrix(src, dst)
                aux = 0
                neighbors = figure_neighbors_seven_core_MCF(row)
                for nei in neighbors:
                    if edge_matrix_path[nei][col]:
                        aux += 1

                sum_of_every_link_xt_component += aux * coefficient * link_length
            if sum_of_every_link_xt_component == 0:
                every_slot_xt.append(-61.93)
            else:
                every_slot_xt.append(add_decibel(noise_matrix[row][col], linear_to_decibel(sum_of_every_link_xt_component)))
        #Light-path XT == maximum XT for any of the slots that form the LP
        maximum_xt = np.max(every_slot_xt)
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            # Get noise matrix for the edge:
            noise_matrix = self.get_noise_edge_matrix(src, dst)
            for slot in occupied_slots:
                row, col = slot
                noise_matrix[row][col] = maximum_xt
            # Update Noise
            self.update_noise_matrix_in_topology(src, dst, noise_matrix)

    def precise_slot_xt_available_spectrum(self, shortest_path, modulation):

        xt_aware_result_matrix = np.zeros((self.matrix_rows, self.matrix_columns))  # Has to start with zeros.

        # Iterate over pairs of nodes in the path
        for i in range(len(shortest_path) - 1):
            src, dst = shortest_path[i], shortest_path[i + 1]

            # Get the matrix associated with the edge
            edge_matrix = self.get_edge_matrix(src, dst)

            # Perform logical OR operation
            xt_aware_result_matrix = np.logical_or(xt_aware_result_matrix, edge_matrix)
            updated_xt_aware_result_matrix = xt_aware_result_matrix



        for core in range(self.matrix_rows):
            for slot in range(self.matrix_columns):
                aux_linear = 0
                if xt_aware_result_matrix[core][slot] == 0:
                    for link in range(len(shortest_path) - 1):
                        src, dst = shortest_path[i], shortest_path[i + 1]
                        edge_matrix = self.get_edge_matrix(src, dst)
                        lightpath_matrix = self.get_lightpath_matrix(src, dst)
                        noise_matrix = self.get_noise_edge_matrix(src, dst)
                        nei1, nei2 = figure_neighbors(core, self.matrix_rows)
                        occ_neigh = 0
                        distance_in_meters = self.get_single_link_distance_meters(src, dst)
                        if edge_matrix[nei1][slot] == 1:
                            occ_neigh += 1
                            self.link_manager.add_link_info(lightpath_matrix[nei1][slot], modulation, noise_matrix[nei1][slot], distance_in_meters)
                        if edge_matrix[nei2][slot] == 1:
                            occ_neigh += 1
                            self.link_manager.add_link_info(lightpath_matrix[nei2][slot], modulation, noise_matrix[nei2][slot], distance_in_meters)
                        aux_linear += occ_neigh * 10 ** -9 * distance_in_meters
                    if aux_linear == 0:
                        projected_slot_xt = -61.93
                        if projected_slot_xt <= Modulation.Modulation.xt_threshold(modulation) and self.link_manager.can_allocate_for_mod():
                            updated_xt_aware_result_matrix[core][slot] = 0
                            self.link_manager.delete_all()
                    else:
                        projected_slot_xt = add_decibel(noise_matrix[core][slot], linear_to_decibel(aux_linear))
                        if projected_slot_xt <= Modulation.Modulation.xt_threshold(modulation) and self.link_manager.can_allocate_for_mod():
                            updated_xt_aware_result_matrix[core][slot] = 0
                            self.link_manager.delete_all()
                else:
                    updated_xt_aware_result_matrix[core][slot] = 1
                    self.link_manager.delete_all()
        return updated_xt_aware_result_matrix

    def precise_slot_xt_available_spectrum_seven_core_MCF(self, shortest_path, modulation):

        xt_aware_result_matrix = np.zeros((self.matrix_rows, self.matrix_columns))  # Has to start with zeros.

        # Iterate over pairs of nodes in the path
        for i in range(len(shortest_path) - 1):
            src, dst = shortest_path[i], shortest_path[i + 1]

            # Get the matrix associated with the edge
            edge_matrix = self.get_edge_matrix(src, dst)

            # Perform logical OR operation
            xt_aware_result_matrix = np.logical_or(xt_aware_result_matrix, edge_matrix)
            updated_xt_aware_result_matrix = xt_aware_result_matrix



        for core in range(self.matrix_rows):
            for slot in range(self.matrix_columns):
                aux_linear = 0
                if xt_aware_result_matrix[core][slot] == 0:
                    for link in range(len(shortest_path) - 1):
                        src, dst = shortest_path[i], shortest_path[i + 1]
                        edge_matrix = self.get_edge_matrix(src, dst)
                        lightpath_matrix = self.get_lightpath_matrix(src, dst)
                        noise_matrix = self.get_noise_edge_matrix(src, dst)
                        neighbors = figure_neighbors_seven_core_MCF(core)
                        occ_neigh = 0
                        distance_in_meters = self.get_single_link_distance_meters(src, dst)
                        for nei in neighbors:
                            if edge_matrix[nei][slot] == 1:
                                occ_neigh += 1
                                self.link_manager.add_link_info(lightpath_matrix[nei][slot], modulation, noise_matrix[nei][slot], distance_in_meters)
                        # Accumulates link XT elements
                        aux_linear += occ_neigh * 10 ** -9 * distance_in_meters
                    if aux_linear == 0:
                        projected_slot_xt = -61.93
                        if projected_slot_xt <= Modulation.Modulation.xt_threshold(modulation) and self.link_manager.can_allocate_for_mod():
                            updated_xt_aware_result_matrix[core][slot] = 0
                            self.link_manager.delete_all()
                    else:
                        projected_slot_xt = add_decibel(noise_matrix[core][slot], linear_to_decibel(aux_linear))
                        if projected_slot_xt <= Modulation.Modulation.xt_threshold(modulation) and self.link_manager.can_allocate_for_mod():
                            updated_xt_aware_result_matrix[core][slot] = 0
                            self.link_manager.delete_all()
                else:
                    updated_xt_aware_result_matrix[core][slot] = 1
                    self.link_manager.delete_all()
        return updated_xt_aware_result_matrix

    #################################################################################
    #################################################################################

    def draw_graph(self):
        # Use the spring layout for a nice layout of the graph
        pos = nx.spring_layout(self.graph)

        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=700)

        # Draw edges
        nx.draw_networkx_edges(self.graph, pos)

        # Draw labels
        nx.draw_networkx_labels(self.graph, pos)

        # Show the graph
        plt.show()
