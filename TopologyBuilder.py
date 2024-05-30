import networkx as nx
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import Parser

class NetworkXGraphBuilder:

    def __init__(self, xml_file, matrix_rows, matrix_columns):
        self.graph = nx.DiGraph()
        self.matrix_rows = matrix_rows
        self.matrix_columns = matrix_columns
        self.edge_matrices = {}

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
            self.graph.add_edge(source, destination, id=link_id, delay=delay, bandwidth=bandwidth, weight=weight, distance=distance, occupancy = 0.0, fragmentation = 0.0, multi_parameter_weight = 0.0)

            # Create matrix for the edge
            matrix = np.zeros((matrix_rows, matrix_columns))
            self.edge_matrices[(source, destination)] = matrix

        #EXTRACT DISTANCE INFO FOR NORMALISATION
        self.distances = [self.graph[src][dst]['distance'] for src, dst in self.graph.edges]
        self.min_distance = min(self.distances)
        self.max_distance = max(self.distances)

        #EXTRACT CALLS INFO
        # Parse XML
        parser_object = Parser.XmlParser(xml_file)
        # Start Metrics Measurement and Calculation
        # All rates practiced (will be useful for fragmentation calculation)
        self.rates = [entry['rate'] for entry in parser_object.get_calls_info()]
        self.slot_capacity = parser_object.get_slots_bandwidth()

    def find_n_shortest_paths(self, graph, source, target, n):
        # Use Jin Y Yen's algorithm to find the n shortest paths O(KNˆ3)
        return list(
            islice(nx.shortest_simple_paths(graph, source, target, weight='weight'), n)
        )


    def find_n_shortest_weighted_paths(self, graph, source, target, n):
        # Use Jin Y Yen's algorithm to find the n shortest paths O(KNˆ3)
        return list(
            islice(nx.shortest_simple_paths(graph, source, target, weight='multi_parameter_weight'), n)
        )

    def get_graph(self):
        return self.graph

    def get_edge_matrix(self, source, destination):
        return self.edge_matrices.get((source, destination))

    def get_all_edge_matrices(self):
        all_matrices = []

        for key, value in self.edge_matrices.items():
            if isinstance(value, np.ndarray):
                all_matrices.append(value.tolist()) #Here I change back to regular arrays. try and keep numpy DSs along the way
        return all_matrices

    def or_matrices_along_path(self, shortest_path):
        result_matrix = np.zeros((self.matrix_rows, self.matrix_columns)) #Has to start with zeros.

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

    def get_physical_distance(self, path_i):
        distance = 0
        # Iterate over pairs of nodes in the path
        for i in range(len(path_i) - 1):
            src, dst = path_i[i], path_i[i + 1]

            # Check pair for distance
            distance += int(self.graph[src][dst]["distance"])

        return distance

    # def set_edge_occupancy(self, source, destination, occupancy):
    #     if self.graph.has_edge(source, destination):
    #         self.graph[source][destination]['occupancy'] = occupancy
    #     else:
    #         raise ValueError(f"Edge from {source} to {destination} does not exist.")

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


    def set_occupancy_of_all_edge_matrices(self):
        for src, dst in self.graph.edges:
            edge_matrix = self.get_edge_matrix(src, dst)
            if edge_matrix is not None:
                self.set_edge_occupancy(src, dst)


    # def set_edge_fragmentation(self, source, destination, fragmentation):
    #     if self.graph.has_edge(source, destination):
    #         self.graph[source][destination]['fragmentation'] = fragmentation
    #     else:
    #         raise ValueError(f"Edge from {source} to {destination} does not exist.")

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


    ##CAN BE MODIFIED TO REFLECT NEW WAYS OF DEFINING EDGE WEIGHT

    def set_edge_custom_weight(self, source, destination):
        if self.graph.has_edge(source, destination):
            normalized_distance = (self.graph[source][destination]['distance'] - self.min_distance) / (self.max_distance-self.min_distance)
            normalized_occupancy = (self.graph[source][destination]['occupancy'] - 0) / (100-0)
            normalized_fragmentation = (self.graph[source][destination]['fragmentation']-0) / (1-0)
            self.graph[source][destination]['multi_parameter_weight'] = normalized_distance * 0.5 + normalized_fragmentation * 0.5  #formula for weighted distance
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



