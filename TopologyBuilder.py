import networkx as nx
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice

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
            self.graph.add_edge(source, destination, id=link_id, delay=delay, bandwidth=bandwidth, weight=weight, distance=distance)

            # Create matrix for the edge
            matrix = np.zeros((matrix_rows, matrix_columns))
            self.edge_matrices[(source, destination)] = matrix

    def find_n_shortest_paths(self, graph, source, target, n):
        # Use Jin Y Yen's algorithm to find the n shortest paths O(KNˆ3)
        return list(
            islice(nx.shortest_simple_paths(graph, source, target, weight='weight'), n)
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



