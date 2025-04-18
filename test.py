import TopologyBuilder
import MMM
import FcaRcsa
import FirstFit

topology = TopologyBuilder.NetworkXGraphBuilder("/Users/ramonoliveira/Desktop/CG - Relatório 3D/SimpleSIm/xml/Image-usa.xml", 7, 320)

topology.find_and_print_shortest_paths()

matrix = [[0,1,0,1],
          [0,0,0,0],
          [1,1,1,1],
          [0,0,0,1]]

ff = FirstFit.FirstFit(matrix)
ff.find_connected_components()
list_of_allocable_regions = ff.get_connected_components()
print(list_of_allocable_regions)
fca_rcsa = FcaRcsa.FcaRcsa(matrix, 1)
fca_rcsa.frag_coef_rcsa()
list_of_allocable_regions = fca_rcsa.output_dict()
print(list_of_allocable_regions)



#


# def calculate_length(distance, occ):
#     return (distance/9)*0.5 + (occ/100)*0.5
#
#
# print(calculate_length(3,45))
#
#
# import networkx as nx
# import matplotlib.pyplot as plt
#
# # Define the graph
# G = nx.Graph()
#
# # Add edges along with their weights
# edges = [
#     ('a', 'b', 56),
#     ('a', 'c', 52),
#     ('a', 'd', 40),
#     ('b', 'd', 21),
#     ('b', 'e', 36),
#     ('c', 'd', 32),
#     ('c', 'f', 20),
#     ('d', 'e', 16),
#     ('d', 'f', 61),
#     ('e', 'f', 37)
# ]
#
# G.add_weighted_edges_from(edges)
#
# # Generate positions for nodes using the Kamada-Kawai layout, which tries to make edge lengths proportional to weights
# pos = nx.kamada_kawai_layout(G, weight='weight')
#
# # Draw the graph
# plt.figure(figsize=(10, 8))
# nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=14, font_weight='bold')
# labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#
# # Display the graph
# plt.show()

import numpy as np
def get_linear_value(db_value):
    return 10 ** (db_value / 10)

def get_db_value(linear_value):
    return 10 * np.log10(linear_value)



