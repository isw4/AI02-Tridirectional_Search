# coding=utf-8
import cPickle as pickle
import random
import unittest

import matplotlib.pyplot as plt
import networkx

from explorable_graph import ExplorableGraph
from search_submission import PriorityQueue, a_star, bidirectional_a_star, \
	bidirectional_ucs, breadth_first_search, uniform_cost_search
from search_submission import three_bidirectional_search, tridirectional_upgraded, custom_heuristic, PriorityQueue, euclidean_dist_heuristic, null_heuristic
from search_submission_tests_grid import save_graph
from search_submission_tests import TestBasicSearch as tbs
from visualize_graph import plot_search


def draw_graph(graph, node_positions=None, start=None, goal=None,
               path=None):
	"""Visualize results of graph search"""
	explored = list(graph.explored_nodes)

	labels = {}
	for node in graph:
		labels[node] = node

	if node_positions is None:
		node_positions = networkx.spring_layout(graph)

	networkx.draw_networkx_nodes(graph, node_positions)
	networkx.draw_networkx_edges(graph, node_positions, style='dashed')
	networkx.draw_networkx_labels(graph, node_positions, labels)

	networkx.draw_networkx_nodes(graph, node_positions, nodelist=explored,
	                             node_color='g')

	if path is not None:
		edges = [(path[i], path[i + 1]) for i in range(0, len(path) - 1)]
		networkx.draw_networkx_edges(graph, node_positions, edgelist=edges,
		                             edge_color='b')

	if start:
		networkx.draw_networkx_nodes(graph, node_positions,
		                             nodelist=[start], node_color='b')

	if goal:
		networkx.draw_networkx_nodes(graph, node_positions,
		                             nodelist=[goal], node_color='y')

	plt.plot()
	plt.show()




if __name__ == "__main__":
	romania = pickle.load(open('romania_graph.pickle', 'rb'))
	save_graph(romania, './figures/romania grid.png', True, True)

	PQ = PriorityQueue()
	PQ.append([0,0,0])
	a, b, c = PQ.pop()
	a, b, c = PQ.pop()