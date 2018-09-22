# coding=utf-8
import cPickle as pickle
import random
import unittest

import matplotlib.pyplot as plt
import networkx

from explorable_graph import ExplorableGraph
from search_submission import PriorityQueue, a_star, bidirectional_a_star, \
	bidirectional_ucs, breadth_first_search, uniform_cost_search
from search_submission import three_bidirectional_search, tridirectional_upgraded, custom_heuristic, PriorityQueue, euclidean_dist_heuristic
from search_submission_tests_grid import save_graph
from search_submission_tests import TestBasicSearch as tbs
from visualize_graph import plot_search




if __name__ == "__main__":
	romania = pickle.load(open('romania_graph.pickle', 'rb'))
	save_graph(romania, './figures/romania grid.png', True, True)

	EXG_romania = ExplorableGraph(romania)
	start = 'a'
	goal = 'u'



	#print("Path found is: {}".format(path))