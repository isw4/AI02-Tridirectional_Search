# coding=utf-8
import cPickle as pickle
import random
import unittest

import matplotlib.pyplot as plt
import networkx

from explorable_graph import ExplorableGraph
from search_submission import PriorityQueue, a_star, bidirectional_a_star, \
	bidirectional_ucs, breadth_first_search, uniform_cost_search, null_heuristic
from search_submission import tridirectional_search, three_bidirectional_search, tridirectional_upgraded, custom_heuristic
from visualize_graph import plot_search


class TestPriorityQueue(unittest.TestCase):
	"""Test Priority Queue implementation"""

	def test_append_and_pop(self):
		"""Test the append and pop functions and that the id_table is being maintained at the same time"""
		queue = PriorityQueue()
		temp_list = []

		for i in xrange(10):
			a = random.randint(0, 10000)
			queue.append((a, 'a'))
			self.assertEqual(len(queue.id_table), i + 1, "ID table is not being pushed to correctly")
			temp_list.append(a)

		temp_list = sorted(temp_list)

		for i in range(0, len(temp_list)):
			popped = queue.pop()
			self.assertEqual(len(queue.id_table), len(temp_list) - (i + 1), "ID table is not being popped correctly")
			self.assertEqual(temp_list[i], popped[0], "Heap not popping the right things")

	def test_append_and_pop_tiebreaker(self):
		"""Test the append and pop elements with the same priority"""
		queue = PriorityQueue()
		entries = [(200, 'a'),
		           (300, 'a'),
		           (300, 'a'),
		           (200, 'a'),
		           (300, 'a')]

		truth = [(200, 0, 'a'),
		         (200, 3, 'a'),
		         (300, 1, 'a'),
		         (300, 2, 'a'),
		         (300, 4, 'a')]

		for i in range(0, len(entries)):
			queue.append(entries[i])

		for item in truth:
			popped = queue.pop()
			self.assertEqual(item, popped, "Heap not popping the right things")

	def test_contains(self):
		""" Test the contains functionality"""
		queue = PriorityQueue()
		truth = [(200, 'a'),
		         (200, 'b'),
		         (300, 'c'),
		         (300, 'd'),
		         (300, 'e')]

		for i in range(0, len(truth)):
			queue.append(truth[i])

		self.assertNotIn('z', queue, "z is not supposed to be in queue")
		self.assertIn('c', queue, "c is supposed to be in queue")

	def test_remove(self):
		"""Tests the remove functionality"""
		queue = PriorityQueue()
		add = [(200, 'a'),
		       (200, 'b'),
		       (300, 'c'),
		       (300, 'd'),
		       (300, 'e')]

		for i in range(0, len(add)):
			queue.append(add[i])

		queue.remove(2) # Removes the node with id==2
		self.assertEqual(len(queue.id_table), len(add) - 1)

		truth = [(200, 0, 'a'),
		         (200, 1, 'b'),
		         (300, 3, 'd'),
		         (300, 4, 'e')]
		for item in truth:
			popped = queue.pop()
			self.assertEqual(item, popped, "Heap not popping the right things")

class TestBasicSearch(unittest.TestCase):
	"""Test the simple search algorithms: BFS, UCS, A*"""

	def setUp(self):
		"""Romania map data from Russell and Norvig, Chapter 3."""
		romania = pickle.load(open('romania_graph.pickle', 'rb'))   # romania is a networkx.Graph object
		self.romania = ExplorableGraph(romania)
		self.romania.reset_search()
	#
	# def test_bfs(self):
	# 	"""Test and visualize breadth-first search"""
	# 	start = 'a'
	# 	goal = 'u'
	#
	# 	node_positions = {n: self.romania.node[n]['pos'] for n in
	# 					  self.romania.node.keys()}
	#
	# 	self.romania.reset_search()
	# 	path = breadth_first_search(self.romania, start, goal)
	#
	# 	self.draw_graph(self.romania, node_positions=node_positions,
	# 					start=start, goal=goal, path=path)
	#
	# 	truth = ['a', 's', 'f', 'b', 'u']
	# 	self.assertEqual(len(truth), len(path), "The length of the path returned is not correct")
	# 	for i in range(0, len(truth)):
	# 		self.assertEqual(truth[i], path[i], "The '" + path[i] + "' node along the path is not correct")
	#
	# def test_bfs_empty_path(self):
	# 	start = "a"
	# 	goal = "a"
	# 	path = breadth_first_search(self.romania, start, goal)
	# 	self.assertEqual(path, [])
	#
	# def test_ucs(self):
	# 	"""Test and visualize uniform-cost search"""
	# 	start = 'a'
	# 	goal = 'u'
	#
	# 	node_positions = {n: self.romania.node[n]['pos'] for n in
	# 					  self.romania.node.keys()}
	#
	# 	self.romania.reset_search()
	# 	path = uniform_cost_search(self.romania, start, goal)
	#
	# 	self.draw_graph(self.romania, node_positions=node_positions,
	# 					start=start, goal=goal, path=path)
	#
	# 	truth = ['a', 's', 'r', 'p', 'b', 'u']
	# 	self.assertEqual(len(truth), len(path), "The length of the path returned is not correct")
	# 	for i in range(0, len(truth)):
	# 		self.assertEqual(truth[i], path[i], "The '" + path[i] + "' node along the path is not correct")

	# def test_a_star(self):
	# 	"""Test and visualize A* search"""
	# 	start = 'z'
	# 	goal = 'u'
	#
	# 	node_positions = {n: self.romania.node[n]['pos'] for n in
	# 					  self.romania.node.keys()}
	#
	# 	self.romania.reset_search()
	# 	path = a_star(self.romania, start, goal)
	#
	# 	self.draw_graph(self.romania, node_positions=node_positions,
	# 					start=start, goal=goal, path=path, title="A* from "+start+" to "+goal)

		# truth = ['a', 's', 'r', 'p', 'b', 'u']
		# self.assertEqual(len(truth), len(path), "The length of the path returned is not correct")
		# for i in range(0, len(truth)):
		# 	self.assertEqual(truth[i], path[i], "The '" + path[i] + "' node along the path is not correct")
#
	@staticmethod
	def draw_graph(graph, node_positions=None, start=None, goal=None,
				   path=None, title=None):
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

		if title is not None: plt.title(title)
		plt.plot()
		plt.show()


class TestBidirectionalSearch(unittest.TestCase):
	"""Test the bidirectional search algorithms: UCS, A*"""

	def setUp(self):
		"""Load Atlanta map data"""
		# atlanta = pickle.load(open('atlanta_osm.pickle', 'rb'))
		# self.atlanta = ExplorableGraph(atlanta)
		# self.atlanta.reset_search()

		"""Romania map data from Russell and Norvig, Chapter 3."""
		romania = pickle.load(open('romania_graph.pickle', 'rb'))
		self.romania = ExplorableGraph(romania)
		self.romania.reset_search()

	# def test_romania_ucs(self):
	# 	"""Test and visualize uniform-cost search"""
	# 	start = 'm'
	# 	goal = 'u'
	#
	# 	node_positions = {n: self.romania.node[n]['pos'] for n in
	# 					  self.romania.node.keys()}
	#
	# 	self.romania.reset_search()
	# 	path = bidirectional_ucs(self.romania, start, goal)
	#
	# 	self.draw_graph(self.romania, node_positions=node_positions,
	# 					start=start, goal=goal, path=path, title="Bi UCS from "+start+" to "+goal)
	#
	# def test_romania_astar(self):
	# 	"""Test and visualize uniform-cost search"""
	# 	start = 'm'
	# 	goal = 'u'
	#
	# 	node_positions = {n: self.romania.node[n]['pos'] for n in
	# 					  self.romania.node.keys()}
	#
	# 	self.romania.reset_search()
	# 	path = bidirectional_a_star(self.romania, start, goal)
	#
	# 	self.draw_graph(self.romania, node_positions=node_positions,
	# 					start=start, goal=goal, path=path, title="Bi A* from "+start+" to "+goal)

	# def test_bidirectional_ucs(self):
	# 	"""Test and generate GeoJSON for bidirectional UCS search"""
	# 	path = bidirectional_ucs(self.atlanta, '69581003', '69581000')
	# 	all_explored = self.atlanta.explored_nodes
	# 	plot_search(self.atlanta, 'atlanta_search_bidir_ucs.json', path,
	# 				all_explored)
#
# 	def test_bidirectional_a_star(self):
# 		"""Test and generate GeoJSON for bidirectional A* search"""
# 		path = bidirectional_a_star(self.atlanta, '69581003', '69581000')
# 		all_explored = self.atlanta.explored_nodes
# 		plot_search(self.atlanta, 'atlanta_search_bidir_a_star.json', path,
# 					all_explored)

	@staticmethod
	def draw_graph(graph, node_positions=None, start=None, goal=None,
				   path=None, title=None):
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

		if title is not None: plt.title(title)
		plt.plot()
		plt.show()


class TestTridirectionalSearch(unittest.TestCase):
	"""Test the bidirectional search algorithms: UCS, A*"""

	def setUp(self):
		"""Load Atlanta map data"""
		# atlanta = pickle.load(open('atlanta_osm.pickle', 'rb'))
		# self.atlanta = ExplorableGraph(atlanta)
		# self.atlanta.reset_search()

		"""Romania map data from Russell and Norvig, Chapter 3."""
		romania = pickle.load(open('romania_graph.pickle', 'rb'))
		self.romania = ExplorableGraph(romania)
		self.romania.reset_search()

	def test_romania_ucs(self):
		"""Test and visualize uniform-cost search"""
		goals = ['a', 'c', 'd']

		node_positions = {n: self.romania.node[n]['pos'] for n in
						  self.romania.node.keys()}

		self.romania.reset_search()
		path = tridirectional_search(self.romania, goals)

		print("Final path: {}".format(path))
		self.draw_graph(self.romania, node_positions=node_positions,
		                goals=goals, path=path,
		                title="Tri A* between " + goals[0] + ", " + goals[1] + ", and " + goals[2])

	# def test_romania_astar(self):
	# 	"""Test and visualize uniform-cost search"""
	# 	goals = ['a', 'f', 'u']
	#
	# 	node_positions = {n: self.romania.node[n]['pos'] for n in
	# 					  self.romania.node.keys()}
	#
	# 	self.romania.reset_search()
	# 	path = bidirectional_a_star(self.romania, goals[0], goals[1])
	#
	# 	self.draw_graph(self.romania, node_positions=node_positions,
	# 					goals=goals, path=path, title="Tri A* between "+goals[0]+", "+goals[1]+", and "+goals[2])

	# def test_bidirectional_ucs(self):
	# 	"""Test and generate GeoJSON for bidirectional UCS search"""
	# 	path = bidirectional_ucs(self.atlanta, '69581003', '69581000')
	# 	all_explored = self.atlanta.explored_nodes
	# 	plot_search(self.atlanta, 'atlanta_search_bidir_ucs.json', path,
	# 				all_explored)
#
# 	def test_bidirectional_a_star(self):
# 		"""Test and generate GeoJSON for bidirectional A* search"""
# 		path = bidirectional_a_star(self.atlanta, '69581003', '69581000')
# 		all_explored = self.atlanta.explored_nodes
# 		plot_search(self.atlanta, 'atlanta_search_bidir_a_star.json', path,
# 					all_explored)

	@staticmethod
	def draw_graph(graph, node_positions=None, goals=None,
				   path=None, title=None):
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

		if goals[0]:
			networkx.draw_networkx_nodes(graph, node_positions,
										 nodelist=[goals[0]], node_color='b')

		if goals[1]:
			networkx.draw_networkx_nodes(graph, node_positions,
										 nodelist=[goals[1]], node_color='y')

		if goals[2]:
			networkx.draw_networkx_nodes(graph, node_positions,
										 nodelist=[goals[2]], node_color='m')

		if title is not None: plt.title(title)
		plt.plot()
		plt.show()


if __name__ == '__main__':
	unittest.main()
