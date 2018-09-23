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

class ExplorableAStarDirection:
	def __init__(self, graph, seed, goal, heuristic):
		self.graph = graph
		self.goal = goal
		self.heuristic = heuristic
		self.explored = set()           # Vertices at the end of a path from the seed that have been expanded
		self.frontier = PriorityQueue() # Vertices at the end of a path from the seed to be expanded
		self.best_cost_to_node = {}     # Best cost from the seed to the node (old paths will be replaced)
		self.frontier_ids = {}          # A list of frontier ids with the key as the last node on the path

		path_cost = 0  # Cost/weight of path
		path_f = path_cost + heuristic(graph, seed, seed)  # TODO determine a good heuristic estimate and fix parameters
		path = [seed]
		id_ = self.frontier.append((path_f, path))

		self.frontier_ids[seed] = {'ids': [id_]}

		self.best_cost_to_node = {}
		self.best_cost_to_node[seed] = {"cost": path_cost, "id": id_, 'path': path}

	def add_to_explored(self, node):
		self.explored |= {node}

	def _fadd_and_update(self, node, path, path_cost):
		"""
		Adding node to the path(copy), adding into the frontier, updating the frontier_id table, and updating
		the best cost to node table

		:param node:
		:param path:
		:param path_cost:
		:return:
		"""
		popped = path[-1]
		cost_to_front = path_cost + self.graph[popped][node]['weight']

		# If the actual path to the node (not estimate) is better than one that has already been seen,
		# remove it from frontier(and frontier id table) to be replaced later
		if node in self.best_cost_to_node and cost_to_front < self.best_cost_to_node[node]['cost']:
			print("Better than old path to {}\nBest: {}".format(node, self.frontier))
			self._fremove(node, self.best_cost_to_node[node]['id'])
			print("Removed: {}".format(self.frontier))

		# If the path to the node is worse than one that has already been seen, prune this path
		elif node in self.best_cost_to_node and cost_to_front >= self.best_cost_to_node[node]['cost']:
			print("Old path is better. Pruning this path")
			return

		# Adding to the frontier
		f_to_front = cost_to_front + self.heuristic(self.graph, node, self.goal)
		path_to_front = list(path)
		path_to_front.append(node)
		id_ = self.frontier.append((f_to_front, path_to_front))

		# Updating the frontier_id table
		if node in self.frontier_ids:
			self.frontier_ids[node]['ids'].append(id_)
		else:
			self.frontier_ids[node] = {'ids': [id_]}

		# Adding to the best cost to node table
		self.best_cost_to_node[node] = {"cost": cost_to_front, "id": id_, "path": path_to_front}

	def _fpop(self):
		"""
		Popping from frontier. Needs to remove from the frontier_id hash table as well

		:return:
		"""
		path_f, id_, path = self.frontier.pop()
		popped = path[-1]
		self.frontier_ids[popped]['ids'].remove(id_)
		if len(self.frontier_ids[popped]['ids']) == 0:
			del self.frontier_ids[popped]
		return path_f, id_, path

	def _fremove(self, node, id_):
		"""
		Removing from frontier

		:param id_: (int) id to remove from frontier priority queue and frontier_ids hash table
		:param node: (char) key to remove from the frontier_ids hash table
		"""
		self.frontier.remove(id_)

		self.frontier_ids[node]['ids'].remove(id_)
		if len(self.frontier_ids[node]['ids']) == 0:
			del self.frontier_ids[node]

	def _combine_path(self, path, path_cost, other_dir):
		print("Path {} with cost {}".format(path, path_cost))
		path = list(path)
		midpoint = path[-1]
		other_path = other_dir.best_cost_to_node[midpoint]['path']
		print("Other path {} with cost {}".format(other_path, other_dir.best_cost_to_node[midpoint]['cost']))
		if path[0] == self.goal:
			part2 = path[::-1]
			combined_path = other_path[:-1]
			combined_path.extend(part2)
		else:
			part2 = other_path[::-1]
			combined_path = path[:-1]
			combined_path.extend(part2)
		combined_cost = path_cost + other_dir.best_cost_to_node[midpoint]['cost']
		return combined_cost, combined_path

	def search_once(self, other_dir, upper_bound):
		combined_cost = None
		combined_path = None
		terminate = False

		path_f, _, path = self._fpop()
		popped = path[-1]
		path_cost = path_f - self.heuristic(self.graph, popped, self.goal)  # Get back the actual cost
		# if path_cost >= upper_bound:
		# 	print("No shorter path can be found than the existing shortest path. Returning")
		# 	terminate = True
		# 	return None, None, terminate

		# Once the goal has been expanded/explored by both, the shortest path from one direction and
		# the other direction to the intermediate has been found. However, the shortest path from one
		# direction to the other has not necessarily been found. Will combine the two paths, and then
		# continue to look at the neighbours.
		if popped in other_dir.explored:
			print("\nMIDPOINT {} POPPED".format(popped))
			combined_cost, combined_path = self._combine_path(path, path_cost, other_dir)
			print("Combined path {} with cost {}".format(combined_path, combined_cost))

		neighbours_iter = self.graph.neighbors_iter(popped) # Getting the neighbours of the last node in the path
		self.add_to_explored(popped)
		print("\nLooking at path {}\nUpdated explored: {}\nUpdated f_id: {}".format(path, self.explored, self.frontier_ids))

		for neighbour in neighbours_iter:
			if neighbour not in self.explored:
				# cost_to_front = path_cost + self.graph[popped][neighbour]['weight']
				print("Neighbour {} found".format(neighbour))
				self._fadd_and_update(neighbour, path, path_cost)

				# # If the actual path to the node (not estimate) is better than one that has already been seen,
				# # remove it from frontier to be replaced later
				# if neighbour in self.best_cost_to_node and cost_to_front < self.best_cost_to_node[neighbour]['cost']:
				# 	print("Better than old path to {}\nBest: {}".format(neighbour, self.frontier))
				# 	self._fremove(neighbour, self.best_cost_to_node[neighbour]['id'])
				# 	print("Removed: {}".format(self.frontier))
				# # If the path to the node is worse than one that has already been seen, prune this path
				# elif neighbour in self.best_cost_to_node and cost_to_front >= self.best_cost_to_node[neighbour]['cost']:
				# 	print("Old path is better. Pruning this path")
				# 	continue
				# # Add path to frontier and update best cost to the node
				# path_to_front = list(path)
				# path_to_front.append(neighbour)
				# id_ = self.frontier.append((f_to_front, path_to_front))
				# self.best_cost_to_node[neighbour] = {"cost": cost_to_front, "id": id_, "path": path_to_front}
				print("Frontier: {}\nf_ids: {}\nBest: {}".format(self.frontier, self.frontier_ids, self.best_cost_to_node))

		return combined_cost, combined_path, terminate

	def phase2(self, other_dir, final_cost, final_path):
		nodes_in_other_frontier = set(other_dir.frontier_ids.keys())
		print(nodes_in_other_frontier)
		nodes_to_search = (nodes_in_other_frontier | other_dir.explored) & self.explored
		print(nodes_to_search)
		while len(nodes_to_search) != 0:
			node = nodes_to_search.pop()
			if node in other_dir.frontier_ids:
				for id_ in other_dir.frontier_ids[node]['ids']:
					path_f, _, path = other_dir.frontier.find(id_)
					popped = path[-1]
					path_cost = path_f - other_dir.heuristic(other_dir.graph, popped, other_dir.goal)  # Get back the actual cost
					combined_cost, combined_path = other_dir._combine_path(path, path_cost, self)
					print("Combined path {} with cost {}".format(combined_path, combined_cost))
					if combined_cost < final_cost:
						final_cost = combined_cost
						final_path = combined_path
		return final_cost, final_path

def bidirectional_a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
	if start == goal: return []
	path = None

	from_start = ExplorableAStarDirection(graph, start, goal, heuristic)
	from_goal = ExplorableAStarDirection(graph, goal, goal, heuristic)

	n = 0
	upper_bound = float("inf")
	while from_start.frontier.size() > 0 and from_goal.frontier.size() > 0:
		if n % 2 == 0:
			temp_cost, temp_path, terminate = from_start.search_once(from_goal, upper_bound)
			if temp_cost is not None:
				print("\n\n\nHERE")
				cost, path = from_start.phase2(from_goal, temp_cost, temp_path)
				break
		else:
			temp_cost, temp_path, terminate = from_goal.search_once(from_start, upper_bound)
			if temp_cost is not None:
				print("\n\n\nHERE")
				cost, path = from_goal.phase2(from_start, temp_cost, temp_path)
				break

		# if temp_cost is not None and temp_cost < upper_bound:
		# 	upper_bound = temp_cost
		# 	path = temp_path

		# f_1, _, path_1 = from_start.frontier.top()
		# f_2, _, path_2 = from_goal.frontier.top()
		# cost_1 = f_1 - heuristic(graph, path_1[-1], goal)
		# cost_2 = f_2 - heuristic(graph, path_2[-1], goal)
		# # Checking for crossover: path_1 = ['a', 'b', 'c'], path_2 = ['c' 'b']. Then distance b-c is counted twice
		# if path_1[-2:] == path_2[-2:][::-1]:
		# 	cost_1 -= graph[path_1[-2]][path_1[-1]]['weight']
		# 	print("Updated cost of path 1 {} to {}".format(path_1, cost_1))
		# 	print("Updated cost of path 2 {} to {}".format(path_2, cost_2))
		# if cost_1 + cost_2 >= upper_bound:
		# 	print("Path 1: {}\nPath 2: {}".format(path_1, path_2))
		# 	print("{} + {} is >= current shortest path cost of {}. Terminating here".format(cost_1, cost_2, upper_bound))
		# 	break

		# if terminate:
		# 	print("TERMINATING")
		# 	break

		n += 1
	return path


if __name__ == "__main__":
	romania = pickle.load(open('romania_graph.pickle', 'rb'))
	save_graph(romania, './figures/romania grid.png', True, True)

	EXG_romania = ExplorableGraph(romania)
	start = 'r'
	goal = 'c'

	path = bidirectional_a_star(EXG_romania, start, goal, heuristic=null_heuristic)
	print("Path found is: {}".format(path))

	node_positions = {n: EXG_romania.node[n]['pos'] for n in EXG_romania.node.keys()}
	draw_graph(EXG_romania, node_positions=node_positions, start=start, goal=goal, path=path)
