# coding=utf-8
import pickle
import random
import unittest

import matplotlib.pyplot as plt
import networkx

from explorable_graph import ExplorableGraph
from search_submission import PriorityQueue, a_star, bidirectional_a_star, \
	bidirectional_ucs, breadth_first_search, uniform_cost_search
from search_submission import three_bidirectional_search, tridirectional_upgraded, custom_heuristic, PriorityQueue
from search_submission_tests_grid import save_graph
from search_submission_tests import TestBasicSearch as tbs
from visualize_graph import plot_search


def uniform_cost_search(graph, start, goal):
	"""
	Warm-up exercise: Implement uniform_cost_search.

	See README.md for exercise description.

	Args:
		graph (ExplorableGraph): Undirected graph to search.
		start (str): Key for the start node.
		goal (str): Key for the end node.

	Returns:
		The best path as a list from the start and goal nodes (including both).
	"""
	if start == goal: return []

	# Init explored
	explored = set()
	# Init frontier
	frontier = PriorityQueue()
	path_cost = 0   # Cost of path (priority)
	path = [start]  # A list of nodes representing the path from the start to the node on the frontier
	id = frontier.append((path_cost, path))
	# Init hash table storing the best cost to the node
	best_cost_to_node = {}
	best_cost_to_node[start] = { "cost":path_cost, "id":id }
	print("Starting off with:\nExplored: {}\nFrontier: {}\nBest Cost:{}".format(explored, frontier, best_cost_to_node))

	while frontier.size() > 0:
		path_cost, _, path = frontier.pop()
		if path[-1] == goal:
			print("\nGoal was popped off the frontier, with a minimum path of {} and cost of {}".format(path, path_cost))
			return path
		neighbours_iter = graph.neighbors_iter(path[-1])
		explored |= {path[-1]}
		print("\nLooking at path: {}".format(path))
		print("Explored updated: {}".format(explored))

		for neighbour in neighbours_iter:
			if neighbour not in explored:
				cost_to_front = path_cost + graph[path[-1]][neighbour]['weight']
				print("Found neighbor {} and accumulated cost to go there is {}".format(neighbour, cost_to_front))
				if neighbour not in best_cost_to_node:
					path_to_front = list(path)
					path_to_front.append(neighbour)
					id = frontier.append((cost_to_front, path_to_front))
					best_cost_to_node[neighbour] = {"cost": cost_to_front, "id": id}
					print("Adding this path")
				elif neighbour in best_cost_to_node and cost_to_front < best_cost_to_node[neighbour]['cost']:
					print("Internal queue: {}".format(frontier.queue))
					frontier.remove(best_cost_to_node[neighbour]['id'])
					print("ID of node removed: {}".format(best_cost_to_node[neighbour]['id']))
					print("Internal queue: {}".format(frontier.queue))
					path_to_front = list(path)
					path_to_front.append(neighbour)
					id = frontier.append((cost_to_front, path_to_front))
					print("Internal queue: {}".format(frontier.queue))
					best_cost_to_node[neighbour] = {"cost":cost_to_front, "id":id }
					print("Adding this path to replace an old path to this node with a higher cost")
				else:
					print("Path to the node {} that has a lower cost already exists. Not adding this path".format(neighbour))
				print("Best costs: {}".format(best_cost_to_node))
	return None


if __name__ == "__main__":
	romania = pickle.load(open('romania_graph.pickle', 'rb'))
	save_graph(romania, './figures/romania grid.png', True, True)

	EXG_romania = ExplorableGraph(romania)
	start = 'a'
	goal = 'u'
	path = uniform_cost_search(EXG_romania, start, goal)
	print("Path found is: {}".format(path))