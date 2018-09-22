# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

from __future__ import division

import heapq
import os
import pickle


class PriorityQueue(object):
	"""
	A queue structure where each element is served in order of priority.

	Elements in the queue are popped based on the priority with higher priority
	elements being served before lower priority elements.  If two elements have
	the same priority, they will be served in the order they were added to the
	queue.

	Traditionally priority queues are implemented with heaps, but there are any
	number of implementation options.

	(Hint: take a look at the module heapq)

	Attributes:
		queue (list): Nodes added to the priority queue.
		current (int): The index of the current node in the queue.
	"""
	REMOVED = "<PATH REMOVED>"

	def __init__(self):
		"""Initialize a new Priority Queue."""
		self.entry_count = 0
		self.queue = []
		self.id_table = {}

	def pop(self):
		"""
		Pop top priority node from queue.

		:returns The node with the highest priority.
		"""
		while self.size() > 0:
			entry = heapq.heappop(self.queue)
			if entry[1] in self.id_table:
				self.id_table.pop(entry[1])
				return tuple(entry)
		return None

	def remove(self, node_id):
		"""
		Remove a node from the queue. (It doesn't actually remove it, it only marks it as removed
		so that if it's at the top of the heap, it won't be popped)

		:param node_id: (int) Index of node in queue.
		"""
		entry = self.id_table.pop(node_id)
		entry[-1] = PriorityQueue.REMOVED

	def __iter__(self):
		"""Queue iterator."""
		return iter(sorted(self.queue))

	def __str__(self):
		"""Priority Queue to string."""
		return 'PQ:%s' % self.queue

	def append(self, node):
		"""
		Append a node to the queue. Typically, the node should be of format (priority, vertex/path).
		The node to be added to the queue will be of format (priority, entry_count, vertex/path).
		entry_count acts as:
			1) a way to break ties in priority, so that the node that was first added will be popped
			   first in the case of a tie
			2) an id for the node

		:param node: Comparable Object to be added to the priority queue.
		:returns the entry_count/id of the node that was just appended
		"""
		# Push onto queue
		entry = [node[0], self.entry_count, node[1]]
		heapq.heappush(self.queue, entry)
		# Push onto id table
		self.id_table[self.entry_count] = entry
		# Update counter
		self.entry_count += 1
		return entry[1]

	def __contains__(self, key):
		"""
		Containment Check operator for 'in'

		Args:
			key: The key to check for in the queue.

		Returns:
			True if key is found in queue, False otherwise.
		"""
		return key in [n for _, _, n in self.queue]

	def __eq__(self, other):
		"""
		Compare this Priority Queue with another Priority Queue.

		Args:
			other (PriorityQueue): Priority Queue to compare against.

		Returns:
			True if the two priority queues are equivalent.
		"""
		return self == other

	def size(self):
		"""
		Get the current size of the queue.

		Returns:
			Integer of number of items in queue.
		"""
		return len(self.queue)

	def clear(self):
		"""Reset queue to empty (no nodes)."""
		self.queue = []

	def top(self):
		"""
		Get the top item in the queue.

		Returns:
			The first item stored in teh queue.
		"""
		return self.queue[0]


def breadth_first_search(graph, start, goal):
	"""
	Warm-up exercise: Implement breadth-first-search.

	See README.md for exercise description.

	Args:
		graph (ExplorableGraph): Undirected graph to search.
		start (str): Key for the start node.
		goal (str): Key for the end node.

	Returns:
		The best path as a list from the start and goal nodes (including both).
		If there is no path, returns None
	"""
	if start == goal: return []

	# Init visited
	visited = set()
	visited |= {start}
	# Init frontier
	frontier = PriorityQueue()
	path_len = 0  # Length of path (priority)
	path = [start]  # A list of nodes representing the path from the start to the node on the frontier
	frontier.append((path_len, path))

	while frontier.size() > 0:
		path_len, _, path = frontier.pop()
		neighbours_iter = graph.neighbors_iter(path[-1])

		for neighbour in neighbours_iter:
			if neighbour == goal:
				path.append(neighbour)
				return path

			if neighbour not in visited:
				visited |= {neighbour}
				path_to_front = list(path)
				path_to_front.append(neighbour)
				frontier.append((path_len + 1, path_to_front))

	return None


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

	while frontier.size() > 0:
		path_cost, _, path = frontier.pop()
		if path[-1] == goal:
			return path
		neighbours_iter = graph.neighbors_iter(path[-1])
		explored |= {path[-1]}

		for neighbour in neighbours_iter:
			if neighbour not in explored:
				cost_to_front = path_cost + graph[path[-1]][neighbour]['weight']
				if neighbour not in best_cost_to_node:
					path_to_front = list(path)
					path_to_front.append(neighbour)
					id = frontier.append((cost_to_front, path_to_front))
					best_cost_to_node[neighbour] = {"cost": cost_to_front, "id": id}
				elif neighbour in best_cost_to_node and cost_to_front < best_cost_to_node[neighbour]['cost']:
					frontier.remove(best_cost_to_node[neighbour]['id'])
					path_to_front = list(path)
					path_to_front.append(neighbour)
					id = frontier.append((cost_to_front, path_to_front))
					best_cost_to_node[neighbour] = {"cost":cost_to_front, "id":id }
	return None


def null_heuristic(graph, v, goal):
	"""
	Null heuristic used as a base line.

	Args:
		graph (ExplorableGraph): Undirected graph to search.
		v (str): Key for the node to calculate from.
		goal (str): Key for the end node to calculate to.

	Returns:
		0
	"""

	return 0


def euclidean_dist_heuristic(graph, v, goal):
	"""
	Warm-up exercise: Implement the euclidean distance heuristic.

	See README.md for exercise description.

	Args:
		graph (ExplorableGraph): Undirected graph to search.
		v (str): Key for the node to calculate from.
		goal (str): Key for the end node to calculate to.

	Returns:
		Euclidean distance between `v` node and `goal` node
	"""

	# TODO: finish this function!
	raise NotImplementedError


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
	"""
	Warm-up exercise: Implement A* algorithm.

	See README.md for exercise description.

	Args:
		graph (ExplorableGraph): Undirected graph to search.
		start (str): Key for the start node.
		goal (str): Key for the end node.
		heuristic: Function to determine distance heuristic.
			Default: euclidean_dist_heuristic.

	Returns:
		The best path as a list from the start and goal nodes (including both).
	"""

	# TODO: finish this function!
	raise NotImplementedError


def bidirectional_ucs(graph, start, goal):
	"""
	Exercise 1: Bidirectional Search.

	See README.md for exercise description.

	Args:
		graph (ExplorableGraph): Undirected graph to search.
		start (str): Key for the start node.
		goal (str): Key for the end node.

	Returns:
		The best path as a list from the start and goal nodes (including both).
	"""

	# TODO: finish this function!
	raise NotImplementedError


def bidirectional_a_star(graph, start, goal,
						 heuristic=euclidean_dist_heuristic):
	"""
	Exercise 2: Bidirectional A*.

	See README.md for exercise description.

	Args:
		graph (ExplorableGraph): Undirected graph to search.
		start (str): Key for the start node.
		goal (str): Key for the end node.
		heuristic: Function to determine distance heuristic.
			Default: euclidean_dist_heuristic.

	Returns:
		The best path as a list from the start and goal nodes (including both).
	"""

	# TODO: finish this function!
	raise NotImplementedError


def tridirectional_search(graph, goals):
	"""
	Exercise 3: Tridirectional UCS Search

	See README.MD for exercise description.

	Args:
		graph (ExplorableGraph): Undirected graph to search.
		goals (list): Key values for the 3 goals

	Returns:
		The best path as a list from one of the goal nodes (including both of
		the other goal nodes).
	"""
	# TODO: finish this function
	raise NotImplementedError


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
	"""
	Exercise 4: Upgraded Tridirectional Search

	See README.MD for exercise description.

	Args:
		graph (ExplorableGraph): Undirected graph to search.
		goals (list): Key values for the 3 goals
		heuristic: Function to determine distance heuristic.
			Default: euclidean_dist_heuristic.

	Returns:
		The best path as a list from one of the goal nodes (including both of
		the other goal nodes).
	"""
	# TODO: finish this function
	raise NotImplementedError


def return_your_name():
	"""Return your name from this function"""
	# TODO: finish this function
	raise NotImplementedError


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
	"""
	Race!: Implement your best search algorithm here to compete against the
	other student agents.

	See README.md for exercise description.

	Args:
		graph (ExplorableGraph): Undirected graph to search.
		start (str): Key for the start node.
		goal (str): Key for the end node.
		data :  Data used in the custom search.
			Default: None.

	Returns:
		The best path as a list from the start and goal nodes (including both).
	"""

	# TODO: finish this function!
	raise NotImplementedError


def three_bidirectional_search(graph, goals, heuristic=euclidean_dist_heuristic):
	"""
	Exercise 5: Use this to test out your implementation for Three Bidirectional Searches to help you with the report.

	See README.MD for exercise description

	Args:
		graph (ExplorableGraph): Undirected graph to search.
		goals (list): Key values for the 3 goals
		heuristic: Function to determine distance heuristic.
			Default: euclidean_dist_heuristic.

	Returns:
		The best path as a list from one of the goal nodes (including both of
		the other goal nodes).
	"""
	pass


def custom_heuristic(graph, v, goal):
	"""
	   Exercise 5: Use this to test out any custom heuristic for comparing Tridirectional vs 3 Bidirectional Searches for the report.
	   See README.md for exercise description.
	   Args:
		   graph (ExplorableGraph): Undirected graph to search.
		   v (str): Key for the node to calculate from.
		   goal (str): Key for the end node to calculate to.
	   Returns:
		   Custom heuristic distance between `v` node and `goal` node
	   """

pass


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
	"""
	Race!: Implement your best search algorithm here to compete against the
	other student agents.

	If you implement this function and submit your code to bonnie, you'll be
	registered for the Race!

	See README.md and the piazza post for exercise description.

	Args:
		graph (ExplorableGraph): Undirected graph to search.
		start (str): Key for the start node.
		goal (str): Key for the end node.
		data :  Data used in the custom search.
			Will be passed your data from load_data(graph).
			Default: None.

	Returns:
		The best path as a list from the start and goal nodes (including both).
	"""

	# TODO: finish this function!
	raise NotImplementedError



def load_data(graph, time_left):
	"""
	Feel free to implement this method. We'll call it only once
	at the beginning of the Race, and we'll pass the output to your custom_search function.
	graph: a networkx graph
	time_left: function you can call to keep track of your remaining time.
		usage: time_left() returns the time left in milliseconds.
		the max time will be 10 minutes.

	* To get a list of nodes, use graph.nodes()
	* To get node neighbors, use graph.neighbors(node)
	* To get edge weight, use graph[node1][node2]['weight']
	"""

	# nodes = graph.nodes()
	return None
