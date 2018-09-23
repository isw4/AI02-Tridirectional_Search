# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

from __future__ import division

import heapq
import os
import cPickle as pickle
import numpy as np


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

	def find(self, node_id):
		"""
		Finds a node from the queue by its id

		:param node_id:
		:return: a copy of the node
		"""
		entry = self.id_table[node_id]
		return tuple(entry)

	def __iter__(self):
		"""Queue iterator."""
		return iter(sorted(self.queue))

	def __str__(self):
		"""Priority Queue to string."""
		return 'PQ:%s' % self.queue

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
	BFS implementation

	:param graph: (ExplorableGraph/Graph) Undirected graph to search.
	:param start: (str) Key for the start node.
	:param goal: (str) Key for the end node.

	:returns
		The best path as a list from the start and goal nodes (including both).
		If there is no path, returns None
	"""
	if start == goal: return []

	# Init visited (vertices that have been looked at, not just expanded)
	visited = set()
	visited |= {start}
	# Init frontier (vertices to be expanded)
	frontier = PriorityQueue()
	path_len = 0  # Length of path (priority)
	path = [start]  # A list of nodes representing the path from the start to the node on the frontier
	frontier.append((path_len, path))

	while frontier.size() > 0:
		path_len, _, path = frontier.pop()
		neighbours_iter = graph.neighbors_iter(path[-1])    # Getting the neighbours of the last node in the path

		for neighbour in neighbours_iter:
			if neighbour == goal:   # Once the goal is seen, there are no shorter paths to consider
				path.append(neighbour)
				return path

			if neighbour not in visited:    # No need to visit neighbours twice
				visited |= {neighbour}
				path_to_front = list(path)  # Copying the path
				path_to_front.append(neighbour)
				frontier.append((path_len + 1, path_to_front))
	return None


def uniform_cost_search(graph, start, goal):
	"""
	UCS implementation

	:param graph: (ExplorableGraph/Graph) Undirected graph to search.
	:param start: (str) Key for the start node.
	:param goal: (str) Key for the end node.

	:returns
		The best path as a list from the start and goal nodes (including both).
		If there is no path, returns None
	"""
	if start == goal: return []

	# Init explored (vertices that have been expanded. Unlike BFS, must continue to visit a node until its expanded)
	explored = set()
	# Init frontier (vertices to be expanded)
	frontier = PriorityQueue()
	path_cost = 0   # Cost/weight of path (priority)
	path = [start]  # A list of nodes representing the path from the start to the node on the frontier
	id_ = frontier.append((path_cost, path))
	# Init hash table storing the best cost to the node (better paths to the same node will replace old paths)
	best_cost_to_node = {}
	best_cost_to_node[start] = { "cost":path_cost, "id":id_ }

	while frontier.size() > 0:
		path_cost, _, path = frontier.pop()
		if path[-1] == goal:    # Once the goal is expanded, there are no more lower cost paths to consider
			return path

		neighbours_iter = graph.neighbors_iter(path[-1])    # Getting the neighbours of the last node in the path
		explored |= {path[-1]}

		for neighbour in neighbours_iter:
			if neighbour not in explored:
				cost_to_front = path_cost + graph[path[-1]][neighbour]['weight']
				# If the path to the node is better than one that has already been seen, remove it from frontier
				# to be replaced later
				if neighbour in best_cost_to_node and cost_to_front < best_cost_to_node[neighbour]['cost']:
					frontier.remove(best_cost_to_node[neighbour]['id'])
				# If the path to the node is worse than one that has already been seen, prune this path
				elif neighbour in best_cost_to_node and cost_to_front >= best_cost_to_node[neighbour]['cost']:
					continue
				# Add path to frontier and update best cost to the node
				path_to_front = list(path)
				path_to_front.append(neighbour)
				id_ = frontier.append((cost_to_front, path_to_front))
				best_cost_to_node[neighbour] = {"cost":cost_to_front, "id":id_ }
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
	v_xy = np.array(graph.node[v]['pos'])
	goal_xy = np.array(graph.node[goal]['pos'])
	return np.linalg.norm(v_xy - goal_xy)


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
	"""
	A* search implementation

	:param graph: (ExplorableGraph/Graph) Undirected graph to search.
	:param start: (str) Key for the start node.
	:param goal: (str) Key for the end node.
	:param heuristic: Function to determine distance heuristic.
			Default: euclidean_dist_heuristic.

	:returns
		The best path as a list from the start and goal nodes (including both).
		If there is no path, returns None
	"""
	if start == goal: return []

	# Init explored (vertices that have been expanded. Unlike BFS, must continue to visit a node until its expanded)
	explored = set()
	# Init frontier (vertices to be expanded)
	frontier = PriorityQueue()
	path_cost = 0   # Cost/weight of path (not used for priority, but used to prune longer paths to the same node)
	path_f = path_cost + heuristic(graph, start, goal)     # Path priority estimate f used for A*
	path = [start]  # A list of nodes representing the path from the start to the node on the frontier
	id_ = frontier.append((path_f, path))
	# Init hash table storing the best cost to the node (better paths to the same node will replace old paths)
	best_cost_to_node = {}
	best_cost_to_node[start] = { "cost":path_cost, "id":id_ }

	while frontier.size() > 0:
		path_f, _, path = frontier.pop()
		if path[-1] == goal:    # Once the goal is expanded, there are no more lower cost paths to consider
			return path

		path_cost = path_f - heuristic(graph, path[-1], goal)   # Get back the actual cost and not the estimate f
		neighbours_iter = graph.neighbors_iter(path[-1])        # Getting the neighbours of the last node in the path
		explored |= {path[-1]}

		for neighbour in neighbours_iter:
			if neighbour not in explored:
				cost_to_front = path_cost + graph[path[-1]][neighbour]['weight']
				f_to_front = cost_to_front + heuristic(graph, neighbour, goal)

				# If the actual path to the node (not estimate) is better than one that has already been seen,
				# remove it from frontier to be replaced later
				if neighbour in best_cost_to_node and cost_to_front < best_cost_to_node[neighbour]['cost']:
					frontier.remove(best_cost_to_node[neighbour]['id'])
				# If the path to the node is worse than one that has already been seen, prune this path
				elif neighbour in best_cost_to_node and cost_to_front >= best_cost_to_node[neighbour]['cost']:
					continue
				# Add path to frontier and update best cost to the node
				path_to_front = list(path)
				path_to_front.append(neighbour)
				id_ = frontier.append((f_to_front, path_to_front))
				best_cost_to_node[neighbour] = {"cost":cost_to_front, "id":id_ }
	return None

###################################################################################
#           BI-DIRECTIONAL ALGORITHMS
###################################################################################

class ExplorableAStarDirection:
	def __init__(self, graph, seed, goal, heuristic, heuristic_goal=None):
		self.graph = graph
		self.goal = goal
		self.heuristic = heuristic
		self.heuristic_goal = heuristic_goal if heuristic_goal is not None else goal
		self.explored = set()           # Vertices at the end of a path from the seed that have been expanded
		self.frontier = PriorityQueue() # Vertices at the end of a path from the seed to be expanded
		self.best_cost_to_node = {}     # Best cost from the seed to the node (old paths will be replaced)
		self.frontier_ids = {}          # A list of frontier ids with the key as the last node on the path

		path_cost = 0  # Cost/weight of path
		path_f = path_cost + heuristic(graph, seed, heuristic_goal)
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
		f_to_front = cost_to_front + self.heuristic(self.graph, node, self.heuristic_goal)
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
		path_cost = path_f - self.heuristic(self.graph, popped, self.heuristic_goal)  # Get back the actual cost
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
		"""
		In phase 2, finds the explored/frontier nodes of the other direction that intersects with the
		explored nodes of this direction. Then for each intersecting node, there is a path from start
		to goal through the intersecting node. The shortest path to the intersecting node from this
		direction is known, but the path from the other direction may not be. Combine the paths, and
		then compare to the shortest known path.

		:param other_dir: (ExplorableAStarDirection) The search in the other direction
		:param final_cost: (int/float) Initialized to the cost of the first path met from both directions
		:param final_path: (list of char) Initialized to the first path met from both directions
		:return:
		"""
		nodes_in_other_frontier = set(other_dir.frontier_ids.keys())
		nodes_to_search = (nodes_in_other_frontier | other_dir.explored) & self.explored
		print("In phase 2 with intersecting nodes: {}".format(nodes_to_search))

		while len(nodes_to_search) != 0:
			node = nodes_to_search.pop()
			if node in other_dir.frontier_ids:
				# Find all paths from the other frontier to this explored set. There shouldn't be another
				# path from the other explored set to this explored set
				for id_ in other_dir.frontier_ids[node]['ids']:
					path_f, _, path = other_dir.frontier.find(id_)
					popped = path[-1]
					path_cost = path_f - other_dir.heuristic(other_dir.graph, popped, other_dir.heuristic_goal)  # Get back the actual cost
					combined_cost, combined_path = other_dir._combine_path(path, path_cost, self)
					print("Combined path {} with cost {}".format(combined_path, combined_cost))

					if combined_cost < final_cost:
						final_cost = combined_cost
						final_path = combined_path
		return final_cost, final_path


def bidir_astar_heuristic(graph, v, goal):
	v_xy = np.array(graph.node[v]['pos'])
	goal_xy = np.array(graph.node[goal]['pos'])
	return np.linalg.norm(v_xy - goal_xy)


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
	# bidirectional_a_star(graph, start, goal, null_heuristic)

	if start == goal: return []
	path = None

	from_start = ExplorableAStarDirection(graph, start, goal, null_heuristic)
	from_goal = ExplorableAStarDirection(graph, goal, goal, null_heuristic)

	n = 0
	upper_bound = float("inf")
	while from_start.frontier.size() > 0 and from_goal.frontier.size() > 0:
		if n % 2 == 0:
			temp_cost, temp_path, terminate = from_start.search_once(from_goal, upper_bound)
			if temp_cost is not None:
				cost, path = from_start.phase2(from_goal, temp_cost, temp_path)
				break
		else:
			temp_cost, temp_path, terminate = from_goal.search_once(from_start, upper_bound)
			if temp_cost is not None:
				cost, path = from_goal.phase2(from_start, temp_cost, temp_path)
				break
		n += 1
	return path


def bidirectional_a_star(graph, start, goal,
						 heuristic=bidir_astar_heuristic):
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
	if start == goal: return []
	path = None

	from_start = ExplorableAStarDirection(graph, start, goal, heuristic, goal)
	from_goal = ExplorableAStarDirection(graph, goal, goal, heuristic, start)

	n = 0
	upper_bound = float("inf")
	while from_start.frontier.size() > 0 and from_goal.frontier.size() > 0:
		if n % 2 == 0:
			temp_cost, temp_path, terminate = from_start.search_once(from_goal, upper_bound)
			if temp_cost is not None:
				cost, path = from_start.phase2(from_goal, temp_cost, temp_path)
				break
		else:
			temp_cost, temp_path, terminate = from_goal.search_once(from_start, upper_bound)
			if temp_cost is not None:
				cost, path = from_goal.phase2(from_start, temp_cost, temp_path)
				break
		n += 1
	return path


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
