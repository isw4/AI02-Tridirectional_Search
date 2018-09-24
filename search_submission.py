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

####################################################################################################################
#                PRIORITY QUEUE AND HEURISTICS                                                                     #
####################################################################################################################

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


def bidir_astar_heuristic(graph, v, goal):
	v_xy = np.array(graph.node[v]['pos'])
	goal_xy = np.array(graph.node[goal]['pos'])
	return np.linalg.norm(v_xy - goal_xy)


def tridir_astar_heuristic(graph, v, goal):
	return 0

####################################################################################################################
#                ONE DIRECTIONAL SEARCH                                                                            #
####################################################################################################################

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

####################################################################################################################
#                BI-DIRECTIONAL SEARCH                                                                             #
####################################################################################################################

class ExplorableBiSearchDirection:
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

		:param node: (char) the node to be added into the frontier
		:param path: (list of char) the path that does not currently include the node to be added. The last node
					 in the path is the node that was last expanded
		:param path_cost: (int) the path cost, not including heuristic
		:return: None
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
			return None

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
		return None

	def _fpop(self):
		"""
		Popping from frontier. Needs to remove from the frontier_id hash table as well

		:return:
			path_f: (int/float) the path cost + heuristic
			id_: (int) id of the node that was just popped from the frontier
			path: (list of char) the path from the starting seed to the frontier node
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
		"""
		Called when this direction pops a new frontier and it is a node that
		the other direction has ALREADY explored. The explored node is guaranteed to be the
		shortest path from that direction to the explored node, and vice versa, making the
		result be the shortest path to the midpoint from each direction (but may not be the
		shortest path from start to goal)

		:param path: (list of char) path of the popped frontier
		:param path_cost: (int) the sum of cost along the path to the popped frontier
		:param other_dir: (ExplorableBiSearchDirection) the other direction
		:return:
			combined_cost: the cost of the popped path + the shortest path to the node from the other dir
			combined_path: the path from start to goal through the node at the end of the popped path
		"""
		print("Path {} with cost {}".format(path, path_cost))
		path = list(path)
		midpoint = path[-1]
		other_path = other_dir.best_cost_to_node[midpoint]['path']
		print("Other path {} with cost {}".format(other_path, other_dir.best_cost_to_node[midpoint]['cost']))

		# Ordering the path correctly so that it's start -> goal and not goal -> start
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

	def search_once(self, other_dir):
		"""
		Phase 1: Search one node from this direction

		:param other_dir: (ExplorableBiSearchDirection) the other direction
		:return:
			combined_cost, combined_path if the explored nodes from each direction meet
			None, None if not
		"""
		combined_cost = None
		combined_path = None

		path_f, _, path = self._fpop()
		popped = path[-1]
		self.add_to_explored(popped)
		path_cost = path_f - self.heuristic(self.graph, popped, self.heuristic_goal)  # Get back the actual cost

		# Once the goal has been expanded/explored by both, the shortest path from one direction and
		# the other direction to the intermediate has been found. However, the shortest path from one
		# direction to the other has not necessarily been found. Will combine the two paths, and then
		# continue to look at the neighbours.
		if popped in other_dir.explored:
			print("\nMIDPOINT {} POPPED".format(popped))
			combined_cost, combined_path = self._combine_path(path, path_cost, other_dir)
			print("Combined path {} with cost {}".format(combined_path, combined_cost))

		# Adding the neighbours of the popped node to the frontier
		neighbours_iter = self.graph.neighbors_iter(popped)
		print("\nLooking at path {}\nUpdated explored: {}\nUpdated f_id: {}".format(path, self.explored, self.frontier_ids))

		for neighbour in neighbours_iter:
			if neighbour not in self.explored:
				print("Neighbour {} found".format(neighbour))
				self._fadd_and_update(neighbour, path, path_cost)
				print("Frontier: {}\nf_ids: {}\nBest: {}".format(self.frontier, self.frontier_ids, self.best_cost_to_node))

		return combined_cost, combined_path

	def phase2(self, other_dir, final_cost, final_path):
		"""
		In phase 2, finds the explored/frontier nodes of the other direction that intersects with the
		explored nodes of this direction. Then for each intersecting node, there is a path from start
		to goal through the intersecting node. The shortest path to the intersecting node from this
		direction is known, but the path from the other direction may not be. Combine the paths, and
		then compare to the shortest known path.

		:param other_dir: (ExplorableBiSearchDirection) The search in the other direction
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
	path = bidirectional_a_star(graph, start, goal, null_heuristic)
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

	from_start = ExplorableBiSearchDirection(graph, start, goal, heuristic, goal)
	from_goal = ExplorableBiSearchDirection(graph, goal, goal, heuristic, start)

	n = 0
	while from_start.frontier.size() > 0 and from_goal.frontier.size() > 0:
		if n % 2 == 0:
			temp_cost, temp_path = from_start.search_once(from_goal)
			if temp_cost is not None:
				cost, path = from_start.phase2(from_goal, temp_cost, temp_path)
				break
		else:
			temp_cost, temp_path = from_goal.search_once(from_start)
			if temp_cost is not None:
				cost, path = from_goal.phase2(from_start, temp_cost, temp_path)
				break
		n += 1
	return path


####################################################################################################################
#                TRI-DIRECTIONAL SEARCH                                                                            #
####################################################################################################################

class TriSearchTracker:
	"""Tracks which tri-search directions have already met"""
	def __init__(self, directions):
		"""
		Tracks whether directions have met

		:param directions: (list of ExplorableTriSearchDirection) with length 3
		"""
		assert len(directions) == 3, "Must have only 3 directions"

		self.directions = directions

		self.met = np.ndarray((3,3))
		self.met[::] = False

		self.direction_ids = {}
		for i in range(0, 3):
			self.direction_ids[directions[i].id] = i

	def meet(self, direction1, direction2):
		"""Marks that two directions have met"""
		id1 = direction1.id
		id2 = direction2.id
		ix1 = self.direction_ids[id1]
		ix2 = self.direction_ids[id2]
		self.met[ix1, ix2] = True
		self.met[ix2, ix1] = True
		return None

	def have_met(self, direction1, direction2):
		"""Returns whether two directions have met"""
		id1 = direction1.id
		id2 = direction2.id
		ix1 = self.direction_ids[id1]
		ix2 = self.direction_ids[id2]
		return self.met[ix1, ix2]

	def get_direction_from_id(self, direction_id):
		ix = self.direction_ids[direction_id]
		return self.directions[ix]

	def get_third_direction_id(self, id1, id2):
		ix1 = self.direction_ids[id1]
		ix2 = self.direction_ids[id2]
		ix3 = ({0, 1, 2} - {ix1, ix2}).pop()
		return self.directions[ix3]

class ExplorableTriSearchDirection:
	"""Class used for tri-directional UCS"""
	def __init__(self, graph, seed):
		self.id = id(self)
		self.name = seed.upper()
		self.graph = graph
		self.origin = seed
		self.explored = set()               # Vertices at the end of a path from the seed that have been expanded
		self.frontier = PriorityQueue()     # Vertices at the end of a path from the seed to be expanded
		self.best_cost_to_node = {}         # Best cost from the seed to the node (old paths will be replaced)
		self.frontier_ids = {}              # A list of frontier ids with the key as the last node on the path

		# call ExplorableTriSearchDirection.setup() after init the 3 directions to set the below up properly
		self.goal1 = None
		self.goal2 = None
		self.tracker = None

		# Init the frontier and frontier id table
		path_cost = 0  # Cost/weight of path
		#path_f = path_cost + heuristic(graph, seed, heuristic_goal)
		path = [seed]
		#id_ = self.frontier.append((path_f, path))
		node_id = self.frontier.append((path_cost, path))
		self.frontier_ids[seed] = {'ids': [node_id]}

		# Init the best cost to node table
		self.best_cost_to_node = {}
		self.best_cost_to_node[seed] = {"cost": path_cost, "id": node_id, 'path': path}

	@staticmethod
	def setup(directions):
		directions[0].goal1 = directions[1]
		directions[0].goal2 = directions[2]
		directions[1].goal1 = directions[0]
		directions[1].goal2 = directions[2]
		directions[2].goal1 = directions[0]
		directions[2].goal2 = directions[1]
		tracker = TriSearchTracker(directions)  # Tracks whether each direction has met the other
		for dir in directions:
			dir.add_tracker(tracker)
		return None

	def add_to_explored(self, node):
		self.explored |= {node}

	def add_tracker(self, tracker):
		"""Adds the tracker that tracks which directions have seen each other"""
		self.tracker = tracker

	def _fadd_and_update(self, node, path, path_cost):
		"""
		Adding node to the path(copy), adding into the frontier, updating the frontier_id table, and updating
		the best cost to node table
		:param node: (char) the node to be added into the frontier
		:param path: (list of char) the path that does not currently include the node to be added. The last node
					 in the path is the node that was last expanded
		:param path_cost: (int) the path cost, not including heuristic
		:return: None
		"""
		popped = path[-1]
		cost_to_front = path_cost + self.graph[popped][node]['weight']

		# If the actual path to the node (not estimate) is better than one that has already been seen,
		# remove it from frontier(and frontier id table) to be replaced later
		if node in self.best_cost_to_node and cost_to_front < self.best_cost_to_node[node]['cost']:
			# print("Better than old path to {}\nBest: {}".format(node, self.frontier))
			self._fremove(node, self.best_cost_to_node[node]['id'])
			# print("Removed: {}".format(self.frontier))

		# If the path to the node is worse than one that has already been seen, prune this path
		elif node in self.best_cost_to_node and cost_to_front >= self.best_cost_to_node[node]['cost']:
			# print("Old path is better. Pruning this path")
			return None

		# Adding to the frontier
		path_to_front = list(path)
		path_to_front.append(node)
		id_ = self.frontier.append((cost_to_front, path_to_front))

		# Updating the frontier_id table
		if node in self.frontier_ids:
			self.frontier_ids[node]['ids'].append(id_)
		else:
			self.frontier_ids[node] = {'ids': [id_]}

		# Adding to the best cost to node table
		self.best_cost_to_node[node] = {"cost": cost_to_front, "id": id_, "path": path_to_front}
		return None

	def _fpop(self):
		"""
		Popping from frontier. Needs to remove from the frontier_id hash table as well
		:return:
			path_f: (int/float) the path cost + heuristic
			id_: (int) id of the node that was just popped from the frontier
			path: (list of char) the path from the starting seed to the frontier node
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

	def _expand(self, node):
		"""
		Expanding the node and getting a neighbours interator for that node. Also adds to
		this direction's explored set
		:param node: (char) node to get neighbours for
		:return: Graph's neighbour iterator for the node
		"""
		self.add_to_explored(node)
		return self.graph.neighbors_iter(node)

	def _combine_path(self, path, path_cost, other_dir):
		"""
		Called when this direction pops a new frontier and it is a node that
		the other direction has ALREADY explored. The explored node is guaranteed to be the
		shortest path from that direction to the explored node, and vice versa, making the
		result be the shortest path to the midpoint from each direction (but may not be the
		shortest path from start to goal)

		:param path: (list of char) path of the popped frontier
		:param path_cost: (int) the sum of cost along the path to the popped frontier
		:param other_dir: (ExplorableBiSearchDirection) the other direction
		:return:
			combined_cost: the cost of the popped path + the shortest path to the node from the other dir
			combined_path: the path from start to goal through the node at the end of the popped path
		"""
		# print("Expanding this path: {}, with cost {}".format(path, path_cost))
		path = list(path)   # Make a copy so the original path will not be changed by accident
		midpoint = path[-1] # The node that was just expanded from one dir, and is already explored in the other dir
		other_path = other_dir.best_cost_to_node[midpoint]['path']  # The best path from other dir to midpoint
		# print("Other path: {}, with cost {}".format(other_path, other_dir.best_cost_to_node[midpoint]['cost']))

		# Ordering the path correctly so that it's origin -> goal1/goal2 and not goal1/goal2 -> origin
		if path[0] == self.origin:
			part2 = other_path[::-1]
			combined_path = path[:-1]
			combined_path.extend(part2)
		else:
			part2 = path[::-1]
			combined_path = other_path[:-1]
			combined_path.extend(part2)

		combined_cost = path_cost + other_dir.best_cost_to_node[midpoint]['cost']
		return combined_cost, combined_path

	def search_once(self):
		combined_cost = None
		combined_path = None
		endpoint_id = None

		path_cost, _, path = self._fpop()
		popped = path[-1]
		# print("\nFROM {}, expanding: {}".format(self.name, popped))

		# TODO what if the expanded node hits both at the same time?
		if popped in self.goal1.explored and not self.tracker.have_met(self, self.goal1):
			# print("MIDPOINT '{}' FOUND BETWEEN '{}' and '{}'".format(popped, self.name, self.goal1.name))
			combined_cost, combined_path = self._combine_path(path, path_cost, self.goal1)
			endpoint_id = self.goal1.id
			self.tracker.meet(self, self.goal1)
			# print("Combined path {} with cost {}".format(combined_path, combined_cost))
		if popped in self.goal2.explored and not self.tracker.have_met(self, self.goal2):
			# print("MIDPOINT '{}' FOUND BETWEEN '{}' and '{}'".format(popped, self.name, self.goal2.name))
			combined_cost, combined_path = self._combine_path(path, path_cost, self.goal2)
			endpoint_id = self.goal2.id
			self.tracker.meet(self, self.goal2)
			# print("Combined path {} with cost {}".format(combined_path, combined_cost))

		# TODO should i check for expanded nodes that are in explored here? if node off frontier is in explored,
		# pop another off the frontier
		# If two directions have already met and the optimal path between the two origins has
		# been calculated, then there is no need to expand frontiers that has been explored
		# by the other direction.
		# if self.tracker.have_met(self, self.goal1) and popped in self.goal1.explored:
		# 	print("Node has already been expanded by {}. Skipping".format(self.goal1.name))
		# 	self.add_to_explored(popped)
		# 	return combined_cost, combined_path, endpoint_id
		# if self.tracker.have_met(self, self.goal2) and popped in self.goal2.explored:
		# 	print("Node has already been expanded by {}. Skipping".format(self.goal2.name))
		# 	self.add_to_explored(popped)
		# 	return combined_cost, combined_path, endpoint_id

		neighbours_iter = self._expand(popped)
		for neighbour in neighbours_iter:
			# Checking my unexplored neighbours (prevents backtracking)
			if neighbour not in self.explored:
				# print("Looking at '{}'".format(neighbour.upper()))
				# # If two directions have already met and the optimal path between the two origins has
				# # been calculated, then there is no need to expand frontiers that has been explored
				# # by the other direction.
				# if self.tracker.have_met(self, self.goal1) and neighbour in self.goal1.explored:
				# 	print("Node has already been expanded by {}. Skipping".format(self.goal1.name))
				# 	continue
				# if self.tracker.have_met(self, self.goal2) and neighbour in self.goal2.explored:
				# 	print("Node has already been expanded by {}. Skipping".format(self.goal2.name))
				# 	continue

				self._fadd_and_update(neighbour, path, path_cost)
				# print("Frontier: {}\nf_ids: {}\nBest: {}".format(self.frontier, self.frontier_ids, self.best_cost_to_node))

		return combined_cost, combined_path, endpoint_id

	def phase2(self, final_cost, final_path, other_dir_id):
		"""
		In phase 2, finds the explored/frontier nodes of the other direction that intersects with the
		explored nodes of this direction. Then for each intersecting node, there is a path from start
		to goal through the intersecting node. The shortest path to the intersecting node from this
		direction is known, but the path from the other direction may not be. Combine the paths, and
		then compare to the shortest known path.

		:param other_dir: (ExplorableBiSearchDirection) The search in the other direction
		:param final_cost: (int/float) Initialized to the cost of the first path met from both directions
		:param final_path: (list of char) Initialized to the first path met from both directions
		:return:
		"""
		other_dir = self.tracker.get_direction_from_id(other_dir_id)
		nodes_in_other_frontier = set(other_dir.frontier_ids.keys())
		nodes_to_search = (nodes_in_other_frontier | other_dir.explored) & self.explored
		# print("In phase 2 with intersecting nodes: {}".format(nodes_to_search))

		while len(nodes_to_search) != 0:
			node = nodes_to_search.pop()
			if node in other_dir.frontier_ids:
				# Find all paths from the other frontier to this explored set. There shouldn't be another
				# path from the other explored set to this explored set
				for id_ in other_dir.frontier_ids[node]['ids']:
					path_cost, _, path = other_dir.frontier.find(id_)
					combined_cost, combined_path = other_dir._combine_path(path, path_cost, self)
					# print("Combined path {} with cost {}".format(combined_path, combined_cost))

					if combined_cost < final_cost:
						final_cost = combined_cost
						final_path = combined_path
		return final_cost, final_path

	def phase3(self, cost_found, path_found, other_dir_id):

		other_dir = self.tracker.get_direction_from_id(other_dir_id)
		# print("This dir name: {}".format(self.name))
		# print("Other dir name: {}".format(other_dir.name))
		# Confirming the cost and path of the directions that just touched
		cost1, path1 = self.phase2(cost_found, path_found, other_dir_id)

		# Attempting to see if other frontiers that haven't touched could touch
		cost2 = float("inf")
		path2 = []
		third_dir = self.tracker.get_third_direction_id(self.id, other_dir_id)
		if self.tracker.have_met(third_dir, other_dir):
			# search the frontiers of third and self
			# print("self front: {}, explored: {}".format(self.frontier, self.explored))
			# print("third dir front: {}, explored: {}".format(third_dir.frontier, third_dir.explored))

			nodes_in_third_frontier = set(third_dir.frontier_ids.keys())
			nodes_in_self_frontier = set(self.frontier_ids.keys())
			nodes_to_search = (nodes_in_third_frontier | third_dir.explored) & (nodes_in_self_frontier | self.explored)

			# print("In phase 3 with intersecting nodes: {}".format(nodes_to_search))

			while len(nodes_to_search) != 0:
				node = nodes_to_search.pop()

				# If in both frontiers, that means that they both have a best cost to node already
				self_best_path = self.best_cost_to_node[node]['path']
				self_best_cost = self.best_cost_to_node[node]['cost']
				combined_cost, combined_path = self._combine_path(self_best_path, self_best_cost, third_dir)
				# print(combined_cost)
				# print(combined_path)

				if combined_cost < cost2:
					cost2 = combined_cost
					path2 = combined_path

		else: # self has met third dir before this
			# search the frontiers of third and other dir
			# print("other dir front: {}, explored: {}".format(other_dir.frontier, other_dir.explored))
			# print("third dir front: {}, explored: {}".format(third_dir.frontier, third_dir.explored))

			nodes_in_third_frontier = set(third_dir.frontier_ids.keys())
			nodes_in_other_frontier = set(other_dir.frontier_ids.keys())
			nodes_to_search = (nodes_in_third_frontier | third_dir.explored) & (nodes_in_other_frontier | other_dir.explored)

			# print("In phase 3 with intersecting nodes: {}".format(nodes_to_search))

			while len(nodes_to_search) != 0:
				node = nodes_to_search.pop()

				# If in both frontiers, that means that they both have a best cost to node already
				other_dir_best_path = other_dir.best_cost_to_node[node]['path']
				other_dir_best_cost = other_dir.best_cost_to_node[node]['cost']
				combined_cost, combined_path = other_dir._combine_path(other_dir_best_path, other_dir_best_cost, third_dir)
				# print(combined_cost)
				# print(combined_path)

				if combined_cost < cost2:
					cost2 = combined_cost
					path2 = combined_path

		return cost1, path1, cost2, path2

	@staticmethod
	def find_best_two_of_three(costs, paths):
		costs = np.array(costs)
		sorted_ix = np.argsort(costs)
		return paths[sorted_ix[0]], paths[sorted_ix[1]]


	@staticmethod
	def stitch_paths(path1, path2):
		full_path = None
		if path1[0] == path2[0]:
			full_path = path1[::-1]
			second_half = path2[1:]
			full_path.extend(second_half)
		elif path1[0] == path2[-1]:
			full_path = path2
			second_half = path1[1:]
			full_path.extend(second_half)
		elif path1[-1] == path2[0]:
			full_path = path1
			second_half = path2[1:]
			full_path.extend(second_half)
		elif path1[-1] == path2[-1]:
			full_path = path1[:-1]
			second_half = path2[::-1]
			full_path.extend(second_half)
		else:
			raise ValueError("Paths returned don't match")
		return full_path

def bidir(graph, start, goal):
	if start == goal: return []
	path = None

	from_start = ExplorableBiSearchDirection(graph, start, goal, null_heuristic)
	from_goal = ExplorableBiSearchDirection(graph, goal, goal, null_heuristic)

	n = 0
	while from_start.frontier.size() > 0 and from_goal.frontier.size() > 0:
		if n % 2 == 0:
			temp_cost, temp_path = from_start.search_once(from_goal)
			if temp_cost is not None:
				cost, path = from_start.phase2(from_goal, temp_cost, temp_path)
				break
		else:
			temp_cost, temp_path = from_goal.search_once(from_start)
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
	assert len(goals) == 3, "Must have only 3 goals"
	if goals[0] == goals[1] == goals [2]: return []
	if goals[0] == goals[1]: return bidir(graph, goals[0], goals[2])
	if goals[0] == goals[2]: return bidir(graph, goals[0], goals[1])
	if goals[1] == goals[2]: return bidir(graph, goals[1], goals[0])
	full_path = None

	# print("Goals are: {}, {}, {}".format(goals[0], goals[1], goals[2]))

	directions = []
	for i in range(0, 3):
		directions.append(ExplorableTriSearchDirection(graph, seed=goals[i]))
	ExplorableTriSearchDirection.setup(directions)

	n = 0
	costs = []
	paths = []
	temp_cost, temp_path, endpoint_id = None, None, None
	front_size = np.array([directions[0].frontier.size(), directions[1].frontier.size(), directions[2].frontier.size()])
	while np.sum(front_size == 0) < 2:
		if directions[n % 3].frontier.size() != 0:
			temp_cost, temp_path, endpoint_id = directions[n % 3].search_once()
		if temp_cost is not None:
			if len(paths) == 0:
				temp_cost, temp_path = directions[n % 3].phase2(temp_cost, temp_path, endpoint_id)
				costs.append(temp_cost)
				paths.append(temp_path)
				temp_cost = None
			elif len(paths) == 1:
				temp_cost1, temp_path1, temp_cost2, temp_path2 = directions[n % 3].phase3(temp_cost, temp_path, endpoint_id)
				costs.append(temp_cost1)
				paths.append(temp_path1)
				costs.append(temp_cost2)
				paths.append(temp_path2)
				final_paths = ExplorableTriSearchDirection.find_best_two_of_three(costs, paths)
				break
		n += 1
		front_size = [directions[0].frontier.size(), directions[1].frontier.size(), directions[2].frontier.size()]
	if final_paths:
		# print("Paths found: \n{}\n{}".format(final_paths[0], final_paths[1]))
		full_path = ExplorableTriSearchDirection.stitch_paths(final_paths[0], final_paths[1])

	return full_path

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
	path = tridirectional_search(graph, goals)
	return path


def return_your_name():
	"""Return your name from this function"""
	return "Isaac Wong"


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
