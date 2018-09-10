# coding=utf-8
from networkx import Graph


class ExplorableGraph(object):
    """
    Keeps track of "explored nodes" i.e. nodes that have been queried from the
    graph.

    Delegates graph operations to a networkx.Graph
    """

    def __init__(self, graph):
        """
        :type graph: Graph
        """
        self.graph = graph
        self._expanded_nodes = dict(zip(graph.nodes(), [0] * len(graph.nodes())))
        self._explored_nodes = set()

    @property
    def explored_nodes(self):
        return self._explored_nodes

    def __getattr__(self, item):
        return getattr(self.graph, item)

    def reset_search(self):
        self._explored_nodes = set()
        self._expanded_nodes = dict(zip(self.graph.nodes(), [0] * len(self.graph.nodes())))

    def __iter__(self):
        self._explored_nodes = set(iter(self.graph.node))
        return self.graph.__iter__()

    def __getitem__(self, n):
        self._explored_nodes |= {n}
        self._expanded_nodes[n] += 1
        return self.graph.__getitem__(n)

    def nodes_iter(self, data=False):
        self._explored_nodes = set(self.graph.nodes_iter())
        return self.graph.nodes_iter(data)

    def neighbors(self, n):
        self._explored_nodes |= {n}
        self._expanded_nodes[n] += 1
        return self.graph.neighbors(n)

    def neighbors_iter(self, n):
        self._explored_nodes |= {n}
        self._expanded_nodes[n] += 1
        return self.graph.neighbors_iter(n)

    def get_expanded_nodes(self):
        return self._expanded_nodes
