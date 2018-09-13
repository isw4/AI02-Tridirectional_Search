# CS 6601: Artificial Intelligence - Assignment 2 - Search - On-campus students

**Warning: this repo is for on-campus students only. The link to the repo for online students is available on Canvas.**

## Setup

Clone this repository:

`git clone https://github.gatech.edu/omscs6601/assignment_2.git`

The submission scripts depend on the presence of 3 python packages - `requests`, `future`, and `nelson`. Install them using the command below:

`pip install -r requirements.txt`

Python 2.7 is recommended and has been tested.

Read [setup.md](./setup.md) for more information on how to effectively manage your git repository and troubleshooting information.

## Overview

Search is an integral part of AI. It helps in problem solving across a wide variety of domains where a solution isn’t immediately clear.  You will implement several graph search algorithms with the goal of solving bi-directional search.

### Submission and Due Date

Submit your code to bonnie for grading by running `submit.py`. **As per every assignment, you can find the output of all of your submissions on bonnie.udacity.com**

You are allowed **two submissions every thirty minutes**.

This assignment is due on Bonnie and Canvas on September 23th, 2018 by 11:59PM UTC-12 (Anywhere on Earth). The deliverables for the assignment are:

• All functions completed in `search_submission.py`

• One-page report named `report-{GT username}.pdf` to be submitted on canvas. Example: "report-jdoe3.pdf" if your GT username is "jdoe3".

**Only the final submission will be used for grading purposes. We cannot use your "best" submission out of all submissions.**

### The Files

While you'll only have to edit and submit **_search_submission.py_**, there are a number of notable files:

1. **_search_submission.py_**: Where you will implement your _PriorityQueue_, _Breadth First Search_, _Uniform Cost Search_, _A* Search_, _Bi-directional Search_
2. **_search_submission_tests.py_**: Sample tests to validate your searches locally.
3. **_search_unit_test.py_**: More detailed tests that run searches from all possible pairs of nodes in the graph
4. **_romania_graph.pickle_**: Serialized graph files for Romania.
5. **_atlanta_osm.pickle_**: Serialized graph files for Atlanta (optional for robust testing for Race!).
6. **_submit.py_**: A script to submit your work.
7. **_explorable_graph.py_**: A wrapper around `networkx` that tracks explored nodes. **FOR DEBUGGING ONLY**
8. **_visualize_graph.py_**: Module to visualize search results. See below on how to use it.
9. **_osm2networkx.py_**: Module used by visualize graph to read OSM networks.


> A note on visualizing results for the Atlanta graph:
>  
> The Atlanta graph is too big to display within a Python window like Romania. As a result, when you run the
> bidirectional tests in **_search_submission_tests.py_**, it generates a JSON file in the GeoJSON format. To see the
> graph, you can upload it to a private GitHub Gist or use [this](http://geojson.io/) site.
>  
> If you want to see how **_visualize_graph.py_** is used, take a look at the class TestBidirectionalSearch in
> **_search_submission_tests.py_**

## Resources

### Provide Resources
* Udacity, [Lesson 2: Search](https://classroom.udacity.com/courses/ud954/lessons/6375179396/concepts/68711451400923)
* R&N slides on [Uninformed Search](https://www.cc.gatech.edu/~thad/6601-gradAI-fall2015/chapter03-clean.pdf)
* [Informed Search](https://www.cc.gatech.edu/~thad/6601-gradAI-fall2015/chapter04a.pdf)

Links from Udacity, below the videos:
* [Finding Optimal Solutions to Rubik's Cube Using Pattern Databases](https://www.cs.princeton.edu/courses/archive/fall06/cos402/papers/korfrubik.pdf)
* [God's Number is 26 in the Quarter-Turn Metric](http://www.cube20.org/qtm/)
* [Reach for A∗: An Efficient Point-to-Point Shortest Path Algorithm](http://www.cc.gatech.edu/~thad/6601-gradAI-fall2015/02-search-01-Astart-ALT-Reach.pdf)
* [Computing the Shortest Path: A∗ Search Meets Graph Theory](http://www.cc.gatech.edu/~thad/6601-gradAI-fall2015/02-search-Goldberg03tr.pdf)
* [Reach-based Routing: A New Approach to Shortest Path Algorithms Optimized for Road Networks](http://www.cc.gatech.edu/~thad/6601-gradAI-fall2015/02-search-Gutman04siam.pdf)

### Relevant Challenge Questions
_Questions will be updated as they are added._

### Other Resources
[Introduction to A*](https://www.redblobgames.com/pathfinding/a-star/introduction.html)

## The Assignment

Your task is to implement several informed search algorithms that will calculate a driving route between two points in Romania with a minimal time and space cost.
There is a `search_submission_tests.py` file to help you along the way. Your searches should be executed with minimal runtime and memory overhead.

We will be using an undirected network representing a map of Romania (and an optional Atlanta graph used for the Race!).

### Warmups
We'll start by implementing some simpler optimization and search algorithms before the real exercises.

#### Warmup 1: Priority queue

_[5 points]_

In all searches that involve calculating path cost or heuristic (e.g. uniform-cost), we have to order our search frontier. It turns out the way that we do this can impact our overall search runtime.

To show this, you'll implement a priority queue and demonstrate its performance benefits. For large graphs, sorting all input to a priority queue is impractical. As such, the data structure you implement should have an amortized O(1) insertion and O(lg n) removal time. It should do better than the naive implementation in our tests (InsertionSortQueue), which sorts the entire list after every insertion.

> Hint:
> The heapq module has been imported for you. Feel free to use it
> Each edge has an associated weight.

#### Warmup 2: BFS

_[5 pts]_

To get you started with handling graphs, implement and test breadth-first search over the test network.

You'll complete this by writing the `breadth_first_search()` method. This returns a path of nodes from a given start node to a given end node, as a list.

For this part, it is optional to use the PriorityQueue as your frontier. You will require it from the next question onwards. You can use it here too if you want to be consistent.

> **Notes**:
> 1. You need to include start and goal in the path.
> 2. If your start and goal are the same then just return [].
> 3. Both of the above are just to keep your results consistent with our test cases.
> 4. You can access all the neighbors of a given node by calling `graph[node]`, or `graph.neighbors(node)` ONLY. 
> 5. To measure your search performance, the `explorable_graph.py` provided keeps track of which nodes you have accessed in this way (this is referred to as the set of 'Explored' nodes). To retrieve the set of nodes you've explored in this way, call `graph.explored_nodes`. If you wish to perform multiple searches on the same graph instance, call `graph.reset_search()` to clear out the current set of 'Explored' nodes. **WARNING**, these functions are intended for debugging purposes only. Calls to these functions will fail on Bonnie.

#### Warmup 3: Uniform-cost search

_[10 points]_

Implement uniform-cost search, using PriorityQueue as your frontier. From now on, PriorityQueue should be your default frontier.

`uniform_cost_search()` should return the same arguments as breadth-first search: the path to the goal node (as a list of nodes).

> **Notes**:
> 1. You can access the weight of an edge using: `graph[node_1][node_2]['weight']`
> 2. You do need to include start and goal in the path.
> 3. If your start and goal are the same then just return []
> 4. We will provide some margin of error in grading the size of your 'Explored' set, but it should be close to the results provided by our reference implementation.
> 5. The above are just to keep your results consistent with our test cases.

#### Warmup 4: A* search

_[10 points]_

Implement A* search using Euclidean distance as your heuristic. You'll need to implement `euclidean_dist_heuristic()` then pass that function to `a_star()` as the heuristic parameter. We provide `null_heuristic()` as a baseline heuristic to test against when calling a_star tests.

> **Hint**:
> You can find a node's position by calling the following to check if the key is available: `graph.node[n]['pos']`

> **Notes**:
> 1. You do need to include start and goal in the path.
> 2. If your start and goal are the same then just return []
> 3. We will provide some margin of error in grading the size of your 'Explored' set, but it should be close to the results provided by our reference implementation.
> 4. The above are just to keep your results consistent with our test cases.

---
### Exercises
The following exercises will require you to implement several kinds of bidirectional searches. The benefits of these algorithms over uninformed or unidirectional search are more clearly seen on larger graphs. As such, during grading, we will evaluate your performance on the map of Atlanta [OpenStreetMap](http://wiki.openstreetmap.org) included in this assignment.

For these exercises, we recommend you take a look at the following resources.

1. [A Star meets Graph Theory](https://github.gatech.edu/omscs6601/assignment_2_online/raw/master/resources/A%20Star%20meets%20Graph%20Theory.pdf)
2. [Bi Directional A Star - Slides](https://github.gatech.edu/omscs6601/assignment_2_online/raw/master/resources/Bi%20Directional%20A%20Star%20-%20Slides.pdf)
3. [Bi Directional A Star with Additive Approx Bounds](https://github.gatech.edu/omscs6601/assignment_2_online/raw/master/resources/Bi%20Directional%20A%20Star%20with%20Additive%20Approx%20Bounds.pdf)
4. [Bi Directional A Star](https://github.gatech.edu/omscs6601/assignment_2_online/raw/master/resources/Bi%20Directional%20A%20Star.pdf)
5. [Search Algorithms Slide Deck](https://github.gatech.edu/omscs6601/assignment_2_online/raw/master/resources/Search%20Algorithms%20Slide%20Deck.pdf)

#### Exercise 1: Bidirectional uniform-cost search

_[15 points]_

Implement bidirectional uniform-cost search. Remember that this requires starting your search at both the start and end states.

`bidirectional_ucs()` should return the path from the start node to the goal node (as a list of nodes).

> **Notes**:
> 1. You do need to include start and goal in the path.
> 2. If your start and goal are the same then just return []
> 3. We will provide some margin of error in grading the size of your 'Explored' set, but it should be close to the results provided by our reference implementation.

> The notes above are to keep your results consistent with our test cases.

#### Exercise 2: Bidirectional A* search

_[20 points]_

Implement bidirectional A* search. Remember that you need to calculate a heuristic for both the start-to-goal search and the goal-to-start search.

To test this function, as well as using the provided tests, you can compare the path computed by bidirectional A star to bidirectional ucs search above.
`bidirectional_a_star()` should return the path from the start node to the goal node, as a list of nodes.

> **Notes**:
> 1. You do need to include start and goal in the path.
> 2. If your start and goal are the same then just return []
> 3. We will provide some margin of error in grading the size of your 'Explored' set, but it should be close to the results provided by our reference implementation.

#### Exercise 3: Tridirectional UCS search

_[19 points]_

Implement tridirectional search in the naive way: starting from each goal node, perform a uniform-cost search and keep
expanding until two of the three searches meet. This should be one continuous path that connects all three nodes.

`tridirectional_search()` should return a path between all three nodes. You can return the path in any order. Eg.
(1->2->3 == 3->2->1). You may also want to look at the Tri-city
search challenge question on Udacity.

> **Notes**:
> 1. You need to include start and goal in the path.
> 2. If any goals are the same then just return [] as the path between them.
> 3. We will provide some margin of error in grading the size of your 'Explored' set, but it should be close to the
     results provided by our reference implementation.

#### Exercise 4: Upgraded Tridirectional search

_[10 points]_

This is the heart of the assignment. Implement tridirectional search in such a way as to consistently improve on the
performance of your previous implementation. This means consistently exploring fewer nodes during your search in order
to reduce runtime.

The specifics are up to you, but we have a few suggestions:
 * Tridirectional A*
 * choosing landmarks and pre-computing reach values
 * ATL (A\*, landmarks, and triangle-inequality)
 * shortcuts (skipping nodes with low reach values)

`tridirectional_upgraded()` should return a path between all three nodes.

> **Notes**:
> 1. You do need to include each goal in the path.
> 2. If any two goals are the same then just return [] as the path between them
> 3. We will provide some margin of error in grading the size of your 'Explored' set, but it should be close to the
     results provided by our reference implementation.
     
     
#### Exercise 5: Comparing Tridirectional search vs 3 Bidirectional Searches

_[5 points]_

Implement three bidirectional searches in `three_bidirectional_search()` using your previous implementation of bidirectional UCS or A-star. 

After you implement the `three_bidirectional_search()` algorithm, it’s time to step back and compare how this and your `tridirectional_search()` algorithm work. Theoretically, tridirectional search _should_ outperform three bidirectional searches- the tridirectional search shares more information after all.

However, this is still an open research question. _Are there cases where three bidirectional searches outperform one tridirectional search, and, if so, how do we detect them?_ These rules should, ideally, generalize to real world maps.

For this write-up, consider both the Uniform Cost heuristic and A* search. For each of these, ask the question “Under what conditions (properties of the graph being explored, different A* heuristics) would three bidirectional outperform one tridirectional?” Without running the algorithm on a graph, but knowing the starting points, can you determine which algorithm will run better? We’re not interested right now whether it would be faster to just run the search algorithm instead of the check you invented, so your check doesn’t have to asymptotically faster than the search algorithm.

Try to formulate a hypothesis stating the conditions under which three bidirectional will outperform tridirectional. It doesn’t have to perfectly predict which will do better, but it should have some predictive power. Then, back this up with some evidence. Run some tests and give some examples that back up your hypothesis.


> **Notes**:

> 1. Your hypothesis may be that tridirectional always outperforms bidirectional. Feel free to make that argument, but back it up with solid reasoning if you choose to do so.

> 2. Performance here is not just about the number of nodes expanded. The three bidirectional search will, after all, expand nodes multiple times. So if you only look at how many nodes were ever expanded, you’ll miss the overlapping nodes that were expanded multiple times. Instead, look at the total number of times nodes were expanded.

> 3. A grid graph (grid.gpickle) and a visualization tool (search_submission_tests_grid.py) are provided in the assignment repository to test your ideas and visualize the difference in explored nodes between bidirectional and tridirectional algorithms.

> 4. You're also welcome to use the optional `custom_heuristic()` method to help you compare this with tridirectional search for other heuristics.


**In summary, submit a pdf on canvas with the following:**

1) For the uniform cost heuristic
    * Describe the conditions under which three bidirectional searches will outperform tridirectional search.
    * Provide examples and evidence to support this.

2) For A* search:
    * Describe the conditions under which three bidirectional searches will outperform tridirectional search.
    * Provide examples and evidence to support this.

_(Try to keep your word count to something that would fit on a single page. Feel free to add pictures on top of this one page limit.)_


     
#### Final Task: Return your name
_[1 point]_

A simple task to wind down the assignment. Return you name from the function aptly called `return_your_name()`.


### The Race!

Here's your chance to show us your best stuff. This part is mandatory if you want to compete in the race for extra credit. Implement `custom_search()` using whatever strategy you like.

**Bonus points are added to the grade for this assignment, not to your overall grade.**

The Race! will be based on Atlanta Pickle data.

## References

Here are some notes you might find useful.
1. [Bonnie: Error Messages](https://docs.google.com/document/d/1hykYneVoV_JbwBjVz9ayFTA6Yr3pgw6JBvzrCgM0vyY/pub)
2. [Bi-directional Search](https://docs.google.com/document/d/14Wr2SeRKDXFGdD-qNrBpXjW8INCGIfiAoJ0UkZaLWto/pub)
3. [Using Landmarks](https://docs.google.com/document/d/1YEptGbSYUtu180MfvmrmA4B6X9ImdI4oOmLaaMRHiCA/pub)

## Frequent Issues and Solutions
1. Make sure you clean up any changes/modifications/additions you make to the networkx graph structure before you exit the search function. Depending on your changes, the auto grader might face difficulties while testing. The best alternative is to create your own data structure(s).
2. If you're having problems (exploring too many nodes) with your Breadth first search implementation, one thing many students have found useful is to re-watch the Udacity videos for an optimization trick mentioned.
3. While submitting to Bonnie, many times the submission goes through even if you get an error on the terminal. You should check the web interface to make sure it’s not gone through before re-submitting. On the other hand, make sure your final submission goes through with Bonnie.
4. Most 'NoneType object ...' errors are because the path you return is not completely connected (a pair of successive nodes in the path are not connected). Or because the path variable itself is empty.
5. Adding unit tests to your code may cause your Bonnie submission to fail. It is best to comment them out when you submit to Bonnie.
6. Make sure you're returning [] for when the source and destination points are the same.
7. Make sure you break ties using FIFO in your priority queue implementation.
