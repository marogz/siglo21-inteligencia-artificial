import networkx as nx
import heapq

# Uninformed Search: Breadth-First Search (BFS)
def bfs(positions_n, start):
    # Create a directed graph
    graph = nx.DiGraph()

    # Add nodes representing positions to the graph
    for i in range(positions_n + 1):
        graph.add_node(i)

    # Add edges between adjacent positions
    for i in range(positions_n):
        graph.add_edge(i, i + 1)
        graph.add_edge(i + 1, i)

    # Initialize sets to keep track of visited nodes and a queue for BFS
    visited = set()
    queue = [start]

    # BFS algorithm
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            yield node  # Yield the current node for BFS
            neighbors = list(graph.neighbors(node))
            # Add unvisited neighbors to the queue
            queue.extend(neighbor for neighbor in neighbors if neighbor not in visited)

# Heuristic Search

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0

    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)

# Define a heuristic function to estimate the cost to the goal
def heuristic(current_position, target_position):
    diff_x = abs(current_position - target_position)
    return diff_x

# A* Search
def astar(positions_n, start, goal):
    # Initialize an open set (priority queue) and a closed set to keep track of explored nodes
    open_set = []
    closed_set = set()

    # Create nodes for the start and goal positions
    start_node = Node(start)
    target_node = Node(goal)

    # Add the start node to the open set
    open_set.append(start_node)

    # A* Search algorithm
    while open_set:
        current_node = heapq.heappop(open_set)

        yield current_node.position  # Yield the current node's position for A* search

        closed_set.add(current_node.position)

        moves = [1, -1]

        for move in moves:
            new_position = current_node.position + move

            # Check if the new position is within bounds and not in the closed set
            if (
                new_position < 0 or new_position >= positions_n + 1 or
                new_position in closed_set
            ):
                continue

            # Create a new node for the new position
            new_node = Node(new_position, current_node)
            new_node.g = current_node.g + 1
            new_node.h = heuristic(new_position, target_node.position)

            exists_in_open_set = False

            # Check if a node with the same position is in the open set with a lower cost
            for node in open_set:
                if node.position == new_node.position and node.g > new_node.g:
                    exists_in_open_set = True
                    break

            # If the new node is not in the open set or has a lower cost, add it to the open set
            if not exists_in_open_set:
                heapq.heappush(open_set, new_node)
