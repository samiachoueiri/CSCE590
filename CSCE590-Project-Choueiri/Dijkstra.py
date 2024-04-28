import heapq  # For priority queue
import sys  # For infinity representation

# Define a class to represent a weighted graph
class WeightedGraph:
    def __init__(self):
        self.edges = {}  # Dictionary of edges and their weights
    
    def add_edge(self, from_node, to_node, cost):
        # Add an edge with a cost from one node to another
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append((to_node, cost))
        
        # Add nodes to ensure all are recognized
        if to_node not in self.edges:
            self.edges[to_node] = []
    
    def get_neighbors(self, node):
        # Get the neighbors of a given node
        return self.edges.get(node, [])

# Dijkstra's algorithm to find the shortest path with minimal cost
def dijkstra(graph, start, goal):
    # Create a priority queue for the open list and a dictionary for the shortest path costs
    open_list = []
    shortest_path = {}  # Dictionary of shortest paths
    # Initialize all nodes with infinite cost
    shortest_path_cost = {node: sys.maxsize for node in graph.edges.keys()}  
    shortest_path_cost[start] = 0  # Cost from start to itself is zero
    parents = {}  # To reconstruct the path

    # Add the start node to the open list
    heapq.heappush(open_list, (0, start))  # (cost, node)

    while open_list:
        # Get the node with the lowest cost from the open list
        current_cost, current_node = heapq.heappop(open_list)

        # If the current node is the goal, reconstruct the path
        if current_node == goal:
            path = []
            step = current_node
            while step:
                path.append(step)
                step = parents.get(step)
            return path[::-1], shortest_path_cost[goal]  # Return the path and the total cost

        # Explore neighbors
        for neighbor, edge_cost in graph.get_neighbors(current_node):
            # Ensure that the neighbor is initialized in the shortest_path_cost
            if neighbor not in shortest_path_cost:
                shortest_path_cost[neighbor] = sys.maxsize  # Set to infinity if not yet initialized

            # Calculate the new cost to reach the neighbor
            new_cost = current_cost + edge_cost
            if new_cost < shortest_path_cost[neighbor]:  # If a shorter path is found
                shortest_path_cost[neighbor] = new_cost
                parents[neighbor] = current_node  # Record the parent node
                heapq.heappush(open_list, (new_cost, neighbor))

    return None, None  # Return None if no path is found

# Create a weighted graph
graph = WeightedGraph()
graph.add_edge("A", "B", 1)
graph.add_edge("A", "C", 4)
graph.add_edge("B", "C", 2)
graph.add_edge("B", "D", 5)
graph.add_edge("C", "D", 1)
graph.add_edge("D", "E", 3)

# Find the shortest path with minimal cost using Dijkstra's algorithm
start = "A"
goal = "E"
path, total_cost = dijkstra(graph, start, goal)

print("Shortest path:", path)
print("Total cost:", total_cost)
