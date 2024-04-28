import heapq  # For priority queue

# Define a Node class to represent each cell in the grid
class Node:
    def __init__(self, x, y, g=0, h=0, parent=None):
        self.x = x
        self.y = y
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic cost from current node to goal
        self.f = g + h  # Total cost (f = g + h)
        self.parent = parent  # To track the path
    
    def __lt__(self, other):
        return self.f < other.f  # For heapq to compare nodes by 'f' value

# Updated heuristic function to handle tuples
def manhattan_heuristic(start, goal):
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

# A* algorithm with the updated heuristic function
def astar(grid, start, goal):
    # Check grid boundaries
    rows = len(grid)
    cols = len(grid[0])

    # Create a priority queue for the open list and a set for the closed list
    open_list = []
    closed_list = set()

    # Add the start node to the open list
    start_node = Node(start[0], start[1], 0, manhattan_heuristic(start, goal))
    heapq.heappush(open_list, start_node)

    # Directions for moving in 4 possible ways (up, down, left, right)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while open_list:
        # Get the node with the lowest f-value from the open list
        current_node = heapq.heappop(open_list)

        # If the goal is reached, reconstruct the path
        if (current_node.x, current_node.y) == goal:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]  # Return the path from start to goal

        # Add the current node to the closed list
        closed_list.add((current_node.x, current_node.y))

        # Explore neighbors
        for direction in directions:
            nx = current_node.x + direction[0]
            ny = current_node.y + direction[1]

            # Check if the neighbor is within the grid and not an obstacle
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                # Create a neighbor node
                g = current_node.g + 1
                h = manhattan_heuristic((nx, ny), goal)
                neighbor = Node(nx, ny, g, h, current_node)

                if (nx, ny) in closed_list:
                    continue

                # Check if the neighbor is already in the open list with a lower f-value
                skip = False
                for open_node in open_list:
                    if (open_node.x == neighbor.x and open_node.y == neighbor.y and open_node.f <= neighbor.f):
                        skip = True
                        break

                if not skip:
                    heapq.heappush(open_list, neighbor)

    return None  # Return None if no path is found

# Example grid (0 = empty cell, 1 = obstacle)
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]

# Start and goal coordinates
start = (0, 0)
goal = (3, 4)

# Find the path using A* algorithm
path = astar(grid, start, goal)

print("Shortest path:", path)
