import matplotlib.pyplot as plt
import heapq
import numpy as np
import time
import pandas as pd

# Define grid size
GRID_SIZE = (20, 30)

# Added helper function for complicated obstacles
def generate_obstacles():
    obstacles = []
    # horizontal obstacle
    obstacles += [(10, i) for i in range(5, 25)]
    # vertical obstacle with a gap at row 10
    obstacles += [(i, 15) for i in range(2, 18) if i != 10]
    # rectangular obstacle in the top right region
    obstacles += [(r, c) for r in range(3, 6) for c in range(20, 25)]
    return obstacles

OBSTACLES = generate_obstacles()

# Define start and goal
START = (4, 6)
GOAL = (12, 20)

# Heuristic: Manhattan distance
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Get valid neighbor cells (4-connected grid)
# outside of map or obstacle cells are not considered as neighbors
def get_neighbors(pos, grid):
    neighbors = []
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] != 1:
            neighbors.append((nx, ny))
    return neighbors

# Reconstruct path from came_from dictionary
def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            return []  # no path found
    path.append(start)
    path.reverse()
    return path

# Dijkstra's Algorithm modified to track visited cells
def dijkstra(grid, start, goal):
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    visited = set()
    while frontier:
        _, current = heapq.heappop(frontier)
        visited.add(current)
        if current == goal:
            break
        for neighbor in get_neighbors(current, grid):
            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(frontier, (new_cost, neighbor))
                came_from[neighbor] = current
    path = reconstruct_path(came_from, start, goal)
    return path, visited

# A* Algorithm modified to track visited cells
def astar(grid, start, goal):
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    visited = set()
    while frontier:
        _, current = heapq.heappop(frontier)
        visited.add(current)
        if current == goal:
            break
        for neighbor in get_neighbors(current, grid):
            new_cost = cost_so_far[current] + 1  # uniform cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current
    path = reconstruct_path(came_from, start, goal)
    return path, visited

# # D* Lite (simulation) – not modified here for visited cells
# def dstarlite(grid, start, goal):
#     dstar = DStarLite(grid.copy(), start, goal, heuristic)
#     dstar.compute_shortest_path()
#     path = dstar.extract_path()
#     visited = dstar.visited
#     return path, visited


# Visualization
def visualize(grid, path_dict, visited_dict):
    fig, axes = plt.subplots(1, len(path_dict), figsize=(18, 6))
    for ax, label in zip(axes, path_dict.keys()):
        display = np.copy(grid).astype(float)
        # Mark obstacles (value 1), free space (0)
        # Overlay visited cells as 0.5 and final path as 0.8
        visited = visited_dict.get(label, set())
        for (x, y) in visited:
            display[x, y] = 0.5
        path = path_dict[label]
        for (x, y) in path:
            display[x, y] = 0.8
        # Mark start and goal explicitly 
        display[START] = 0.7
        display[GOAL] = 0.9
        ax.imshow(display, cmap='viridis', origin='lower', alpha=0.8)
        ax.set_title(label)
    plt.show()

def visualize_map(grid):
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, cmap='viridis', origin='lower', alpha=0.8) # changed cmap to viridis for colorful visualization                
    markers = ['go', 'bo']
    labels = ['Start', 'Goal']
    plt.title("Map Visualization")
    plt.axis('off')
    plt.show()

# Main execution
if __name__ == '__main__':
    grid = np.zeros(GRID_SIZE)
    for x, y in OBSTACLES:
        grid[x, y] = 1

    start_time = time.time()
    path_dijkstra, visited_dijkstra = dijkstra(grid, START, GOAL)
    dijkstra_time = time.time() - start_time

    start_time = time.time()
    path_astar, visited_astar = astar(grid, START, GOAL)
    astar_time = time.time() - start_time

    # start_time = time.time()
    # path_dstarlite, visited_dstarlite = dstarlite(grid, START, GOAL)
    # dstarlite_time = time.time() - start_time


    df = pd.DataFrame({
        "Algorithm": ["Dijkstra", "A*", "D* Lite"],
        "Path Length": [len(path_dijkstra), len(path_astar)],
        "Computation Time (s)": [dijkstra_time, astar_time]
    })
    print(df)

    # Prepare dictionaries for visualization
    paths = {
        "Dijkstra": path_dijkstra,
        "A*": path_astar,
    }
    visited_sets = {
        "Dijkstra": visited_dijkstra,
        "A*": visited_astar,  # currently empty sets
    }
    
    visualize(grid, paths, visited_sets)
    # visualize_map(grid)