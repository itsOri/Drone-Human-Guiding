import math
import random
import cv2
import numpy as np
import logging

# Get logger from main module
logger = logging.getLogger(__name__)


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def calculate_new_point(nearest_node, random_point, delta, img_width, img_height):
    """Calculate a new point in the direction of random_point from nearest_node"""
    new_point = Node(nearest_node.x, nearest_node.y)
    
    angle = math.atan2(random_point.y - nearest_node.y, random_point.x - nearest_node.x)
    
    new_point.x += int(delta * math.cos(angle))
    new_point.y += int(delta * math.sin(angle))
    
    # Ensure the new point is within the image bounds
    new_point.x = max(0, min(img_width - 1, new_point.x))
    new_point.y = max(0, min(img_height - 1, new_point.y))
    
    return new_point


def is_collision_free(node1, node2, img):
    """Check if the path between two nodes is collision-free"""
    x1, y1 = node1.x, node1.y
    x2, y2 = node2.x, node2.y
    
    # Use Bresenham's line algorithm to check all points along the line
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        # Check if current point is in obstacle (black pixel = 0)
        if img[y1, x1] == 0:
            return False
        
        if x1 == x2 and y1 == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    
    return True


def rrt_star(start, goal, img, max_iter=2000, delta=15, radius=30):
    """
    RRT* path planning algorithm
    
    Args:
        start: Starting node
        goal: Goal node
        img: Binary image (255=free space, 0=obstacle)
        max_iter: Maximum iterations
        delta: Step size for extending tree
        radius: Radius for rewiring nodes
    
    Returns:
        List of (x, y) coordinates representing the path
    """
    img_height, img_width = img.shape
    nodes = [start]
    
    # Early termination if goal is reached
    goal_threshold = delta * 2
    
    for iteration in range(max_iter):
        # Bias towards goal 10% of the time
        if random.random() < 0.1:
            random_point = Node(goal.x, goal.y)
        else:
            random_point = Node(random.randint(0, img_width - 1), random.randint(0, img_height - 1))
        
        # Find nearest node
        nearest_node = min(nodes, key=lambda n: calculate_distance(n, random_point))
        
        # Calculate new point
        new_point = calculate_new_point(nearest_node, random_point, delta, img_width, img_height)
        
        # Skip if new point is in obstacle
        if img[new_point.y, new_point.x] == 0:
            continue
        
        # Skip if path to new point is not collision-free
        if not is_collision_free(nearest_node, new_point, img):
            continue
        
        # Find nearby nodes for optimal parent selection
        near_nodes = [node for node in nodes if calculate_distance(node, new_point) <= radius]
        
        # Choose best parent (minimum cost)
        min_cost_node = nearest_node
        min_cost = nearest_node.cost + calculate_distance(nearest_node, new_point)
        
        for node in near_nodes:
            if is_collision_free(node, new_point, img):
                cost = node.cost + calculate_distance(node, new_point)
                if cost < min_cost:
                    min_cost_node = node
                    min_cost = cost
        
        new_point.parent = min_cost_node
        new_point.cost = min_cost
        nodes.append(new_point)
        
        # Rewire nearby nodes
        for node in near_nodes:
            if node == new_point:
                continue
            if is_collision_free(new_point, node, img):
                cost = new_point.cost + calculate_distance(node, new_point)
                if cost < node.cost:
                    node.parent = new_point
                    node.cost = cost
        
        # Check if we've reached the goal
        if calculate_distance(new_point, goal) < goal_threshold:
            goal.parent = new_point
            goal.cost = new_point.cost + calculate_distance(new_point, goal)
            nodes.append(goal)
            break
    
    # Find path by backtracking from nearest node to goal
    nearest_to_goal = min(nodes, key=lambda n: calculate_distance(n, goal))
    
    path = []
    current_node = nearest_to_goal
    while current_node:
        path.append((current_node.x, current_node.y))
        current_node = current_node.parent
    
    return path[::-1]  # Reverse to get path from start to goal


def draw_path_on_image(image, path, path_color=(0, 255, 255), thickness=2):
    """
    Draw the RRT* path on an image
    
    Args:
        image: BGR image to draw on
        path: List of (x, y) coordinates
        path_color: Color for the path (BGR format, default cyan)
        thickness: Line thickness
    
    Returns:
        Image with path drawn
    """
    if len(path) < 2:
        return image
    
    # Draw path
    for i in range(len(path) - 1):
        pt1 = (int(path[i][0]), int(path[i][1]))
        pt2 = (int(path[i + 1][0]), int(path[i + 1][1]))
        cv2.line(image, pt1, pt2, path_color, thickness)
    
    return image


def find_rrt_path(img, start_coords, goal_coords, max_iter=2000, delta=15, radius=30):
    """
    Find RRT* path from start to goal
    
    Args:
        img: Binary image (grayscale or BGR, will be converted to binary)
        start_coords: Tuple of (x, y) for start position
        goal_coords: Tuple of (x, y) for goal position
        max_iter: Maximum RRT* iterations
        delta: Step size
        radius: Rewiring radius
    
    Returns:
        List of (x, y) coordinates representing the path, or None if no path found
    """
    if start_coords is None:
        logger.warning("[RRT*] Start point is None")
        return None
    
    if goal_coords is None:
        logger.warning("[RRT*] Goal point is None")
        return None
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
    
    # Ensure binary image (255 = free, 0 = obstacle)
    _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    
    start_x, start_y = start_coords
    goal_x, goal_y = goal_coords
    
    # Create nodes
    start = Node(start_x, start_y)
    goal = Node(goal_x, goal_y)
    
    # Run RRT*
    path = rrt_star(start, goal, img_binary, max_iter=max_iter, delta=delta, radius=radius)
    
    if len(path) < 2:
        logger.debug(f"[RRT*] Path too short or not found (length: {len(path)})")
        return None
    
    logger.debug(f"[RRT*] Path found with {len(path)} points")
    return path


# Example usage for testing
if __name__ == "__main__":
    # Create a test image with obstacles
    test_img = np.ones((480, 640), dtype=np.uint8) * 255
    
    # Add some obstacles
    cv2.rectangle(test_img, (200, 100), (250, 400), 0, -1)
    cv2.rectangle(test_img, (400, 50), (450, 350), 0, -1)
    
    # Define start and goal
    start = (50, 240)
    goal = (590, 240)
    
    # Find path
    path = find_rrt_path(test_img, start, goal)
    
    if path:
        # Draw on color image
        img_color = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
        img_with_path = draw_path_on_image(img_color, path)
        
        # Draw start and goal
        cv2.circle(img_with_path, start, 5, (0, 0, 255), -1)  # Red
        cv2.circle(img_with_path, goal, 5, (0, 255, 0), -1)   # Green
        
        cv2.imwrite('rrt_test_result.png', img_with_path)
        logger.info("Test completed. Check rrt_test_result.png")
