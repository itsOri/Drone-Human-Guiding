import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import cv2
import os
import sys

# Add parent directory to path to import RRTStar_New
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.RRTStar_New import draw_path_on_image, Node, calculate_distance, calculate_new_point, is_collision_free
import random
import time

def find_colored_points(img):
    """
    Find red and green points in the image
    Returns: (red_point, green_point) as (x, y) tuples
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Red color range (red wraps around in HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 | mask_red2
    
    # Green color range
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find centers of red and green regions
    red_coords = np.where(mask_red > 0)
    green_coords = np.where(mask_green > 0)
    
    red_point = None
    green_point = None
    
    if len(red_coords[0]) > 0:
        red_y = int(np.mean(red_coords[0]))
        red_x = int(np.mean(red_coords[1]))
        red_point = (red_x, red_y)
    
    if len(green_coords[0]) > 0:
        green_y = int(np.mean(green_coords[0]))
        green_x = int(np.mean(green_coords[1]))
        green_point = (green_x, green_y)
    
    return red_point, green_point

def create_binary_image(img, red_point, green_point):
    """
    Convert image to binary (white=free space, black=obstacles)
    Make sure colored dots and their surroundings are in free space
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Make sure start and goal points are in free space by drawing white circles
    # Increase clearance for dense obstacle maps
    if red_point is not None:
        cv2.circle(binary, red_point, 30, 255, -1)
    if green_point is not None:
        cv2.circle(binary, green_point, 30, 255, -1)
    
    return binary

def capture_snapshot(nodes, goal, iteration, elapsed_time):
    """Helper function to capture a snapshot of the current best path"""
    # Check if goal is in the tree
    if goal in nodes:
        # Goal reached, use it
        nearest_to_goal = goal
        status = "Goal reached"
    else:
        # Find closest node to goal
        nearest_to_goal = min(nodes, key=lambda n: calculate_distance(n, goal))
        status = f"Closest to goal (dist: {calculate_distance(nearest_to_goal, goal):.1f})"
    
    path = []
    current_node = nearest_to_goal
    while current_node:
        path.append((current_node.x, current_node.y))
        current_node = current_node.parent
    
    path_reversed = path[::-1]  # Reverse to get start to goal
    print(f"  Snapshot at iteration {iteration}: {len(path)} points, cost: {nearest_to_goal.cost:.2f}, {status}, time: {elapsed_time:.2f}s")
    return path_reversed, elapsed_time

def rrt_star_progressive(start, goal, img, max_iter, snapshot_iters, delta=15, radius=30):
    """
    RRT* that captures snapshots at specific iterations
    
    Args:
        start: Starting node
        goal: Goal node
        img: Binary image (255=free space, 0=obstacle)
        max_iter: Maximum iterations (should be max of snapshot_iters)
        snapshot_iters: List of iteration counts to capture snapshots
        delta: Step size for extending tree
        radius: Radius for rewiring nodes
    
    Returns:
        Tuple of (snapshots dictionary with paths, snapshot_times dictionary)
    """
    start_time = time.time()
    
    img_height, img_width = img.shape
    nodes = [start]
    goal_threshold = delta * 2
    
    snapshots = {}
    snapshot_times = {}
    snapshot_iters_list = sorted(list(snapshot_iters))
    print(f"Will capture snapshots at: {snapshot_iters_list}")
    
    nodes_added = 0  # Track successful node additions
    
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
            # Still check for snapshot
            current_iter = iteration + 1
            if current_iter in snapshot_iters and current_iter not in snapshots:
                elapsed = time.time() - start_time
                path, snap_time = capture_snapshot(nodes, goal, current_iter, elapsed)
                snapshots[current_iter] = path
                snapshot_times[current_iter] = snap_time
            continue
        
        # Skip if path to new point is not collision-free
        if not is_collision_free(nearest_node, new_point, img):
            # Still check for snapshot
            current_iter = iteration + 1
            if current_iter in snapshot_iters and current_iter not in snapshots:
                elapsed = time.time() - start_time
                path, snap_time = capture_snapshot(nodes, goal, current_iter, elapsed)
                snapshots[current_iter] = path
                snapshot_times[current_iter] = snap_time
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
        nodes_added += 1
        
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
            if goal not in nodes:
                goal.parent = new_point
                goal.cost = new_point.cost + calculate_distance(new_point, goal)
                nodes.append(goal)
        
        # Capture snapshot at specified iterations
        current_iter = iteration + 1
        if current_iter in snapshot_iters and current_iter not in snapshots:
            elapsed = time.time() - start_time
            path, snap_time = capture_snapshot(nodes, goal, current_iter, elapsed)
            snapshots[current_iter] = path
            snapshot_times[current_iter] = snap_time
    
    # Ensure all requested snapshots were captured
    for snap_iter in snapshot_iters:
        if snap_iter not in snapshots:
            print(f"  Warning: Snapshot at iteration {snap_iter} was not captured")
    
    total_time = time.time() - start_time
    
    print(f"Completed all {max_iter} iterations. Total nodes: {len(nodes)} (added: {nodes_added})")
    print(f"Snapshots captured at: {sorted(snapshots.keys())}")
    print(f"Total execution time: {total_time:.2f} seconds")
    return snapshots, snapshot_times

def run_rrt_with_iterations(img_path, iterations_list):
    """
    Run RRT* with different iteration values and display results
    
    Args:
        img_path: Path to the input image
        iterations_list: List of iteration values to test
    """
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return
    
    print(f"Image loaded: {img.shape}")
    
    # Find colored points
    red_point, green_point = find_colored_points(img)
    
    if red_point is None or green_point is None:
        print("Error: Could not find red or green points in the image")
        print(f"Red point: {red_point}, Green point: {green_point}")
        return
    
    print(f"Start (red): {red_point}")
    print(f"Goal (green): {green_point}")
    
    # Create binary image for path planning
    binary_img = create_binary_image(img, red_point, green_point)
    
    # Check if start and goal are in free space
    print(f"\nBinary image stats:")
    print(f"  - Free space pixels (255): {np.sum(binary_img == 255)}")
    print(f"  - Obstacle pixels (0): {np.sum(binary_img == 0)}")
    print(f"  - Start point value: {binary_img[red_point[1], red_point[0]]}")
    print(f"  - Goal point value: {binary_img[green_point[1], green_point[0]]}")
    
    # Create nodes
    start = Node(red_point[0], red_point[1])
    goal = Node(green_point[0], green_point[1])
    
    # Run RRT* once and capture snapshots at different iterations
    print(f"\n--- Running RRT* progressively up to {max(iterations_list)} iterations ---")
    # Increase delta and radius for better exploration in dense obstacle maps
    snapshots, snapshot_times = rrt_star_progressive(start, goal, binary_img, max(iterations_list), 
                                                     set(iterations_list), delta=40, radius=80)
    
    # Create figure with subplots - adjust grid based on number of iterations
    num_iterations = len(iterations_list)
    if num_iterations <= 4:
        rows, cols = 2, 2
        figsize = (16, 12)
    elif num_iterations <= 6:
        rows, cols = 2, 3
        figsize = (24, 12)
    else:
        rows = (num_iterations + 2) // 3
        cols = 3
        figsize = (24, 8 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle('RRT* Path Planning - Progressive Improvement', 
                 fontsize=24, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Display snapshots for each iteration value
    for idx, iter_count in enumerate(iterations_list):
        print(f"\n--- Displaying snapshot at {iter_count} iterations ---")
        
        path = snapshots.get(iter_count, None)
        
        # Create visualization image
        vis_img = img.copy()
        
        if path and len(path) > 1:
            print(f"Path has {len(path)} points")
            # Draw path in purple
            vis_img = draw_path_on_image(vis_img, path, path_color=(255, 0, 255), thickness=3)
        else:
            print("No path available")
        
        # Draw start and goal points (make them visible)
        cv2.circle(vis_img, red_point, 8, (0, 0, 255), -1)  # Red start
        cv2.circle(vis_img, green_point, 8, (0, 255, 0), -1)  # Green goal
        
        # Convert BGR to RGB for matplotlib
        vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        
        # Display in subplot with timing info
        time_str = f" ({snapshot_times.get(iter_count, 0):.2f}s)" if iter_count in snapshot_times else ""
        axes[idx].imshow(vis_img_rgb)
        axes[idx].set_title(f'Iterations: {iter_count}{time_str}', fontsize=22, fontweight='bold')
        axes[idx].axis('off')
        
        # Add legend
        if path and len(path) > 1:
            axes[idx].text(10, 50, f'Path length: {len(path)} points', 
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          fontsize=16)
    
    # Hide any unused subplots
    for idx in range(len(iterations_list), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save the result
    output_path = os.path.join(os.path.dirname(img_path), 'rrt_comparison_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Results saved to: {output_path}")
    plt.close()

def compare_delta_values(img_path, delta_values, fixed_iterations=1500):
    """
    Compare RRT* performance with different delta (step size) values
    
    Args:
        img_path: Path to the input image
        delta_values: List of delta values to test
        fixed_iterations: Fixed number of iterations for all runs
    """
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return
    
    print(f"Image loaded: {img.shape}")
    
    # Find colored points
    red_point, green_point = find_colored_points(img)
    
    if red_point is None or green_point is None:
        print("Error: Could not find red or green points in the image")
        print(f"Red point: {red_point}, Green point: {green_point}")
        return
    
    print(f"Start (red): {red_point}")
    print(f"Goal (green): {green_point}")
    
    # Create binary image for path planning
    binary_img = create_binary_image(img, red_point, green_point)
    
    print(f"\nBinary image stats:")
    print(f"  - Free space pixels (255): {np.sum(binary_img == 255)}")
    print(f"  - Obstacle pixels (0): {np.sum(binary_img == 0)}")
    
    # Create nodes
    start = Node(red_point[0], red_point[1])
    goal = Node(green_point[0], green_point[1])
    
    # Store results for each delta
    results = {}
    
    # Create figure with subplots - adjust grid based on number of deltas
    num_deltas = len(delta_values)
    if num_deltas <= 4:
        rows, cols = 2, 2
        figsize = (20, 16)
    elif num_deltas <= 6:
        rows, cols = 2, 3
        figsize = (24, 12)
    else:
        rows = (num_deltas + 2) // 3
        cols = 3
        figsize = (24, 8 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(f'RRT* Path Planning - Delta Comparison ({fixed_iterations} iterations)', 
                 fontsize=24, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Run RRT* for each delta value
    for idx, delta in enumerate(delta_values):
        print(f"\n--- Running RRT* with delta={delta}, iterations={fixed_iterations} ---")
        
        # Set random seed for consistency
        random.seed(42)
        
        # Run RRT* with this delta value
        snapshots, snapshot_times = rrt_star_progressive(
            start, goal, binary_img, 
            fixed_iterations, 
            {fixed_iterations},  # Only one snapshot at the end
            delta=delta, 
            radius=delta * 2  # Scale radius with delta
        )
        
        # Get the path and time
        path = snapshots.get(fixed_iterations, None)
        exec_time = snapshot_times.get(fixed_iterations, 0)
        
        results[delta] = {
            'path': path,
            'time': exec_time,
            'points': len(path) if path else 0
        }
        
        # Create visualization image
        vis_img = img.copy()
        
        if path and len(path) > 1:
            print(f"Path found with {len(path)} points in {exec_time:.2f}s")
            # Draw path in purple
            vis_img = draw_path_on_image(vis_img, path, path_color=(255, 0, 255), thickness=3)
        else:
            print("No path found")
        
        # Draw start and goal points
        cv2.circle(vis_img, red_point, 8, (0, 0, 255), -1)  # Red start
        cv2.circle(vis_img, green_point, 8, (0, 255, 0), -1)  # Green goal
        
        # Convert BGR to RGB for matplotlib
        vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        
        # Display in subplot
        axes[idx].imshow(vis_img_rgb)
        axes[idx].set_title(f'Delta: {delta} ({exec_time:.2f}s)', fontsize=22, fontweight='bold')
        axes[idx].axis('off')
        
        # Add legend
        if path and len(path) > 1:
            axes[idx].text(10, 50, f'Path length: {len(path)} points', 
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          fontsize=16)
    
    # Hide any unused subplots
    for idx in range(len(delta_values), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save the result
    output_path = os.path.join(os.path.dirname(img_path), 'rrt_delta_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Results saved to: {output_path}")
    plt.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - Delta Comparison")
    print("=" * 60)
    for delta in delta_values:
        res = results[delta]
        print(f"Delta {delta:3d}: {res['points']:3d} points, {res['time']:6.2f}s")

def compare_radius_values(img_path, radius_values, fixed_iterations=1500, fixed_delta=40):
    """
    Compare RRT* performance with different radius (rewiring neighborhood) values
    
    Args:
        img_path: Path to the input image
        radius_values: List of radius values to test
        fixed_iterations: Fixed number of iterations for all runs
        fixed_delta: Fixed delta (step size) for all runs
    """
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return
    
    print(f"Image loaded: {img.shape}")
    
    # Find colored points
    red_point, green_point = find_colored_points(img)
    
    if red_point is None or green_point is None:
        print("Error: Could not find red or green points in the image")
        print(f"Red point: {red_point}, Green point: {green_point}")
        return
    
    print(f"Start (red): {red_point}")
    print(f"Goal (green): {green_point}")
    
    # Create binary image for path planning
    binary_img = create_binary_image(img, red_point, green_point)
    
    print(f"\nBinary image stats:")
    print(f"  - Free space pixels (255): {np.sum(binary_img == 255)}")
    print(f"  - Obstacle pixels (0): {np.sum(binary_img == 0)}")
    
    # Create nodes
    start = Node(red_point[0], red_point[1])
    goal = Node(green_point[0], green_point[1])
    
    # Store results for each radius
    results = {}
    
    # Create figure with subplots - adjust grid based on number of radius values
    num_radius = len(radius_values)
    if num_radius <= 4:
        rows, cols = 2, 2
        figsize = (20, 16)
    elif num_radius <= 6:
        rows, cols = 2, 3
        figsize = (24, 12)
    else:
        rows = (num_radius + 2) // 3
        cols = 3
        figsize = (24, 8 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(f'RRT* Path Planning - Radius Comparison ({fixed_iterations} iterations, delta={fixed_delta})', 
                 fontsize=24, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Run RRT* for each radius value
    for idx, radius in enumerate(radius_values):
        print(f"\n--- Running RRT* with radius={radius}, delta={fixed_delta}, iterations={fixed_iterations} ---")
        
        # Set random seed for consistency
        random.seed(42)
        
        # Run RRT* with this radius value
        snapshots, snapshot_times = rrt_star_progressive(
            start, goal, binary_img, 
            fixed_iterations, 
            {fixed_iterations},  # Only one snapshot at the end
            delta=fixed_delta, 
            radius=radius
        )
        
        # Get the path and time
        path = snapshots.get(fixed_iterations, None)
        exec_time = snapshot_times.get(fixed_iterations, 0)
        
        results[radius] = {
            'path': path,
            'time': exec_time,
            'points': len(path) if path else 0
        }
        
        # Create visualization image
        vis_img = img.copy()
        
        if path and len(path) > 1:
            print(f"Path found with {len(path)} points in {exec_time:.2f}s")
            # Draw path in purple
            vis_img = draw_path_on_image(vis_img, path, path_color=(255, 0, 255), thickness=3)
        else:
            print("No path found")
        
        # Draw start and goal points
        cv2.circle(vis_img, red_point, 8, (0, 0, 255), -1)  # Red start
        cv2.circle(vis_img, green_point, 8, (0, 255, 0), -1)  # Green goal
        
        # Convert BGR to RGB for matplotlib
        vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        
        # Display in subplot
        axes[idx].imshow(vis_img_rgb)
        axes[idx].set_title(f'Radius: {radius} ({exec_time:.2f}s)', fontsize=22, fontweight='bold')
        axes[idx].axis('off')
        
        # Add legend
        if path and len(path) > 1:
            axes[idx].text(10, 50, f'Path length: {len(path)} points', 
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          fontsize=16)
    
    # Hide any unused subplots
    for idx in range(len(radius_values), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save the result
    output_path = os.path.join(os.path.dirname(img_path), 'rrt_radius_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Results saved to: {output_path}")
    plt.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - Radius Comparison")
    print("=" * 60)
    for radius in radius_values:
        res = results[radius]
        print(f"Radius {radius:3d}: {res['points']:3d} points, {res['time']:6.2f}s")

def main():
    # Set random seed for consistent results
    random.seed(42)
    
    # Path to the test image
    img_path = "/Users/nathan/Documents/Semester6/Project_drone/Drone-Human-Guiding/tests/Screenshot 2025-12-18 at 15.30.54.png"
    
    # Choose which comparison to run
    print("=" * 60)
    print("RRT* Path Planning Comparison")
    print("=" * 60)
    
    # Option 1: Compare different delta values (step sizes)
    # delta_values = [5, 15, 30, 60, 100, 150]
    # compare_delta_values(img_path, delta_values, fixed_iterations=1500)
    
    # Option 2: Compare different radius values (rewiring neighborhood)
    radius_values = [10, 30, 60, 100, 150, 200]
    compare_radius_values(img_path, radius_values, fixed_iterations=1500, fixed_delta=40)
    
    # Option 3: Compare different iteration counts (commented out)
    # iterations_list = [50, 100, 500, 1000, 1500, 10000]
    # run_rrt_with_iterations(img_path, iterations_list)

if __name__ == "__main__":
    main()
