import numpy as np
import matplotlib.pyplot as plt
import os

# Distance penalty curve constant (from template_matching_wrapper.py)
DISTANCE_PENALTY_CURVE = 0.00001

def get_distance_penalty(center1, center2):
    """
    Calculates an exponential penalty based on the Euclidean distance between two centers.
    Penalty is normalized in the range 0 to 1.
    """
    if center1 is None or center2 is None:
        return 0
    distance = np.linalg.norm(np.array(center1) - np.array(center2))
    penalty = (1 - np.exp(-DISTANCE_PENALTY_CURVE * (distance**2)))
    return penalty

def plot_penalty_contour(reference_point, grid_size=500, width=640, height=480):
    """
    Creates a contour plot showing the penalty function around a reference point.
    
    Args:
        reference_point: tuple (x, y) - the reference point
        grid_size: int - resolution of the grid
        width: int - width of the frame (x-axis)
        height: int - height of the frame (y-axis)
    """
    # Create a grid of points matching frame dimensions
    x_min, x_max = 0, width
    y_min, y_max = 0, height
    
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Calculate penalty for each point in the grid
    Z = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            point = (X[i, j], Y[i, j])
            Z[i, j] = get_distance_penalty(reference_point, point)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define levels for filled contours (more levels for smooth color transitions)
    filled_levels = np.linspace(Z.min(), Z.max(), 32)
    
    # Define levels for contour lines (subset of filled levels for clarity)
    # Choose every other level from filled_levels to get lines on color boundaries
    line_levels = filled_levels[::2]  # Every 2nd level
    
    # Contour plot with lines aligned to color boundaries
    contour_filled = ax1.contourf(X, Y, Z, levels=filled_levels, cmap='viridis', alpha=0.7)
    contour_lines = ax1.contour(X, Y, Z, levels=line_levels, colors='black', alpha=0.8, linewidths=1.5)
    ax1.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')  # Add labels to contour lines
    
    # Add special red contour for 0.12 penalty (if it's within the range)
    if Z.min() <= 0.12 <= Z.max():
        contour_012 = ax1.contour(X, Y, Z, levels=[0.12], colors='red', linewidths=3.0, linestyles='dashed')
        # ax1.clabel(contour_012, inline=True, fontsize=10, fmt='%.2f', colors='red')
    
    ax1.plot(reference_point[0], reference_point[1], 'r*', markersize=25, label='Reference Point', 
             markeredgecolor='white', markeredgewidth=2)
    ax1.set_xlabel('X coordinate', fontsize=12)
    ax1.set_ylabel('Y coordinate', fontsize=12)
    ax1.set_title(f'Distance Penalty Contour ($\\beta$ = {DISTANCE_PENALTY_CURVE})', fontsize=14)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(contour_filled, ax=ax1)
    cbar.set_label('Penalty Value', fontsize=12)
    
    # 3D surface plot
    from mpl_toolkits.mplot3d import Axes3D
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax2.scatter([reference_point[0]], [reference_point[1]], [0], color='red', s=100, marker='*', label='Reference Point')
    ax2.set_xlabel('X coordinate', fontsize=10)
    ax2.set_ylabel('Y coordinate', fontsize=10)
    ax2.set_zlabel('Penalty Value', fontsize=10)
    ax2.set_title('Distance Penalty 3D Surface', fontsize=14)
    ax2.view_init(elev=30, azim=45)
    
    # Add colorbar for 3D plot
    cbar2 = plt.colorbar(surf, ax=ax2, shrink=0.5)
    cbar2.set_label('Penalty', fontsize=10)
    
    plt.tight_layout()
    
    # Save in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'distance_penalty_visualization.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print some statistics
    print(f"\nPenalty Statistics:")
    print(f"Reference Point: {reference_point}")
    print(f"Distance Penalty Curve (β): {DISTANCE_PENALTY_CURVE}")
    print(f"Penalty range in plot: {Z.min():.4f} to {Z.max():.4f}")
    print(f"Penalty at reference point: {get_distance_penalty(reference_point, reference_point):.4f}")
    print(f"Penalty at 100 pixels away: {get_distance_penalty(reference_point, (reference_point[0] + 100, reference_point[1])):.4f}")
    print(f"Penalty at 200 pixels away: {get_distance_penalty(reference_point, (reference_point[0] + 200, reference_point[1])):.4f}")
    print(f"Penalty at 300 pixels away: {get_distance_penalty(reference_point, (reference_point[0] + 300, reference_point[1])):.4f}")
    print(f"\nRed contour at 0.12 penalty:")
    # Calculate distance for 0.12 penalty: 0.12 = 1 - exp(-β*d)
    # exp(-β*d) = 0.88
    # -β*d = ln(0.88)
    # d = -ln(0.88)/β
    distance_012 = -np.log(1 - 0.12) / DISTANCE_PENALTY_CURVE
    print(f"Distance for 0.12 penalty: {distance_012:.1f} pixels")

def plot_penalty_vs_distance():
    """
    Creates a simple 1D plot showing penalty as a function of distance.
    """
    distances = np.linspace(0, 450, 1000)
    penalties = [1 - np.exp(-DISTANCE_PENALTY_CURVE * (d**2)) for d in distances]
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances, penalties, 'b-', linewidth=2)
    plt.xlabel('Distance (pixels)', fontsize=12)
    plt.ylabel('Penalty Value', fontsize=12)
    plt.title(f'Distance Penalty Function ($\\beta$ = {DISTANCE_PENALTY_CURVE})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'distance_penalty_1d.png'), dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Frame dimensions from main.py
    WIDTH = 640
    HEIGHT = 480
    
    # Example usage: Set a reference point and plot the penalty contour
    reference_point = (320, 240)  # Center of the frame
    
    print("Creating distance penalty visualization...")
    print(f"Frame size: {WIDTH}x{HEIGHT}")
    print(f"Using reference point: {reference_point}")
    
    # Create 1D plot
    plot_penalty_vs_distance()
    
    # Create 2D contour plot with frame dimensions
    plot_penalty_contour(reference_point, grid_size=500, width=WIDTH, height=HEIGHT)
    
    print("\nVisualization saved!")
