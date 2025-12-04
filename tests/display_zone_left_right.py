import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_diagram():
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set the background color to white
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # --- Configuration ---
    # We will use a coordinate system that matches the labels generally
    # Inverting Y later to match the "Y down" look
    
    # Dimensions
    rect_width = 80
    rect_height = 250
    
    # Left Blue Rectangle Position
    left_rect_x = 90
    left_rect_y = 100 
    
    # Right Red Rectangle Position
    right_rect_x = 410
    right_rect_y = 100
    
    # Central Rounded Rectangle Position
    center_rect_x = 180
    center_rect_y = 220
    center_rect_width = 220
    center_rect_height = 120
    
    # Axis positions
    axis_origin_x = 100
    axis_origin_y = 380 # Where the axes cross
    
    # --- 1. Draw The Rectangles ---
    
    # Left Rectangle (Blue)
    rect_left = patches.Rectangle(
        (left_rect_x, left_rect_y), rect_width, rect_height,
        linewidth=3, edgecolor='#0ea5e9', facecolor='none'
    )
    ax.add_patch(rect_left)
    
    # Right Rectangle (Red) - Same size and y-position as requested
    rect_right = patches.Rectangle(
        (right_rect_x, right_rect_y), rect_width, rect_height,
        linewidth=3, edgecolor='#ef4444', facecolor='none'
    )
    ax.add_patch(rect_right)
    
    # Central Rounded Rectangle (Black)
    # BoxStyle 'Round' with pad=0.2 gives rounded corners
    fancy_box = patches.FancyBboxPatch(
        (center_rect_x, center_rect_y), center_rect_width, center_rect_height,
        boxstyle="round,pad=10,rounding_size=20",
        linewidth=4, edgecolor='black', facecolor='none'
    )
    ax.add_patch(fancy_box)

    # --- 2. Add Labels to Rectangles ---
    
    # "left" text - aligned at top inside blue box
    ax.text(left_rect_x + 20, left_rect_y + 30, 'left', 
            fontsize=16, color='#0ea5e9', fontweight='bold')

    # "right" text - aligned at top inside red box (same height)
    ax.text(right_rect_x + 20, right_rect_y + 30, 'right', 
            fontsize=16, color='#ef4444', fontweight='bold')
    
    # --- 2b. Add dotted lines inside black central rectangle ---
    
    # Blue dotted line in black rectangle (further from left side)
    center_left_dotted_x = center_rect_x + 65
    plt.plot([center_left_dotted_x, center_left_dotted_x], 
             [left_rect_y, left_rect_y + rect_height], 
             color='#0ea5e9', linestyle='--', linewidth=2)
    ax.text(center_left_dotted_x + 20, left_rect_y + 20, 'stop\nleft', 
            fontsize=10, color='#0ea5e9', fontweight='bold', ha='center')
    
    # Red dotted line in black rectangle (further from right side)
    center_right_dotted_x = center_rect_x + center_rect_width - 65
    plt.plot([center_right_dotted_x, center_right_dotted_x], 
             [right_rect_y, right_rect_y + rect_height], 
             color='#ef4444', linestyle='--', linewidth=2)
    ax.text(center_right_dotted_x - 20, right_rect_y + 20, 'stop\nright', 
            fontsize=10, color='#ef4444', fontweight='bold', ha='center')

    # --- 3. Draw the Custom Axes ---
    
    # Draw Y-axis line (Vertical)
    # Starts from top of central area down past the origin
    plt.plot([axis_origin_x, axis_origin_x], [120, 420], color='black', linewidth=2.5)
    
    # Draw Y-axis Arrow head at bottom
    ax.arrow(axis_origin_x, 420, 0, 1, head_width=15, head_length=15, fc='black', ec='black', clip_on=False)
    ax.text(axis_origin_x + 20, 440, "Y", fontsize=24, color='black')

    # Draw X-axis line (Horizontal)
    plt.plot([50, 430], [axis_origin_y, axis_origin_y], color='black', linewidth=2.5)
    
    # Draw X-axis Arrow head at right
    ax.arrow(430, axis_origin_y, 1, 0, head_width=15, head_length=15, fc='black', ec='black', clip_on=False)
    ax.text(450, axis_origin_y + 20, "X", fontsize=24, color='black')

    # --- 4. Draw Ticks and Labels ---
    
    # Y-axis ticks (Left side)
    # Positions based on image: 325, 360, 395
    # We map these values to our y-coordinates. 
    # Let's align them roughly with the center box logic.
    y_ticks = [
        (220, "325"),
        (280, "360"),
        (340, "395")
    ]
    
    for y_pos, label in y_ticks:
        # Tick mark
        plt.plot([axis_origin_x - 10, axis_origin_x + 10], [y_pos, y_pos], color='black', linewidth=2)
        # Label (aligned to the left of the axis)
        ax.text(axis_origin_x - 40, y_pos + 5, label, fontsize=12, color='black')

    # X-axis ticks (Bottom)
    # Positions: 285, 320, 355
    x_ticks = [
        (center_rect_x, "285"),       # left side of center box
        (center_rect_x + center_rect_width/2, "320"), # middle of center box
        (center_rect_x + center_rect_width, "355") # right side of center box
    ]
    
    for x_pos, label in x_ticks:
        # Tick mark
        plt.plot([x_pos, x_pos], [axis_origin_y - 10, axis_origin_y + 10], color='black', linewidth=2)
        # Label (aligned below the axis)
        ax.text(x_pos - 10, axis_origin_y + 35, label, fontsize=12, color='black')

    # --- 5. Final Plot Settings ---
    
    # Invert Y axis to match the computer graphics coordinate system (0 at top, increasing downwards)
    plt.gca().invert_yaxis()
    
    # Remove standard chart border and ticks
    ax.axis('off')
    
    # Set limits to frame everything nicely
    ax.set_xlim(0, 550)
    ax.set_ylim(480, 50) # Inverted limits (bottom, top)
    
    plt.tight_layout()
    plt.savefig('diagram.png', dpi=300, bbox_inches='tight')
    print("Diagram saved as 'diagram.png'")

if __name__ == "__main__":
    draw_diagram()