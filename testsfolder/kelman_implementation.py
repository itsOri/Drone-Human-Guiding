import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# Global variables
N_PREDICTIONS = 20  # Number of future positions to predict
DECAY_RATE = 0.8    # Velocity decay rate (0.9 = 10% reduction per step)
MIN_HISTORY_SIZE = 5 # Minimum number of frames to start predicting


def predict_trajectory(coordinates, n_predictions=N_PREDICTIONS, decay_rate=DECAY_RATE):
    """
    Predict future trajectory using Kalman Filter with velocity damping.
    
    Parameters:
    -----------
    coordinates : list of tuples
        List of (x, y) coordinate tuples representing observed positions.
        Example: [(x1, y1), (x2, y2), ...]
    n_predictions : int, optional
        Number of future positions to predict. Defaults to N_PREDICTIONS global.
    decay_rate : float, optional
        Velocity decay rate for predictions (0-1). Defaults to DECAY_RATE global.
        
    Returns:
    --------
    list of tuples
        List of (x, y) predicted coordinate tuples.
    """
    # Convert list of tuples to numpy array
    coords_array = np.array(coordinates)
    n_measurements = len(coordinates)
    
    if n_measurements <= MIN_HISTORY_SIZE:
        return []
    
    # Initialize Kalman Filter
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0
    
    # State transition matrix (constant velocity model)
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1,  0],
                     [0, 0, 0,  1]])
    
    # Measurement matrix (we only observe position, not velocity)
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    
    # Covariance matrices
    kf.P *= 500.0                # State covariance
    kf.R = np.eye(2) * 0.4       # Measurement noise
    kf.Q = np.eye(4) * 0.01      # Process noise
    
    # Initialize state with first measurement
    kf.x = np.array([coords_array[0, 0], coords_array[0, 1], 0, 0])
    
    # Filter all measurements to get current state estimate
    for i in range(n_measurements):
        z = coords_array[i]
        kf.predict()
        kf.update(z)
    
    # Predict future positions with velocity damping
    predicted_positions = []
    for _ in range(n_predictions):
        kf.x[2:] *= decay_rate  # Apply damping to velocity components
        kf.predict()
        predicted_positions.append((kf.x[0], kf.x[1]))
    
    return predicted_positions


def demo_example():
    """
    Demonstration of Kalman Filter tracking with the original example.
    This function runs when the script is executed directly.
    """
    # Generate synthetic 2D measurements
    np.random.seed(42)
    t = np.arange(20)
    true_positions = np.vstack([t, 0.5 * t + np.sin(t / 2)])
    measurements = true_positions + np.random.randn(2, 20) * 0.3
    
    # Convert measurements to list of tuples
    measurement_list = [(measurements[0, i], measurements[1, i]) for i in range(20)]
    
    # Initialize Kalman Filter for visualization
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1,  0],
                     [0, 0, 0,  1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= 500.0
    kf.R = np.eye(2) * 0.4
    kf.Q = np.eye(4) * 0.01
    kf.x = np.array([measurements[0, 0], measurements[1, 0], 0, 0])
    
    # Filter measurements
    filtered_positions = []
    for i in range(20):
        z = measurements[:, i]
        kf.predict()
        kf.update(z)
        filtered_positions.append(kf.x[:2].copy())
    filtered_positions = np.array(filtered_positions)
    
    # Use the predict_trajectory function
    predicted_positions = predict_trajectory(measurement_list)
    predicted_positions = np.array(predicted_positions)
    
    # Plot everything
    plt.figure(figsize=(8, 6))
    plt.plot(measurements[0], measurements[1], 'ro', label='Measurements')
    plt.plot(filtered_positions[:, 0], filtered_positions[:, 1], 'b-', label='Filtered path')
    plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], 'gx--', label=f'Predicted next {N_PREDICTIONS} (damped)')
    plt.legend()
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('2D Person Tracking with Kalman Filter (Damped Prediction)')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    demo_example()