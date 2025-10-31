import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# Generate synthetic 2D measurements
np.random.seed(42)
t = np.arange(20)
true_positions = np.vstack([t, 0.5 * t + np.sin(t / 2)])
measurements = true_positions + np.random.randn(2, 20) * 0.3

# Initialize Kalman Filter
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

# Predict next 10 positions with decaying steps
predicted_positions = []
damping = 0.9  # reduce velocity by 10% each step
for _ in range(10):
    kf.x[2:] *= damping  # apply damping to velocity
    kf.predict()
    predicted_positions.append(kf.x[:2].copy())
predicted_positions = np.array(predicted_positions)

# Plot everything
plt.figure(figsize=(8, 6))
plt.plot(measurements[0], measurements[1], 'ro', label='Measurements')
plt.plot(filtered_positions[:, 0], filtered_positions[:, 1], 'b-', label='Filtered path')
plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], 'gx--', label='Predicted next 10 (damped)')
plt.legend()
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('2D Person Tracking with Kalman Filter (Damped Prediction)')
plt.grid(True)
plt.show()