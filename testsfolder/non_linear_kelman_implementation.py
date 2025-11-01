import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter

# ============================================================
# 1. Simulation setup
# ============================================================

dt = 0.1
n_meas = 20
n_pred = 15
true_r = 10
true_omega = 0.25  # radians per second

# True circular path (x, y)
thetas = np.arange(0, n_meas * dt * true_omega, true_omega * dt)
true_positions = np.array([
    [true_r * np.cos(theta), true_r * np.sin(theta)] for theta in thetas
])

# Add Gaussian measurement noise
np.random.seed(42)
measurements = true_positions + np.random.randn(*true_positions.shape) * 0.4


# ============================================================
# 2. Linear Kalman Filter (constant velocity model)
# ============================================================

kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
kf.H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])
kf.P *= 500.
kf.R = np.eye(2) * 0.25
kf.Q = np.eye(4) * 0.05
kf.x = np.array([true_r, 0, 0, true_r * true_omega])  # [x, y, vx, vy]

kf_positions = []

for z in measurements:
    kf.predict()
    kf.update(z)
    kf_positions.append(kf.x[:2].copy())

# Predict 15 future positions
for _ in range(n_pred):
    kf.predict()
    kf_positions.append(kf.x[:2].copy())

kf_positions = np.array(kf_positions)


# ============================================================
# 3. Extended Kalman Filter (nonlinear circular model)
# ============================================================

def fx(x, dt):
    """State transition for circular motion."""
    r, theta, omega = x
    return np.array([r, theta + omega * dt, omega])

def hx(x):
    """Measurement: polar → Cartesian."""
    r, theta, omega = x
    return np.array([r * np.cos(theta), r * np.sin(theta)])

def jacobian_F(x, dt):
    return np.array([
        [1, 0, 0],
        [0, 1, dt],
        [0, 0, 1]
    ])

def jacobian_H(x):
    r, theta, omega = x
    return np.array([
        [np.cos(theta), -r * np.sin(theta), 0],
        [np.sin(theta),  r * np.cos(theta), 0]
    ])

ekf = ExtendedKalmanFilter(dim_x=3, dim_z=2)
ekf.x = np.array([9.5, 0.0, 0.3])  # [r, θ, ω]
ekf.P *= 10.
ekf.R = np.eye(2) * 0.25
ekf.Q = np.eye(3) * 1e-3

ekf_positions = []

# Filter updates for 20 measurements
for z in measurements:
    ekf.F = jacobian_F(ekf.x, dt)
    ekf.H = jacobian_H(ekf.x)
    ekf.predict()
    ekf.update(z, HJacobian=jacobian_H, Hx=hx)
    r, theta, _ = ekf.x
    ekf_positions.append([r * np.cos(theta), r * np.sin(theta)])

# Predict 15 future positions
for _ in range(n_pred):
    ekf.F = jacobian_F(ekf.x, dt)
    ekf.predict()
    r, theta, _ = ekf.x
    ekf_positions.append([r * np.cos(theta), r * np.sin(theta)])

ekf_positions = np.array(ekf_positions)


# ============================================================
# 4. Extended ground truth for visualization
# ============================================================

thetas_future = np.arange(0, (n_meas + n_pred) * dt * true_omega, true_omega * dt)
true_positions_ext = np.array([
    [true_r * np.cos(theta), true_r * np.sin(theta)] for theta in thetas_future
])


# ============================================================
# 5. Plot results
# ============================================================

plt.figure(figsize=(7,7))
plt.plot(true_positions_ext[:,0], true_positions_ext[:,1], 'g-', label='True Path')
plt.plot(measurements[:,0], measurements[:,1], 'bx', alpha=0.6, label='Measurements')
plt.plot(kf_positions[:,0], kf_positions[:,1], 'r--', label='Linear KF (track + predict)')
plt.plot(ekf_positions[:,0], ekf_positions[:,1], 'm-', label='EKF (track + predict)')
plt.legend()
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Circular Motion: Linear KF vs EKF (20 track + 15 predict)')
plt.axis('equal')
plt.grid(True)
plt.show()


# ============================================================
# 6. Print tracking and prediction RMSE
# ============================================================

# Compare only over the first 20 measurements
kf_rmse = np.sqrt(np.mean(np.sum((kf_positions[:n_meas] - true_positions)**2, axis=1)))
ekf_rmse = np.sqrt(np.mean(np.sum((ekf_positions[:n_meas] - true_positions)**2, axis=1)))

print(f"Tracking RMSE (20 points):")
print(f"  Linear KF : {kf_rmse:.3f}")
print(f"  Extended KF: {ekf_rmse:.3f}")