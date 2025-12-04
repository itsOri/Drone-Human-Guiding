import numpy as np
import matplotlib.pyplot as plt

def hysteresis_path(values, low_threshold, high_threshold):
    """
    Simulate a binary output with hysteresis
    going through the input values in order.
    """
    state = 0  # current output
    outputs = []

    for v in values:
        if state == 0 and v > high_threshold:
            state = 1
        elif state == 1 and v < low_threshold:
            state = 0
        outputs.append(state)

    return np.array(outputs)

# Input range
x_forward = np.linspace(-1.5, 1.5, 400)
x_backward = x_forward[::-1]

# Thresholds similar to a central band for error
low_thr = -0.3
high_thr = 0.3

# Forward and backward paths
y_forward = hysteresis_path(x_forward, low_thr, high_thr)
y_backward = hysteresis_path(x_backward, low_thr, high_thr)

# Bring backward path back to original order on x axis
x_loop = np.concatenate([x_forward, x_backward])
y_loop = np.concatenate([y_forward, y_backward[::-1]])

plt.figure(figsize=(6, 5))
plt.plot(x_forward, y_forward, linewidth=2)
plt.plot(x_backward, y_backward, linewidth=2)

# Optional guides for thresholds
plt.axvline(low_thr, linestyle="--")
plt.axvline(high_thr, linestyle="--")

plt.xlabel("Input (for example tracking error)")
plt.ylabel("Output (for example control state)")
plt.title("Example hysteresis loop with two thresholds")
plt.grid(True)
plt.tight_layout()
plt.show()