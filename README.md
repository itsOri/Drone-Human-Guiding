# Drone Human Guiding System

A sophisticated autonomous drone-based project for tracking and guiding humans to a destination. This system serves as a foundation for robotic assistants and navigation aids for visually impaired persons.

## 📋 Table of Contents
- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Software Architecture](#software-architecture)
- [Installation & Setup](#installation--setup)
- [Running the System](#running-the-system)
- [Key Features](#key-features)
- [Core Components](#core-components)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This project implements an intelligent drone system that can:
- **Track a user** autonomously using computer vision and object detection
- **Navigate the user** to a target person or location
- **Avoid obstacles** dynamically (both static and moving objects)
- **Re-identify lost targets** using template matching when tracking is lost
- **Predict trajectories** using Kalman filtering for smooth tracking
- **Plan optimal paths** using RRT* (Rapidly-exploring Random Tree Star) algorithm

The system uses the **DJI Tello EDU drone** combined with a computer running advanced computer vision algorithms to provide real-time autonomous guidance.

---

## 🛠 Hardware Requirements

### DJI Tello EDU Drone
- **Model**: DJI Tello EDU
- **Camera**: 720p HD video transmission at 30fps
- **Flight Time**: ~13 minutes
- **Range**: Up to 100 meters
- **Battery**: 1100mAh rechargeable

### Computer Requirements
- **OS**: Windows, macOS, or Linux
- **RAM**: Minimum 8GB (16GB recommended for YOLO processing)
- **CPU**: Intel i5 or equivalent (i7 or better recommended)
- **GPU**: Optional but recommended (CUDA-compatible GPU for faster YOLO inference)
- **WiFi**: 2.4GHz WiFi adapter required for drone connection

### Connectivity
The computer connects to the drone via **WiFi Direct**:
1. The Tello drone creates its own WiFi network (SSID: TELLO-XXXXXX)
2. Computer connects to this network (no internet access while connected)
3. Communication happens over UDP on the drone's IP: `192.168.10.1`

---

## 🏗 Software Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Main Control Loop                       │
│                      (main.py)                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Drone      │  │    YOLO      │  │   Kalman     │    │
│  │  Controller  │  │   Tracking   │  │   Filter     │    │
│  │ (djitellopy) │  │ (YOLOv11)    │  │ (FilterPy)   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Template    │  │     RRT*     │  │   Video      │    │
│  │  Matching    │  │    Planner   │  │   Output     │    │
│  │   (OpenCV)   │  │  (Threading) │  │  (3 Streams) │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Drone Video Stream (720p @ 30fps)
        │
        ▼
┌──────────────────┐
│   Frame Capture  │  ← Raw RGB frames
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  YOLO Detection  │  ← YOLOv11s-seg (segmentation + tracking)
│  & Tracking      │    Classes: Person (0), Car (2)
└────────┬─────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌──────────────┐   ┌──────────────┐
│   Person     │   │   Target     │
│   Tracking   │   │   Tracking   │
└──────┬───────┘   └───────┬──────┘
       │                   │
       │  (If lost)        │ (If lost)
       ▼                   ▼
┌──────────────────────────────────┐
│    Template Matching Engine      │
│  (Re-identify based on appearance)│
└──────────────────┬────────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│      Kalman Filter Predictor     │
│   (Predict trajectory when lost) │
└──────────────────┬────────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│        RRT* Path Planner         │
│ (Calculate path from user→target)│
└──────────────────┬────────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│      Drone Control Commands      │
│   (Yaw, Forward/Back velocities) │
└──────────────────────────────────┘
```

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Drone-Human-Guiding
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required Packages:**
- `djitellopy==2.5.0` - DJI Tello drone SDK
- `opencv_python==4.11.0.86` - Computer vision library
- `opencv_python_headless==4.11.0.86` - OpenCV without GUI dependencies
- `matplotlib==3.9.3` - Plotting library
- `numpy==2.0.1` - Numerical computing
- `ultralytics==8.3.129` - YOLOv11 framework
- `lap>=0.5.12` - Linear assignment problem solver (for tracking)
- `filterpy>=1.4.5` - Kalman filter implementation

### 3. Download YOLO Models
The project uses pre-trained YOLO models stored in `yolo_models/`:
- `yolo11s-seg.pt` - Main model (segmentation + detection)
- `yolo11n-seg.pt` - Nano model (faster, less accurate)
- `yolo11s.pt` - Detection only (no segmentation)
- `yolo11n.pt` - Nano detection only

These models are automatically loaded from the `yolo_models/` directory.

### 4. Connect to Drone
1. **Power on** the DJI Tello EDU drone
2. **Connect** your computer to the drone's WiFi network
   - Network Name: `TELLO-XXXXXX`
   - Password: (if required, check drone manual)
3. **Verify connection** - No internet will be available while connected to drone

---

## 🚀 Running the System

### Basic Execution

```bash
python main.py
```

### What Happens on Startup

1. **Drone Connection**: System connects to Tello and displays battery level
2. **Video Stream**: Drone camera stream initializes (720p @ 30fps)
3. **YOLO Model**: YOLOv11s-seg model loads for person/car detection
4. **Takeoff**: Drone automatically takes off to configured height (default: 400cm)
5. **ID Selection**: Console prompts for USER person ID to track
6. **Target Selection**: Console prompts for TARGET person ID to navigate to
7. **Autonomous Tracking**: System begins tracking and navigation

### User Input During Operation

During execution, you'll be prompted to enter IDs:

```
[INPUT] Enter USER Person ID to follow: 5
[INPUT] Enter Target Person ID to follow: 13
```

These IDs correspond to YOLO tracking IDs visible in the video feed.

## ⚙️ Key Features

### 1. **YOLO Object Detection & Tracking**
- **Model**: YOLOv11s-seg (segmentation variant)
- **Tracking Algorithm**: ByteTrack
- **Detected Classes**: 
  - `Person (0)` - Primary tracking target
  - `Car (2)` - Obstacle detection
- **Frame Rate**: Processes every frame (~30fps)
- **Persistence**: Maintains consistent IDs across frames

### 2. **Template Matching for Re-identification**
When a tracked person is lost (occlusion, leaves frame, etc.):
- System collects **20 most recent images** of each detected person
- Uses **normalized cross-correlation** with preprocessing:
  - Resize to common width (preserving aspect ratio)
  - Normalize pixel values
  - Apply Gaussian blur (kernel size: 31x31)
  - Test with multiple rotations (0°, +3°, -3°)
- **Distance Penalty**: Penalizes candidates far from last known position
- **Scoring**: Combines template match score (30%) + distance penalty (70%)
- **Threshold**: `THRESHOLD_BEST_MATCH = 0.1` (configurable)
- **Windowed Matching**: Accumulates scores over 15 frames before deciding

**Template Matching Flow:**
```
Person Lost → Wait 15 frames → Collect candidate scores
                    ↓
            Best score < threshold?
                    ↓
          YES ─────→ Re-acquire with new ID
                    ↓
          NO ──────→ Try again (max 3 rounds)
                    ↓
                Manual re-entry required
```

### 3. **Kalman Filter Trajectory Prediction**
Uses `filterpy` library to predict future positions when tracking is lost:
- **State Vector**: [x, y, vx, vy] (position + velocity)
- **Measurement**: [x, y] (only position observed)
- **Model**: Constant velocity with damping
- **Decay Rate**: 0.8 (velocity reduces by 20% per prediction)
- **Predictions**: Generates 20 future positions
- **Minimum History**: Requires 5+ frames of history to predict

**Prediction Integration:**
1. Person lost in frame N
2. Generate trajectory from last 30 frames of history
3. Use predictions as "last seen position" for subsequent frames
4. Updates template matching search area based on predictions
5. Visualizes predicted position with yellow/cyan markers

### 4. **RRT* Path Planning**
Rapidly-exploring Random Tree Star algorithm for optimal path planning:
- **Threaded Execution**: Runs asynchronously to avoid blocking main loop
- **Update Frequency**: Recalculates every 5 frames (configurable)
- **Parameters**:
  - `max_iter=1000` - Maximum tree expansion iterations
  - `delta=15` - Step size for tree growth (pixels)
  - `radius=30` - Rewiring radius for optimization
- **Input**: Binary segmentation map (white=free space, black=obstacles)
- **Output**: List of (x,y) waypoints from user to target
- **Visualization**: Purple/magenta path overlay on video

**Path Planning Process:**
```
Black/White Segmented Frame
    (Person & Car obstacles)
            ↓
    RRT* Algorithm
            ↓
    Optimal Path Found
            ↓
    Draw on all 3 video outputs
```

### 5. **Hysteresis Control System**
Prevents oscillation ("ping-pong" behavior) in drone movements:
- **Zones**: Divides tracking rectangle into regions with margins
- **Margin Size**: 33% of rectangle size (configurable)
- **States**: Tracks `up/down/left/right` movement states
- **Logic**: Once movement starts, continues until well inside target zone
- **Result**: Smooth, stable tracking without jitter

### 6. **Multi-Stream Video Output**
Records 3 synchronized video streams:
1. **Clean Video** (`clean_video_<timestamp>.avi`)
   - Raw drone camera feed
   - RGB format, 640x480 @ 20fps
   - Unmodified frames

2. **Annotated Video** (`annotated_video_<timestamp>.avi`)
   - YOLO bounding boxes with IDs
   - Tracking indicators:
     - 🔴 Red dot: Selected person center
     - 🟢 Green dot: Target person center
     - 🟡 Yellow dot: Lost selected person (predicted)
     - 🔵 Cyan dot: Lost target person (predicted)
   - RRT* path visualization (purple)
   - Control zone rectangle
   - Frame counter
   - Movement commands (FORWARD, ROTATE, etc.)

3. **Black/White Segmented** (`black_white_segmented_<timestamp>.avi`)
   - Binary obstacle map
   - White background = free space
   - Black = obstacles (persons + cars)
   - Colored dots for tracked persons
   - RRT* path overlay

### 7. **Comprehensive Logging**
All events logged to file: `videos/drone_log_<timestamp>.log`
- **Levels**: DEBUG, INFO, WARNING, ERROR
- **Captures**:
  - Frame-by-frame tracking status
  - Template matching scores
  - Kalman predictions
  - RRT* path calculations
  - Drone commands and velocities
  - ID changes and re-acquisitions
  - Battery status

---

## 🔧 Core Components

### `main.py` - Main Control System (1640 lines)
**Core Classes:**
- `Person` - Encapsulates tracking state for user/target
  - Stores ID history, coordinates, tracking status
  - Manages template matching candidates
  - Tracks Kalman predictions

- `RRTPathPlanner` - Threaded path planning
  - Asynchronous path calculation
  - Non-blocking queue-based communication
  - Automatic path updates

**Key Functions:**
- `start_drone()` - Initialize and connect to Tello
- `get_frame()` - Capture and preprocess video frames
- `find_id_in_frame()` - Track person with template matching fallback
- `control_drone()` - Calculate and send movement commands
- `create_bw_segmented_frame()` - Generate obstacle map

### `components/template_matching.py` - Re-identification Engine
**Functions:**
- `extract_all_visible_persons()` - Extract person images from frame
- `update_persons_dict()` - Maintain rolling history of person appearances
- `find_best_match()` - Perform template matching with distance penalty
- `get_distance_penalty()` - Exponential penalty based on distance

**Configuration:**
- `MAX_RECENT_EXTRACTIONS = 20` - Keep 20 most recent images per person
- `THRESHOLD_BEST_MATCH = 0.1` - Re-identification threshold
- `DISTANCE_PENALTY_CURVE = 0.002` - Penalty curve steepness
- `DIST_PENALTY_WEIGHT = 0.7` - 70% weight on distance, 30% on appearance

### `components/template_matching_function.py` - Matching Algorithm
**Core Algorithm:**
```python
def template_match(template_obj, image):
    """
    Returns normalized score (0-1, lower is better)
    Steps:
    1. Resize both images to same width (preserve aspect ratio)
    2. Normalize: (image - mean) / std
    3. Apply Gaussian blur (31x31 kernel)
    4. Test with 3 rotations: 0°, +3°, -3°
    5. Perform cross-correlation with padding
    6. Apply size penalty for aspect ratio differences
    7. Return minimum score across rotations
    """
```

**Key Features:**
- Handles scale variations via resizing
- Rotation invariance (±3°)
- Normalized cross-correlation with padding
- Size penalty to discourage mismatched aspect ratios

### `components/kelman_implementation.py` - Trajectory Prediction
**Main Function:**
```python
def predict_trajectory(coordinates, n_predictions=20, decay_rate=0.8):
    """
    Predict future trajectory using Kalman Filter
    
    Args:
        coordinates: List of (x,y) tuples (history)
        n_predictions: Number of future positions
        decay_rate: Velocity decay factor (0.8 = 20% reduction/step)
    
    Returns:
        List of (x,y) predicted coordinates
    """
```

**Kalman Filter Configuration:**
- **State Dimension**: 4 (x, y, vx, vy)
- **Measurement Dimension**: 2 (x, y observed only)
- **State Covariance (P)**: 500.0 (initial uncertainty)
- **Measurement Noise (R)**: 0.4 (sensor accuracy)
- **Process Noise (Q)**: 0.01 (model uncertainty)
- **dt**: 1.0 (assume 1 frame = 1 time unit)

### `components/RRTStar_New.py` - Path Planning
**Main Algorithm:**
```python
def rrt_star(start, goal, img, max_iter=2000, delta=15, radius=30):
    """
    RRT* pathfinding with optimization
    
    Key Features:
    - Random sampling with 10% goal bias
    - Nearest neighbor search
    - Collision checking via Bresenham's line algorithm
    - Parent selection based on minimum cost
    - Dynamic rewiring for optimization
    - Early termination when goal reached
    """
```

**Node Class:**
```python
class Node:
    x, y        # Position
    parent      # Parent node in tree
    cost        # Cost from start node
```

**Collision Detection:**
- Uses binary image (255=free, 0=obstacle)
- Bresenham's line algorithm checks every pixel along path
- Rejects paths that touch black pixels (obstacles)

---

## 🎛 Configuration

### Main Configuration Variables (`main.py`)

```python
# Video Settings
WIDTH = 640
HEIGHT = 480

# Tracking Settings
MAX_LOST_ID_FRAMES = 20           # Max frames before giving up
COORDINATE_HISTORY_SIZE = 30      # Frames to keep for Kalman
TEMPLATE_M_WAITING_FRAMES_WINDOW = 15  # Template matching window
MAX_LOST_ROUNDS = 3                # Template matching attempts

# Drone Control
YAW_MOVING_VELOCITY = 5           # Rotation speed
FB_MOVING_VELOCITY = 5            # Forward/back speed
DRONE_START_HEIGHT = 400          # Starting altitude (cm)

# Control Rectangle (tracking zone)
RECT_CENTER = (WIDTH // 2, HEIGHT // 2 + 120)
RECT_SIZE = {"w": 70, "h": 70}
MIN_RECT_SIZE = 50

# Features
HISTEREZIS_ENABLED = True         # Enable smooth control
TARGET_TRACKING_ENABLED = True    # Enable target navigation
SAVE_FILE = True                  # Save video outputs

# RRT* Settings
ENABLE_RRT_PATH = True
RRT_FRAME_SKIP = 5                # Recalculate every N frames
RRT_MAX_ITER = 1000
RRT_DELTA = 15
RRT_RADIUS = 30
RRT_PATH_COLOR = (255, 0, 255)    # Purple/Magenta

# YOLO Classes
BASE_DETECTION_CLASSES = [0, 2]   # Person, Car
```

### Template Matching Configuration

```python
# In components/template_matching.py
MAX_RECENT_EXTRACTIONS = 20
THRESHOLD_BEST_MATCH = 0.1
DISTANCE_PENALTY_CURVE = 0.002
DIST_PENALTY_WEIGHT = 0.7

# In components/template_matching_function.py
KERNEL_SIZE = 31                  # Gaussian blur kernel
```

### Kalman Filter Configuration

```python
# In components/kelman_implementation.py
N_PREDICTIONS = 20                # Future positions to predict
DECAY_RATE = 0.8                  # Velocity decay per step
MIN_HISTORY_SIZE = 5              # Minimum frames to start predicting
```

---

## 📁 Project Structure

```
Drone-Human-Guiding/
│
├── main.py                       # Main control system (1640 lines)
├── basic_control.py              # Basic drone control examples
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── components/                   # Core algorithms
│   ├── template_matching.py              # Re-identification system
│   ├── template_matching_function.py     # Matching algorithm
│   ├── kelman_implementation.py          # Kalman prediction
│   ├── RRTStar_New.py                    # Path planning
│   └── yolo_track_test.py                # YOLO testing utilities
│
├── yolo_models/                  # Pre-trained models
│   ├── yolo11s-seg.pt           # Main segmentation model
│   ├── yolo11n-seg.pt           # Nano segmentation
│   ├── yolo11s.pt               # Detection only
│   └── yolo11n.pt               # Nano detection
│
├── videos/                       # Output directory (auto-created)
│   ├── clean_video_<timestamp>.avi
│   ├── annotated_video_<timestamp>.avi
│   ├── black_white_segmented_<timestamp>.avi
│   └── drone_log_<timestamp>.log
│
├── photos/                       # Captured person images
│   └── person_id_<id>_capture_<n>.png
│
├── persons_segmented/            # Real-time person extractions
│   └── person_id_<id>.png
│
└── Documentation/                # Design documents
    ├── COORDINATE_FORMAT_CHANGE.md
    ├── KALMAN_FLOW_DIAGRAM.md
    ├── KALMAN_PREDICTION_INTEGRATION.md
    └── KALMAN_TEST_SCENARIOS.md
```

---

## 🔍 Algorithm Details

### YOLO Detection Pipeline

```python
# YOLOv11 Segmentation + Tracking
model = YOLO("./yolo_models/yolo11s-seg.pt")
results = model.track(
    frame,
    persist=True,              # Maintain tracking IDs
    verbose=False,             # Quiet mode
    tracker="bytetrack.yaml",  # ByteTrack algorithm
    classes=[0, 2]             # Person, Car
)
```

**YOLOv11s-seg Characteristics:**
- **Architecture**: YOLOv11 Small with segmentation head
- **Parameters**: ~11M
- **Speed**: ~15-30 FPS on CPU (depends on hardware)
- **Accuracy**: High precision for person detection
- **Output**: Bounding boxes + pixel-level segmentation masks + tracking IDs

**ByteTrack Algorithm:**
- Kalman filter-based tracking
- Handles occlusions robustly
- Maintains ID consistency across frames
- Low-confidence detections used for tracking (not shown)

### Template Matching Mathematics

**Preprocessing:**
```
I_normalized = (I - μ) / σ

Where:
- I = input image
- μ = mean pixel value
- σ = standard deviation
```

**Cross-Correlation with Padding:**
```
C(x,y) = Σ[T(i,j) × I(x+i, y+j)]

Where:
- T = template (flipped horizontally/vertically)
- I = padded image
- (x,y) = search position
```

**Distance Penalty:**
```
P_dist = 1 - e^(-α × distance)

Where:
- α = DISTANCE_PENALTY_CURVE (0.002)
- distance = Euclidean distance from last seen position
```

**Final Score:**
```
Score = (1-w) × correlation_score + w × distance_penalty

Where:
- w = DIST_PENALTY_WEIGHT (0.7)
- Lower score = better match
```

### Kalman Filter Equations

**State Transition (Prediction):**
```
x_{k+1} = F × x_k
P_{k+1} = F × P_k × F^T + Q

Where:
x = [x, y, vx, vy]^T
F = [[1, 0, dt, 0 ],     dt = 1 (frame interval)
     [0, 1, 0,  dt],
     [0, 0, 1,  0 ],
     [0, 0, 0,  1 ]]
```

**Update (Measurement):**
```
K = P × H^T × (H × P × H^T + R)^{-1}
x = x + K × (z - H × x)
P = (I - K × H) × P

Where:
z = [x_measured, y_measured]^T
H = [[1, 0, 0, 0],
     [0, 1, 0, 0]]
```

**Velocity Damping (for prediction):**
```
vx_{k+1} = decay_rate × vx_k
vy_{k+1} = decay_rate × vy_k

decay_rate = 0.8 (velocity reduces by 20% each step)
```

### RRT* Cost Calculation

**Node Cost:**
```
cost(n) = cost(parent) + distance(parent, n)

Where distance is Euclidean distance
```

**Rewiring Decision:**
```
For each neighbor node m:
    new_cost = cost(n) + distance(n, m)
    if new_cost < cost(m):
        m.parent = n
        m.cost = new_cost
```

**Collision Check (Bresenham's Line):**
```python
# Checks every pixel along line from node1 to node2
# Returns False if any pixel is black (obstacle)
# Returns True if entire path is white (free space)
```

---

## 🐛 Troubleshooting

### Connection Issues

**Problem**: Cannot connect to drone
```
Solution:
1. Ensure drone is powered on (LED lights flashing)
2. Check WiFi connection to TELLO-XXXXXX network
3. Verify no other device is connected to drone
4. Restart drone and reconnect
5. Check firewall settings (UDP ports must be open)
```

**Problem**: Video stream not working
```
Solution:
1. Restart drone stream: drone.streamoff() then drone.streamon()
2. Check if other apps are using the camera
3. Increase time.sleep(2) after streamon() to 5 seconds
4. Verify drone battery > 20%
```

### Performance Issues

**Problem**: Low FPS / Lagging
```
Solution:
1. Use GPU acceleration for YOLO (install CUDA + PyTorch GPU)
2. Reduce RRT_FRAME_SKIP to 10 or higher
3. Disable RRT path planning: ENABLE_RRT_PATH = False
4. Use yolo11n-seg.pt (nano model) instead of yolo11s-seg.pt
5. Close other resource-intensive applications
```

**Problem**: Template matching too slow
```
Solution:
1. Reduce MAX_RECENT_EXTRACTIONS from 20 to 10
2. Increase TEMPLATE_M_WAITING_FRAMES_WINDOW to 20
3. Reduce KERNEL_SIZE in template_matching_function.py
```

### Tracking Issues

**Problem**: Person ID keeps changing
```
Solution:
1. Ensure good lighting conditions
2. Reduce occlusions in scene
3. Lower THRESHOLD_BEST_MATCH to accept more matches (e.g., 0.15)
4. Increase MAX_RECENT_EXTRACTIONS for better templates
```

**Problem**: Template matching never succeeds
```
Solution:
1. Check log file for actual matching scores
2. Adjust THRESHOLD_BEST_MATCH based on logged suggestions
3. Reduce DIST_PENALTY_WEIGHT if distance penalty too harsh
4. Verify person templates are being collected (check persons_segmented/)
```

**Problem**: Kalman predictions are inaccurate
```
Solution:
1. Increase MIN_HISTORY_SIZE for more stable predictions
2. Adjust DECAY_RATE (higher = faster deceleration)
3. Tune Kalman filter covariances (P, R, Q) in kelman_implementation.py
4. Ensure COORDINATE_HISTORY_SIZE is sufficient (>30)
```

### Drone Control Issues

**Problem**: Drone oscillates / jittery movement
```
Solution:
1. Ensure HISTEREZIS_ENABLED = True
2. Increase hysteresis margins in control_drone()
3. Reduce YAW_MOVING_VELOCITY and FB_MOVING_VELOCITY
4. Increase RECT_SIZE for larger dead zone
```

**Problem**: Drone loses person frequently
```
Solution:
1. Adjust RECT_CENTER and RECT_SIZE for better tracking window
2. Slow down drone speeds (reduce velocities)
3. Increase DRONE_START_HEIGHT for wider field of view
4. Enable DYNAMIC_RECT to adapt to person size
```

### Output Issues

**Problem**: Video files are corrupted / won't play
```
Solution:
1. Ensure proper shutdown (don't force-quit)
2. Try different codec: change 'MJPG' to 'XVID' in start_recording()
3. Check disk space available
4. Verify OpenCV installation: pip install --upgrade opencv-python
```

**Problem**: Colors are wrong in saved videos
```
Solution:
1. Check RGB/BGR conversions in cv2.imwrite() calls
2. Verify annotated_frame color format before writing
3. Frames from YOLO are RGB, ensure proper conversion to BGR
```

---

## 📊 Expected Performance

### Typical Metrics
- **Detection FPS**: 15-30 (depends on hardware)
- **Tracking Accuracy**: 90%+ in good conditions
- **Re-identification Success**: 70-85% (with good templates)
- **Path Planning Time**: 50-200ms per calculation
- **Kalman Prediction Accuracy**: 80-95% for short-term (5-10 frames)
- **Battery Life**: ~10-13 minutes flight time

### Optimal Operating Conditions
- **Lighting**: Bright, even lighting (avoid backlighting)
- **Background**: Contrasting background for better segmentation
- **Distance**: 2-5 meters from target person
- **Speed**: Walking pace or slower
- **Environment**: Open space with minimal obstacles
- **Interference**: No WiFi congestion on 2.4GHz band

---

## 🎓 Research & References

### Key Technologies
1. **YOLOv11**: Real-time object detection and segmentation
   - Ultralytics implementation
   - ByteTrack for object tracking persistence

2. **Template Matching**: Cross-correlation based re-identification
   - Normalized cross-correlation (NCC)
   - Multi-scale and rotation-invariant matching

3. **Kalman Filtering**: Optimal state estimation
   - Linear Kalman Filter for position + velocity
   - Predictive tracking under occlusion

4. **RRT* Algorithm**: Asymptotically optimal path planning
   - Sampling-based motion planning
   - Dynamic obstacle avoidance

### Applications
- **Assistive Technology**: Guide visually impaired persons
- **Robotics**: Mobile robot navigation and human-following
- **Surveillance**: Autonomous tracking in security systems
- **Retail**: Customer behavior analysis and assistance
- **Healthcare**: Patient monitoring and guidance

---

## 📝 License

This project is developed as part of an academic research project for autonomous drone navigation and human guidance systems.

---

## 🤝 Contributing

For improvements or bug fixes:
1. Test changes thoroughly with actual hardware
2. Verify all video outputs are correct
3. Check log files for errors
4. Update documentation as needed
5. Follow existing code style and structure

---

## 📞 Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review log files in `videos/` directory
3. Verify hardware connections and battery levels
4. Test with `basic_control.py` to isolate drone issues

---

**Last Updated**: November 2025  
**Version**: 1.0  
**Status**: Active Development

