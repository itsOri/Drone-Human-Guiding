# Kalman Filter Trajectory Prediction Integration

## Overview
Integrated the Kalman filter-based trajectory prediction from `kelman_implementation.py` into the main tracking system to improve template matching when a person is temporarily lost from the frame.

## Changes Made

### 1. Import Addition
- Added import: `from components.kelman_implementation import predict_trajectory`

### 2. Container Structure Updates
Both `selected_id_container` and `selected_target_container` now include:
- `"predicted_trajectory": []` - List of predicted (x, y) positions from Kalman filter

### 3. Logic Flow in `find_id_in_frame()`

#### When Person is Found (coords exist):
- Clears `predicted_trajectory` list (reset predictions)
- Updates `center_history` with current position
- Normal tracking continues

#### When Person is Lost (first frame, lost_counter == 1):
- Calls `predict_trajectory(center_history)` to generate future position predictions
- Stores predictions in `predicted_trajectory` list
- Logs number of predictions generated (or if insufficient history)

#### While Person Remains Lost (lost_counter > 1):
- **If predictions available:**
  - Pops the first predicted position from the list
  - Stores as center coordinates: `(center_x, center_y)`
  - Updates `last_seen_center_coords` with predicted position
  - Logs predicted position and remaining predictions count

- **If predictions list is empty:**
  - Falls back to using the last known `coords`
  - Extracts center from full bbox: `(center_x, center_y)`
  - Updates `last_seen_center_coords` with last known center position
  - Logs fallback to last known position

#### When Person is Completely Lost (lost_counter >= MAX_LOST_ID_FRAMES):
- Attempts one final template matching using:
  - Predicted position if available (pops from list)
  - Last known position if no predictions
- Clears all predictions and resets tracking

## Key Features

### Prediction Strategy
1. **First loss frame**: Generate trajectory predictions using coordinate history
2. **Subsequent frames**: Pop and use predictions one at a time
3. **Fallback**: Use last known position when predictions exhausted
4. **Reset**: Clear predictions when person is re-acquired

### Coordinate Format
Predictions are stored as `(x, y)` tuples from Kalman filter and stored directly as center coordinates for template matching:
```python
(center_x, center_y)
```
This simplified format is stored in the `last_seen_center_coords` field.

### Logging
All Kalman-related operations are logged with `[KALMAN]` prefix for easy tracking:
- Prediction generation
- Number of historical points used
- Predicted positions being used
- Remaining predictions count
- Fallback to last known position

## Benefits

1. **Improved Re-identification**: Template matching has better location hints from predicted trajectory
2. **Smooth Tracking**: Predictions follow expected movement path rather than static last position
3. **Graceful Degradation**: Falls back to last known position when predictions unavailable
4. **No Breaking Changes**: Maintains backward compatibility with existing template matching

## Requirements

- Minimum 5 frames of coordinate history to generate predictions (configurable in `kelman_implementation.py`)
- Coordinate history maintained in `center_history` field (max COORDINATE_HISTORY_SIZE=30 frames)

## Usage

No changes needed to use this feature - it's automatically integrated into the tracking system. When a person is lost:

1. System generates predictions from movement history
2. Each frame uses next predicted position for template matching
3. Template matching searches near predicted location
4. When person is re-acquired, tracking resumes normally

## Parameters (from kelman_implementation.py)

- `N_PREDICTIONS = 20`: Number of future positions to predict
- `DECAY_RATE = 0.90`: Velocity decay rate (10% reduction per step)
- `MIN_HISTORY_SIZE = 5`: Minimum frames needed to start predicting

