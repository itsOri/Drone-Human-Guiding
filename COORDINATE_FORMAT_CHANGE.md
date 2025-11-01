# Coordinate Format Change - Summary

## Changes Made

### Key Update: Simplified `last_seen_coords` to `last_seen_center_coords`

**Previous Format:**
```python
last_seen_coords = (x1, y1, x2, y2, center_x, center_y)  # Full bounding box + center
```

**New Format:**
```python
last_seen_center_coords = (center_x, center_y)  # Center coordinates only
```

---

## Rationale

1. **Simplified Data Structure**: Template matching primarily needs the center position to calculate distances, not the full bounding box
2. **Clearer Naming**: `last_seen_center_coords` explicitly indicates it contains center coordinates
3. **Consistency**: Kalman predictions output `(x, y)` tuples - storing them directly without conversion
4. **Reduced Complexity**: No need to estimate bounding box dimensions when storing predictions

---

## Code Changes in `main.py`

### 1. Container Initialization (line ~542)
```python
# Before
if "last_seen_coords" not in selected_id_container:
    selected_id_container["last_seen_coords"] = None

# After
if "last_seen_center_coords" not in selected_id_container:
    selected_id_container["last_seen_center_coords"] = None
```

### 2. When Person is Found (line ~582)
```python
# Before
selected_id_container["last_seen_coords"] = None

# After
selected_id_container["last_seen_center_coords"] = None
```

### 3. Storing Predicted Position (line ~611)
```python
# Before
bbox_size = 50
selected_id_container["last_seen_coords"] = (
    int(predicted_x - bbox_size), int(predicted_y - bbox_size),
    int(predicted_x + bbox_size), int(predicted_y + bbox_size),
    int(predicted_x), int(predicted_y)
)

# After
selected_id_container["last_seen_center_coords"] = (int(predicted_x), int(predicted_y))
```

### 4. Storing Last Known Position (line ~617)
```python
# Before
selected_id_container["last_seen_coords"] = selected_id_container["coords"]

# After
if selected_id_container["coords"] is not None:
    x1, y1, x2, y2, center_x, center_y = selected_id_container["coords"]
    selected_id_container["last_seen_center_coords"] = (center_x, center_y)
```

### 5. Final Attempt - Predicted (line ~632)
```python
# Before
bbox_size = 50
selected_id_container["last_seen_coords"] = (
    int(predicted_x - bbox_size), int(predicted_y - bbox_size),
    int(predicted_x + bbox_size), int(predicted_y + bbox_size),
    int(predicted_x), int(predicted_y)
)

# After
selected_id_container["last_seen_center_coords"] = (int(predicted_x), int(predicted_y))
```

### 6. Final Attempt - Last Known (line ~636)
```python
# Before
selected_id_container["last_seen_coords"] = selected_id_container["coords"]

# After
x1, y1, x2, y2, center_x, center_y = selected_id_container["coords"]
selected_id_container["last_seen_center_coords"] = (center_x, center_y)
```

### 7. In `find_new_id()` Function (line ~1037)
```python
# Before
selected_id_last_center = selected_id_container.get("last_seen_coords", None)

# After
selected_id_last_center = selected_id_container.get("last_seen_center_coords", None)
```

---

## Updated Container Structure

```python
selected_id_container = {
    "id": [],
    "history": [],
    "id_was_seen": False,
    "lost_counter": 0,
    "followed_id": None,
    "coords": (x1, y1, x2, y2, center_x, center_y),  # Full bbox when person visible
    "last_seen_center_coords": (center_x, center_y),  # Center only when lost
    "center_history": [(x1, y1), (x2, y2), ...],      # Historical centers
    "predicted_trajectory": [(px1, py1), (px2, py2), ...]  # Kalman predictions
}
```

---

## Impact on Template Matching

The `template_matching.find_best_match()` function receives `last_seen_center_coords` as:
- **Input**: `(center_x, center_y)` tuple
- **Usage**: Calculate distance between this center and detected person centers
- **No changes needed**: Template matching already expects center coordinates for distance calculation

---

## Benefits of This Change

1. ✅ **Clearer semantics**: Name explicitly indicates center coordinates
2. ✅ **Simpler code**: No need to construct fake bounding boxes around predictions
3. ✅ **Direct storage**: Kalman predictions stored without conversion
4. ✅ **Consistent format**: Always `(x, y)` tuple, no mixed formats
5. ✅ **Less error-prone**: No arbitrary bbox_size constant (was 50px)

---

## Testing Checklist

- [x] All occurrences of `last_seen_coords` renamed to `last_seen_center_coords`
- [x] Format changed from 6-tuple to 2-tuple throughout
- [x] Extraction of center from `coords` when converting
- [x] No linter errors
- [x] Documentation updated (3 files)

---

## Files Modified

1. **main.py** - 7 locations updated
2. **KALMAN_PREDICTION_INTEGRATION.md** - Coordinate format section
3. **KALMAN_FLOW_DIAGRAM.md** - Data structure example
4. **KALMAN_TEST_SCENARIOS.md** - Debug commands and limitations

---

## Backward Compatibility

⚠️ **Breaking Change**: Any code that previously accessed `last_seen_coords` must be updated to use `last_seen_center_coords` and expect a 2-tuple instead of 6-tuple.

However, since this is a newly added feature (Kalman prediction integration), there should be no existing dependencies on this field outside of the modified code.

