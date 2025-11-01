# Kalman Filter Prediction - Test Scenarios

## Test Scenario 1: Person Walking in Straight Line

**Setup:**
- Person walking horizontally from left to right at constant speed
- Track for 10+ frames, then person gets temporarily occluded

**Expected Behavior:**
```
Frame 1-10:  Person visible, building center_history
Frame 11:    Person lost → Generate predictions (linear right movement)
Frame 12-15: Use predictions p1-p4 → Should point further right
Frame 16:    Person reappears → Template matching succeeds using predicted location
```

**Log Output to Watch For:**
```
[KALMAN] Generating trajectory prediction with 10 historical points
[KALMAN] Generated 20 predicted positions
[KALMAN] Using predicted position: (450, 240), 19 predictions remaining
[KALMAN] Using predicted position: (455, 240), 18 predictions remaining
```

---

## Test Scenario 2: Person Stops and Gets Occluded

**Setup:**
- Person walks, then stops for a few frames
- Gets occluded while stationary

**Expected Behavior:**
```
Frame 1-8:  Person walking (velocity in history)
Frame 9-12: Person stops (zero velocity in recent history)
Frame 13:   Person lost → Generate predictions with damping
Frame 14-18: Predictions should show gradual slowdown
             Later predictions should cluster near last position
```

**Why This Tests Damping:**
The Kalman filter with DECAY_RATE=0.9 should predict the person continues briefly in their original direction but rapidly decelerates.

---

## Test Scenario 3: Insufficient History

**Setup:**
- Select a person ID
- Person gets lost within first 5 frames (before enough history)

**Expected Behavior:**
```
Frame 1-3:   Person visible, only 3 positions in center_history
Frame 4:     Person lost → Cannot generate predictions (< MIN_HISTORY_SIZE=5)
Frame 5-8:   Fallback to last known position
```

**Log Output:**
```
[KALMAN] Not enough history (need >5 frames), using last known position
[KALMAN] No predictions available, using last known position
```

---

## Test Scenario 4: Long Occlusion (Exhaust Predictions)

**Setup:**
- Person tracked for 15 frames
- Gets occluded for 25+ frames (longer than N_PREDICTIONS=20)

**Expected Behavior:**
```
Frame 1-15:   Person visible
Frame 16:     Lost → Generate 20 predictions
Frame 17-36:  Use predictions p1-p20 (pop one per frame)
Frame 37-40:  Predictions exhausted → Fallback to last known position
```

**Log Output:**
```
[KALMAN] Using predicted position: (500, 300), 0 predictions remaining
[KALMAN] No predictions available, using last known position
[KALMAN] No predictions available, using last known position
```

---

## Test Scenario 5: Person Changes Direction Before Loss

**Setup:**
- Person walks right, then turns and walks left
- Gets occluded shortly after direction change

**Expected Behavior:**
- Predictions should follow the NEW direction (left)
- Kalman filter considers recent velocity in center_history
- Should NOT predict continued right movement

**Verification:**
Compare predicted x-coordinates - should be decreasing (moving left), not increasing.

---

## Test Scenario 6: Both Selected and Target Lost Simultaneously

**Setup:**
- Track both selected person and target person
- Both get occluded at the same time

**Expected Behavior:**
- Each container independently generates predictions
- Both use their own center_history
- Template matching for each uses their respective predicted positions
- No interference between the two tracking systems

---

## Test Scenario 7: Person Found via Template Matching

**Setup:**
- Person lost for 3 frames
- Template matching successfully re-identifies person with new YOLO ID

**Expected Behavior:**
```
Frame 10:   Person (ID=5) lost
            → Generate predictions using ID=5's center_history
Frame 11:   Use prediction p1 for template matching
            → Template matching finds person with new ID=12
            → Clear predicted_trajectory
            → Reset center_history = []
Frame 12:   Start building new center_history for ID=12
```

**Log Output:**
```
[TEMPLATE_MATCHING] Re-acquired target! Old ID: 5, New ID: 12
[TEMPLATE_MATCHING] Reset coordinate history for new ID 12
```

---

## How to Observe in Logs

Search for these patterns in the log file:

```bash
# Check prediction generation
grep "\[KALMAN\] Generating" drone_log_*.log

# Check prediction usage
grep "\[KALMAN\] Using predicted" drone_log_*.log

# Check fallback cases
grep "\[KALMAN\] No predictions available" drone_log_*.log

# Check final attempts
grep "\[KALMAN\] Final attempt" drone_log_*.log

# Count how many times predictions were used
grep -c "\[KALMAN\] Using predicted" drone_log_*.log
```

---

## Success Metrics

1. **Prediction Accuracy**: 
   - When person reappears, they should be close to predicted position
   - Template matching success rate should improve

2. **Smooth Tracking**:
   - last_seen_center_coords should follow a smooth trajectory
   - No sudden jumps to irrelevant locations

3. **Performance**:
   - Kalman prediction should be fast (< 1ms typically)
   - No noticeable FPS drop

4. **Robustness**:
   - Handles edge cases gracefully (no history, exhausted predictions)
   - Falls back to last known position when needed

---

## Debug Commands During Testing

While running the drone system:

```python
# Add these temporary debug prints in main loop if needed:
logger.info(f"[DEBUG] center_history length: {len(selected_id_container['center_history'])}")
logger.info(f"[DEBUG] predicted_trajectory length: {len(selected_id_container['predicted_trajectory'])}")
logger.info(f"[DEBUG] last_seen_center_coords: {selected_id_container.get('last_seen_center_coords')}")
```

---

## Known Limitations

1. **Prediction assumes smooth motion**: Won't predict sudden direction changes
2. **Bounded by MAX_LOST_ID_FRAMES (20)**: Person must be found within 20 frames
3. **Center coordinates only**: Stores only (x, y) center, no bounding box information
4. **Requires history**: First 5 frames of tracking can't use predictions

These are acceptable trade-offs for the improved tracking capability.

