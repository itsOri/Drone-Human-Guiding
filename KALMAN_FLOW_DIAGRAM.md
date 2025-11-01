# Kalman Filter Trajectory Prediction - Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PERSON TRACKING STATE MACHINE                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        PERSON IS VISIBLE                             │
│                                                                      │
│  • Update coords with current position                              │
│  • Add (center_x, center_y) to center_history                       │
│  • Keep last 30 positions in history                                │
│  • Clear predicted_trajectory = []                                  │
│  • lost_counter = 0                                                 │
│                                                                      │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           │ Person disappears from frame
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              FIRST FRAME - PERSON LOST (lost_counter = 1)           │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ IF center_history has > 5 positions:                       │    │
│  │   predicted_trajectory = predict_trajectory(center_history)│    │
│  │   Log: "Generated N predicted positions"                   │    │
│  │                                                             │    │
│  │ ELSE:                                                       │    │
│  │   predicted_trajectory = []                                │    │
│  │   Log: "Not enough history, using last known position"     │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           │ Person still not found
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│         SUBSEQUENT FRAMES - PERSON STILL LOST (lost_counter 2-20)   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ IF predicted_trajectory is NOT empty:                       │   │
│  │   predicted_pos = predicted_trajectory.pop(0)  ◄─── USE     │   │
│  │   last_seen_center_coords = (pred_x, pred_y)                │   │
│  │   Log: "Using predicted position: (x, y)"                   │   │
│  │                                                              │   │
│  │ ELSE:                                                        │   │
│  │   last_seen_center_coords = (cx, cy) from coords ◄── FALLBACK│  │
│  │   Log: "No predictions available, using last known"         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Template Matching uses last_seen_center_coords to search nearby    │
│                                                                      │
└──────────────────┬────────────────────────────┬──────────────────────┘
                   │                            │
         Person found via                       │ lost_counter >= 20
         template matching                       │
                   │                            ▼
                   │              ┌────────────────────────────────────┐
                   │              │   PERSON COMPLETELY LOST           │
                   │              │                                    │
                   │              │  • One final template match attempt│
                   │              │  • Use predicted_trajectory or     │
                   │              │    last known coords               │
                   │              │  • Clear predicted_trajectory = [] │
                   │              │  • Reset tracking, ask for new ID  │
                   │              └────────────────────────────────────┘
                   │
                   ▼
    ┌───────────────────────────────┐
    │   PERSON RE-ACQUIRED          │
    │                               │
    │  • New ID assigned            │
    │  • Reset center_history = []  │
    │  • Clear predictions          │
    │  • Resume normal tracking     │
    └───────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                    PREDICTION VISUALIZATION
═══════════════════════════════════════════════════════════════════════

Frame Timeline:
  
  [Frame -5]  [Frame -4]  [Frame -3]  [Frame -2]  [Frame -1]  [Frame 0]
     ●           ●           ●           ●           ●          LOST!
     │           │           │           │           │
     └───────────┴───────────┴───────────┴───────────┴──────────┐
                      center_history                            │
                      (last 30 positions)                       │
                                                                │
                                                                ▼
                                                    ┌──────────────────┐
                                                    │ Kalman Filter    │
                                                    │ Prediction       │
                                                    └──────────────────┘
                                                                │
                ┌───────────────────────────────────────────────┘
                │
                ▼
  [Frame 1]   [Frame 2]   [Frame 3]   [Frame 4]   [Frame 5]
     ◯           ◯           ◯           ◯           ◯      ← Predicted
     │           │           │           │           │        positions
     Pop         Pop         Pop         Pop         Pop      (used for
  (Use first) (Use 2nd)  (Use 3rd)   (Use 4th)   (Use 5th)  template
                                                              matching)

Legend:
  ●  = Actual detected position (in center_history)
  ◯  = Kalman predicted position (in predicted_trajectory)


═══════════════════════════════════════════════════════════════════════
                     DATA STRUCTURE OVERVIEW
═══════════════════════════════════════════════════════════════════════

selected_id_container = {
    "center_history": [
        (320, 240),    # Frame -5
        (325, 238),    # Frame -4
        (330, 235),    # Frame -3
        (335, 233),    # Frame -2
        (340, 230)     # Frame -1 (last seen)
    ],
    
    "predicted_trajectory": [  # Generated when lost_counter == 1
        (345, 228),    # Prediction for frame 1
        (350, 226),    # Prediction for frame 2
        (355, 224),    # Prediction for frame 3
        ...            # Up to 20 predictions
    ],
    
    "last_seen_center_coords": (
        345,  # center_x (predicted_x)
        228   # center_y (predicted_y)
    ),
    
    "lost_counter": 3,  # Frames since person was last detected
    "coords": None      # Cleared when person is lost
}
```

## Key Advantages of This Approach

1. **Smooth Trajectory Following**: Predictions follow the person's movement direction and velocity
2. **Velocity Damping**: Predictions gradually slow down (DECAY_RATE=0.9) - realistic for stopping
3. **Progressive Updates**: Each frame uses the next prediction in sequence
4. **Graceful Fallback**: If predictions run out or weren't generated, use last known position
5. **Automatic Reset**: When person is found again, clear predictions and start fresh

## Example Scenario

```
Frame 100: Person walking right at 5 pixels/frame → Added to center_history
Frame 101: Person walking right at 5 pixels/frame → Added to center_history
Frame 102: Person walking right at 5 pixels/frame → Added to center_history
...
Frame 110: Person walking right at 5 pixels/frame → Added to center_history

Frame 111: Person LOST (occluded behind obstacle)
           → Kalman filter predicts: person continues right but slowing down
           → Generated 20 predictions: [p1, p2, p3, ..., p20]
           → Use p1 for template matching

Frame 112: Person still LOST
           → Use p2 for template matching (person moved further right)

Frame 113: Person still LOST  
           → Use p3 for template matching

Frame 114: Person RE-APPEARS at position matching p4!
           → Template matching succeeds with new ID
           → Tracking resumes with better accuracy
```

