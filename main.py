"""
TODO Steps:

1. Fix Histeresis
2. Define what target to use
3. Detect target in image
4. Detect objects in the way, segmentation or else
4. Implement basic algo of shortest path rrt or a newer algo implement from papers with code.
5. give dynamic instructions to user ( could be an app with an arrow that directs user to target and avoids obstacles)

Ideas: had interpolation between frames to smooth the tracking but it was not working well.
The recognition of target was not working after interpolation it was messing up with yolo recognition
because it was moving even tho yolo wasnt getting new frames( to reduce computational load).
Now we are not interpolating and we are just getting the coordinates from the YOLO frame.  
We also tried taking every Nth frame to reduce computational load and pass them to yolo 
but it was not working well.
We added multiple classes to detect, like car, backpack, umbrella. 
Car is a must for future tasks the other two where for target detection but the detection on those classes was bad.
**SIMPLIFIED: Now only detecting person (class 0) and car (class 2) for better performance** 


obstacles static and dynamic (people moving)
and destination object static then hidden.

# YOLO Classes Reference (only using 0 and 2):
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush

remove backpack and umbrella from detection DONE
dont put selected and target on cars DONE
dont make template matching with diff type DONE
put point on target id DONE
check template matching on target id that it works 
make clearer prompt for input of target and selected id DONE
do logs also in a file , and check if it gets updated in live DONE`

"""

import cv2
import time
from djitellopy import Tello
from ultralytics import YOLO
import threading
import numpy as np
import os
import logging
from datetime import datetime 
from testsfolder import template_matching

#TODO change container to class
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Create videos directory if it doesn't exist
VIDEOS_DIR = "videos"
os.makedirs(VIDEOS_DIR, exist_ok=True)
OUTPUT_FILENAME = os.path.join(VIDEOS_DIR, f'clean_video_{TIMESTAMP}.avi')
OUTPUT_ANNOTATED_FILENAME = os.path.join(VIDEOS_DIR, f'annotated_video_{TIMESTAMP}.avi')
LOG_FILENAME = os.path.join(VIDEOS_DIR, f'drone_log_{TIMESTAMP}.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILENAME),
    ]
)
logger = logging.getLogger(__name__)


WIDTH = 640
HEIGHT = 480
selected_id = None

RECT_CENTER = (WIDTH // 2, HEIGHT // 2 + 50)
rect_size = {"w": 100, "h": 100}
MIN_RECT_SIZE = 50
CONST_RECT_TOP_LEFT = (RECT_CENTER[0] - rect_size['w'] // 2, RECT_CENTER[1] - rect_size['h'] // 2)
CONST_RECT_BOTTOM_RIGHT = (RECT_CENTER[0] + rect_size['w'] // 2, RECT_CENTER[1] + rect_size['h'] // 2)
DYNAMIC_RECT = False
SAVE_FILE = True
HISTEREZIS_ENABLED = True # Fixed hysteresis logic to prevent ping-pong behavior
TARGET_TRACKING_ENABLED = True # Set to True to enable target selection and tracking logic

# === PERFORMANCE OPTIMIZATION SETTINGS ===
YOLO_FRAME_SKIP = 2  # Process every Nth frame (1=every frame, 2=every 2nd frame, 3=every 3rd frame)
TEMPLATE_MATCHING_FRAME_SKIP = 1  # Update templates every Nth YOLO frame (reduces template collection frequency)
ENABLE_PERFORMANCE_MONITORING = True  # Show FPS and processing time stats

#RECT_TOP_LEFT = (RECT_CENTER[0] - RECT_W // 2, RECT_CENTER[1] - RECT_H // 2)
#RECT_BOTTOM_RIGHT = (RECT_CENTER[0] + RECT_W // 2, RECT_CENTER[1] + RECT_H // 2)

BLACK = (0, 0, 0)

YAW_MOVING_VELOCITY = 20
FB_MOVING_VELOCITY = 30
DRONE_START_HEIGHT = 250 # 200 is good for outside
MAX_LOST_ID_FRAMES = 20

# === YOLO DETECTION CLASSES ===
# Always detect these classes
BASE_DETECTION_CLASSES = [0, 2]  # 0: person, 2: car
BASE_CLASS_NAMES = {
    0: "person",
    2: "car"
}


def RECT_TOP_LEFT():
    if(DYNAMIC_RECT is False): 
        return CONST_RECT_TOP_LEFT
    return (RECT_CENTER[0] - rect_size['w'] // 2, RECT_CENTER[1] - rect_size['h'] // 2)

def RECT_BOTTOM_RIGHT():
    if(DYNAMIC_RECT is False): 
        return CONST_RECT_BOTTOM_RIGHT
    return (RECT_CENTER[0] + rect_size['w'] // 2, RECT_CENTER[1] + rect_size['h'] // 2)

def start_drone():
    drone = Tello()
    drone.connect()
    drone.LOGGER.setLevel("ERROR")  # or "WARNING"
    drone.for_back_velocity = 0
    drone.left_right_velocity = 0
    drone.up_down_velocity = 0
    drone.yaw_velocity = 0
    drone.speed = 0
    drone.TIME_BTW_RC_CONTROL_COMMANDS = 0.1

    logger.info(f"Battery: {drone.get_battery()}%")

    drone.streamoff()
    drone.streamon()

    time.sleep(2)  # Give stream a moment to start

    return drone


def start_recording(filename):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    if(SAVE_FILE):
        out = cv2.VideoWriter(filename, fourcc, 20.0, (WIDTH, HEIGHT))
        logger.info("Starting video capture. Press 'q' to stop.")
    
        return out
    return None


def get_frame(drone):
    frame = drone.get_frame_read().frame
    if frame is None:
        return None

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame_rgb


def should_quit():
    # Wait for 1ms for a keypress, mask to get last 8 bits, and check if 'q' was pressed
    # If 'q' is pressed, exit the video capture loop

    return cv2.waitKey(1) & 0xFF == ord('q')

# outs: lists of outputs to release
def exit(outs, drone):
    logger.info("Stopping capture, releasing resources.")
    for out in outs:
        out.release()
    cv2.destroyAllWindows()
    drone.streamoff()
    logger.info(f"Battery: {drone.get_battery()}%")


def launch_drone(drone):
    # do takeoff
    drone.takeoff()
    drone.move_up(DRONE_START_HEIGHT) # starting height
    return

def shutdown_drone(drone):
    drone.land()
    return

# def get_id(result):
# 	return
    
def target_thread(target_id_container):
    target_input = input("[INPUT] Enter Target Person ID to navigate to: ")
    try:
        target_id = int(target_input)
        target_id_container["id"].append(target_id)
        target_id_container["history"].append(target_id)
        target_id_container["lost_counter"] = 0
        target_id_container["id_was_seen"] = False
        target_id_container["target_class"] = None  # Reset target class for new target
        logger.info(f"Target ID: {target_id_container['id']}")

    except ValueError:
        logger.warning("Invalid ID entered for target.")
        target_id_container["id"] = []
        target_id_container["target_class"] = None
    

# TODO unite this func with target thread using generic container or string param for text display to user
def input_thread(selected_id_container):
    user_input = input("[INPUT] Enter Person ID to follow: ")
    try:
        selected_id_container["id"].append(int(user_input))
        selected_id_container["history"].append(int(user_input))
        selected_id_container["lost_counter"] = 0
        selected_id_container["id_was_seen"] = False
        logger.info(f"Selected Person ID: {selected_id_container['id']}")

    except ValueError:
        logger.warning("Invalid ID entered for selected person.")
        selected_id_container["id"] = []


# TODO check its generic func
def init_id_getter_thread(container):
    """
    Initializes and starts a background thread to continuously get user input.
    """
    thread = threading.Thread(target=input_thread, args=(container,))
    thread.daemon = True
    thread.start()
    return thread

def get_coordinates_by_id(results, target_id):
    """
    Given YOLO tracking results and a target ID,
    returns the bounding box coordinates for that ID as (x1, y1, x2, y2, center_x, center_y).
    If not found, returns None.
    """
    if(not target_id): return None
    for box in results[0].boxes:
        # box.id is a tensor, so convert to int if necessary
        if hasattr(box, 'id') and box.id is not None:
            # For YOLOv8, box.id is usually a 1-element tensor, so extract the scalar
            box_id = int(box.id.item()) if hasattr(box.id, 'item') else int(box.id)
            if box_id == target_id:
                # xyxy = [x1, y1, x2, y2], as float; convert to int if you want
                coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, coords)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                return (x1, y1, x2, y2, center_x, center_y)
    return None

def get_ids_in_frame(results):
    """
    Extracts tracking IDs from YOLOv8 inference results for all detected objects.

    Parameters:
        results: YOLOv8 result object (from model.predict or model(...))

    Returns:
        List of integer tracking IDs detected in the frame.
    """
    if len(results) == 0 or results[0].boxes is None:
        return []

    # Get all tracking IDs for detected objects (person, car)
    all_ids = [int(box.id.item()) for box in results[0].boxes if hasattr(box, 'id') and box.id is not None]

    return all_ids


def get_object_class_name(class_id):
    """
    Returns the class name for a given class ID.
    """
    return BASE_CLASS_NAMES.get(class_id, f"class_{class_id}")


def get_target_class_from_id(results, target_id):
    """
    Get the class ID of a specific target ID from YOLO results.
    Returns None if target not found.
    """
    if not results or not results[0].boxes or target_id is None:
        return None
        
    for box in results[0].boxes:
        if hasattr(box, 'id') and box.id is not None:
            box_id = int(box.id.item()) if hasattr(box.id, 'item') else int(box.id)
            if box_id == target_id:
                return int(box.cls.item())
    return None


def update_target_class(target_container, results):
    """
    Update the target_class in container once we identify what type of object the target is.
    Only supports person and car classes.
    """
    if not target_container.get("id") or not target_container["id"]:
        return
        
    target_ids = target_container["id"]
    if not target_ids:
        return
        
    # Try to identify the class of the first target ID
    target_id = target_ids[0] if isinstance(target_ids, list) else target_ids
    target_class = get_target_class_from_id(results, target_id)
    
    if target_class is not None and target_class in BASE_CLASS_NAMES:
        target_container["target_class"] = target_class
        class_name = get_object_class_name(target_class)
        logger.info(f"Target ID {target_id} identified as {class_name} (class {target_class})")
    else:
        # Target not found or not a supported class
        target_container["target_class"] = None


def get_object_info(results, target_id):
    """
    Returns information about a specific tracked object.
    
    Returns:
        dict with 'class_id', 'class_name', 'coords' or None if not found
    """
    if not target_id:
        return None
        
    for box in results[0].boxes:
        if hasattr(box, 'id') and box.id is not None:
            box_id = int(box.id.item()) if hasattr(box.id, 'item') else int(box.id)
            if box_id == target_id:
                class_id = int(box.cls.item())
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                return {
                    'class_id': class_id,
                    'class_name': get_object_class_name(class_id),
                    'coords': (x1, y1, x2, y2, center_x, center_y)
                }
    return None


def get_target_id_in_frame(results, target_ids):
    """
    Returns the first target ID found in the frame that is in target_ids list.
    Only returns IDs that are persons (class 0).

    Parameters:
        results: Detection results for the current frame.
        target_ids (list): IDs to search for.

    Returns:
        int or None: Matching target ID (person only) or None if not found.
    """
    ids_in_frame = get_ids_in_frame(results)
    logger.info(f"ids_in_frame: {ids_in_frame}")
    logger.info(f"target: {target_ids}")
    
    # Only allow person IDs (class 0)
    for box in results[0].boxes:
        if hasattr(box, 'id') and box.id is not None:
            box_id = int(box.id.item()) if hasattr(box.id, 'item') else int(box.id)
            if box_id in target_ids:
                # Check if it's a person (class 0)
                class_id = int(box.cls.item())
                if class_id == 0:
                    return box_id
    
    logger.error("Error - target id not in frame (or not a person)")
    return None

def find_id_in_frame(results, selected_id_container, persons_dict, all_persons_in_frame):
    """
    Checks if an ID is selected and processes tracking results to get coordinates.
    Stores the coordinates in selected_id_container["coords"] if found.
    If the ID is not found, resets selection and input thread status.
    Returns updated (id_selected).
    """
    is_id_found = False
    
    # Ensure all required keys exist in the container
    if "id_was_seen" not in selected_id_container:
        selected_id_container["id_was_seen"] = False
    if "lost_counter" not in selected_id_container:
        selected_id_container["lost_counter"] = 0
    if "followed_id" not in selected_id_container:
        selected_id_container["followed_id"] = None
    if "coords" not in selected_id_container:
        selected_id_container["coords"] = None
    
    target_id = get_target_id_in_frame(results, selected_id_container["history"])
    logger.debug(f"Initial target_id from frame: {target_id}")
    logger.debug(f"Current followed_id: {selected_id_container.get('followed_id')}")
    logger.debug(f"ID was seen before: {selected_id_container.get('id_was_seen')}")
    
    if not target_id and selected_id_container["id_was_seen"]:
        logger.debug(f"Target not found in frame, attempting template matching...")
        # Keep the old followed_id for template matching, don't overwrite it yet
        target_id = find_new_id(persons_dict, selected_id_container, all_persons_in_frame)
        logger.debug(f"Template matching result: {target_id}")

    # Only update followed_id after template matching attempt
    if target_id:
        logger.debug(f"Updating followed_id from {selected_id_container.get('followed_id')} to {target_id}")
    selected_id_container["followed_id"] = target_id
    coords = get_coordinates_by_id(results, target_id)
    
    if coords:
        selected_id_container["coords"] = coords  # Store or update the coordinates
        x1, y1, x2, y2, center_x, center_y = coords
        logger.info(f"Frame: Found ID {target_id} at ({x1}, {y1}), ({x2}, {y2}), center: ({center_x}, {center_y})")
        # You can add more logic here (e.g., draw something special or send commands)
        is_id_found = True
        selected_id_container["id_was_seen"] = True
        selected_id_container["lost_counter"] = 0
    elif selected_id_container["id_was_seen"] and selected_id_container["lost_counter"] < MAX_LOST_ID_FRAMES:
        selected_id_container["lost_counter"] += 1
        logger.info(f"Counter value: {selected_id_container['lost_counter']}")
        is_id_found = True
    else:
        logger.warning(f"Frame: ID {target_id} not found in this frame.")
        # Only clear followed_id if template matching also failed
        if not target_id:  # target_id is None, meaning template matching failed too
            selected_id_container["followed_id"] = None
        selected_id_container["id"] = []
        selected_id_container["coords"] = None  # Clear coordinates when target is lost
        logger.info("[INPUT] Please enter a new ID to select.")
        is_id_found = False
    
    return is_id_found
        

def is_alive(thread):
    if(thread is None): return False
    return thread.is_alive()


def draw_frame_addons(annotated_frame, coords, target_coords=None):
    # Draw rectangle
    cv2.rectangle(annotated_frame, RECT_TOP_LEFT(), RECT_BOTTOM_RIGHT(), (0, 0, 255), 2)
    
    if(coords != None):
        x1, y1, x2, y2, center_x, center_y = coords
        # Draw red point at selected ID coords
        cv2.circle(annotated_frame, (center_x, center_y), radius=3, color=(0, 0, 255), thickness=-1)
    
    if(target_coords != None):
        x1, y1, x2, y2, target_center_x, target_center_y = target_coords
        # Draw green point at target ID coords
        cv2.circle(annotated_frame, (target_center_x, target_center_y), radius=3, color=(0, 255, 0), thickness=-1)


def reset_hysteresis(histerezis_on):
    """Reset all hysteresis states"""
    histerezis_on["down"] = False
    histerezis_on["up"] = False
    histerezis_on["left"] = False
    histerezis_on["right"] = False

def control_drone(drone, coords, frame, histerezis_enabled, histerezis_on):

    if coords is None:
        # Reset hysteresis when no target is found
        reset_hysteresis(histerezis_on)
        drone.for_back_velocity = 0
        drone.yaw_velocity = 0
        return 0,0,0,0
    
    x1, y1, x2, y2, center_x, center_y = coords
    min_object_size = max(MIN_RECT_SIZE, min(y2-y1, x2-x1))
    rect_size['w'], rect_size['h'] = min_object_size, min_object_size
    logger.debug(f"Rectangle size: w={rect_size['w']}, h={rect_size['h']}")

    # Y-axis control with improved hysteresis
    hysteresis_margin = rect_size['h'] // 4  # 25% of rectangle height for hysteresis
    
    if histerezis_enabled:
        # Forward movement logic with hysteresis
        if center_y < RECT_TOP_LEFT()[1] - hysteresis_margin or histerezis_on["up"]:
            if center_y < RECT_TOP_LEFT()[1] + hysteresis_margin:  # Continue until well inside
                cv2.putText(frame, "FORWARD", 
                            (RECT_TOP_LEFT()[0] + 20, RECT_TOP_LEFT()[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 2)
                drone.for_back_velocity = FB_MOVING_VELOCITY
                histerezis_on["up"] = True
                histerezis_on["down"] = False
            else:
                histerezis_on["up"] = False
                drone.for_back_velocity = 0
        # Backward movement logic with hysteresis
        elif center_y > RECT_BOTTOM_RIGHT()[1] + hysteresis_margin or histerezis_on["down"]:
            if center_y > RECT_BOTTOM_RIGHT()[1] - hysteresis_margin:  # Continue until well inside
                cv2.putText(frame, "BACKWARD", 
                            (RECT_TOP_LEFT()[0] + 20, RECT_TOP_LEFT()[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 2)
                drone.for_back_velocity = -FB_MOVING_VELOCITY
                histerezis_on["down"] = True
                histerezis_on["up"] = False
            else:
                histerezis_on["down"] = False
                drone.for_back_velocity = 0
        else:
            drone.for_back_velocity = 0
            histerezis_on["up"] = False
            histerezis_on["down"] = False
    else:
        # Simple logic without hysteresis
        if center_y < RECT_TOP_LEFT()[1]:
            cv2.putText(frame, "FORWARD", 
                        (RECT_TOP_LEFT()[0] + 20, RECT_TOP_LEFT()[1] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 2)
            drone.for_back_velocity = FB_MOVING_VELOCITY
        elif center_y > RECT_BOTTOM_RIGHT()[1]:
            cv2.putText(frame, "BACKWARD", 
                        (RECT_TOP_LEFT()[0] + 20, RECT_TOP_LEFT()[1] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 2)
            drone.for_back_velocity = -FB_MOVING_VELOCITY
        else:
            drone.for_back_velocity = 0


    # X-axis control with improved hysteresis
    hysteresis_margin_x = rect_size['w'] // 4  # 25% of rectangle width for hysteresis
    
    if histerezis_enabled:
        # Left rotation logic with hysteresis
        if center_x < RECT_TOP_LEFT()[0] - hysteresis_margin_x or histerezis_on["left"]:
            if center_x < RECT_TOP_LEFT()[0] + hysteresis_margin_x:  # Continue until well inside
                cv2.putText(frame, "<- ROTATE LEFT <-",
                    (RECT_TOP_LEFT()[0] - 110, (RECT_TOP_LEFT()[1] + RECT_BOTTOM_RIGHT()[1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2)
                drone.yaw_velocity = -YAW_MOVING_VELOCITY
                histerezis_on["left"] = True
                histerezis_on["right"] = False
            else:
                histerezis_on["left"] = False
                drone.yaw_velocity = 0
        # Right rotation logic with hysteresis
        elif center_x > RECT_BOTTOM_RIGHT()[0] + hysteresis_margin_x or histerezis_on["right"]:
            if center_x > RECT_BOTTOM_RIGHT()[0] - hysteresis_margin_x:  # Continue until well inside
                cv2.putText(frame, "-> ROTATE RIGHT ->",
                            (RECT_BOTTOM_RIGHT()[0] + 20, (RECT_TOP_LEFT()[1] + RECT_BOTTOM_RIGHT()[1]) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2)
                drone.yaw_velocity = YAW_MOVING_VELOCITY
                histerezis_on["right"] = True
                histerezis_on["left"] = False
            else:
                histerezis_on["right"] = False
                drone.yaw_velocity = 0
        else:
            drone.yaw_velocity = 0
            histerezis_on["left"] = False
            histerezis_on["right"] = False
    else:
        # Simple logic without hysteresis
        if center_x < RECT_TOP_LEFT()[0]:
            cv2.putText(frame, "<- ROTATE LEFT <-",
                (RECT_TOP_LEFT()[0] - 110, (RECT_TOP_LEFT()[1] + RECT_BOTTOM_RIGHT()[1]) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2)
            drone.yaw_velocity = -YAW_MOVING_VELOCITY
        elif center_x > RECT_BOTTOM_RIGHT()[0]:
            cv2.putText(frame, "-> ROTATE RIGHT ->",
                        (RECT_BOTTOM_RIGHT()[0] + 20, (RECT_TOP_LEFT()[1] + RECT_BOTTOM_RIGHT()[1]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2)
            drone.yaw_velocity = YAW_MOVING_VELOCITY
        else:
            drone.yaw_velocity = 0

    return drone.left_right_velocity, drone.for_back_velocity, drone.up_down_velocity, drone.yaw_velocity
    




def move_drone(drone):
    key = cv2.waitKey(1) & 0xFF

    if key == ord('t'):
        drone.takeoff()
    elif key == ord('l'):
        drone.land()
    elif key == ord('w'):
        drone.move_forward(60)
    elif key == ord('s'):
        drone.move_back(30)
    elif key == ord('a'):
        drone.move_left(30)
    elif key == ord('d'):
        drone.move_right(30)
    elif key == ord('u'):  # UP arrow
        logger.debug("up")
        drone.move_up(60)
    elif key == ord('j'):  # DOWN arrow
        drone.move_down(30)
    elif key == 81:  # LEFT arrow
        drone.rotate_counter_clockwise(30)
    elif key == 83:  # RIGHT arrow
        drone.rotate_clockwise(30)

    
def save_segmented_persons(results, frame):
    # Make sure the output folder exists
    output_dir = "persons_segmented"
    os.makedirs(output_dir, exist_ok=True)

    for result in results:
        # For each detection

        # checks if any of the data is None, if one of them is None goes to next result
        if (not result.boxes.cls) or (not result.masks.data) or (not result.boxes.id) or (not result.boxes.xyxy):
            continue

        # save segmented person in folder output_dir
        for mask, cls, track_id, box in zip(result.masks.data, result.boxes.cls, result.boxes.id, result.boxes.xyxy):
            if int(cls) == 0:  # 0 = person (COCO)
                x1, y1, x2, y2 = map(int, box.tolist())
                person_mask = mask.cpu().numpy()
                mask_uint8 = (person_mask * 255).astype("uint8")

                # Crop both mask and image to bbox
                mask_cropped = mask_uint8[y1:y2, x1:x2]
                # Frame is in RGB; convert to BGR for saving
                person_crop = frame[y1:y2, x1:x2]

                # Apply the mask: make background black
                person_crop_masked = cv2.bitwise_and(person_crop, person_crop, mask=mask_cropped)

                # Save with id in filename
                save_path = os.path.join(output_dir, f"person_id_{int(track_id)}.png")
                # Convert from RGB to BGR for saving
                person_crop_masked_bgr = cv2.cvtColor(person_crop_masked, cv2.COLOR_RGB2BGR)
                logger.info("[SAVING]Saving frame")
                cv2.imwrite(save_path, person_crop_masked_bgr)


def extract_all_visible_objects(results, original_frame):
    """
    Extract all visible persons (class 0 only) from the current frame for template matching.
    Cars are detected but not extracted for template matching.
    
    Returns:
        dict: {person_id: {'image': cropped_image, 'class_id': class_id}}
    """
    if results[0].boxes is None or results[0].boxes.id is None or results[0].masks is None:
        return {}
    
    all_objects = {}
    track_ids = results[0].boxes.id.int().cpu().tolist()
    class_ids = results[0].boxes.cls.int().cpu().tolist()
    
    for i, (track_id, class_id) in enumerate(zip(track_ids, class_ids)):
        # Only extract persons (class 0) for template matching
        if class_id != 0:
            continue
            
        try:
            mask = results[0].masks[i]
            binary_mask = mask.data[0].cpu().numpy().astype("uint8")
            extracted_object = cv2.bitwise_and(original_frame, original_frame, mask=binary_mask)
            
            box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            cropped_object = extracted_object[y1:y2, x1:x2]
            
            if cropped_object.size > 0:
                all_objects[track_id] = {
                    'image': cropped_object,
                    'class_id': class_id
                }
        except Exception as e:
            logger.warning(f"Failed to extract person ID {track_id}: {e}")
            continue
    
    return all_objects


def update_objects_dict(all_objects_in_frame, objects_dict, max_recent_extractions):
    """
    Update the objects dictionary with new object images.
    Similar to update_persons_dict but for all object types.
    """
    for object_id, object_info in all_objects_in_frame.items():
        object_image = object_info['image']
        
        if object_id not in objects_dict:
            objects_dict[object_id] = []
        
        # Add the new image to the list
        objects_dict[object_id].append(object_image)
        
        # Keep only the most recent extractions
        if len(objects_dict[object_id]) > max_recent_extractions:
            objects_dict[object_id] = objects_dict[object_id][-max_recent_extractions:]


def find_new_id(template_dict, selected_id_container, all_objects_in_frame):
    selected_id_to_track = selected_id_container["followed_id"]
    templates = template_dict.get(selected_id_to_track, [])
    replacement_id = -1
    
    logger.info(f"[TEMPLATE_MATCHING] Lost ID: {selected_id_to_track}")
    logger.info(f"[TEMPLATE_MATCHING] Available templates: {len(templates)}")
    logger.info(f"[TEMPLATE_MATCHING] Current objects in frame: {list(all_objects_in_frame.keys()) if all_objects_in_frame else 'None'}")
    
    if templates and all_objects_in_frame:
        logger.warning(f"Target ID {selected_id_to_track} lost. Attempting to re-acquire...")
        logger.info(f"[TEMPLATE_MATCHING] === ATTEMPTING RE-IDENTIFICATION ===")
        
        # Convert all_objects_in_frame format for template matching
        objects_for_matching = {}
        if isinstance(list(all_objects_in_frame.values())[0], dict):
            # New format: {id: {'image': img, 'class_id': cls}}
            objects_for_matching = {obj_id: obj_info['image'] for obj_id, obj_info in all_objects_in_frame.items()}
        else:
            # Old format: {id: image}
            objects_for_matching = all_objects_in_frame
        
        replacement_id = template_matching.find_best_match(objects_for_matching, templates)
        if replacement_id != -1:
            logger.info(f"✓ Re-acquired target! Old ID: {selected_id_to_track}, New ID: {replacement_id}")
            selected_id_container["followed_id"] = replacement_id
            selected_id_container["history"].append(replacement_id)
            return replacement_id
        else:
            logger.info(f"[TEMPLATE_MATCHING] ❌ No suitable match found (all scores above threshold)")
            logger.info(f"[TEMPLATE_MATCHING] Current threshold: {template_matching.THRESHOLD_BEST_MATCH}")
            logger.info(f"[TEMPLATE_MATCHING] 💡 To tune threshold, check the scores above and adjust THRESHOLD_BEST_MATCH in template_matching.py")
    else:
        if not templates:
            logger.info(f"[TEMPLATE_MATCHING] No templates available for ID {selected_id_to_track}")
        if not all_objects_in_frame:
            logger.info(f"[TEMPLATE_MATCHING] No objects detected in current frame")
    
    return None


def get_coords(container):
    coords = container.get("coords")
    lost_counter = container.get("lost_counter", 0)

    if coords and lost_counter > 0:
        coords = None

    return coords

def init_input_thread(selected_id_container, user_getter_thread):
    if((not selected_id_container["id"]) and not is_alive(user_getter_thread)):
        user_getter_thread = init_id_getter_thread(selected_id_container)
    return user_getter_thread


class PerformanceMonitor:
    def __init__(self):
        self.frame_times = []
        self.yolo_times = []
        self.template_times = []
        self.last_fps_print = time.time()
        self.frame_count = 0
        
    def start_frame(self):
        self.frame_start = time.time()
        
    def start_yolo(self):
        self.yolo_start = time.time()
        
    def end_yolo(self):
        if hasattr(self, 'yolo_start'):
            self.yolo_times.append(time.time() - self.yolo_start)
            
    def start_template(self):
        self.template_start = time.time()
        
    def end_template(self):
        if hasattr(self, 'template_start'):
            self.template_times.append(time.time() - self.template_start)
            
    def end_frame(self):
        if hasattr(self, 'frame_start'):
            self.frame_times.append(time.time() - self.frame_start)
            self.frame_count += 1
            
        # Print stats every 5 seconds
        if time.time() - self.last_fps_print > 5.0:
            self.print_stats()
            self.last_fps_print = time.time()
            
    def print_stats(self):
        if self.frame_times:
            avg_fps = 1.0 / (sum(self.frame_times[-30:]) / min(30, len(self.frame_times)))
            avg_frame_time = sum(self.frame_times[-30:]) / min(30, len(self.frame_times)) * 1000
            
            yolo_time = 0
            template_time = 0
            if self.yolo_times:
                yolo_time = sum(self.yolo_times[-10:]) / min(10, len(self.yolo_times)) * 1000
            if self.template_times:
                template_time = sum(self.template_times[-10:]) / min(10, len(self.template_times)) * 1000
                
            logger.info(f"[PERFORMANCE] FPS: {avg_fps:.1f} | Frame: {avg_frame_time:.1f}ms | YOLO: {yolo_time:.1f}ms | Templates: {template_time:.1f}ms")



def main():
    # Print to terminal only - inform user where logs are saved
    print("="*60)
    print(f"Drone Control System Starting...")
    print(f"Log file: {LOG_FILENAME}")
    print("="*60)
    print()
    
    logger.info("="*60)
    logger.info("DRONE CONTROL SYSTEM STARTED")
    logger.info(f"Timestamp: {TIMESTAMP}")
    logger.info(f"Video files: {OUTPUT_FILENAME}, {OUTPUT_ANNOTATED_FILENAME}")
    logger.info(f"Log file: {LOG_FILENAME}")
    logger.info("="*60)

    # thread for id selection 
    selected_id_container = {
        "id": [], 
        "history": [], 
        "id_was_seen": False, 
        "lost_counter": 0, 
        "followed_id": None, 
        "coords": None
    }
    selected_target_container = {
        "id": [],  # Target uses list like selected_id for consistency
        "history": [], 
        "id_was_seen": False, 
        "lost_counter": 0, 
        "followed_id": None, 
        "coords": None,
        "target_class": None  # Will be set once we identify the target object type
    }

    persons_dict = {}  # Template dictionary for persons (following)
    targets_dict = {}  # Template dictionary for targets (navigation destinations)
    model = YOLO("./yolo_models/yolo11s-seg.pt")
    
    # thread for id selection 
    user_getter_thread = None
    target_getter_thread = None

    model = YOLO("./yolo_models/yolo11s-seg.pt")

    drone = start_drone()
    out_clean = start_recording(OUTPUT_FILENAME)
    out_annotated = start_recording(OUTPUT_ANNOTATED_FILENAME)

    launch_drone(drone)

    # Performance monitoring
    perf_monitor = PerformanceMonitor() if ENABLE_PERFORMANCE_MONITORING else None
    
    # Frame processing variables
    frame_counter = 0
    yolo_frame_counter = 0
    last_yolo_results = None

    histerezis_on = {
        "down": False,
        "up": False,
        "left": False,
        "right": False
    }
    
    logger.info(f"[OPTIMIZATION] YOLO processing every {YOLO_FRAME_SKIP} frames")
    logger.info(f"[OPTIMIZATION] Template matching every {TEMPLATE_MATCHING_FRAME_SKIP} YOLO frames")
    
    try:
        while True:
            if perf_monitor:
                perf_monitor.start_frame()
                
            frame_counter += 1
            frame = get_frame(drone)
            if frame is None:
                logger.warning("No frame received.")
                continue
            
            # Determine if we should process YOLO on this frame
            should_process_yolo = (frame_counter % YOLO_FRAME_SKIP == 1)
            should_update_templates = (yolo_frame_counter % TEMPLATE_MATCHING_FRAME_SKIP == 1)
            
            if should_process_yolo:
                if perf_monitor:
                    perf_monitor.start_yolo()
                    
                yolo_frame_counter += 1
                # logger.info(f"[OPTIMIZATION] Processing YOLO frame {yolo_frame_counter}")
                
                # Always detect person and car classes
                class_names = [get_object_class_name(cls) for cls in BASE_DETECTION_CLASSES]
                logger.info(f"[OPTIMIZATION] Detecting classes: {class_names} {BASE_DETECTION_CLASSES}")
                
                results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml", classes=BASE_DETECTION_CLASSES)
                last_yolo_results = results
                annotated_frame = results[0].plot()
                
                # Update target class information if we have a target
                if TARGET_TRACKING_ENABLED and selected_target_container["id"]:
                    update_target_class(selected_target_container, results)
                
                if perf_monitor:
                    perf_monitor.end_yolo()
                
                # Template matching (only on selected YOLO frames)
                if should_update_templates:
                    if perf_monitor:
                        perf_monitor.start_template()
                        
                    logger.info(f"[OPTIMIZATION] Updating templates (frame {yolo_frame_counter})")
                    
                    # 1. Extract all persons visible in the current frame (for following)
                    all_persons_in_frame = template_matching.extract_all_visible_persons(results, frame)
                    
                    # 2. Extract all objects visible in the current frame (for targets)
                    all_objects_in_frame = extract_all_visible_objects(results, frame)
                    
                    # Debug: Show extracted objects
                    if all_persons_in_frame:
                        logger.info(f"[TEMPLATE_MATCHING] Extracted {len(all_persons_in_frame)} persons: {list(all_persons_in_frame.keys())}")
                    if all_objects_in_frame:
                        objects_str = ", ".join([f"ID{id}({get_object_class_name(info['class_id'])})" for id, info in all_objects_in_frame.items()])
                        logger.info(f"[TEMPLATE_MATCHING] Extracted objects: {objects_str}")
                    
                    # 3. Update template dictionaries
                    if all_persons_in_frame:
                        template_matching.update_persons_dict(all_persons_in_frame, persons_dict, template_matching.MAX_RECENT_EXTRACTIONS)
                        # Debug: Show current template counts
                        template_counts = {pid: len(templates) for pid, templates in persons_dict.items()}
                        if template_counts:
                            logger.info(f"[TEMPLATE_MATCHING] Person template counts: {template_counts}")
                    
                    if all_objects_in_frame:
                        update_objects_dict(all_objects_in_frame, targets_dict, template_matching.MAX_RECENT_EXTRACTIONS)
                        # Debug: Show target template counts
                        target_template_counts = {oid: len(templates) for oid, templates in targets_dict.items()}
                        if target_template_counts:
                            logger.info(f"[TEMPLATE_MATCHING] Target template counts: {target_template_counts}")
                    
                    if perf_monitor:
                        perf_monitor.end_template()
                else:
                    # Use empty dictionaries when not updating templates
                    all_persons_in_frame = {}
                    all_objects_in_frame = {}
            else:
                # Use last YOLO results and create annotated frame from original
                results = last_yolo_results
                annotated_frame = frame.copy()
                all_persons_in_frame = {}
                all_objects_in_frame = {}
                
                # Draw previous detections on skipped frames if available
                if results is not None and results[0].boxes is not None:
                    annotated_frame = results[0].plot()

            user_getter_thread = init_input_thread(selected_id_container, user_getter_thread)

            # Initialize variables
            is_id_found = False
            is_target_found = False

            if(selected_id_container["id"]):
                # Target tracking logic (controlled by TARGET_TRACKING_ENABLED flag)
                if TARGET_TRACKING_ENABLED:
                    # Use different input thread for target (asks for target objects)
                    if((not selected_target_container["id"]) and not is_alive(target_getter_thread)):
                        target_getter_thread = threading.Thread(target=target_thread, args=(selected_target_container,))
                        target_getter_thread.daemon = True
                        target_getter_thread.start()
                    #follow target - now with template matching support
                    is_target_found = find_id_in_frame(results, selected_target_container, targets_dict, all_objects_in_frame)
                
                # saves the coordinates in container IMPORTANT DONT COMMENT
                is_id_found = find_id_in_frame(results, selected_id_container, persons_dict, all_persons_in_frame)
                # TODO: pivot to new function that will take care of scanning the potential target objects

            # Display detected objects (persons and cars)
            if results[0].boxes is not None:
                detected_objects = {}
                for box in results[0].boxes:
                    if hasattr(box, 'id') and box.id is not None:
                        obj_id = int(box.id.item())
                        class_id = int(box.cls.item())
                        class_name = get_object_class_name(class_id)
                        detected_objects[obj_id] = class_name
                
                if detected_objects:
                    objects_str = ", ".join([f"ID{id}({cls})" for id, cls in detected_objects.items()])
                    logger.info(f"[DETECTED] {objects_str}")

            logger.info(f"[Selected ID] {selected_id_container['id']}")
            if TARGET_TRACKING_ENABLED:
                logger.info(f"[Target ID] {selected_target_container['id']}")
            
            # Get target coordinates if target tracking is enabled
            target_coords = None
            if(is_target_found):
                target_coords = get_coords(selected_target_container)
                if target_coords is not None:
                    x1, y1, x2, y2, target_center_x, target_center_y = target_coords
                    # Get target object info
                    target_info = get_object_info(results, selected_target_container.get("followed_id"))
                    if target_info:
                        logger.info(f"Target ({target_info['class_name']}) Center coordinates: ({target_center_x}, {target_center_y})")
                    else:
                        logger.info(f"Target Center coordinates: ({target_center_x}, {target_center_y})")
                else:
                    logger.warning("Target found but coordinates are None (target recently lost)")
            #TODO - add path finding between user and target

            # Get coordinates directly (no interpolation)
            coords = get_coords(selected_id_container)

            drone.left_right_velocity, drone.for_back_velocity, drone.up_down_velocity, drone.yaw_velocity = control_drone(drone, coords, annotated_frame, HISTEREZIS_ENABLED, histerezis_on)
            
            # keep alive command
            drone.send_control_command("command")

            # SEND VELOCITY VALUES TO TELLO
            
            if drone.send_rc_control:
                drone.send_rc_control(drone.left_right_velocity, drone.for_back_velocity, drone.up_down_velocity, drone.yaw_velocity)

            move_drone(drone)

            # Draw frame addons with both selected ID (red) and target ID (green) points
            draw_frame_addons(annotated_frame, coords, target_coords)

            cv2.imshow('Tello Video Feed', annotated_frame)
            out_annotated.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            out_clean.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


            if should_quit():
                shutdown_drone(drone)
                break

            if perf_monitor:
                perf_monitor.end_frame()

            time.sleep(1/30)  # reduce CPU load, aiming for 30fps

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Shutting down...")
        logger.info("Interrupted by user.")


    exit([out_clean, out_annotated], drone)
    
    print()
    print("="*60)
    print("Drone Control System Stopped")
    print(f"Log file saved: {LOG_FILENAME}")
    print("="*60)

    return


if __name__ == '__main__':
    main()			