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

- remove backpack and umbrella from detection DONE
- dont put selected and target on cars DONE
- dont make template matching with diff type DONE
- put point on target id DONE
- check template matching on target id that it works 
- make clearer prompt for input of target and selected id DONE
- do logs also in a file , and check if it gets updated in live DONE
- add penalty to template matching basded on distance from selected id
- stop working with dicts and start working with classes for better readability and maintainability
- use cursor to improve code readability
- fix RGB to BGR in videos
"""

import cv2
import time
from djitellopy import Tello
from ultralytics import YOLO
import threading
import queue
import numpy as np
import os
import logging
from datetime import datetime 
from testsfolder import template_matching
from testsfolder import RRTStar_New as rrt
from testsfolder.kelman_implementation import predict_trajectory
import subprocess


class Person:
    """
    Represents a tracked person with all tracking-related information.
    Used for both selected person (to follow) and target person (to navigate to).
    """
    def __init__(self, person_type):
        self.id = []  # List of IDs entered by user
        self.history = []  # History of all IDs this person has been tracked as
        self.id_was_seen = False  # Whether this person has been seen at least once
        self.lost_counter = 0  # Counter for consecutive frames where person is lost
        self.followed_id = None  # Currently followed ID (may differ from input after template matching)
        self.coords = None  # Current bounding box coordinates (x1, y1, x2, y2, center_x, center_y)
        self.last_seen_center_coords = None  # Last known center coordinates when lost
        self.center_history = []  # History of center coordinates for the last N frames
        self.predicted_trajectory = []  # Kalman filter predicted positions when person is lost
        self.best_match_candidates = {}  # Template matching candidates: {id: score}
        self.lost_rounds = 0  # Number of template matching rounds completed while lost
        self.type = person_type # type of person: USER or TARGET
        self.other = None

    def reset_details_on_new_id(self):
        """
        Reset all tracking details to initial state when replacing id with a new one.
        """
        self.lost_counter = 0
        self.id_was_seen = False

    def reset_when_seen(self):
        """
        Reset all tracking details to initial state when person is seen.
        """
        self.id_was_seen = True
        self.lost_counter = 0
        self.best_match_candidates = {}
        self.last_seen_center_coords = None
        self.predicted_trajectory = []



TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Create videos directory if it doesn't exist
VIDEOS_DIR = "videos"
os.makedirs(VIDEOS_DIR, exist_ok=True)
OUTPUT_FILENAME = os.path.join(VIDEOS_DIR, f'clean_video_{TIMESTAMP}.avi')
OUTPUT_ANNOTATED_FILENAME = os.path.join(VIDEOS_DIR, f'annotated_video_{TIMESTAMP}.avi')
OUTPUT_BW_SEGMENTED_FILENAME = os.path.join(VIDEOS_DIR, f'black_white_segmented_{TIMESTAMP}.avi')
LOG_FILENAME = os.path.join(VIDEOS_DIR, f'drone_log_{TIMESTAMP}.log')

# Configure logging - this will be used by all modules including template_matching
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILENAME),
    ],
    force=True  # Override any existing logging configuration
)
logger = logging.getLogger(__name__)

# Ensure template_matching module logs are also captured
logging.getLogger('testsfolder.template_matching').setLevel(logging.DEBUG)
logging.getLogger('template_matching').setLevel(logging.DEBUG)


WIDTH = 640
HEIGHT = 480
selected_id = None

RECT_CENTER = (WIDTH // 2, HEIGHT // 2 + 120)
RECT_SIZE = {"w": 70, "h": 70}
MIN_RECT_SIZE = 50
CONST_RECT_TOP_LEFT = (RECT_CENTER[0] - RECT_SIZE['w'] // 2, RECT_CENTER[1] - RECT_SIZE['h'] // 2)
CONST_RECT_BOTTOM_RIGHT = (RECT_CENTER[0] + RECT_SIZE['w'] // 2, RECT_CENTER[1] + RECT_SIZE['h'] // 2)
DYNAMIC_RECT = False
SAVE_FILE = True
HISTEREZIS_ENABLED = True # Fixed hysteresis logic to prevent ping-pong behavior
TARGET_TRACKING_ENABLED = True # Set to True to enable target selection and tracking logic


# === PERFORMANCE OPTIMIZATION SETTINGS ===
TEMPLATE_MATCHING_FRAME_SKIP = 1  # Update templates every Nth YOLO frame (reduces template collection frequency)
ENABLE_PERFORMANCE_MONITORING = True  # Show FPS and processing time stats

# === RRT* PATH PLANNING SETTINGS ===
ENABLE_RRT_PATH = True  # Enable RRT* path planning from selected to target
RRT_FRAME_SKIP = 5  # Calculate RRT* path every N frames (1=every frame, 5=every 5th frame)
RRT_MAX_ITER = 1000  # Maximum iterations for RRT* (lower = faster but less optimal)
RRT_DELTA = 15  # Step size for RRT* tree expansion
RRT_RADIUS = 30  # Radius for rewiring in RRT*
RRT_PATH_COLOR = (255, 0, 255)  # Magenta/Purple color for path (BGR)

#RECT_TOP_LEFT = (RECT_CENTER[0] - RECT_W // 2, RECT_CENTER[1] - RECT_H // 2)
#RECT_BOTTOM_RIGHT = (RECT_CENTER[0] + RECT_W // 2, RECT_CENTER[1] + RECT_H // 2)

BLACK = (0, 0, 0)

YAW_MOVING_VELOCITY = 5
FB_MOVING_VELOCITY = 5
DRONE_START_HEIGHT = 400 # 200 is good for outside, 70 for inside , 450 good outside
MAX_LOST_ID_FRAMES = 20
COORDINATE_HISTORY_SIZE = 30  # Number of frames to keep coordinate history for tracked persons

TEMPLATE_M_WAITING_FRAMES_WINDOW = 15  # Number of lost frames to wait before checking best template match
MAX_LOST_ROUNDS = 3

# === YOLO DETECTION CLASSES ===
# Always detect these classes
BASE_DETECTION_CLASSES = [0, 2]  # 0: person, 2: car
BASE_CLASS_NAMES = {
    0: "person",
    2: "car"
}
#------
NOTEPAD_PATH = r"C:\Program Files (x86)\Notepad++\notepad++.exe"

#------- Functions -------#
def open_log_in_editor(log_filename):
    """
    Opens a new PowerShell window showing the last 20 lines of the log file,
    continuously updating (like 'tail -f' in Linux).
    """
    # Make sure the log file exists
    if not os.path.exists(log_filename):
        logger.error(f"Log file not found: {log_filename}")
        return

    # PowerShell command to show last 20 lines and follow the file
    powershell_cmd = f'powershell -NoExit -Command "Get-Content -Path \'{log_filename}\' -Tail 20 -Wait"'

    try:
        # Open PowerShell in a new window
        subprocess.Popen(["powershell", "-NoExit", "-Command", f"Get-Content -Path '{log_filename}' -Tail 20 -Wait"])
        logger.info(f"Opened PowerShell tail view for: {log_filename}")
    except Exception as e:
        logger.warning(f"Failed to open PowerShell tail view: {e}")

def RECT_TOP_LEFT():
    if(DYNAMIC_RECT is False): 
        return CONST_RECT_TOP_LEFT
    return (RECT_CENTER[0] - RECT_SIZE['w'] // 2, RECT_CENTER[1] - RECT_SIZE['h'] // 2)

def RECT_BOTTOM_RIGHT():
    if(DYNAMIC_RECT is False): 
        return CONST_RECT_BOTTOM_RIGHT
    return (RECT_CENTER[0] + RECT_SIZE['w'] // 2, RECT_CENTER[1] + RECT_SIZE['h'] // 2)

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
    
def is_person_class(results, id):
    """
    Checks if the ID is a person class in the results.
    Returns True if the ID is a person class, False otherwise.
    """
    for box in results[0].boxes:
        if hasattr(box, 'id') and box.id is not None:
            box_id = int(box.id.item()) if hasattr(box.id, 'item') else int(box.id)
            if box_id == id:
                return int(box.cls.item()) == 0

    return False

def input_thread(current_person, other_person, results):
    user_input = input(f"[INPUT] Enter {current_person.type.capitalize()} Person ID to follow: ")
    try:
        selected_id = int(user_input)
        
        # Check if this ID is the same as target ID
        if other_person.id and selected_id in other_person.history:
            print(f"[ERROR] {current_person.type} ID {selected_id} is the same as {other_person.type} ID. USER and TARGET must be different.")
            logger.warning(f"Rejected {current_person.type} ID {selected_id} - same as {other_person.type} ID")
            return
        
        if not is_person_class(results, selected_id):
            logger.error(f"ID {selected_id} is not a person. Skipping tracking.")
            return
        
        current_person.id.append(selected_id)
        current_person.history.append(selected_id)
        current_person.reset_details_on_new_id()
        logger.info(f"{current_person.type} Person ID: {current_person.id}")

    except ValueError:
        logger.warning(f"Invalid ID entered for {current_person.type} person.")
        current_person.id = []


# TODO check its generic func
def init_id_getter_thread(selected_person, target_person, results):
    """
    Initializes and starts a background thread to continuously get user input for selected ID.
    """
    thread = threading.Thread(target=input_thread, args=(selected_person, target_person, results))
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


def get_selected_id_in_frame(results, person):
    """
    Returns the first target ID found in the frame that is in target_ids list.
    Only returns IDs that are persons (class 0).

    Parameters:
        results: Detection results for the current frame.
        target_ids (list): IDs to search for.

    Returns:
        int or None: Matching target ID (person only) or None if not found.
    """
    target_ids = person.history
    ids_in_frame = get_ids_in_frame(results)
    logger.info(f"ids found in current frame: {ids_in_frame}")
    logger.info(f"{person.type} history ids are: {target_ids}")
    
    # ids detected that are in target ids
    ids_detected = []
    # Only allow person IDs (class 0)
    for box in results[0].boxes:
        if hasattr(box, 'id') and box.id is not None:
            box_id = int(box.id.item()) if hasattr(box.id, 'item') else int(box.id)

            if box_id in target_ids:
                # Check if it's a person (class 0)
                class_id = int(box.cls.item())
                if class_id == 0:
                    ids_detected.append(box_id)
    
    if len(ids_detected) > 0:
        return min(ids_detected)
    
    logger.error("Error - target id not in frame (or not a person)")
    return None

def find_id_in_frame(results, person, persons_dict, all_persons_in_frame, all_persons_centers_dict, exclude_ids=None, tracking_type="Selected"):
    """
    Checks if an ID is selected and processes tracking results to get coordinates.
    Make sure the id that is being tracked is a person.
    Stores the coordinates in person.coords if found.
    If the ID is not found, resets selection and input thread status.
    Returns updated (id_selected).
    
    Args:
        person: Person object containing tracking information
        exclude_ids: List of IDs to exclude from template matching (e.g., the opposite tracked ID)
        tracking_type: String "Selected" or "Target" for logging purposes
    """
    is_id_found = False
    
    if exclude_ids is None:
        exclude_ids = []

    selected_id = get_selected_id_in_frame(results, person)

    # enters here in case selected_id that appeared in past frames was lost
    if (not selected_id) and person.id_was_seen:
        # in case window is over, take the best template matching result below threshold.
        # if no result found, go to next round. if max rounds reached, enter the selected_id manually.
        if person.lost_counter % TEMPLATE_M_WAITING_FRAMES_WINDOW == 0 and person.lost_counter != 0:
            person.lost_rounds += 1
            followed_id = person.followed_id
            logger.info(f"[{tracking_type}] Template matching window expired for ID {followed_id}. Round {person.lost_rounds}/{MAX_LOST_ROUNDS}")
            # Get current IDs in frame for filtering candidates
            current_ids_in_frame = get_ids_in_frame(results)
            # Note: choose_best_match will clear best_match_candidates automatically
            selected_id = choose_best_match(person, current_ids_in_frame)
            if selected_id is None:
                logger.info(f"[{tracking_type}] No suitable template match found, going to next round.")
                #if max rounds reached, enter the selected_id manually
                if person.lost_rounds == MAX_LOST_ROUNDS:
                    # Prompt user for new ID input
                    logger.warning(f"[{tracking_type}] All {MAX_LOST_ROUNDS} rounds finished without a match. Manual ID entry required.")
            else:
                logger.info(f"[{tracking_type}] Best match ID {selected_id} passed threshold!")
                # Update person with the new followed ID
                person.followed_id = selected_id
                person.history.append(selected_id)
                person.center_history = []
                person.lost_rounds = 0
                logger.info(f"[{tracking_type}] ✓ Re-acquired! Old ID: {followed_id}, New ID: {selected_id}")
        # in case window isnt over, do template matching and append scores to a dict
        else:
            logger.debug(f"Target not found in frame, attempting template matching...")
            # Keep the old followed_id for template matching, don't overwrite it yet
            find_new_id(persons_dict, person, all_persons_in_frame, all_persons_centers_dict, exclude_ids)
            logger.debug(f"Template matching result: {selected_id}")

    # Only update followed_id when we have a valid selected_id (don't overwrite with None)
    if selected_id:
        logger.debug(f"Updating followed_id from {person.followed_id} to {selected_id}")
        person.followed_id = selected_id

    logger.debug(f"Initial selected_id from frame: {selected_id}")
    logger.debug(f"Current followed_id: {person.followed_id}")
    logger.debug(f"ID was seen before: {person.id_was_seen}")

    # Use followed_id to get coordinates (this persists even when selected_id is None during template matching)
    coords = get_coordinates_by_id(results, person.followed_id)
    
    # if id is found -> we have coords of this person
    logger.debug(f"[{tracking_type}] coords: {coords}")
    logger.debug(f"[{tracking_type}] id_was_seen: {person.id_was_seen}")
    logger.debug(f"[{tracking_type}] lost_rounds < MAX_LOST_ROUNDS: {person.lost_rounds} < {MAX_LOST_ROUNDS} = {person.lost_rounds < MAX_LOST_ROUNDS}")
    
    if coords:
        person.coords = coords  # Store or update the coordinates
        x1, y1, x2, y2, center_x, center_y = coords
        followed_id = person.followed_id
        logger.info(f"[{tracking_type}] Frame: Found ID {followed_id} at ({x1}, {y1}), ({x2}, {y2}), center: ({center_x}, {center_y})")
        
        # Update coordinate history
        person.center_history.append((center_x, center_y))
        # Keep only the last COORDINATE_HISTORY_SIZE frames
        if len(person.center_history) > COORDINATE_HISTORY_SIZE:
            person.center_history.pop(0)
        
        # You can add more logic here (e.g., draw something special or send commands)
        is_id_found = True
        person.reset_when_seen()
    elif person.id_was_seen and person.lost_rounds < MAX_LOST_ROUNDS:
        person.lost_counter += 1
        person.coords = None  # Clear coordinates immediately when person is lost
        followed_id = person.followed_id
        logger.info(f"[{tracking_type}] ID {followed_id} lost. Counter: {person.lost_counter}/{TEMPLATE_M_WAITING_FRAMES_WINDOW}, Round: {person.lost_rounds}/{MAX_LOST_ROUNDS}")
        logger.info(f"[{tracking_type}] Current best_match_candidates: {person.best_match_candidates}")
        
        # Use Kalman filter prediction when person is lost
        if person.lost_counter == 1:
            # First time person is lost - generate trajectory predictions
            center_history = person.center_history
            if center_history and len(center_history) > 0:
                logger.info(f"[KALMAN] Generating trajectory prediction with {len(center_history)} historical points")
                predicted_positions = predict_trajectory(center_history)
                person.predicted_trajectory = predicted_positions
                
                if predicted_positions:
                    logger.info(f"[KALMAN] Generated {len(predicted_positions)} predicted positions")
                else:
                    logger.info(f"[KALMAN] Not enough history (need >5 frames), using last known position")
            else:
                logger.info(f"[KALMAN] No center history available for prediction")
                person.predicted_trajectory = []
        
        # Update last_seen_center_coords with predicted trajectory or last known position
        if person.predicted_trajectory:
            # Pop the first predicted coordinate and use it
            predicted_pos = person.predicted_trajectory.pop(0)
            predicted_x, predicted_y = predicted_pos
            # Store as (center_x, center_y) format for template matching
            person.last_seen_center_coords = (int(predicted_x), int(predicted_y))
            logger.info(f"[KALMAN] Using predicted position: ({int(predicted_x)}, {int(predicted_y)}), {len(person.predicted_trajectory)} predictions remaining")
            if not person.predicted_trajectory:
                logger.info(f"[KALMAN] All predicted positions used up.")
                person.center_history.append((int(predicted_x), int(predicted_y)))
        else:
            # No predictions available, use last known coords from center_history
            if person.center_history:
                # Use the last item in center_history - this is the last seen center
                center_x, center_y = person.center_history[-1]
                person.last_seen_center_coords = (center_x, center_y)
                logger.info(f"[KALMAN] No predictions available, using last known position from history: ({center_x}, {center_y})")
            else:
                # Edge case: no history and no predictions (person selected but never seen)
                person.last_seen_center_coords = None
                logger.error(f"[KALMAN] No predictions and no center history available - template matching will fail")
        
        is_id_found = True
    else:
        logger.warning(f"Frame: ID {selected_id} not found in this frame.")
        # Only clear followed_id if template matching also failed
        if not selected_id:  # selected_id is None, meaning template matching failed too
            person.followed_id = None
        person.id = []        
        person.coords = None  # Clear coordinates when target is lost
        person.predicted_trajectory = []  # Clear any remaining predictions
        person.lost_rounds = 0
        logger.info("[INPUT] Please enter a new ID to select.")
        is_id_found = False
    
    return is_id_found
        

def is_alive(thread):
    if(thread is None): return False
    return thread.is_alive()


def draw_frame_addons(annotated_frame, coords, target_coords=None, selected_last_seen=None, target_last_seen=None):
    # Draw rectangle
    cv2.rectangle(annotated_frame, RECT_TOP_LEFT(), RECT_BOTTOM_RIGHT(), (0, 0, 255), 2)
    
    if(coords != None):
        x1, y1, x2, y2, center_x, center_y = coords
        # Draw red point at selected ID coords
        cv2.circle(annotated_frame, (center_x, center_y), radius=3, color=(0, 0, 255), thickness=-1)
    elif selected_last_seen is not None:
        # Draw yellow point at last seen position when selected person is lost
        center_x, center_y = selected_last_seen
        cv2.circle(annotated_frame, (center_x, center_y), radius=5, color=(0, 255, 255), thickness=-1)
        cv2.putText(annotated_frame, "LOST", (center_x + 10, center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    if(target_coords != None):
        x1, y1, x2, y2, target_center_x, target_center_y = target_coords
        # Draw green point at target ID coords
        cv2.circle(annotated_frame, (target_center_x, target_center_y), radius=3, color=(0, 255, 0), thickness=-1)
    elif target_last_seen is not None:
        # Draw cyan point at last seen position when target person is lost
        target_center_x, target_center_y = target_last_seen
        cv2.circle(annotated_frame, (target_center_x, target_center_y), radius=5, color=(255, 255, 0), thickness=-1)
        cv2.putText(annotated_frame, "TARGET LOST", (target_center_x + 10, target_center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


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
    RECT_SIZE['w'], RECT_SIZE['h'] = min_object_size, min_object_size
    logger.debug(f"Rectangle size: w={RECT_SIZE['w']}, h={RECT_SIZE['h']}")

    # Y-axis control with improved hysteresis
    hysteresis_margin = RECT_SIZE['h'] // 3  # 25% of rectangle height for hysteresis
    
    if histerezis_enabled:
        histerezis_margin_y = RECT_SIZE['h'] // 3
        histerezis_margin_x = RECT_SIZE['w'] // 3
        # Forward movement logic with hysteresis
        if center_y < RECT_TOP_LEFT()[1] or histerezis_on["up"]:
            if center_y < RECT_TOP_LEFT()[1] + histerezis_margin_y:  # Continue until well inside
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
        elif center_y > RECT_BOTTOM_RIGHT()[1] or histerezis_on["down"]:
            if center_y > RECT_BOTTOM_RIGHT()[1] - histerezis_margin_y:  # Continue until well inside
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
    
    if histerezis_enabled:
        # Left rotation logic with hysteresis
        if center_x < RECT_TOP_LEFT()[0] or histerezis_on["left"]:
            if center_x < RECT_TOP_LEFT()[0] + histerezis_margin_x:  # Continue until well inside
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
        elif center_x > RECT_BOTTOM_RIGHT()[0] or histerezis_on["right"]:
            if center_x > RECT_BOTTOM_RIGHT()[0] - histerezis_margin_x:  # Continue until well inside
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


def create_bw_segmented_frame(results, frame_shape, selected_id=None, target_id=None, 
                               selected_coords=None, target_coords=None, rrt_path=None):
    """
    Create a black and white segmented frame where detected objects (persons and cars) 
    are black and the background is white. Selected and target IDs are represented 
    as colored dots instead of full segmentation. Optionally draws RRT* path between them.
    
    Args:
        results: YOLO detection results with masks
        frame_shape: Shape of the original frame (height, width, channels)
        selected_id: ID of the selected person to follow (will be red dot)
        target_id: ID of the target person to navigate to (will be green dot)
        selected_coords: Coordinates (x1, y1, x2, y2, cx, cy) of selected ID
        target_coords: Coordinates (x1, y1, x2, y2, cx, cy) of target ID
        rrt_path: Pre-calculated RRT* path (list of (x, y) tuples), or None
    
    Returns:
        numpy.ndarray: Black and white segmented frame (BGR for colored dots/path)
        selected_center: Center coordinates of selected ID (for RRT planning)
        target_center: Center coordinates of target ID (for RRT planning)
    """
    height, width = frame_shape[:2]
    
    # Start with a white frame (grayscale)
    bw_frame = np.ones((height, width), dtype=np.uint8) * 255
    
    # If no masks detected, convert to BGR and return
    if results[0].masks is None or results[0].boxes is None:
        return cv2.cvtColor(bw_frame, cv2.COLOR_GRAY2BGR), bw_frame, None, None
    
    # Check if we have class and ID information
    if results[0].boxes.cls is None:
        return cv2.cvtColor(bw_frame, cv2.COLOR_GRAY2BGR), bw_frame, None, None
    
    try:
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        
        # Get tracking IDs if available
        track_ids = []
        if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        
        # Store selected and target centers for later
        selected_center = None
        target_center = None
        
        # Iterate through all masks and paint detected objects black
        for i, class_id in enumerate(class_ids):
            # Get tracking ID for this detection
            current_track_id = track_ids[i] if i < len(track_ids) else None
            
            # Skip selected and target IDs - we'll draw dots for them instead
            if current_track_id in [selected_id, target_id]:
                # Store center for path planning
                box = results[0].boxes[i]
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                
                if current_track_id == selected_id:
                    selected_center = center
                elif current_track_id == target_id:
                    target_center = center
                continue
            
            # Only process persons (class 0) and cars (class 2)
            if class_id in [0, 2]:
                mask = results[0].masks[i]
                binary_mask = mask.data[0].cpu().numpy().astype("uint8")
                
                # Resize mask to frame size if needed
                if binary_mask.shape != (height, width):
                    binary_mask = cv2.resize(binary_mask, (width, height))
                
                # Set detected object pixels to black (0)
                bw_frame[binary_mask > 0] = 0
        
        # Convert grayscale to BGR to allow colored dots and path
        bw_frame_bgr = cv2.cvtColor(bw_frame, cv2.COLOR_GRAY2BGR)
        
        # Draw RRT* path if provided (pre-calculated in separate thread)
        if ENABLE_RRT_PATH and rrt_path and len(rrt_path) > 1:
            try:
                # Draw path on the image
                bw_frame_bgr = rrt.draw_path_on_image(bw_frame_bgr, rrt_path, RRT_PATH_COLOR, thickness=3)
            except Exception as e:
                logger.warning(f"[RRT*] Error drawing path: {e}")
        
        # Draw colored dots for selected and target IDs
        for i, box in enumerate(results[0].boxes):
            if hasattr(box, 'id') and box.id is not None:
                obj_id = int(box.id.item())
                
                # Get center coordinates
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Draw red dot for selected ID (on top of path)
                if obj_id == selected_id:
                    cv2.circle(bw_frame_bgr, (center_x, center_y), radius=6, color=(0, 0, 255), thickness=-1)
                
                # Draw green dot for target ID (on top of path)
                elif obj_id == target_id:
                    cv2.circle(bw_frame_bgr, (center_x, center_y), radius=6, color=(0, 255, 0), thickness=-1)
        
        return bw_frame_bgr, bw_frame, selected_center, target_center
                
    except Exception as e:
        logger.warning(f"Error creating BW segmented frame: {e}")
        return cv2.cvtColor(bw_frame, cv2.COLOR_GRAY2BGR), bw_frame, None, None


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


def find_new_id(template_dict, person, all_objects_in_frame, all_persons_centers_dict, exclude_ids=None):
    """
    Perform template matching to update best_match_candidates in person.
    Does not return a replacement ID - instead updates the person's best_match_candidates dict.
    
    Args:
        template_dict: Dictionary of templates for each ID
        person: Person object with the lost ID information
        all_objects_in_frame: Current objects detected in frame
        all_persons_centers_dict: Dictionary of person IDs to their center coordinates
        exclude_ids: List of IDs to exclude from matching (e.g., the opposite tracked ID)
    """
    selected_id_to_track = person.followed_id
    templates = template_dict.get(selected_id_to_track, [])
    selected_id_last_center = person.last_seen_center_coords

    if exclude_ids is None:
        exclude_ids = []
    
    if not templates:
        logger.info(f"[TEMPLATE_MATCHING] No templates available for ID {selected_id_to_track}")
        return
    
    if not all_objects_in_frame:
        logger.info(f"[TEMPLATE_MATCHING] No objects detected in current frame")
        return
    
    # Convert all_objects_in_frame format for template matching
    # and filter out excluded IDs and non-person objects
    objects_for_matching = {}
    if isinstance(list(all_objects_in_frame.values())[0], dict):
        # New format: {id: {'image': img, 'class_id': cls}}
        # Only include persons (class 0) for template matching
        objects_for_matching = {
            obj_id: obj_info['image'] 
            for obj_id, obj_info in all_objects_in_frame.items() 
            if obj_id not in exclude_ids and obj_info.get('class_id') == 0
        }
    else:
        # Old format: {id: image}
        objects_for_matching = {
            obj_id: img 
            for obj_id, img in all_objects_in_frame.items() 
            if obj_id not in exclude_ids
        }
    
    if not objects_for_matching:
        logger.info(f"[TEMPLATE_MATCHING] No valid objects after excluding IDs {exclude_ids}")
        return
    
    # Update best_match_candidates in person (does not return a value)
    template_matching.find_best_match(person, objects_for_matching, templates, selected_id_last_center, all_persons_centers_dict)


def choose_best_match(person, current_ids_in_frame):
    """
    Selects the ID with the lowest score from best_match_candidates that is currently in the frame.
    Returns ID if score is below threshold and ID is in frame, None otherwise.
    Always clears the candidates dict.
    """
    candidates = person.best_match_candidates
    
    logger.info(f"[TEMPLATE_MATCHING] === WINDOW EXPIRED - CHOOSING BEST MATCH ===")
    logger.info(f"[TEMPLATE_MATCHING] All accumulated candidates over window: {candidates}")
    logger.info(f"[TEMPLATE_MATCHING] IDs currently in frame: {current_ids_in_frame}")
    
    # Always clear the dict at the end of the window
    person.best_match_candidates = {}
    
    if not candidates:
        logger.info(f"[TEMPLATE_MATCHING] No candidates accumulated during window")
        return None
    
    # Filter candidates to only those currently in frame
    valid_candidates = {id: score for id, score in candidates.items() if id in current_ids_in_frame and id not in person.other.id}
    
    if not valid_candidates:
        logger.info(f"[TEMPLATE_MATCHING] No valid candidates in frame (all candidates left the scene)")
        return None
    
    # Find ID with minimum score
    min_id = min(valid_candidates, key=valid_candidates.get)
    min_score = valid_candidates[min_id]
    
    logger.info(f"[TEMPLATE_MATCHING] Valid candidates in frame: {valid_candidates}")
    logger.info(f"[TEMPLATE_MATCHING] Best candidate: ID {min_id} with score {min_score:.1f}")
    logger.info(f"[TEMPLATE_MATCHING] Threshold: {template_matching.THRESHOLD_BEST_MATCH}")
    
    if min_score <= template_matching.THRESHOLD_BEST_MATCH:
        logger.info(f"[TEMPLATE_MATCHING] ✓ MATCH ACCEPTED: {min_score:.1f} <= {template_matching.THRESHOLD_BEST_MATCH}")
        return min_id
    else:
        logger.info(f"[TEMPLATE_MATCHING] ❌ MATCH REJECTED: {min_score:.1f} > {template_matching.THRESHOLD_BEST_MATCH}")
        logger.info(f"[TEMPLATE_MATCHING] SUGGESTION: Consider threshold >= {min_score + 100:.0f} to accept this match")
        return None


def init_input_thread(current_person, other_person, current_getter_thread, results):
    if((not current_person.id) and not is_alive(current_getter_thread)):
        current_getter_thread = init_id_getter_thread(current_person, other_person, results)
    return current_getter_thread




class RRTPathPlanner:
    """
    Threaded RRT* path planner to avoid blocking the main loop.
    Calculates paths asynchronously and provides the latest available path.
    """
    def __init__(self):
        self.request_queue = queue.Queue(maxsize=1)  # Only keep the latest request
        self.result_queue = queue.Queue(maxsize=1)   # Only keep the latest result
        self.current_path = None
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info("[RRT*] Path planner thread started")
    
    def _worker(self):
        """Background worker that processes path planning requests"""
        while self.is_running:
            try:
                # Wait for a new request (with timeout to allow checking is_running)
                request = self.request_queue.get(timeout=0.1)
                
                bw_frame, selected_center, target_center, max_iter, delta, radius = request
                
                # Calculate the path
                try:
                    path = rrt.find_rrt_path(
                        bw_frame,
                        selected_center,
                        target_center,
                        max_iter=max_iter,
                        delta=delta,
                        radius=radius
                    )
                    
                    # Put result in queue (discard old result if any)
                    if not self.result_queue.empty():
                        try:
                            self.result_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.result_queue.put(path)
                    
                except Exception as e:
                    logger.warning(f"[RRT*] Error in worker thread: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[RRT*] Unexpected error in worker: {e}")
    
    def request_path(self, bw_frame, selected_center, target_center, 
                     max_iter=1000, delta=15, radius=30):
        """
        Request a new path calculation. Non-blocking.
        Discards previous request if not yet processed.
        """
        if selected_center is None or target_center is None:
            return
        
        # Discard old request if any
        if not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except queue.Empty:
                pass
        
        # Submit new request
        try:
            self.request_queue.put_nowait((
                bw_frame.copy(),  # Make a copy to avoid race conditions
                selected_center,
                target_center,
                max_iter,
                delta,
                radius
            ))
        except queue.Full:
            pass  # Queue full, skip this request
    
    def get_latest_path(self):
        """
        Get the latest calculated path if available.
        Non-blocking. Returns None if no path is ready.
        """
        try:
            # Get the latest result without blocking
            self.current_path = self.result_queue.get_nowait()
        except queue.Empty:
            pass  # No new path, keep using current one
        
        return self.current_path
    
    def clear_path(self):
        """
        Clear the current path and any pending calculations.
        Called when target or selected ID is lost.
        """
        self.current_path = None
        # Clear any pending requests
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except queue.Empty:
                break
        # Clear any pending results
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
    
    def stop(self):
        """Stop the worker thread"""
        self.is_running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        logger.info("[RRT*] Path planner thread stopped")


def init_logs():
    logger.info("="*60)
    logger.info("DRONE CONTROL SYSTEM STARTED")
    logger.info(f"Timestamp: {TIMESTAMP}")
    logger.info(f"Video files:")
    logger.info(f"  - Clean: {OUTPUT_FILENAME}")
    logger.info(f"  - Annotated: {OUTPUT_ANNOTATED_FILENAME}")
    logger.info(f"  - BW Segmented: {OUTPUT_BW_SEGMENTED_FILENAME}")
    logger.info(f"Log file: {LOG_FILENAME}")
    logger.info("="*60)

def update_templates(results, frame, persons_dict):
    """
    Update the templates logic.
    """
    all_persons_in_frame, all_persons_centers_dict = template_matching.extract_all_visible_persons(results, frame)
    if all_persons_in_frame:
        template_matching.update_persons_dict(all_persons_in_frame, persons_dict, template_matching.MAX_RECENT_EXTRACTIONS)
        template_counts = {pid: len(templates) for pid, templates in persons_dict.items()}
        logger.info(f"[TEMPLATE_MATCHING] Person template counts: {template_counts}")
        return all_persons_in_frame, all_persons_centers_dict
    else:
        logger.info(f"[TEMPLATE_MATCHING] No persons in frame")
        return {}, {}

def main():
    # Print to terminal only - inform user where logs are saved
    print("="*60)
    print(f"Drone Control System Starting...")
    print(f"Log file: {LOG_FILENAME}")
    print("="*60)
    print()
    
    # open_log_in_editor(LOG_FILENAME)

    init_logs()

    # Create Person objects for tracking
    user_person = Person(person_type="USER")
    target_person = Person(person_type="TARGET")
    user_person.other = target_person
    target_person.other = user_person

    persons_dict = {}  # Template dictionary for persons (used for both following and target tracking)
    
    # thread for id selection 
    user_getter_thread = None
    target_getter_thread = None

    model = YOLO("./yolo_models/yolo11s-seg.pt")

    drone = start_drone()
    out_clean = start_recording(OUTPUT_FILENAME)
    out_annotated = start_recording(OUTPUT_ANNOTATED_FILENAME)
    out_bw_segmented = start_recording(OUTPUT_BW_SEGMENTED_FILENAME)

    launch_drone(drone)

    # Performance monitoring
    
    # RRT* path planner (threaded)
    rrt_planner = RRTPathPlanner() if ENABLE_RRT_PATH else None
    
    # Frame processing variables
    frame_counter = 0
    yolo_frame_counter = 0
    rrt_frame_counter = 0

    histerezis_on = {
        "down": False,
        "up": False,
        "left": False,
        "right": False
    }
    
    logger.info(f"[TRACKING] Coordinate history size: {COORDINATE_HISTORY_SIZE} frames")
    if ENABLE_RRT_PATH:
        logger.info(f"[RRT*] Path planning enabled (threaded, async, every {RRT_FRAME_SKIP} frames)")
    
    try:
        while True:

                
            frame_counter += 1
            frame = get_frame(drone)
            if frame is None:
                logger.warning("No frame received.")
                continue
            
            # Process YOLO on every frame
            yolo_frame_counter += 1
            
            # Always detect person and car classes
            class_names = [get_object_class_name(cls) for cls in BASE_DETECTION_CLASSES]
            logger.info(f"[OPTIMIZATION] Detecting classes: {class_names} {BASE_DETECTION_CLASSES}")
            
            results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml", classes=BASE_DETECTION_CLASSES)

            annotated_frame = results[0].plot(line_width=1)
            
            # # filter out cars boxes AND masks from results so will not be seen in the frame
            # # IMPORTANT: Must filter both boxes and masks to keep them synchronized!
            # car_indices = (results[0].boxes.cls == 2)
            # results[0].boxes = results[0].boxes[~car_indices]
            
            # # Filter masks too if they exist
            # if results[0].masks is not None:
            #     results[0].masks.data = results[0].masks.data[~car_indices]

            logger.info(f"[OPTIMIZATION] Updating templates (frame {yolo_frame_counter})")
            
            # Extract all persons visible in the current frame (for both following and target tracking)
            # Only persons (class 0) are used for template matching
            all_persons_in_frame, all_persons_centers_dict = update_templates(results, frame, persons_dict)
                
            user_getter_thread = init_input_thread(user_person, target_person, user_getter_thread, results)

            # Initialize variables
            is_id_found = False
            is_target_found = False

            # Only process tracking if we have valid YOLO results
            if (results is not None) and user_person.id and (user_person.history is not None):
                # Target tracking logic (controlled by TARGET_TRACKING_ENABLED flag)
                if TARGET_TRACKING_ENABLED:
                    # Use different input thread for target (asks for target objects)
                    target_getter_thread = init_input_thread(target_person, user_person, target_getter_thread, results)
                    
                    #follow target - now with template matching support
                    # Exclude selected ID history from target template matching
                    # Only use persons for template matching (persons_dict is used for both selected and target)
                    is_target_found = find_id_in_frame(results, target_person, persons_dict, all_persons_in_frame, all_persons_centers_dict,
                                                       exclude_ids=user_person.history, tracking_type="Target")
                
                # saves the coordinates in person IMPORTANT DONT COMMENT
                # Exclude target ID history from selected template matching
                is_id_found = find_id_in_frame(results, user_person, persons_dict, all_persons_in_frame, all_persons_centers_dict,
                                               exclude_ids=target_person.history, tracking_type="Selected")
                # TODO: pivot to new function that will take care of scanning the potential target objects

            # Display detected objects (persons and cars)
            if results is not None and results[0].boxes is not None:
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

            # Show both original input ID and currently followed ID (may differ after template matching)
            user_person_ids = user_person.id
            user_person_followed_id = user_person.followed_id
            if user_person_followed_id and user_person_followed_id not in user_person_ids:
                logger.info(f"[User ID] Input: {user_person_ids}, Currently Following: {user_person_followed_id}")
            else:
                logger.info(f"[User ID] {user_person_ids}")
            
            if TARGET_TRACKING_ENABLED:
                target_input_ids = target_person.id
                target_followed_id = target_person.followed_id
                if target_followed_id and target_followed_id not in target_input_ids:
                    logger.info(f"[Target ID] Input: {target_input_ids}, Currently Following: {target_followed_id}")
                else:
                    logger.info(f"[Target ID] {target_input_ids}")
            
            # Get target coordinates if target tracking is enabled
            target_coords = None
            if(is_target_found):
                target_coords = target_person.coords
                if target_coords is not None:
                    x1, y1, x2, y2, target_center_x, target_center_y = target_coords
                    # Get target object info
                    target_info = get_object_info(results, target_person.followed_id)
                    if target_info:
                        logger.info(f"Target ({target_info['class_name']}) Center coordinates: ({target_center_x}, {target_center_y})")
                    else:
                        logger.info(f"Target Center coordinates: ({target_center_x}, {target_center_y})")
                else:
                    logger.warning("Target found but coordinates are None (target recently lost)")

            # Get coordinates directly (no interpolation)
            coords = user_person.coords

            drone.left_right_velocity, drone.for_back_velocity, drone.up_down_velocity, drone.yaw_velocity = control_drone(drone, coords, annotated_frame, HISTEREZIS_ENABLED, histerezis_on)
            logging.info(f"drone.for_back_velocity: {drone.for_back_velocity}, drone.yaw_velocity: {drone.yaw_velocity}, ")


            # keep alive command
            drone.send_control_command("command")

            # SEND VELOCITY VALUES TO TELLO
            
            if drone.send_rc_control:
                drone.send_rc_control(drone.left_right_velocity, drone.for_back_velocity, drone.up_down_velocity, drone.yaw_velocity)

            move_drone(drone)

            # Get last seen coordinates for visualization when persons are lost
            selected_last_seen = user_person.last_seen_center_coords
            target_last_seen = target_person.last_seen_center_coords if TARGET_TRACKING_ENABLED else None
            
            # Get current RRT path for drawing on annotated frame
            current_rrt_path = None
            if rrt_planner:
                selected_actually_present = (user_person.lost_counter == 0 and 
                                            user_person.coords is not None)
                target_actually_present = (TARGET_TRACKING_ENABLED and 
                                          target_person.lost_counter == 0 and 
                                          target_person.coords is not None)
                both_ids_present = selected_actually_present and (target_actually_present if TARGET_TRACKING_ENABLED else False)
                
                if both_ids_present:
                    current_rrt_path = rrt_planner.get_latest_path()
            
            # Draw RRT path on annotated frame if available
            if ENABLE_RRT_PATH and current_rrt_path and len(current_rrt_path) > 1:
                try:
                    annotated_frame = rrt.draw_path_on_image(annotated_frame, current_rrt_path, RRT_PATH_COLOR, thickness=3)
                except Exception as e:
                    logger.warning(f"[RRT*] Error drawing path on annotated frame: {e}")
            
            # Draw frame addons with both selected ID (red) and target ID (green) points
            # When lost, draw last seen position in different colors (yellow for selected, cyan for target)
            draw_frame_addons(annotated_frame, coords, target_coords, selected_last_seen, target_last_seen)

            # Draw frame counter on screen at top left
            cv2.putText(annotated_frame, f"Frame: {frame_counter}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Log frame counter
            logger.info(f"[FRAME] Frame counter: {frame_counter}")

            # cv2.imshow('Tello Video Feed', annotated_frame)
            cv2.imshow('Tello Video Feed', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            

            # out_annotated.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            # out_clean.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            out_annotated.write(annotated_frame) # TODO CHECK COLORS GOOD IN SAVED VIDEOS
            out_clean.write(frame)
            
            # Create and write BW segmented frame with selected/target as colored dots and RRT* path
            if results is not None:
                selected_followed_id = user_person.followed_id
                target_followed_id = target_person.followed_id if TARGET_TRACKING_ENABLED else None
                
                # Check if IDs are actually present in the current frame (not just tracked with lost counter)
                # This ensures path is deleted immediately when ID disappears, not after MAX_LOST_ID_FRAMES
                selected_actually_present = (user_person.lost_counter == 0 and 
                                            user_person.coords is not None)
                target_actually_present = (TARGET_TRACKING_ENABLED and 
                                          target_person.lost_counter == 0 and 
                                          target_person.coords is not None)
                
                # Get latest calculated path (non-blocking)
                # This returns the last calculated path (cached) if no new path is ready
                # Clear the path immediately if either ID is not actually present in frame
                current_rrt_path = None
                both_ids_present = selected_actually_present and (target_actually_present if TARGET_TRACKING_ENABLED else False)
                
                if rrt_planner and both_ids_present:
                    current_rrt_path = rrt_planner.get_latest_path()
                elif rrt_planner and not both_ids_present:
                    # Clear the path immediately when IDs disappear from frame
                    logger.debug(f"[RRT*] Clearing path - Selected present: {selected_actually_present}, Target present: {target_actually_present}")
                    rrt_planner.clear_path()
                
                # Create BW frame and get centers for next RRT calculation
                bw_segmented_frame, bw_frame_gray, sel_center, tgt_center = create_bw_segmented_frame(
                    results, 
                    frame.shape, 
                    selected_followed_id, 
                    target_followed_id,
                    coords,
                    target_coords,
                    current_rrt_path  # Use cached path or newly calculated path if ready
                )
                out_bw_segmented.write(bw_segmented_frame)
                
                # Request new path calculation every RRT_FRAME_SKIP frames (async, non-blocking)
                should_calculate_rrt = (frame_counter % RRT_FRAME_SKIP == 0)
                if rrt_planner and sel_center and tgt_center:
                    if should_calculate_rrt:
                        rrt_frame_counter += 1
                        logger.debug(f"[RRT*] Requesting NEW path calculation (frame {rrt_frame_counter})")
                        rrt_planner.request_path(
                            bw_frame_gray,
                            sel_center,
                            tgt_center,
                            max_iter=RRT_MAX_ITER,
                            delta=RRT_DELTA,
                            radius=RRT_RADIUS
                        )
                    # When should_calculate_rrt is False, we simply reuse the last path
                    # (already retrieved above via get_latest_path())
            else:
                # Write blank white frame when no YOLO results yet
                blank_frame = np.ones((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) * 255
                out_bw_segmented.write(blank_frame)


            if should_quit():
                shutdown_drone(drone)
                break



            time.sleep(1/30)  # reduce CPU load, aiming for 30fps

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Shutting down...")
        logger.info("Interrupted by user.")
    
    finally:
        # Stop RRT planner thread
        if rrt_planner:
            rrt_planner.stop()

    exit([out_clean, out_annotated, out_bw_segmented], drone)
    
    print()
    print("="*60)
    print("Drone Control System Stopped")
    print(f"Log file saved: {LOG_FILENAME}")
    print("="*60)

    return


if __name__ == '__main__':
    main()
# GUI_RADIUS_SIZE=10
# def draw_user_GUI(frame, center, color=(255, 0, 0), thickness=2):
#     """
#     Draws a circle on the given frame at the specified center with the given radius.
#     Args:
#         frame: The image/frame to draw on (numpy array)
#         center: Tuple (x, y) for the center of the circle
#         radius: Integer radius of the circle (global parameter)
#         color: BGR color tuple (default blue)
#         thickness: Thickness of the circle outline (default 2)
#     """
#     cv2.circle(frame, center, GUI_RADIUS_SIZE, color, thickness)