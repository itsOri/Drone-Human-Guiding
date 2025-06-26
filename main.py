import cv2
import time
from djitellopy import Tello
from ultralytics import YOLO
import threading
import numpy as np
import os
from datetime import datetime 

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FILENAME = f'clean_video_{TIMESTAMP}.avi'
OUTPUT_ANNOTATED_FILENAME = f'annotated_video_{TIMESTAMP}.avi'


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
HISTEREZIS_ENABLED = False # TODO!!, its doing a ping pong between both sides and not turning off

#RECT_TOP_LEFT = (RECT_CENTER[0] - RECT_W // 2, RECT_CENTER[1] - RECT_H // 2)
#RECT_BOTTOM_RIGHT = (RECT_CENTER[0] + RECT_W // 2, RECT_CENTER[1] + RECT_H // 2)

BLACK = (0, 0, 0)

YAW_MOVING_VELOCITY = 15
FB_MOVING_VELOCITY = 25
DRONE_START_HEIGHT = 200

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

	drone.for_back_velocity = 0
	drone.left_right_velocity = 0
	drone.up_down_velocity = 0
	drone.yaw_velocity = 0
	drone.speed = 0
	drone.TIME_BTW_RC_CONTROL_COMMANDS = 0.1

	print(f"Battery: {drone.get_battery()}%")

	drone.streamoff()
	drone.streamon()

	time.sleep(2)  # Give stream a moment to start

	return drone


def start_recording(filename):
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	if(SAVE_FILE):
		out = cv2.VideoWriter(filename, fourcc, 20.0, (WIDTH, HEIGHT))
		print("[INFO] Starting video capture. Press 'q' to stop.")
	
	return out


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
	print("[INFO] Stopping capture, releasing resources.")
	for out in outs:
		out.release()
	cv2.destroyAllWindows()
	drone.streamoff()
	print(f"Battery: {drone.get_battery()}%")


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
	target_input = input("[INPUT] Enter Target ID: ")
	try:
		target_id_container["id"] = int(target_input)
		print(f"[INFO] Target ID: {target_id_container['id']}")

	except ValueError:
		print("[WARN] Invalid ID entered.")
		target_id_container["id"] = None
	

# TODO unite this func with target thread using generic container or string param for text display to user
def input_thread(selected_id_container):
	user_input = input("[INPUT] Enter ID to select: ")
	try:
		selected_id_container["id"].append(int(user_input))
		selected_id_container["id_was_seen"] = False
		print(f"[INFO] Selected ID: {selected_id_container['id']}")

	except ValueError:
		print("[WARN] Invalid ID entered.")
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
    Extracts class IDs from YOLOv8 inference results.

    Parameters:
        results: YOLOv8 result object (from model.predict or model(...))

    Returns:
        List of integer class IDs detected in the frame.
    """
    if len(results) == 0 or results[0].boxes is None:
        return []

    class_ids = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
    return class_ids


def get_target_id_in_frame(results, target_ids):
	"""
    Returns the first target ID found in the frame that is in target_ids list

    Parameters:
        results: Detection results for the current frame.
        target_ids (list): IDs to search for.

    Returns:
        int or None: Matching target ID or None if not found.
    """
	ids_in_frame = get_ids_in_frame(results)
	for id in ids_in_frame:
		if( id.isin(target_ids)):
			return id
	print("Error - target id not in frame")
	return None

def find_id_in_frame(results, selected_id_container):
	"""
	Checks if an ID is selected and processes tracking results to get coordinates.
	Stores the coordinates in selected_id_container["coords"] if found.
	If the ID is not found, resets selection and input thread status.
	Returns updated (id_selected).
	"""
	if selected_id_container["id"]:
		target_id = get_target_id_in_frame(results, selected_id_container["id"])
		selected_id_container["followed_id"] = target_id
		coords = get_coordinates_by_id(results, target_id)
		selected_id_container["coords"] = coords  # Store or update the coordinates
		if coords:
			x1, y1, x2, y2, center_x, center_y = coords
			print(f"[INFO] Frame: Found ID {target_id} at ({x1}, {y1}), ({x2}, {y2}), center: ({center_x}, {center_y})")
			# You can add more logic here (e.g., draw something special or send commands)
			return True
		else:
			print(f"[WARN] Frame: ID {target_id} not found in this frame.")
			selected_id_container["id"] = []
			selected_id_container["followed_id"] = None
			print("[INPUT] Please enter a new ID to select.")
			return False
		

def is_alive(thread):
	if(thread is None): return False
	return thread.is_alive()


def draw_frame_addons(annotated_frame, coords):
	# Draw rectangle
	cv2.rectangle(annotated_frame, RECT_TOP_LEFT(), RECT_BOTTOM_RIGHT(), (0, 0, 255), 2)
	
	if(coords != None):
		x1, y1, x2, y2, center_x, center_y = coords
		# Draw red point at coords
		cv2.circle(annotated_frame, (center_x, center_y), radius=3, color=(0, 0, 255), thickness=-1)


def control_drone(drone, coords, frame, histerezis_enabled, histerezis_on):

	if coords is None:
		return 0,0,0,0
	
	x1, y1, x2, y2, center_x, center_y = coords
	min_object_size = max(MIN_RECT_SIZE, min(y2-y1, x2-x1))
	rect_size['w'], rect_size['h'] = min_object_size, min_object_size
	print(rect_size['w'], rect_size['h'])

	# if person above rectangle, need to go forward
	if center_y < RECT_TOP_LEFT()[1] :
		cv2.putText(frame, "FORWARD", 
					(RECT_TOP_LEFT()[0] + 20, RECT_TOP_LEFT()[1] - 20), 
					cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 2)
		drone.for_back_velocity = FB_MOVING_VELOCITY
		if histerezis_enabled: histerezis_on["up"] = True

	# if person below rectangle, need to go backward
	elif center_y > RECT_BOTTOM_RIGHT()[1] :
		cv2.putText(frame, "BACKWARD", 
					(RECT_TOP_LEFT()[0] + 20, RECT_TOP_LEFT()[1] - 20), 
					cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 2)
		drone.for_back_velocity = -FB_MOVING_VELOCITY
		if histerezis_enabled: histerezis_on["down"] = True
	else:
		
		if histerezis_enabled:
			#y-axis histerezis effects
			if(histerezis_on["up"]):
				if(center_y > RECT_TOP_LEFT()[1] + rect_size['h']//2):
					histerezis_on["up"] = False
					drone.for_back_velocity = 0
				else:
					drone.for_back_velocity = FB_MOVING_VELOCITY
			elif(histerezis_on["down"]):
				if(center_y < RECT_BOTTOM_RIGHT()[1] - rect_size['h']//2):
					histerezis_on["down"] = False
					drone.for_back_velocity = 0
				else:
					drone.for_back_velocity = -FB_MOVING_VELOCITY
		else:
			drone.for_back_velocity = 0


	if center_x < RECT_TOP_LEFT()[0]:
		cv2.putText(frame, "<- ROTATE LEFT <-",
			(RECT_TOP_LEFT()[0] - 110, (RECT_TOP_LEFT()[1] + RECT_BOTTOM_RIGHT()[1]) // 2),
			cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2)
		drone.yaw_velocity = -YAW_MOVING_VELOCITY
		if histerezis_enabled: histerezis_on["left"] = True
	# ROTATE RIGHT if center_x is to the right of the rectangle
	elif center_x > RECT_BOTTOM_RIGHT()[0]:
		cv2.putText(frame, "-> ROTATE RIGHT ->",
					(RECT_BOTTOM_RIGHT()[0] + 20, (RECT_TOP_LEFT()[1] + RECT_BOTTOM_RIGHT()[1]) // 2),
					cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2)		
		drone.yaw_velocity = YAW_MOVING_VELOCITY
		if histerezis_enabled: histerezis_on["right"] = True
	else:
		if histerezis_enabled:
			
			#x-axis histerezis effects
			if(histerezis_on["left"]):
				if(center_x > RECT_TOP_LEFT()[0] + rect_size['w']//2):
					histerezis_on["left"] = False
					drone.yaw_velocity = 0
				else:
					drone.yaw_velocity = -YAW_MOVING_VELOCITY
			elif(histerezis_on["right"]):
				if(center_x < RECT_BOTTOM_RIGHT()[0] - rect_size['w']//2):
					histerezis_on["right"] = False
					drone.yaw_velocity = 0
				else:
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
		print("up")
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
				print("[SAVING...]Saving frame")
				cv2.imwrite(save_path, person_crop_masked_bgr)

def find_new_id(selected_id_container):
	pass

def main():
	print("START!")

	# thread for id selection 
	selected_id_container = {"id": []}
	getter_thread = None
	# id_selected = selected_id_container["id"]

	model = YOLO("./yolo_models/yolo11s-seg.pt")

	drone = start_drone()
	out_clean = start_recording(OUTPUT_FILENAME)
	out_annotated = start_recording(OUTPUT_ANNOTATED_FILENAME)



	launch_drone(drone)


	try:
		while True:
			frame = get_frame(drone)
			if frame is None:
				print("[WARN] No frame received.")
				continue
			
			results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml", classes=[0])
			# if(selected_id_container["id"]): save_segmented_persons(results, frame)
			annotated_frame = results[0].plot() # this is the frame with boxes and ids after track()


			if((not selected_id_container["id"]) and not is_alive(getter_thread)):
				getter_thread = init_id_getter_thread(selected_id_container)

			if(selected_id_container["id"]):
				# saves the coordinates in container IMPORTANT DONT COMMENT
				is_id_found = find_id_in_frame(results, selected_id_container)
				# TODO: pivot to new function that will take care of scanning the potential target objects
				if not is_id_found and selected_id_container["id_was_seen"]:
					is_id_found = find_new_id(selected_id_container)
				#first time found the id in frame
				elif is_id_found and not selected_id_container["id_was_seen"]:
					selected_id_container["id_was_seen"] = True


			print(f"[SELECTED ID] {selected_id_container['id']}")

			coords = selected_id_container.get("coords")
			if coords:
				x1, y1, x2, y2, center_x, center_y = coords
				# Do actions using coords!
				print(f"Center coordinates: ({center_x}, {center_y})")

			histerezis_on = {
				"down": False,
				"up": False,
				"left": False,
				"right": False
			}
			drone.left_right_velocity, drone.for_back_velocity, drone.up_down_velocity, drone.yaw_velocity = control_drone(drone, coords, annotated_frame, HISTEREZIS_ENABLED, histerezis_on)
			
			# keep alive command
			drone.send_control_command("command")

			# SEND VELOCITY VALUES TO TELLO
			
			if drone.send_rc_control:
				drone.send_rc_control(drone.left_right_velocity, drone.for_back_velocity, drone.up_down_velocity, drone.yaw_velocity)

			move_drone(drone)

			draw_frame_addons(annotated_frame, coords)

			cv2.imshow('Tello Video Feed', annotated_frame)
			out_annotated.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
			out_clean.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


			if should_quit():
				shutdown_drone(drone)
				break

			time.sleep(1/30)  # reduce CPU load, aiming for 30fps

	except KeyboardInterrupt:
		print("[INFO] Interrupted by user.")


	exit([out_clean, out_annotated], drone)

	return


if __name__ == '__main__':
	main()			