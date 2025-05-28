import cv2
import time
from djitellopy import Tello
from ultralytics import YOLO
import threading


WIDTH = 640
HEIGHT = 480
selected_id = None


def start_drone():
	drone = Tello()
	drone.connect()
	print(f"Battery: {drone.get_battery()}%")

	drone.streamoff()
	drone.streamon()

	time.sleep(2)  # Give stream a moment to start

	return drone


def start_recording():
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	out = cv2.VideoWriter('tello_capture.avi', fourcc, 20.0, (WIDTH, HEIGHT))

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


def exit(out, drone):
	print("[INFO] Stopping capture, releasing resources.")
	out.release()
	cv2.destroyAllWindows()
	drone.streamoff()


def launch_drone(drone):
	# do takeoff
	return


# def get_id(result):
# 	return
	

def input_thread(selected_id_container):
	user_input = input("[INPUT] Enter ID to select: ")
	try:
		selected_id_container["id"] = int(user_input)
		print(f"[INFO] Selected ID: {selected_id_container['id']}")

	except ValueError:
		print("[WARN] Invalid ID entered.")
		selected_id_container["id"] = None


def init_id_getter_thread(selected_id_container):
	"""
	Initializes and starts a background thread to continuously get user input.
	"""
	thread = threading.Thread(target=input_thread, args=(selected_id_container))
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


def find_id_in_frame(results, selected_id_container):
	"""
	Checks if an ID is selected and processes tracking results to get coordinates.
	Stores the coordinates in selected_id_container["coords"] if found.
	If the ID is not found, resets selection and input thread status.
	Returns updated (id_selected).
	"""
	if selected_id_container["id"]:
		target_id = selected_id_container["id"]
		coords = get_coordinates_by_id(results, target_id)
		selected_id_container["coords"] = coords  # Store or update the coordinates
		if coords:
			x1, y1, x2, y2, center_x, center_y = coords
			print(f"[INFO] Frame: Found ID {target_id} at ({x1}, {y1}), ({x2}, {y2}), center: ({center_x}, {center_y})")
			# You can add more logic here (e.g., draw something special or send commands)
			return True
		else:
			print(f"[WARN] Frame: ID {target_id} not found in this frame.")
			selected_id_container["id"] = None
			print("[INPUT] Please enter a new ID to select.")
			return False
		
def is_alive(thread):
	if(thread is None): return False
	return thread.is_alive()

def main():
	print("START!")

	# thread for id selection 
	selected_id_container = {"id": None}
	getter_thread = None
	id_selected = selected_id_container["id"] == None

	model = YOLO("./yolo_models/yolo11s-seg.pt")

	drone = start_drone()
	out = start_recording()
	launch_drone(drone)

	try:
		while True:
			frame = get_frame(drone)
			if frame is None:
				print("[WARN] No frame received.")
				continue
			
			results = model.track(frame, persist=True, verbose=False)
			annotated_frame = results[0].plot() # this is the frame with boxes and ids after track()


			cv2.imshow('Tello Video Feed', annotated_frame)
			out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

			if((selected_id_container["id"] is None) and not is_alive(getter_thread)):
				getter_thread = init_id_getter_thread(selected_id_container)

			if(selected_id_container["id"]):
				is_id_found = find_id_in_frame(results, selected_id_container)

			coords = selected_id_container.get("coords")
			if coords:
				x1, y1, x2, y2, center_x, center_y = coords
				# Do actions using coords!
				print(f"Center coordinates: ({center_x}, {center_y})")
			
			if should_quit():
				break

			time.sleep(1/30)  # reduce CPU load, aiming for 30fps

	except KeyboardInterrupt:
		print("[INFO] Interrupted by user.")


	exit(out, drone)

	return


if __name__ == '__main__':
	main()