import cv2
import os
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import logging
import subprocess

# Get logger from main module (will use main's configuration)
# When imported from main.py, this will use the root logger configured in main
# When run standalone, it will create its own logger
logger = logging.getLogger(__name__)

try:
    # Try importing from same directory (when run from testsfolder)
    import template_matching_function as template_match_file
except ImportError:
    try:
        # Try importing from testsfolder package (when run from parent directory)
        from testsfolder import template_matching_function as template_match_file
    except ImportError:
        # Last resort: add current directory to path and import
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        import template_matching_function as template_match_file

# --- GLOBAL CONSTANTS & CONFIGURATION ---
PERSON_OUTPUT_FOLDER = f'./persons_data_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
#VIDEO_PATH = r"C:\Users\orifr\OneDrive - Technion\Documents\semester_f\project\drone_project_archives\Archive\clean_video_2025-10-17_17-25-20.avi"
VIDEO_PATH = "../videos/clean_video_2025-10-30_17-03-09.avi"

MAX_RECENT_EXTRACTIONS = 20
THRESHOLD_BEST_MATCH = 1200

# --- MAIN DATA STORE ---
# This dictionary will hold the image histories for ALL detected persons.
# Keys are person IDs, values are lists of images.
persons_dict = {}

# (extract_person_from_frame function can be removed if not used elsewhere, but we'll keep it for now)
def extract_person_from_frame(results, target_id, original_frame, frame_number):
	"""
	Extract a specific person (class 0 only) from frame by ID.
	Returns None if ID not found, not a person, or extraction fails.
	"""
	if results[0].boxes is None or results[0].boxes.id is None:
		return None
	
	track_ids = results[0].boxes.id.int().cpu().tolist()
	if target_id not in track_ids:
		return None
	
	person_index = track_ids.index(target_id)
	
	# Check if this is a person (class 0)
	if results[0].boxes.cls is not None:
		class_ids = results[0].boxes.cls.int().cpu().tolist()
		if class_ids[person_index] != 0:
			logger.warning(f"ID {target_id} is not a person (class {class_ids[person_index]}), skipping extraction.")
			return None
	
	if results[0].masks is None:
		logger.warning(f"No masks found for frame {frame_number}, cannot extract person.")
		return None
	
	mask = results[0].masks[person_index]
	binary_mask = mask.data[0].cpu().numpy().astype("uint8")
	extracted_person = cv2.bitwise_and(original_frame, original_frame, mask=binary_mask)
	box = results[0].boxes.xyxy[person_index].cpu().numpy().astype(int)
	x1, y1, x2, y2 = box
	cropped_person = extracted_person[y1:y2, x1:x2]
	if cropped_person.size == 0:
		return None
	return cropped_person


def extract_all_visible_persons(results, original_frame):
	"""
	Extract all visible persons (class 0 only) from the current frame for template matching.
	Cars and other objects are NOT extracted.
	
	Returns:
		extracted_persons_dict: {person_id: cropped_image}
		extracted_persons_centers_dict: {person_id: (center_x, center_y)}
	"""
	if results[0].boxes is None or results[0].boxes.id is None or results[0].masks is None:
		return {}, {}
	
	# Get class IDs to filter for persons only
	if results[0].boxes.cls is None:
		return {}, {}
	
	track_ids = results[0].boxes.id.int().cpu().tolist()
	class_ids = results[0].boxes.cls.int().cpu().tolist()
	all_masks = results[0].masks
	all_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
	extracted_persons_dict = {}
	extracted_persons_centers_dict = {}
	
	for i in range(len(track_ids)):
		current_id = track_ids[i]
		class_id = class_ids[i]
		
		# Only extract persons (class 0) for template matching
		if class_id != 0:
			continue
		
		mask = all_masks[i]
		binary_mask = mask.data[0].cpu().numpy().astype("uint8")
		extracted_person = cv2.bitwise_and(original_frame, original_frame, mask=binary_mask)
		x1, y1, x2, y2 = all_boxes[i]
		cropped_person = extracted_person[y1:y2, x1:x2]
		if cropped_person.size > 0:
			extracted_persons_dict[current_id] = cropped_person
			center_x = int((x1 + x2) / 2)
			center_y = int((y1 + y2) / 2)
			extracted_persons_centers_dict[current_id] = (center_x, center_y)
	
	return extracted_persons_dict, extracted_persons_centers_dict


def update_persons_dict(all_persons_in_frame, persons_dict, max_extractions):
	"""
	Updates the main dictionary with new extractions for each person.
	"""
	for person_id, image in all_persons_in_frame.items():
		if person_id not in persons_dict:
			persons_dict[person_id] = []
		persons_dict[person_id].append(image)
		if len(persons_dict[person_id]) > max_extractions:
			persons_dict[person_id].pop(0)


def save_all_persons(persons_dict, base_output_folder):
	"""
	Saves all collected images from the persons_dict to disk, organized
	in sub-folders by person ID.
	"""
	if not persons_dict:
		logger.info("Person dictionary is empty. Nothing to save.")
		return
	logger.info(f"Saving all collected person images to '{base_output_folder}'...")
	os.makedirs(base_output_folder, exist_ok=True)
	for person_id, image_list in persons_dict.items():
		person_folder = os.path.join(base_output_folder, f"person_{person_id}")
		os.makedirs(person_folder, exist_ok=True)
		logger.info(f"  > Saving {len(image_list)} images for Person ID {person_id}...")
		for i, img in enumerate(image_list):
			filename = os.path.join(person_folder, f"capture_{i+1}.png")
			# Images are already in BGR format (extracted from OpenCV frame), so save directly
			cv2.imwrite(filename, img)
	logger.info("All images saved successfully.")


def get_distance_penalty(center1, center2):
    """
    Calculates an exponential penalty based on the Euclidean distance between two centers.
    Penalty is in the range 0 to 10000.
    """
    if center1 is None or center2 is None:
        return 0
    distance = np.linalg.norm(np.array(center1) - np.array(center2))
    # Scale and exponentiate: adjust alpha for sensitivity
    alpha = 0.001  # Tune this value for desired curve
    penalty = int(10000 * (1 - np.exp(-alpha * distance)))
    return penalty

def find_best_match(other_persons_dict, selected_id_templates, selected_id_last_center, all_persons_centers_dict):
	logger.info(f"[TEMPLATE_MATCHING] Starting match with {len(selected_id_templates)} templates against {len(other_persons_dict)} persons (class 0 only)")
	logger.info(f"[TEMPLATE_MATCHING] Using threshold: {THRESHOLD_BEST_MATCH}")

	min_score = -1
	min_id = -1
	all_scores = {}  # Store all scores for debugging
	detailed_scores = {}  # Store individual template scores for each ID
	
	for id, extracted_image in other_persons_dict.items():
		# Assume each extracted_image is a dict: {'image': img}
		if isinstance(extracted_image, dict):
			candidate_img = extracted_image.get('image')
			candidate_center = extracted_image.get('center')
		else:
			candidate_img = extracted_image
			candidate_center = None
		# Calculate penalty
		candidate_center = all_persons_centers_dict[id]
		current_penalty = get_distance_penalty(candidate_center, selected_id_last_center)
		current_scores = [	template_match_file.template_match(template, candidate_img) + current_penalty
							for template in selected_id_templates]
		current_score = min(current_scores)
		
		# Store detailed information
		all_scores[id] = current_score
		detailed_scores[id] = {
			'best_score': current_score,
			'penalty_score': current_penalty,
			'all_template_scores': current_scores,
			'worst_score': max(current_scores),
			'avg_score': sum(current_scores) / len(current_scores)
		}
		
		if min_score == -1 or current_score < min_score:
			min_score = current_score
			min_id = id
	
	# Log detailed score analysis
	logger.info(f"[TEMPLATE_MATCHING] === DETAILED SCORE ANALYSIS ===")
	for id in sorted(all_scores.keys()):
		scores_info = detailed_scores[id]
		logger.info(f"[TEMPLATE_MATCHING] ID {id:2d}: BEST={scores_info['best_score']:6.1f} | AVG={scores_info['avg_score']:6.1f} | WORST={scores_info['worst_score']:6.1f} | PENALTY={scores_info['penalty_score']:6.1f}")
		# Log first few individual scores to see the range
		sample_scores = scores_info['all_template_scores'][:5]  # Show first 5 template scores
		logger.info(f"[TEMPLATE_MATCHING]      Sample scores: {[f'{s:.1f}' for s in sample_scores]}")
	
	logger.info(f"[TEMPLATE_MATCHING] === SUMMARY ===")
	logger.info(f"[TEMPLATE_MATCHING] All match scores (best): {all_scores}")
	logger.info(f"[TEMPLATE_MATCHING] Best overall: {min_score:.1f} (ID {min_id})")
	logger.info(f"[TEMPLATE_MATCHING] Threshold: {THRESHOLD_BEST_MATCH}")
	
	# Show threshold analysis
	if min_id != -1:
		margin = THRESHOLD_BEST_MATCH - min_score
		logger.info(f"[TEMPLATE_MATCHING] Margin: {margin:.1f} ({'PASS' if margin >= 0 else 'FAIL'})")
		
		if min_score <= THRESHOLD_BEST_MATCH:
			logger.info(f"[TEMPLATE_MATCHING] MATCH ACCEPTED: Score {min_score:.1f} <= threshold {THRESHOLD_BEST_MATCH}")
			return min_id
		else:
			logger.info(f"[TEMPLATE_MATCHING] MATCH REJECTED: Score {min_score:.1f} > threshold {THRESHOLD_BEST_MATCH}")
			logger.info(f"[TEMPLATE_MATCHING] SUGGESTION: Consider threshold >= {min_score + 100:.0f} to accept this match")
	else:
		logger.info(f"[TEMPLATE_MATCHING] NO CANDIDATES FOUND")
	
	return -1


def display_video_frame_by_frame():
	"""
	Opens a video, tracks all persons, stores their image histories,
	and saves them at the end.
	"""
	# This is the specific person we might want to re-identify if lost.
	# The data collection happens for everyone regardless.
	selected_id_to_track = 1

	if not os.path.isfile(VIDEO_PATH):
		logger.error(f"Video file not found at: {VIDEO_PATH}")
		return

	video = cv2.VideoCapture(VIDEO_PATH)
	if not video.isOpened():
		logger.error("Could not open video file.")
		return

	logger.info("Video opened successfully. Press 'q' to quit.")
	model = YOLO("../yolo_models/yolo11s-seg.pt")
	frame_number = 0
	last_seen_center_id_to_track = None
	try:
		while True:
			success, frame = video.read()
			if not success:
				logger.info("Reached the end of the video.")
				break

			# YOLO automatically converts BGR to RGB internally, so pass the frame as-is
			results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml", classes=[0])
			
			# --- CORE LOGIC ---
			# 1. Extract all persons visible in the current frame.
			# Extract from the original BGR frame
			all_persons_in_frame, all_persons_centers_dict = extract_all_visible_persons(results, frame)
			
			if(all_persons_centers_dict.get(selected_id_to_track) is not None):
				last_seen_center_id_to_track = all_persons_centers_dict.get(selected_id_to_track)

			# 2. Update our main data store with these new images.
			if all_persons_in_frame:
				update_persons_dict(all_persons_in_frame, persons_dict, MAX_RECENT_EXTRACTIONS)

			# --- RE-IDENTIFICATION LOGIC (Optional, runs alongside data collection) ---
			# Check if our specifically tracked person is lost.
			if selected_id_to_track not in all_persons_in_frame.keys():
				# Get the known templates for our lost person from the main dictionary
				templates = persons_dict.get(selected_id_to_track, [])
				
				# Only search if we have templates and there are other people to check
				if templates and all_persons_in_frame and last_seen_center_id_to_track :
					logger.warning(f"Target ID {selected_id_to_track} lost. Attempting to re-acquire...")
					#there is error with the key here, need to add additional checks and print statements
					replacement_id = find_best_match(all_persons_in_frame, templates, last_seen_center_id_to_track ,all_persons_centers_dict)
					if replacement_id != -1:
						logger.info(f"✓ Re-acquired target! Old ID: {selected_id_to_track}, New ID: {replacement_id}")
						selected_id_to_track = replacement_id
			# --- END OF RE-ID LOGIC ---

			annotated_frame = results[0].plot()
			# results[0].plot() returns RGB format, need to convert to BGR for cv2.imshow
			cv2.imshow('Tello Video Feed', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
			frame_number += 1

			if cv2.waitKey(1) & 0xFF == ord('q'):
				logger.info("'q' pressed. Exiting.")
				break

	except KeyboardInterrupt:
		logger.info("Interrupted by user (Ctrl+C).")

	finally:
		logger.info("Releasing video resources.")
		video.release()
		cv2.destroyAllWindows()
		# Use the new save function to save everything.
		save_all_persons(persons_dict, PERSON_OUTPUT_FOLDER)

def setup_standalone_logging():
	"""Setup logging when running template_matching.py standalone (not from main.py)"""
	timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	videos_dir = os.path.join(project_root, "videos")
	log_filename = os.path.join(videos_dir, f'template_matching_log_{timestamp}.log')
	
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s',
		handlers=[
			logging.FileHandler(log_filename),
		]
	)
	logger.info(f"Template matching standalone mode - Log file: {log_filename}")
	print(f"Log file: {log_filename}")  # Print to console only for standalone mode

if __name__ == "__main__":
	setup_standalone_logging()
	display_video_frame_by_frame()