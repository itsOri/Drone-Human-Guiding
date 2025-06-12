import cv2
import os
import numpy as np
from ultralytics import YOLO
from datetime import datetime

import template_matching_function as template_match_file

# --- GLOBAL CONSTANTS & CONFIGURATION ---
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
PERSON_OUTPUT_FOLDER = f'./persons_data_{TIMESTAMP}'
VIDEO_PATH = "../tello_capture_ori_trees.avi"
MAX_RECENT_EXTRACTIONS = 20
THRESHOLD_BEST_MATCH = 2000

# --- MAIN DATA STORE ---
# This dictionary will hold the image histories for ALL detected persons.
# Keys are person IDs, values are lists of images.
person_dict = {}

# (extract_person_from_frame function can be removed if not used elsewhere, but we'll keep it for now)
def extract_person_from_frame(results, target_id, original_frame, frame_number):
	# ... (your function is fine, no changes needed)
	if results[0].boxes is None or results[0].boxes.id is None:
		return None
	track_ids = results[0].boxes.id.int().cpu().tolist()
	if target_id not in track_ids:
		return None
	person_index = track_ids.index(target_id)
	if results[0].masks is None:
		print(f"[WARN] No masks found for frame {frame_number}, cannot extract person.")
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
	# ... (your function is fine, no changes needed here, just removed unused frame_number)
	if results[0].boxes is None or results[0].boxes.id is None or results[0].masks is None:
		return {}
	track_ids = results[0].boxes.id.int().cpu().tolist()
	all_masks = results[0].masks
	all_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
	extracted_persons_dict = {}
	for i in range(len(track_ids)):
		current_id = track_ids[i]
		mask = all_masks[i]
		binary_mask = mask.data[0].cpu().numpy().astype("uint8")
		extracted_person = cv2.bitwise_and(original_frame, original_frame, mask=binary_mask)
		x1, y1, x2, y2 = all_boxes[i]
		cropped_person = extracted_person[y1:y2, x1:x2]
		if cropped_person.size > 0:
			extracted_persons_dict[current_id] = cropped_person
	return extracted_persons_dict


def update_person_dict(all_persons_in_frame, person_dict, max_extractions):
	"""
	Updates the main dictionary with new extractions for each person.
	"""
	for person_id, image in all_persons_in_frame.items():
		if person_id not in person_dict:
			person_dict[person_id] = []
		person_dict[person_id].append(image)
		if len(person_dict[person_id]) > max_extractions:
			person_dict[person_id].pop(0)


def save_all_persons(person_dict, base_output_folder):
	"""
	Saves all collected images from the person_dict to disk, organized
	in sub-folders by person ID.
	"""
	if not person_dict:
		print("[INFO] Person dictionary is empty. Nothing to save.")
		return
	print(f"\n[INFO] Saving all collected person images to '{base_output_folder}'...")
	os.makedirs(base_output_folder, exist_ok=True)
	for person_id, image_list in person_dict.items():
		person_folder = os.path.join(base_output_folder, f"person_{person_id}")
		os.makedirs(person_folder, exist_ok=True)
		print(f"  > Saving {len(image_list)} images for Person ID {person_id}...")
		for i, img in enumerate(image_list):
			filename = os.path.join(person_folder, f"capture_{i+1}.png")
			cv2.imwrite(filename, img)
	print("[INFO] All images saved successfully.")


def find_best_match(other_persons_dict, selected_id_templates):
	# ... (your function is fine, no changes needed)
	min_score = -1
	min_id = -1
	for id, extracted_image in other_persons_dict.items():
		current_scores = [template_match_file.template_match(template, extracted_image) for template in selected_id_templates]
		current_score = min(current_scores)
		if min_score == -1 or current_score < min_score:
			min_score = current_score
			min_id = id
	print(f"Current lowest score: {min_score}, ID: {min_id}")
	if min_id != -1 and min_score <= THRESHOLD_BEST_MATCH:
		return min_id
	return -1


def display_video_frame_by_frame():
	"""
	Opens a video, tracks all persons, stores their image histories,
	and saves them at the end.
	"""
	# This is the specific person we might want to re-identify if lost.
	# The data collection happens for everyone regardless.
	selected_id_to_track = 31

	if not os.path.isfile(VIDEO_PATH):
		print(f"[ERROR] Video file not found at: {VIDEO_PATH}")
		return

	video = cv2.VideoCapture(VIDEO_PATH)
	if not video.isOpened():
		print("[ERROR] Could not open video file.")
		return

	print("[INFO] Video opened successfully. Press 'q' to quit.")
	model = YOLO("../yolo_models/yolo11s-seg.pt")
	frame_number = 0

	try:
		while True:
			success, frame = video.read()
			if not success:
				print("[INFO] Reached the end of the video.")
				break

			results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml", classes=[0])
			
			# --- CORE LOGIC ---
			# 1. Extract all persons visible in the current frame.
			all_persons_in_frame = extract_all_visible_persons(results, frame)
			
			# 2. Update our main data store with these new images.
			if all_persons_in_frame:
				update_person_dict(all_persons_in_frame, person_dict, MAX_RECENT_EXTRACTIONS)

			# --- RE-IDENTIFICATION LOGIC (Optional, runs alongside data collection) ---
			# Check if our specifically tracked person is lost.
			if selected_id_to_track not in all_persons_in_frame.keys():
				# Get the known templates for our lost person from the main dictionary
				templates = person_dict.get(selected_id_to_track, [])
				
				# Only search if we have templates and there are other people to check
				if templates and all_persons_in_frame:
					print(f"[WARN] Target ID {selected_id_to_track} lost. Attempting to re-acquire...")
					replacement_id = find_best_match(all_persons_in_frame, templates)
					if replacement_id != -1:
						print(f"[SUCCESS] Re-acquired target! Old ID: {selected_id_to_track}, New ID: {replacement_id}")
						selected_id_to_track = replacement_id
			# --- END OF RE-ID LOGIC ---

			annotated_frame = results[0].plot()
			cv2.imshow('Tello Video Feed', annotated_frame)
			frame_number += 1

			if cv2.waitKey(1) & 0xFF == ord('q'):
				print("[INFO] 'q' pressed. Exiting.")
				break

	except KeyboardInterrupt:
		print("[INFO] Interrupted by user (Ctrl+C).")

	finally:
		print("[INFO] Releasing video resources.")
		video.release()
		cv2.destroyAllWindows()
		# Use the new save function to save everything.
		save_all_persons(person_dict, PERSON_OUTPUT_FOLDER)


if __name__ == "__main__":
	display_video_frame_by_frame()