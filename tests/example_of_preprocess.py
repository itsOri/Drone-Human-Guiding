import cv2
import numpy as np
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Use relative paths from parent directory
template_path = os.path.join(parent_dir, "photos/person_id_13_capture_1.png")
img1_path = os.path.join(parent_dir, "photos/person_id_13_capture_14.png")

# Read images
template = cv2.imread(template_path)
img1 = cv2.imread(img1_path)

# Check if images were loaded successfully
if template is None or img1 is None:
    print("Error: Could not load images")
    exit(1)

# do gaussian blur on both images
template_blurred = cv2.GaussianBlur(template, (16, 16), 0)
img1_blurred = cv2.GaussianBlur(img1, (16, 16), 0)

# show the original images
cv2.imshow("template", template)
cv2.imshow("img1", img1)

# save images locally in the same folder as the script
cv2.imwrite(os.path.join(script_dir, "template_blurred.png"), template_blurred)
cv2.imwrite(os.path.join(script_dir, "img1_blurred.png"), img1_blurred)

# show the blurred images
cv2.imshow("template_blurred", template_blurred)
cv2.imshow("img1_blurred", img1_blurred)

# Wait for key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()