import cv2
import os
import time

def display_video_frame_by_frame():
    """
    Opens a video file, reads it frame by frame, and displays it on the screen.
    Includes a placeholder for you to add your per-frame calculations.
    """
    video_path = "../tello_captureori2.avi"

    # --- 1. Check if the video file exists ---
    if not os.path.isfile(video_path):
        print(f"[ERROR] Video file not found at: {video_path}")
        return

    # --- 2. Open the video file ---
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("[ERROR] Could not open video file.")
        return

    print("[INFO] Video opened successfully. Press 'q' to quit.")

    try:
        while True:
            # --- 3. Read one frame from the video ---
            # .read() returns two values: 
            #   - a boolean (True if frame was read successfully)
            #   - the frame itself
            success, frame = video.read()

            # If success is False, it means we've reached the end of the video
            if not success:
                print("[INFO] Reached the end of the video.")
                break


            # --- 4. Display the frame in a window ---
            cv2.imshow('Tello Video Feed', frame)

            # --- 5. Wait for a key press (CRITICAL STEP) ---
            # cv2.waitKey(1) waits for 1 millisecond. It is REQUIRED for cv2.imshow() to work.
            # It also returns the key that was pressed.
            # We check if the pressed key's ASCII value is the same as 'q'.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 'q' pressed. Exiting.")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user (Ctrl+C).")

    finally:
        # --- 6. Clean up: release the video and close all windows ---
        print("[INFO] Releasing video resources.")
        video.release()
        cv2.destroyAllWindows()

# --- Run the function ---
if __name__ == "__main__":
    display_video_frame_by_frame()