import cv2
import time
from djitellopy import Tello

def main():
    drone = Tello()
    drone.connect()
    print(f"Battery: {drone.get_battery()}%")

    drone.streamoff()
    drone.streamon()

    time.sleep(2)  # Give stream a moment to start

    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('tello_capture.avi', fourcc, 20.0, (width, height))

    print("[INFO] Starting video capture. Press 'q' to stop.")

    try:
        while True:
            frame = drone.get_frame_read().frame
            if frame is None:
                print("[WARN] No frame received.")
                continue

            frame = cv2.resize(frame, (width, height))
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('Tello Video Feed', frame_rgb)
            out.write(frame_rgb)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(1/30)  # reduce CPU load, aiming for 30fps

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")

    print("[INFO] Stopping capture, releasing resources.")
    out.release()
    cv2.destroyAllWindows()
    drone.streamoff()

# def main():
# 	print("test")
      
# 	drone = Tello()

# 	drone.connect()
# 	drone.TIME_BTW_RC_CONTROL_COMMANDS = 500
      
# 	print("battery:", drone.get_battery())

# 	drone.streamoff()
# 	drone.streamon()

# 	# This runs a background thread that continuously grabs frames from the drone’s video feed.
# 	frame_read = drone.get_frame_read()

# 	drone.takeoff()

# 	if False:
# 		drone.move_left(100)
# 		drone.move_forward(100)
# 		drone.move_right(100)
# 		drone.move_back(100)
    
# 	# drone.rotate_counter_clockwise(90)
	

# 	drone.land()
      
# 	while True:
# 		# Get the latest frame
# 		frame = frame_read.frame

# 		# Show it in a window
# 		cv2.imshow("Tello Camera", frame)

# 		# Quit when 'q' is pressed
# 		if cv2.waitKey(1) & 0xFF == ord('q'):
# 			break



def main2():
    drone = Tello()
    drone.connect()
    print(f"Battery: {drone.get_battery()}%")

    drone.streamoff()
    drone.streamon()
    time.sleep(2)

    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('tello_capture.avi', fourcc, 20.0, (width, height))

    print("[INFO] Starting video capture. Use WASD/arrows to fly, 't' to take off, 'l' to land, 'q' to quit.")

    try:
        while True:
            frame = drone.get_frame_read().frame
            if frame is None:
                continue

            frame = cv2.resize(frame, (width, height))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('Tello Video Feed', frame_rgb)
            out.write(frame_rgb)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('t'):
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
            elif key == 82:  # UP arrow
                drone.move_up(60)
            elif key == 84:  # DOWN arrow
                drone.move_down(30)
            elif key == 81:  # LEFT arrow
                drone.rotate_counter_clockwise(30)
            elif key == 83:  # RIGHT arrow
                drone.rotate_clockwise(30)

            time.sleep(0.1)  # reduce CPU load

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")

    print("[INFO] Stopping capture, releasing resources.")
    out.release()
    cv2.destroyAllWindows()
    drone.streamoff()



if __name__ == '__main__':
    main()