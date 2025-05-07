# this is a file for running basic commands with the drone controls
# Including fly, up, down, land...
from djitellopy import Tello
import cv2



def main():
	print("test")
      
	drone = Tello()

	drone.connect()
	drone.TIME_BTW_RC_CONTROL_COMMANDS = 2
      
	print("battery:", drone.get_battery())

	drone.streamoff()
	drone.streamon()
     
	# This runs a background thread that continuously grabs frames from the drone’s video feed.
	# frame_read = drone.get_frame_read()

	drone.takeoff()

	drone.move_left(100)
	drone.move_forward(100)
	drone.move_right(100)
	drone.move_back(100)
    
	# drone.rotate_counter_clockwise(90)
	

	drone.land()
      
	while True:
		# Get the latest frame
		frame = frame_read.frame

		# Show it in a window
		cv2.imshow("Tello Camera", frame)

		# Quit when 'q' is pressed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


if __name__ == '__main__':
    main()