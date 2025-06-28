from ultralytics import YOLO

# we've trieid different 

model = YOLO("yolo11s-seg.pt")
results = model.track("tello_captureori2.avi", save=True, show=True)