from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.track("../tello_captureori2.avi", save=True, show=True)