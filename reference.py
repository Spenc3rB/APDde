import sys
import cv2
import numpy as np
import time
from ultralytics import YOLO
import triangulation as tri
import calibration

# Camera setup
right_camera_index = 1
left_camera_index = 0
resolution = (320, 320)

cap_right = cv2.VideoCapture(right_camera_index)
cap_left = cv2.VideoCapture(left_camera_index)

cap_right.set(3, resolution[0])
cap_right.set(4, resolution[1])
cap_left.set(3, resolution[0])
cap_left.set(4, resolution[1])

# Stereo vision setup parameters
stereo_parameters = {
    'FRAME_RATE': 30,
    'B': 15,
    'F': 3.67,
    'ALPHA': 70.42
}

# YOLO model setup
model = YOLO("best.pt")

# Main program loop
try:
    while cap_right.isOpened() and cap_left.isOpened():
        _, frame_right = cap_right.read()
        _, frame_left = cap_left.read()

        start = time.time()

        # Calibration
        frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

        # YOLO prediction
        results_left, results_right = model.predict(frame_left, imgsz=320), model(frame_right, imgsz=320)

        xyxys_left, xyxys_right = [], []
        right_points, left_points = [], []

        for result_left, result_right in zip(results_left, results_right):
            left_boxes = result_left.boxes.cpu().numpy()
            right_boxes = result_right.boxes.cpu().numpy()

            xyxys_left.extend(left_boxes.xyxy)
            xyxys_right.extend(right_boxes.xyxy)

            left_points.extend([(int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)) for xyxy in xyxys_left])
            right_points.extend([(int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)) for xyxy in xyxys_right])

        # Depth calculation and visualization
        if left_points and right_points:
            for right_point, left_point, xyxyl, xyxyr in zip(right_points, left_points, xyxys_left, xyxys_right):
                depth = tri.find_depth(left_point, right_point, frame_right, frame_left, stereo_parameters['B'], stereo_parameters['F'], stereo_parameters['ALPHA'])
                cv2.rectangle(frame_left, (int(xyxyl[0]), int(xyxyl[1])), (int(xyxyl[2]), int(xyxyl[3])), (0, 255, 0), 2)
                cv2.putText(frame_left, f"{depth:.2f}", (int(xyxyl[0]), int(xyxyl[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display FPS
        fps = 1 / (time.time() - start)
        cv2.putText(frame_left, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display frames
        cv2.imshow("Frame", frame_left)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the cameras
    cap_right.release()
    cap_left.release()
    cv2.destroyAllWindows()