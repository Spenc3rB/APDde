# Stereo Vision Depth Estimation with YOLO - Part of APDde [Autonomous Pedestrian Dection with Depth Estimation](https://gtihub.com/Electric-Go-Kart)

![Example video](https://github.com/Spenc3rB/YOLOv8Depth/assets/101066043/0ee56077-a8ca-46f9-b68e-89c77874a969)

This repository contains a Python script for stereo vision depth estimation using the YOLO (You Only Look Once) object detection model. The system employs two cameras, a right and a left camera, to capture stereo images and calculates the depth of detected objects in real-time. Depth is estimated based on triangulation principles.

## Requirements

Make sure to install the following Python libraries before running the script:

- OpenCV
- NumPy
- Ultralytics (YOLO)

## Setup

Adjust the camera parameters in the script to match your setup:

```python
right_camera_index = 1
left_camera_index = 0
resolution = (320, 320)
```

Set the stereo vision parameters according to your camera setup:

```python
stereo_parameters = {
    'FRAME_RATE': 30,
    'B': 15,       # Baseline (distance between the two camera centers)
    'F': 3.67,     # Focal length of the cameras
    'ALPHA': 70.42 # Stereo vision alpha parameter
}
```

Configure the YOLO model with the desired weights file (e.g., "best.pt"):

```python
model = YOLO("best.pt")
```

## Main Program Loop

The main program loop captures frames from the right and left cameras, performs calibration, runs YOLO predictions, and calculates depth using triangulation. The depth information is then visualized on the left camera frame, including bounding boxes and depth values.

## Usage

Run the script (in progress):

```bash
python reference.py
```

Press 'q' to exit the application.

## Note

Inference was performed on an Intel i7-13700K processor.

## Acknowledgments

- YOLO: [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
- Triangulation Module: [Custom Triangulation Module](https://github.com/niconielsen32/ComputerVision/)

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
