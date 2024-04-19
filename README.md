
# Social Distance Detection using YOLO and OpenCV
This Python script detects and alerts for potential violations of social distancing rules in a given video. It uses the YOLO (You Only Look Once) object detection model to detect people in the video frames and calculates the distance between them. If the distance between two people is less than a predefined threshold (in this case, 100 pixels), it highlights them as a potential danger zone.
## Prerequisites

- Python 3.11
- OpenCV
- Numpy library






##  Installation
1.  Clone the repository 
```python
git clone https://github.com/punamcancodee/Social_Distancing_Detector/blob/main/Social_Distance.py

cd Social_Distancing_Detector
```

2. Install the required libraries:
```python
pip install opencv-python numpy
```



## Usage

1. Download the YOLOv4 weights (yolov4.weights), configuration file (yolov4.cfg), and COCO names file (coco.names) from the official YOLO website.
2. Place the downloaded files in the same directory as the Python script.
3. Run the Python script:
```python
python social_distance_detection.py
```
4. The script will start processing the specified video (pedestrian.mp4 by default) and display the output with bounding boxes around detected people and danger alerts if social distancing is violated.

## Customization

- You can adjust the threshold distance for detecting social distancing violations by changing the distance < 100 condition in the code.
- Modify the video file path in the cv2.VideoCapture() function to use a different video source.
## Demo

Here is the link to the full video 
https://drive.google.com/file/d/1xRBewWFCMR4h4UfU-CBk1pyEyAYVvSRw/view?usp=drive_link