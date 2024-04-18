import numpy as np
import cv2

# Load YOLO
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()

# Function to detect objects
def detect_objects(img):
    height, width, channels = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return indexes, boxes, class_ids

# Function to calculate distance
def calculate_distance(boxes, indexes):
    distances = []
    coordinates = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            center1_x = x + w // 2
            center1_y = y + h // 2
            for j in range(i + 1, len(boxes)):
                if j in indexes:
                    x, y, w, h = boxes[j]
                    center2_x = x + w // 2
                    center2_y = y + h // 2
                    distance = np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
                    distances.append(distance)
                    coordinates.append(((center1_x, center1_y), (center2_x, center2_y)))
    return distances, coordinates

# Read video
cap = cv2.VideoCapture("pedestrian.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    indexes, boxes, class_ids = detect_objects(frame)
    distances, coordinates = calculate_distance(boxes, indexes)

    # Draw boxes and distance
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for distance, (coord1, coord2) in zip(distances, coordinates):
        if distance < 100:
            mid_x = (coord1[0] + coord2[0]) // 2
            mid_y = (coord1[1] + coord2[1]) // 2
            cv2.putText(frame, f"Danger! Distance: {distance}", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Social Distance Detection', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):  # Decreased waiting time to 20 milliseconds
        break

cap.release()
cv2.destroyAllWindows()
