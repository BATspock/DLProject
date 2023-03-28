import cv2
import time
import sys
import numpy as np
from PIL import Image
import cv2

# Load the model
model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Load the input image
image = cv2.imread('test3.png')

classes = []
with open("/Users/nimishamittal/Documents/USC/CSCI566_DL/DL_Project/classes.txt", "r") as f:
    classes = [cname.strip() for cname in f.readlines()]

# Get the image dimensions
height, width, _ = image.shape

# Create a blob from the image
blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True)

# Set the input for the model
model.setInput(blob)

# Get the output layer names
output_layers = model.getUnconnectedOutLayersNames()

# Run inference through the model
outputs = model.forward(output_layers)

# Get the bounding boxes, confidences, and class IDs
boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = center_x - w // 2
            y = center_y - h // 2
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Perform non-maximum suppression to remove overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indices)

# Draw the bounding boxes and labels on the image
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

for i in indices:
    print(i)
    # i = i[0]
    box = boxes[i]
    x, y, w, h = box
    label = str(classes[class_ids[i]])
    color = colors[i]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
