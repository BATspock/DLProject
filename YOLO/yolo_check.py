import cv2
import time
import sys
import numpy as np
from PIL import Image
import cv2
import itertools

text_prompt = "dog cat"

def detect_object(class_label, outputs, width, height, classes):
    
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

                # class name
            
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    ############################################################
    # CODE FOR TESTING 2 OBJECTS
    ############################################################
    # classes_final = [classes[class_ids[i]] for i in indices if classes[class_ids[i]] in classes_to_compare]

    #check if the class_label in the text prompt are in the image
    for i in indices:
        if classes[class_ids[i]] == class_label:
            return True, boxes
    return False, boxes
    
        
def relation_objects(class_label, direction, boxes, width, height):
    # Get the indices of the objects we want to compare
    obj1_idx = 0

    x1, y1, x2, y2 = boxes[obj1_idx]
    object1_center = ((x1 + x2) / 2, (y1 + y2) / 2)
    label_obj1 = class_label

    # check based on the direction
    if direction == 'left':
        if object1_center[0] < width/2:
            return True
        else:
            return False
    elif direction == 'right':
        if object1_center[0] > width/2:
            return True
        else:
            return False
    elif direction == 'top':
        if object1_center[1] < height/2:
            return True
        else:
            return False
    elif direction == 'bottom':
        if object1_center[1] > height/2:
            return True
        else:
            return False
    else:
        return False


############################################################
# CODE FOR TESTING BOUNDING BOXES
############################################################
    # Draw the bounding boxes and labels on the image
    # colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    # for i in indices:
    #     print(i)
    #     # i = i[0]
    #     box = boxes[i]
    #     x, y, w, h = box
    #     label = str(classes[class_ids[i]])
    #     color = colors[i]
    #     cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    #     cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # # Show the image
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    input_image = "test4.png"
    class_label = "dog"
    direction = "left"

    # Load the model
    model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

    # Load the input image
    image = cv2.imread(input_image)

    classes = []
    with open("classes.txt", "r") as f:
        classes = [cname.strip() for cname in f.readlines()]

    ############################################################
    # CODE FOR TESTING 2 OBJECTS
    ############################################################
    # find classes of tokens in text_prompt
    # classes_to_compare = []
    # for token in text_prompt.split():
    #     if token in classes:
    #         classes_to_compare.append(token)
    # print(classes_to_compare)

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

    bool_present, bounding_box = detect_object(class_label, outputs, width, height, classes)

    if bool_present:
        print(class_label," is present")
        print(relation_objects(class_label, direction, bounding_box, width, height))
    