import cv2
import time
import sys
import numpy as np
from PIL import Image
import cv2
import itertools
import spacy
nlp = spacy.load("en_core_web_sm")


def extract_object_moved(text):
    # Parse the text prompt using the Spacy nlp object
    doc = nlp(text)
    
    # Initialize variables to store the object and direction
    object_moved = None
    direction = None
    
    # Iterate through the parsed output to find the verb and its direct object
    for token in doc:
        # Identify the root verb of the sentence that indicates movement
        if token.pos_ == "VERB" and (token.dep_ == "ROOT" or token.dep_ == "acl"):
            verb = token
            # Iterate through the children of the verb to find the direct object
            for child in verb.children:
                if child.dep_ == "dobj":
                    # Store the text of the direct object as the object being moved
                    object_moved = child.text
                elif child.dep_ == "prep":
                    # Identify the preposition that indicates the direction
                    prep = child
                    # Iterate through the children of the preposition to find the direction
                    for grandchild in prep.children:
                        if grandchild.dep_ == "pobj":
                            # Store the text of the preposition's object as the direction
                            direction = grandchild.text
    
    # Return the object being moved and direction as a tuple
    return object_moved, direction



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
        if object1_center[1] > height/2:
            return True
        else:
            return False
    elif direction == 'bottom':
        if object1_center[1] < height/2:
            return True
        else:
            return False
    else:
        return False


if __name__ == "__main__":
    input_image = "test6.jpeg"
    text = "Move the orange to the top."
    class_label, direction = extract_object_moved(text)

    # Load the model
    model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

    # Load the input image
    image = cv2.imread(input_image)

    classes = []
    with open("classes.txt", "r") as f:
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

    bool_present, bounding_box = detect_object(class_label, outputs, width, height, classes)
    print("Given object:", class_label)
    print("Given direction:", direction)
    if bool_present:
        print(class_label,"is present")
        if relation_objects(class_label, direction, bounding_box, width, height) == True:
            print(class_label, "is on the", direction, "side of the image")
        else:
            print(class_label, "is not on the", direction, "side of the image")
    else:
        print(class_label, "not detected")
    