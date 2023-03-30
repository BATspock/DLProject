import cv2
import os
import time
import sys
import numpy as np
from PIL import Image
import cv2
import itertools
import spacy


def extract_object_and_direction(sentence, class_list, direction_list, exceptions):
    #Extract words and bigrams from sentence that are present in class list
    object_list = []
    nlp = spacy.load("en_core_web_sm")

    #Check if any exceptions are present in sentence and if present remove the space between them
    for exception in exceptions:
        if exception in sentence:
            sentence = sentence.replace(exception, exception.replace(" ", ""))

    doc = nlp(sentence)
    for i, token in enumerate(doc):
        if token.text in class_list:
            object_list.append(token.text)
        elif token.i == len(doc) - 1:
            break
        elif token.text + " " + token.nbor(1).text in class_list:
            object_list.append(token.text + " " + token.nbor(1).text)

    #If bigram is present in class list, remove all elements after second element
    if len(object_list)>2:
        for i, object in enumerate(object_list):
            if " " in object:
                object_list = object_list[:2]
                break
    
    #Extract direction from sentence that is present in direction list
    direction = ""
    for token in doc:
        if token.text in direction_list:
            direction = token.text
            break
    
    return object_list, direction


def detect_object(classes_final, outputs, width, height, classes):
    
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

                if class_id not in class_ids:
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    # print(len(boxes))
    # Perform non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print("checking",len(boxes))


    ############################################################
    # CODE FOR TESTING 2 OBJECTS
    ############################################################
    class_label = [classes[class_ids[i]] for i in indices if classes[class_ids[i]] in classes_final]
    class_ind = [class_ids[i] for i in indices if classes[class_ids[i]] in classes_final]
    print(class_label, class_ind)

    bounding_boxes = {}
    print("indices", indices)
    
    
    #check if the class_label in the text prompt are in the image
    if len(class_ind)==1:
        class_cur = classes[class_ind[0]]

        if class_cur == classes_final[0]:
            bounding_boxes[0] = boxes[0]
            bounding_boxes[1] = boxes[1]
        else:
            bounding_boxes[0] = boxes[1]
            bounding_boxes[1] = boxes[0]
    else:
        class_ind0 = classes[class_ind[0]]
        class_ind1 = classes[class_ind[1]]
        print("check", class_ind0, class_ind1)
        ind0, ind1 = -1, -1

        if classes_final[0] == class_ind0:
            ind0 = 0
            bounding_boxes[0] = boxes[0]
        elif classes_final[0] == class_ind1:
            ind0 = 1
            bounding_boxes[0] = boxes[1]

        if classes_final[1] == class_ind0:
            ind1 = 0
            bounding_boxes[1] = boxes[0]
        elif classes_final[1] == class_ind1:
            ind1 = 1
            bounding_boxes[1] = boxes[1]
        
        if ind0 == -1 and ind1!=-1:
            if ind1 == 0:
                bounding_boxes[0] = boxes[1]
            else:
                bounding_boxes[0] = boxes[0]
        
        if ind1 == -1 and ind0!=-1:
            if ind0 == 0:
                bounding_boxes[1] = boxes[1]
            else:
                bounding_boxes[1] = boxes[0]

    if len(bounding_boxes)==2:
        return True, bounding_boxes, classes_final
    return False, {}, classes_final

    
        
def relation_objects(image,class_label, direction, boxes):
    # Get the indices of the objects we want to compare

    x1_obj1, y1_obj1, x2_obj1, y2_obj1 = boxes[0]
    coordinates = [x1_obj1, y1_obj1, x1_obj1 + x2_obj1, y1_obj1 + y2_obj1]
    object1_center = ((x1_obj1 + x2_obj1) / 2, (y1_obj1 + y2_obj1) / 2)
    label_obj1 = class_label

    x1_obj2, y1_obj2, x2_obj2, y2_obj2 = boxes[1]
    object2_center = ((x1_obj2 + x2_obj2) / 2, (y1_obj2 + y2_obj2) / 2)
    label_obj2 = class_label

    print(object1_center, object2_center)

    # check based on the direction
    if direction == 'left':
        if object1_center[0] > object2_center[0]:
            return True
        else:
            return False
    elif direction == 'right':
        if object1_center[0] < object2_center[0]:
            return True
        else:
            return False
    elif direction == 'above':
        if object1_center[1] < object2_center[1]:
            return True
        else:
            return False
    elif direction == 'below':
        if object1_center[1] > object2_center[1]:
            return True
        else:
            return False
    else:
        return False



if __name__ == "__main__":
    input_folder = 'images_final_2_objects'

    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    # image_files = image_files[22:24]
    num_images = len(image_files)

    classes = []
    with open("YOLO/classes.txt", "r") as f:
        classes = [cname.strip() for cname in f.readlines()]

    correct_count = 0
    directions = ["above", "below", "left", "right"]
    exceptions = ["dining table", "tv monitor", "potted plant"]

    exception_list = []
    exception_no = []
    not_detected_list = []

    for i, image_file in enumerate(image_files):
        print("#" * 50, "Image", i + 1)
        
        image_path = os.path.join(input_folder, image_file)
        input_image = cv2.imread(image_path)

        text = image_file.split('.')[0]
        # remove last character
        text = text[:-2]
        print("Image: ", text)
        try:
            class_labels, direction = extract_object_and_direction(text,classes,directions,exceptions)
            print("Given object:", class_labels)

            model = cv2.dnn.readNetFromDarknet('YOLO/yolov3.cfg', 'YOLO/yolov3.weights')
            
            height, width, _ = input_image.shape
            blob = cv2.dnn.blobFromImage(input_image, 1/255, (416, 416), swapRB=True)
            model.setInput(blob)
            output_layers = model.getUnconnectedOutLayersNames()
            outputs = model.forward(output_layers)

            bool_present, bounding_box, classes_final = detect_object(class_labels, outputs, width, height, classes)
            print("Given object:", class_labels)
            print("Given direction:", direction)

            if bool_present:
                print(class_labels,"are present")
                if relation_objects(input_image, class_labels, direction, bounding_box):
                    print(classes_final[0], "is on the", direction, "of the ", classes_final[1])
                    correct_count += 1
                else:
                    print(classes_final[0], "is not on the", direction, "of the ", classes_final[1])
            else:
                not_detected_list.append(image_file)
                print(class_labels, "not detected")
        except:
            print("Exception")
            exception_list.append(image_file)
            exception_no.append(i)

        
        print("#" * 50)

print("Exceptions:", exception_list)
print("Num exceptions:", len(exception_list))
print("Exception no:", exception_no)

print("Not detected:", not_detected_list)
print("Num not detected:", len(not_detected_list))


if correct_count == 0:
    print("Accuracy: 0%")
else:
    print("Accuracy:", correct_count/num_images*100, "%")
        
    
