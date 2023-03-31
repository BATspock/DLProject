import cv2
import time
import sys
import numpy as np
from PIL import Image
import itertools

class ObjectDetector:
    def __init__(self):
        self._config = 'YOLO/yolov3.cfg'
        self._weights = 'YOLO/yolov3.weights'
        self.coco_file = 'YOLO/classes.txt'

        self._coco_classes = []
        with open(self.coco_file, "r") as f:
            self._coco_classes = [cname.strip() for cname in f.readlines()]
        print(self._coco_classes)
        self._model = cv2.dnn.readNetFromDarknet(self._config, self._weights)


    def is_consistent(self, input_image_path, prompt_labels, prompt_direction)->bool:  
        assert(type(input_image_path) == str)
        assert(type(prompt_labels) == list)
        assert(type(prompt_direction) == str)

        # Model setup for each image      
        input_image = cv2.imread(input_image_path)
        height, width, _ = input_image.shape
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (416, 416), swapRB=True)
        self._model.setInput(blob)
        output_layers = self._model.getUnconnectedOutLayersNames()
        outputs = self._model.forward(output_layers)

        # Detect objects
        bool_present, bounding_boxes, _ = self.detect_object(prompt_labels, outputs, width, height)

        if bool_present:
            if self.relation_objects(input_image, prompt_direction, bounding_boxes):
                print(prompt_labels[0], "is on the", prompt_direction, "of the ", prompt_labels[1])
                return True
            else:
                print(prompt_labels[0], "is not on the", prompt_direction, "of the ", prompt_labels[1])
                return False
        return False


    def detect_object(self,classes_final, outputs, width, height)->(bool, list, list):
        assert(type(classes_final) == list)
        assert(type(outputs) == tuple)
        assert(type(width) == int)
        assert(type(height) == int)

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


        ############################################################
        # CODE FOR TESTING 2 OBJECTS
        ############################################################
        class_label = [self._coco_classes[class_ids[i]] for i in indices if self._coco_classes[class_ids[i]] in classes_final]
        class_ind = [class_ids[i] for i in indices if self._coco_classes[class_ids[i]] in classes_final]
        print(class_label, class_ind)

        bounding_boxes = {}
        print("indices", indices)
        
        
        #check if the class_label in the text prompt are in the image
        if len(class_ind)==1:
            class_cur = self._coco_classes[class_ind[0]]

            if class_cur == classes_final[0]:
                bounding_boxes[0] = boxes[0]
                bounding_boxes[1] = boxes[1]
            else:
                bounding_boxes[0] = boxes[1]
                bounding_boxes[1] = boxes[0]
        else:
            class_ind0 = self._coco_classes[class_ind[0]]
            class_ind1 = self._coco_classes[class_ind[1]]
    
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

      
    def relation_objects(self,input_image, direction, bounding_boxes)->bool:
        assert(type(input_image) == np.ndarray)
        assert(type(direction) == str)
        assert(type(bounding_boxes) == dict)

        # Get the indices of the objects we want to compare
        x1_obj1, y1_obj1, x2_obj1, y2_obj1 = bounding_boxes[0]
        coordinates = [x1_obj1, y1_obj1, x1_obj1 + x2_obj1, y1_obj1 + y2_obj1]
        object1_center = ((x1_obj1 + x2_obj1) / 2, (y1_obj1 + y2_obj1) / 2)

        x1_obj2, y1_obj2, x2_obj2, y2_obj2 = bounding_boxes[1]
        object2_center = ((x1_obj2 + x2_obj2) / 2, (y1_obj2 + y2_obj2) / 2)

        # check based on the direction
        if direction == 'right':
            if object1_center[0] > object2_center[0]:
                return True
            else:
                return False
        elif direction == 'left':
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
    input_image_path = "dalle_api_images/a bear above a truck_0.jpg"
    prompt_labels = ['bear', 'truck']
    prompt_direction = "above"

    object_detector = ObjectDetector()
    object_detector.is_consistent(input_image_path, prompt_labels, prompt_direction)