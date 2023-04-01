import cv2
import time
import sys
import numpy as np
from PIL import Image
import itertools
import os
import csv

class Filter2Objects:
    def __init__(self, images_folder, output_folder):
        self.images_folder = images_folder
        self.output_folder = output_folder

        # make directory for output images
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Load the model
        self.model = cv2.dnn.readNetFromDarknet('YOLO/yolov3.cfg', 'YOLO/yolov3.weights')

        self.classes = []
        with open("YOLO/classes.txt", "r") as f:
            self.classes = [cname.strip() for cname in f.readlines()]
            
    
    def detect_object(self, outputs, width, height)->list:
        assert(type(outputs) == tuple)
        assert(type(width) == int)
        assert(type(height) == int)

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
        return boxes

    
    def images_more2_objects(self)->None:
        total_images = 0
        count_images = 0
        for filename in os.listdir(self.images_folder):
            image = cv2.imread(os.path.join(self.images_folder, filename))
    
            # Get the image dimensions
            height, width, _ = image.shape

            # Create a blob from the image
            blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True)

            # Set the input for the model
            self.model.setInput(blob)

            # Get the output layer names
            output_layers = self.model.getUnconnectedOutLayersNames()

            # Run inference through the model
            outputs = self.model.forward(output_layers)

            bounding_box = self.detect_object(outputs, width, height)

            if len(bounding_box) >=2:
                # save images with 2 objects in a new folder
                cv2.imwrite(os.path.join(self.output_folder, filename), image)
                count_images += 1
            
            total_images += 1
        
        print(f"Total images: {total_images}")
        print(f"Number of images with 2 objects: {count_images}")



if __name__ == "__main__":
    """
    input_folder_to_images: folder with images to check
    destination_folder: folder to save images with 2 objects
    """
    input_folder_to_images = 'pix2pix_imgs'
    destination_folder = 'pix2pix_imgs_2objects'

    detect_2objects = Filter2Objects(input_folder_to_images, destination_folder)
    detect_2objects.images_more2_objects()


