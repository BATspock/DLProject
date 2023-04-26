#Import the necessary packages
import os
import cv2
import numpy as np
import time
import random
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import torch
import supervision as sv

#Defining global variables
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = 'D:\Downloads'
IMAGE_PATH = 'D:\College Work\CSCI 566\Project\Code\Data Extraction\code\dall e api images'
IMAGE_NAME = 'a hair drier above a mouse_2.jpg'

#Initialize the predictor
sam = sam_model_registry[MODEL_TYPE](checkpoint = CHECKPOINT_PATH).to(device = DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

os.chdir(IMAGE_PATH)
image_bgr = cv2.imread(IMAGE_NAME)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

#Predict the masks
sam_result = mask_generator.generate(image_rgb)

#Visualize the masks
mask_annotator = sv.MaskAnnotator()

detections = sv.Detections.from_sam(sam_result = sam_result)

annotated_img = mask_annotator.annotate(scene = image_rgb.copy(), detections = detections)

sv.plot_images_grid(images = [image_bgr, annotated_img],
                    grid_size = (1, 2),
                    titles = ["Original Image", "Predicted Masks"])
