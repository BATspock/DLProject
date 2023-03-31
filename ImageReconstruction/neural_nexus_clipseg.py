#Ensure that clip and transformers are installed
# !pip install clip
# !pip install -q transformers

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import requests
import os
import torch
import spacy
import numpy as np
import clip
import cv2
from typing import Tuple


class getMask(object):

    def __init__(self, classes_file, image_str:str, direction_list: list, exceptions: list):
        
        assert(str == type(image_str))

        self._image = Image.open(image_str)
        self._direction_list = direction_list
        self._exceptions = exceptions
        # remove _ and numbers from image_str
        self._prompt = image_str.split('_')[0]
        self._coco_class_list = None

        with open(classes_file, 'r') as f:
            self._coco_class_list = f.read().splitlines()
        #initliazing segmenter
        self._processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self._model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    def get_smaller_mask(self, mask1, mask2):
        
        mask1 = np.squeeze(mask1).astype(np.uint8)
        mask2 = np.squeeze(mask2).astype(np.uint8)
        mask1 = Image.fromarray(mask1)
        mask2 = Image.fromarray(mask2)
        
        mask1 = np.array(mask1)
        mask2 = np.array(mask2)
        # threshold mask to form binary image

        mask1 = cv2.resize(mask1, (self._image.size[0], self._image.size[1]))
        mask2 = cv2.resize(mask2, (self._image.size[0], self._image.size[1]))

        ret, mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
        ret, mask2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)
        # return mask with lesser number of zero pixels

        if np.count_nonzero(mask1) > np.count_nonzero(mask2):
            return mask1
        else:
            return mask2

    #Function to extract objects and direction from sentence
    def extract_object_and_direction(self)-> Tuple[list, str]:
        #Extract words and bigrams from sentence that are present in class list
        object_list = []
        nlp = spacy.load("en_core_web_sm")

        #Check if any exceptions are present in sentence and if present remove the space between them
        for exception in self._exceptions:
            if exception in self._prompt:
                self._prompt = self._prompt.replace(exception, exception.replace(" ", ""))

        doc = nlp(self._prompt)
        #print(doc)
        for i, token in enumerate(doc):
            if token.text in self._coco_class_list:
                object_list.append(token.text)
            elif token.i == len(doc) - 1:
                break
            elif token.text + " " + token.nbor(1).text in self._coco_class_list:
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
            if token.text in self._direction_list:
                direction = token.text
                break
        
        return object_list, direction
    
    def get_clip_mask(self)->np.ndarray:
        
        objects, _ = self.extract_object_and_direction()
        print(objects)
        inputs = self._processor(text=objects, images=[self._image] * len(objects), padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
        preds = outputs.logits.unsqueeze(1)
        # get only the first object
        mask_np_1 = preds[0].numpy()
        mask_np_2 = preds[1].numpy()
    
        mask_np = self.get_smaller_mask(mask_np_1, mask_np_2)
        # Convert the tensor to a numpy array with the data type `uint8`
        seg_array = np.squeeze(mask_np).astype(np.uint8)
        
        return seg_array
    
if __name__ == "__main__":
    # set the path to the classes file
    maskObj = getMask('../YOLO/classes.txt', 'a tvmonitor to the right of a teddy bear_2.jpeg', ['left', 'right', 'up', 'down'], ["dining table", "tv monitor", "potted plant"])
    mask = maskObj.get_clip_mask()
    cv2.imwrite('newmasktest.png', mask)