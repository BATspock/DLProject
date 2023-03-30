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

#Function to extract objects and direction from sentence
def extract_object_and_direction(sentence: str, class_list: list, direction_list: list, exceptions: list)-> Tuple[list, str]:
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

#Initiating CLIPseg model from hugging face
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

#Read coco class list
with open('/content/coco_class_list.txt', 'r') as f:
    coco_class_list = f.read().splitlines()

#Read base image here
image = Image.open("{Image path here}")

#Get image prompt by taking the name of the file and removing numbers
image_prompt = ""

#Need to get all objects from prompt text present in coco list
objects, direction = extract_object_and_direction(image_prompt, coco_class_list, ["above", "below", "left", "right"], ["dining table", "tv monitor", "potted plant]"])

#Get all objects in image
inputs = processor(text=objects, images=[image] * len(objects), padding="max_length", return_tensors="pt")

#Mask Prediction
with torch.no_grad():
  outputs = model(**inputs)
preds = outputs.logits.unsqueeze(1)

#Visualising the image and the object segmenetation masks
# _, ax = plt.subplots(1, len(objects) + 1, figsize=(3*(len(objects) + 1), 2))
# [a.axis('off') for a in ax.flatten()]
# ax[0].imshow(image)
# [ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(objects))];
# [ax[i+1].text(0, -15, prompt) for i, prompt in enumerate(objects)];

#Iterate through mask predictions and convert them into numpy arrays and store masks as image files
for i in range(len(preds)):
  mask_np = preds[i].numpy()
  
  # stack all four arrays along the depth dimension to get a 4-channel image
  #mask_array = np.stack([mask1_np, mask2_np, mask3_np, mask4_np], axis=-1)

  # Convert the tensor to a numpy array with the data type `uint8`
  seg_array = np.squeeze(mask_np).astype(np.uint8)

  # Create a PIL Image object from the numpy array
  seg_image = Image.fromarray(seg_array)

  rgb_img = seg_image.convert('RGB')

  # Save the image as a JPEG file
  rgb_img.save(f'{objects[i]} mask.jpg')

#Read Mask and perform downstream tasks