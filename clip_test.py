from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import os

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# read all files in dalle_imgs_2objects
image_files = [f for f in os.listdir('Dalle_correct_images') if f.endswith('.jpg')]
directions = {"left": "right", "right": "left", "above": "below", "below": "above"}

total_prob = 0 
total_images = len(image_files)


for image_file in image_files:
    # load image
    image = Image.open(os.path.join('Dalle_correct_images', image_file))
    
    # get file name without extension
    image_name = os.path.splitext(image_file)[0][:-2]
    print(image_name)

    negate_image_name = ' '.join([directions.get(word, word) for word in image_name.split()])


    inputs = processor(text=[image_name, negate_image_name], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    total_prob += probs[0][1].item()

print(total_prob/total_images)

