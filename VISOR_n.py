# Visor for n objects
# positional_words = ["above",
#                     "below",
#                     "beside",
#                     "between",
#                     "beyond",
#                     "near",
#                     "far from",
#                     "outside",
#                     "inside",
#                     "in front of",
#                     "behind",
#                     "on top of",
#                     "underneath",
#                     "next to",
#                     "adjacent to",
#                     "to the right of",
#                     "to the left of"]

# Dog to the left of cat to the left of table

# objects:[dog,cat,table]
# positions:[left,left]


# lamp on the right of a book to the left of a bird

# objects:[lamp, book, bird]
# positions:[right left]

# table on top of the floor

# objects: [table, floor]
# positions: [top]


import numpy as np
import cv2
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import sys
import os

sys.path.append('/Users/nimishamittal/Documents/USC/CSCI566_DL/DLProject')
import Visor
import Visor.Visor_NER_Data_Preparation as visor


class VISOR_n(object):
    def __init__(self, img, objects, directions):
        assert(type(img) == str)
        assert(type(objects) == list)
        assert(type(directions) == list)
        
        self._img = Image.open(img)
        self._objects = objects
        self._directions = directions

        # create mask dictonary

        self._objectMasks = {}
        self._maskCoord = {}

        self._processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self._model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    
    # get variables
    def __get_image(self):
        return self._img
    def __get_objects(self):
        return self._objects
    def __get_directions(self):
        return self._directions
    

    def _get_mask(self, object):
        
        assert(type(object) == str)

        inputs = self._processor(text=[object], images=[self._img], padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
        preds = outputs.logits.unsqueeze(1)

        mask = preds.numpy()
        seg_array = np.squeeze(mask).astype(np.uint8)
        seg_image = Image.fromarray(seg_array)
        rgb_img = seg_image.convert('RGB')
        np_array = np.asarray(rgb_img)
        # print(np_array.shape)
        # convert np_array to black and white
        bw_array = cv2.cvtColor(np_array, cv2.COLOR_BGR2GRAY)
        #threshold the masks
        ret, bw_array = cv2.threshold(bw_array, 127, 255, cv2.THRESH_BINARY)
        #flip the mask because we need white mask
        bw_array = cv2.bitwise_not(bw_array)
        return bw_array   

    def _get_object_mask(self):

        for obj in self._objects:
            self._objectMasks[obj] = self._get_mask(obj)

        return self._objectMasks
    
    def _get_mask_coordinates(self, mask):
        assert(type(mask) == np.ndarray)

        # get center of mass of mask
        M = cv2.moments(mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        return (cX, cY)
    

    def _get_objMask_coordinates(self):
        
        self._get_object_mask()
        
        for obj in self._objects:
            self._maskCoord[obj] = self._get_mask_coordinates(self._objectMasks[obj])
        
        return self._maskCoord
    

    def _check_directions(self):
        # check if directions are valid
        self._get_objMask_coordinates()
        correct_directions = 0
        #Assumption for N objects we will have N-1 directions
        j=0
        for i in range(len(self._directions)):
            # if we check for i th index in diretion list check objects for i and i+1 in object list
            # get coordinates of i and i+1 object
            if  self._directions[i] == "left":
                cx_1, cy_1 = self._maskCoord[self._objects[j]]
                cx_2, cy_2 = self._maskCoord[self._objects[j+1]]

                if cx_1 < cx_2:
                    correct_directions += 1
                j+=1

            elif self._directions[i] == "right":
                cx_1, cy_1 = self._maskCoord[self._objects[j]]
                cx_2, cy_2 = self._maskCoord[self._objects[j+1]]

                if cx_1 > cx_2:
                    correct_directions += 1
                j+=1

            elif self._directions[i] == "above" or self._directions[i] == "on top of" or self._directions[i] == "over":

                cx_1, cy_1 = self._maskCoord[self._objects[j]]
                cx_2, cy_2 = self._maskCoord[self._objects[j+1]]

                if cy_1 < cy_2:
                    correct_directions += 1
                j+=1

            elif self._directions[i] == "below" or self._directions[i] == "underneath" or self._directions[i] == "under":
                    
                cx_1, cy_1 = self._maskCoord[self._objects[j]]
                cx_2, cy_2 = self._maskCoord[self._objects[j+1]]

                if cy_1 > cy_2:
                    correct_directions += 1
                j+=1
            
            elif self._directions[i] == "between" or self._directions[i] == "inbetween":
                # check if the objects are in the same row or column
                main_objx, main_objy = self._maskCoord[self._objects[j]]
                cx_2, cy_2 = self._maskCoord[self._objects[j+1]]
                cx_3, cy_3 = self._maskCoord[self._objects[j+2]]

                #horizontal check
                if (cx_2< main_objx and cx_3>main_objx) or (cx_2>main_objx and cx_3<main_objx):
                    correct_directions += 1

                # elif (cy_2< main_objy and cy_3>main_objy) or (cy_2>main_objy and cy_3<main_objy):
                #     correct_directions += 1

                j+=2

        return correct_directions/len(self._directions)


def extract_objects(img_str):
    prompt = img_str.split(".")[0][:-2]
    print(prompt)
    _, objects, directions = visor.annotate_prompts([prompt])
    print(objects)
    print(directions)
    print("#"*50)
    return objects, directions

if __name__ == "__main__":

    # obj = ["lamp","book","bird", "lamp", "bird"]
    # dire = ["right", "between", "left"]
    # img_str = "test2 lamp on the right of a book to the left of a bird.png"

    image_directory = 'Dalle_object_images/dalle_api_images_2_objects'
    image_list = os.listdir(image_directory)

    # only keep images with extension .png
    image_list = [img_str for img_str in image_list if img_str.endswith(".png") or img_str.endswith(".jpg") or img_str.endswith(".jpeg")]

    sum_value = 0

    for img_str in image_list:
        obj, dir = extract_objects(img_str)

        # concatenate image directory and image name
        img_name = os.path.join(image_directory, img_str)
        visorn = VISOR_n(img=img_name, objects=obj , directions=dir) 
        coordinates = visorn._get_objMask_coordinates()

        # plot coordinates on an image
        dummy_img = cv2.imread(img_name)
        dummy_img = cv2.resize(dummy_img, (352, 352))

        for obj in obj:
            cv2.circle(dummy_img, coordinates[obj], 5, (0, 0, 255), -1)
        # cv2.imshow("image", dummy_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        sum_value += visorn._check_directions()
        
    
    visor_value = sum_value/len(image_list)
    print("Visor value: ", visor_value)
    
    # visorn = VISOR_n(img=img_str, objects=obj , directions=dire) 
    # coordinates = visorn._get_objMask_coordinates()
    # # plot coordinates on an image
    # dummy_img = cv2.imread(img_str)
    # dummy_img = cv2.resize(dummy_img, (352, 352))
    # for obj in obj:
    #     cv2.circle(dummy_img, coordinates[obj], 5, (0, 0, 255), -1)
    # cv2.imshow("image", dummy_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(visorn._check_directions())
    
    
    # masks = visorn._get_object_mask()
    # for obj in obj:
    #     # plot moment of mask on the images
    #     M = cv2.moments(masks[obj])
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])
    #     cv2.circle(masks[obj], (cX, cY), 5, (0, 0, 255), -1)
    #     cv2.imshow("image", masks[obj])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # print(visorn._check_directions())
    