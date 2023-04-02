import cv2
import os
import time
import sys
import numpy as np
from PIL import Image
import cv2
import itertools
import spacy

sys.path.append('/Users/nimishamittal/Documents/USC/CSCI566_DL/DLProject')
import YOLO
from YOLO.object_detect_yolo import ObjectDetector

sys.path.append('/Users/nimishamittal/Documents/USC/CSCI566_DL/DLProject/ImageReconstruction')
import ImageReconstruction
from ImageReconstruction.neural_nexus_clipseg import getMask

class Visor:
    def __init__(self, input_folder):
        self._input_folder = input_folder
        self._image_files = [f for f in os.listdir(self._input_folder) if f.endswith('.jpg')]
        self._total_images = len(self._image_files)

        self._directions = ["above", "below", "left", "right"]
        self._exceptions = ["dining table", "tv monitor", "potted plant"]

        self._exception_list = []
        self._exception_no = []
        self._not_detected_list = []


    def extract_object_and_direction(self,sentence, class_list, direction_list, exceptions):
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

    
    def preprocess_prompt_folder(self, image_file)->str:
        assert(type(image_file) == str)
        prompt_text = image_file.split('.')[0]
        if self._input_folder == 'dalle_imgs_2objects' or self._input_folder == 'dalle_api_images':
            prompt_text = prompt_text[:-2]
        else:
            prompt_text = prompt_text[:-4]
        return prompt_text

    
    def run_visor_pipeline(self):
        self._correct_count = 0
        object_detector = ObjectDetector()

        for i, image_file in enumerate(self._image_files):
            print("#" * 50, "Image", i + 1)
        
            image_path = os.path.join(self._input_folder, image_file)
            
            prompt_text = self.preprocess_prompt_folder(image_file)
            # label_direction_extract = getMask(classes_file=object_detector._coco_classes, image_str = image_file, direction_list = self._directions, exceptions = self._exceptions)



            print("Image: ", prompt_text)
            try:
                prompt_labels, prompt_direction = self.extract_object_and_direction(prompt_text, object_detector._coco_classes, self._directions, self._exceptions)
                print("Given object:", prompt_labels)
                print("Given direction:", prompt_direction)

                bool_consistent = object_detector.is_consistent(image_path, prompt_labels, prompt_direction)
                if bool_consistent:
                    self._correct_count += 1
                    print("Correct")
                else:
                    print("Incorrect")
            except:
                print("Exception")
                self._exception_list.append(image_file)
                self._exception_no.append(i)
        
            print("#" * 50)
        
        print("Num exceptions:", len(self._exception_list))
        print("Exceptions:", self._exception_list)
        print("Exception no:", self._exception_no)

        print("Not detected:", self._not_detected_list)
        print("Num not detected:", len(self._not_detected_list))

        self.calc_accuracy()

    
    def calc_accuracy(self):
        if self._correct_count == 0:
            print("Accuracy: 0%")
        else:
            print("Accuracy:", self._correct_count/self._total_images*100, "%")



if __name__ == "__main__":
    """
    # This name is for the folder containing the images from dalle. Keep the folder name as it is if using dalle
    input_folder = 'dalle_imgs_2objects' 
    # initial installation: python -m spacy download en_core_web_sm
    # spacy: version 3.5.1
    """
    input_folder = 'dalle_imgs_2objects'
    visor = Visor(input_folder=input_folder)
    visor.run_visor_pipeline()
        
    
