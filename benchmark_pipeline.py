import sys
import os
sys.path.append('/Users/nimishamittal/Documents/USC/CSCI566_DL/DLProject')
import YOLO
from YOLO.object_detect_yolo import ObjectDetector

import Visor
from Visor.visor import Visor


class Benchmark_Pipeline:
    def __init__(self, input_folder, output_folder):
        self._input_folder = input_folder
        self._output_folder = output_folder
        self._exception_list = []
        self._incorrect_images = 0
        self._correct_images = 0

    def run_pipeline(self):
        
        object_detector = ObjectDetector()
        visor = Visor(self._input_folder)

        for i, image_file in enumerate(visor._image_files):
            print("#" * 50, "Image", i + 1)
        
            image_path = os.path.join(self._input_folder, image_file)
            
            prompt_text = visor.preprocess_prompt_folder(image_file)
    
            print("Image: ", prompt_text)
            try:
                prompt_labels, prompt_direction = visor.extract_object_and_direction(prompt_text, object_detector._coco_classes, visor._directions, visor._exceptions)
                print("Given object:", prompt_labels)
                print("Given direction:", prompt_direction)

                bool_consistent = object_detector.is_consistent(image_path, prompt_labels, prompt_direction)
                if bool_consistent:
                    self._correct_images += 1
                    print("Correct")
                else:
                    self._incorrect_images += 1
                    # write this image to a output_folder
                    cv2.imwrite(os.path.join(self._output_folder, image_file), image)
                    
                    
            except:
                print("Exception")
                self._exception_list.append(image_file)
        
            print("#" * 50)

        print("Num exceptions:", len(self._exception_list))
        print("Num incorrect images:", self._incorrect_images)
        print("Num correct images:", self._correct_images)
        print("TOtal images:", len(visor._image_files))


if __name__=="__main__":
    input_folder = "dalle_api_images"
    output_folder = "dalle_api_images_incorrect"
    benchmark_pipeline = Benchmark_Pipeline(input_folder, output_folder)
    benchmark_pipeline.run_pipeline()
        
        
