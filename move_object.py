from YOLO.yolo_check import relation_objects
from YOLO.yolo_check import detect_object
import numpy as np
import cv2


class MoveObject(object):
    def __init__(self, bounding_box: list, image: np.ndarray, direction: str):
        self._bounding_box = bounding_box
        self._image = image
        self._direction = direction
        self._maskImage = None
        

    def move_object(self)->np.ndarray:
        # get bouding boxes of the object to be moved
        center_x, center_y, width_bb, height_bb = self._bounding_box
        # get image center
        height = self._image.shape[0]
        width = self._image.shape[1]
        image_center = (height / 2, width / 2)
        print(image_center)
        # if direction is left
        # traslate the object such the the right side of the object is at the center of the image
        if self._direction == 'right':
            # get the the distance by which the object has to be moved
            distance = int(center_y - image_center[1]) + int(width_bb / 2)
            # print("Distance: ", distance)
            # get the new coordinates of the object
            for i in range(int(center_x-(width_bb/2)), int(center_x+(width_bb/2))):
                for j in range(int(center_y-(height_bb/2)), int(center_y+(height_bb/2))):
                    # print("Original coordinates: ", i, j)
                    # print("New coordinates: ", i, j-distance)
                    if j-distance < 0:
                        continue
                    else:
                        for c in range(3):
                            self._image[i][j-distance][c] = self._image[i][j][c]
                            self._image[i][j][c]=0

        if self._direction == 'left':
            # get the the distance by which the object has to be moved
            distance = int(image_center[1] - center_y) + int(width_bb / 2)
            # print("Distance: ", distance)
            # get the new coordinates of the object
            for i in range(int(center_x-(width_bb/2)), int(center_x+(width_bb/2))):
                for j in range(int(center_y-(height_bb/2)), int(center_y+(height_bb/2))):
                    # print("Original coordinates: ", i, j)
                    # print("New coordinates: ", i, j+distance)
                    if j+distance > width-1:
                        continue
                    else:
                        for c in range(3):
                            self._image[i][j+distance][c] = self._image[i][j][c]
                            self._image[i][j][c]=0
                    
        if self._direction == 'top':
            # get the the distance by which the object has to be moved
            distance = int(center_x - image_center[0]) + int(height_bb / 2)
            # print("Distance: ", distance)
            # get the new coordinates of the object
            for i in range(int(center_x-(width_bb/2)), int(center_x+(width_bb/2))):
                for j in range(int(center_y-(height_bb/2)), int(center_y+(height_bb/2))):
                    if i-distance < 0:
                        continue
                    else:
                        for c in range(3):
                            self._image[i-distance][j][c] = self._image[i][j][c]
                            self._image[i][j][c]=0    

        if self._direction == 'bottom':
            # get the the distance by which the object has to be moved
            distance = int(image_center[0] - center_x) + int(height_bb / 2)
            # print("Distance: ", distance)
            # get the new coordinates of the object
            for i in range(int(center_x-(width_bb/2)), int(center_x+(width_bb/2))):
                for j in range(int(center_y-(height_bb/2)), int(center_y+(height_bb/2))):
                    if (i+distance) > height-1:
                        continue
                    else:
                        for c in range(3):
                            self._image[i+distance][j][c] = self._image[i][j][c]
                            self._image[i][j][c]=0
                    
        return self._image
    
    def create_mask(self)->np.ndarray:
        # get bouding boxes of the object to be moved
        center_x, center_y, width_bb, height_bb = self._bounding_box
        self._maskImage = np.zeros((self._image.shape[0], self._image.shape[1]), dtype=np.uint8)
        # everything insid the bounding box is white
        for i in range(int(center_x-(width_bb/2)), int(center_x+(width_bb/2))):
            for j in range(int(center_y-(height_bb/2)), int(center_y+(height_bb/2))):
                
                    self._maskImage[i][j] = 255
        return self._maskImage

if __name__ == "__main__":

    input_image = "YOLO/test4.png"
    class_label = "cat"
    direction = "right"

    # Load the model
    model = cv2.dnn.readNetFromDarknet('YOLO/yolov3.cfg', 'YOLO/yolov3.weights')

    # Load the input image
    image = cv2.imread(input_image)

    classes = []
    with open("YOLO/classes.txt", "r") as f:
        classes = [cname.strip() for cname in f.readlines()]

    
    height, width, _ = image.shape

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True)

    # Set the input for the model
    model.setInput(blob)

    # Get the output layer names
    output_layers = model.getUnconnectedOutLayersNames()

    # Run inference through the model
    outputs = model.forward(output_layers)

    bool_present, bounding_box = detect_object(class_label, outputs, width, height, classes)

    Bool, label_bb = relation_objects(class_label, direction, bounding_box, width, height)
    print(label_bb)
    image = cv2.imread('YOLO/test4.png')
    print(type (image), image.shape)
    bounding_box = label_bb#bounding_box#[280,1020,  254, 260]
    move_object = MoveObject(bounding_box, image, direction)
    image = move_object.move_object()
    mask = move_object.create_mask()
    cv2.imshow('Image', image)
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()