import cv2
import numpy as np



class MoveObject(object):

    def __init__(self, original_image:np.ndarray, mask:np.ndarray, direction:str):
        self._original_image = original_image
        self._mask = mask
        self._direction = direction
        self._flipped_image = None
        self._translated_y_mask = None


        self._original_image_copy = self._original_image.copy()

        assert(type(self._original_image) == np.ndarray)
        assert(3 == self._original_image.shape[2])

        assert(type(self._mask) == np.ndarray)
        assert(2 == len(self._mask.shape))

        assert(type(self._direction) == str)

        self._mask = cv2.resize(self._mask, (self._original_image.shape[1], self._original_image.shape[0]))

    
    
    def get_background_object_reversed_right_left(self,flip_combine_teddy, thresholded_mask)->np.ndarray:

        flip_mask = cv2.flip(thresholded_mask, 1)
        
        array = np.zeros((self._original_image.shape[0], self._original_image.shape[1], 3), dtype=np.uint8)

        for i in range(self._original_image.shape[0]):
            for j in range(self._original_image.shape[1]):
                if thresholded_mask[i][j] == 0:
                    for c in range(3):
                        array[i][j][c] = 0
                elif flip_mask[i][j] == 0:
                    for c in range(3):
                        array[i][j][c] = flip_combine_teddy[i][j][c]
                else:
                    for c in range(3):
                        array[i][j][c] = self._original_image_copy[i][j][c]
        return array


    def get_shifted_image_left_right(self)->np.ndarray:

        # threshold mask to form binary image
        ret, thresholded_mask = cv2.threshold(self._mask, 127, 255, cv2.THRESH_BINARY)
        # get separate channels of the original image
        img_r = self._original_image[:,:,0]
        img_g = self._original_image[:,:,1]
        img_b = self._original_image[:,:,2]

        # get coordinates inside mask and outside of mask should be zero
        img_r[thresholded_mask == 255] = 0
        img_g[thresholded_mask == 255] = 0
        img_b[thresholded_mask == 255] = 0


        combined_image = cv2.merge((img_r, img_g, img_b))
        #cv2.imshow('combined',combined_image)
        #flip tbe image
        flip_combined_image= cv2.flip(combined_image, 1)

        new_image = self.get_background_object_reversed_right_left(flip_combined_image, thresholded_mask)
        return new_image
        
    def translate_mask_y(self, img:np.ndarray, mask:np.ndarray)->np.ndarray:
        
        ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        reversed_mask = cv2.bitwise_not(mask)
        M = cv2.moments(reversed_mask)

        # Calculate the center of mass
        cy = int(M['m01']/M['m00'])

        # cacluate the pixel distance in y direction from the center of mass to the center of the image
        y_dist = cy - img.shape[0]/2
        #translate the mask by the distance in the y direction to the opposite direction
    
        # shift the image by the distance in the y direction to the opposite direction
        M = np.float32([[1, 0, 0], [0, 1, -2*y_dist]])

        translated_y_mask = cv2.warpAffine(reversed_mask, M, (self._original_image.shape[0], self._original_image.shape[0]))
        self._translated_y_mask = cv2.bitwise_not(translated_y_mask)
    
        translated_y = cv2.warpAffine(img, M, (self._original_image.shape[0], self._original_image.shape[0]))

        return translated_y

    def combine_translated_y_image(self, translated_y_image:np.ndarray, thresholded_mask:np.ndarray)->np.ndarray:
        
        array = np.zeros((self._original_image.shape[0], self._original_image.shape[1], 3), dtype=np.uint8)

        for i in range(self._original_image.shape[0]):
            for j in range(self._original_image.shape[1]):
                if thresholded_mask[i][j] == 0:
                    for c in range(3):
                        array[i][j][c] = 0
                elif self._translated_y_mask[i][j] == 0:
                    for c in range(3):
                        array[i][j][c] = translated_y_image[i][j][c]
                else:
                    for c in range(3):
                        array[i][j][c] = self._original_image_copy[i][j][c]

        return array
                

    def get_shifted_image_up_down(self)->np.ndarray:

        ret, thresholded_mask = cv2.threshold(self._mask, 127, 255, cv2.THRESH_BINARY)
        # get separate channels of the original image
        img_r = self._original_image[:,:,0]
        img_g = self._original_image[:,:,1]
        img_b = self._original_image[:,:,2]

        # get coordinates inside mask and outside of mask should be zero
        img_r[thresholded_mask == 255] = 0
        img_g[thresholded_mask == 255] = 0
        img_b[thresholded_mask == 255] = 0

        combined_image = cv2.merge((img_r, img_g, img_b))
        translated_y_image = self.translate_mask_y(combined_image, thresholded_mask)

        new_image = self.combine_translated_y_image(translated_y_image, thresholded_mask)

        return new_image
    
    def translate_object(self):
        if self._direction == 'left' or self._direction == 'right':
            return self.get_shifted_image_left_right()
        
        elif self._direction == 'top' or self._direction == 'bottom':
            return self.get_shifted_image_up_down()
        else:
            raise ValueError('Direction must be left, right, up or down')

    def get_mask(self)->np.ndarray:
        return self._mask
    
    def get_image(self)->np.ndarray:
        return self._original_image       
    

if __name__ == "__main__":
    
    
    img_test_top_bottom = cv2.imread('a toilet below an umbrella_2.jpg')
    mask_test_top_bottom = cv2.imread('newmasktest.png',0)
    
    
    # cv2.imshow('Top Bottom OG', img_test_top_bottom)
    # cv2.imshow('mask top bottom', mask_test_top_bottom)

    # test top and bottom

    move_object_top_bottom= MoveObject(img_test_top_bottom, mask_test_top_bottom, 'top')
    # new_image = move_object.get_shifted_image_left_right()
    new_image_top_bottom = move_object_top_bottom.translate_object()
    
    # top and bottom test
    img_test_left_right = cv2.imread('a tvmonitor to the right of a teddy bear_2.jpeg')
    mask_test_left_right = cv2.imread('new_mask.png',0)

    move_object_left_right= MoveObject(img_test_left_right, mask_test_left_right, 'left')
    new_image_left_right = move_object_left_right.translate_object()

    cv2.imshow('Top Bottom', new_image_top_bottom)
    cv2.imshow('Left Right', new_image_left_right)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
