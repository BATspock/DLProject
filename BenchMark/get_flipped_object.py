import cv2
import numpy as np



class MoveObject(object):

    def __init__(self, original_image:np.ndarray, mask:np.ndarray, direction:str):
        self._original_image = original_image
        self._mask = mask
        self._direction = direction
        self._flipped_image = None


        self._original_image_copy = self._original_image.copy()

        assert(type(self._original_image) == np.ndarray)
        assert(3 == self._original_image.shape[2])

        assert(type(self._mask) == np.ndarray)
        assert(2 == len(self._mask.shape))

        assert(type(self._direction) == str)

        self._mask = cv2.resize(self._mask, (self._original_image.shape[1], self._original_image.shape[0]))

    
    
    def get_background_object_reversed(self,flip_combine_teddy, thresholded_mask)->np.ndarray:

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

        new_image = self.get_background_object_reversed(flip_combined_image, thresholded_mask)

        # cv2.imshow('teddy',combine_teddy)
        #cv2.imshow('flip',flip_combined_image)
        #cv2.imshow('new',new_image)
        # cv2.imshow('background',background)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return new_image
    
    def get_mask(self)->np.ndarray:
        return self._mask
    
    def get_image(self)->np.ndarray:
        return self._original_image       
    

if __name__ == "__main__":
    img = cv2.imread('color.jpeg')
    mask = cv2.imread('mask_bear.jpeg',0)

    move_object = MoveObject(img, mask, 'right')
    new_image = move_object.get_shifted_image_left_right()
