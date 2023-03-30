import cv2
import numpy as np




def get_background_object_reversed(original_image, flip_combine_teddy, mask_bear):

    flip_mask = cv2.flip(mask_bear, 1)
    array = np.zeros((original_image.shape[0], original_image.shape[1], 3), dtype=np.uint8)

    for i in range(original_image.shape[0]):
        for j in range(original_image.shape[1]):
            if mask_bear[i][j] == 0:
                for c in range(3):
                    array[i][j][c] = 0
            elif flip_mask[i][j] == 0:
                for c in range(3):
                    array[i][j][c] = flip_combine_teddy[i][j][c]
            else:
                for c in range(3):
                    array[i][j][c] = original_image[i][j][c]
    return array


color = cv2.imread('color.jpeg')
color_copy = color.copy()
cv2.imshow('color',color)
mask_bear = cv2.imread('mask_bear.jpeg', 0)

# interpolate mask to form image of 256x256
mask_bear = cv2.resize(mask_bear, (256, 256))
cv2.imshow('mask_bear',mask_bear)

# threshold mask to form binary image
ret, mask_bear = cv2.threshold(mask_bear, 127, 255, cv2.THRESH_BINARY)



img_r = color[:,:,0]
img_g = color[:,:,1]
img_b = color[:,:,2]

img_r_bg = color[:,:,0]
img_g_bg = color[:,:,1]
img_b_bg = color[:,:,2]

# get coordinates inside mask and outside of mask should be zero
img_r[mask_bear == 255] = 0
img_g[mask_bear == 255] = 0
img_b[mask_bear == 255] = 0




combine_teddy = cv2.merge((img_r, img_g, img_b))
#flip tbe image
flip_combine_teddy = cv2.flip(combine_teddy, 1)
cv2.imshow('original',color)
new_image = get_background_object_reversed(color_copy, flip_combine_teddy, mask_bear)

# cv2.imshow('teddy',combine_teddy)
cv2.imshow('flip',flip_combine_teddy)
cv2.imshow('new',new_image)
# cv2.imshow('background',background)


cv2.waitKey(0)
cv2.destroyAllWindows()