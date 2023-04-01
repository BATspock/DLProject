import cv2
import numpy as np

img = cv2.imread('a cup above a kite_1.jpg')
mask = cv2.imread('newmasktest.png',0)

ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

reversed_mask = cv2.bitwise_not(mask)

img_r = img[:,:,0]
img_g = img[:,:,1]
img_b = img[:,:,2]


M = cv2.moments(reversed_mask)

# Calculate the center of mass
cy = int(M['m01']/M['m00'])

# cacluate the pixel distance in y direction from the center of mass to the center of the image
y_dist = cy - img.shape[0]/2
#translate the mask by the distance in the y direction to the opposite direction



# shift the image by the distance in the y direction to the opposite direction
M = np.float32([[1, 0, 0], [0, 1, -2*y_dist]])

translated_y_mask = cv2.warpAffine(reversed_mask, M, (img.shape[0], img.shape[1]))

arr = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if reversed_mask[i,j] == 255:
            for c in range(3):
                arr[i,j,c] = img[i,j,c]


# translate arr by y_dist
M = np.float32([[1, 0, 0], [0, 1, -2*y_dist]])
translated_y = cv2.warpAffine(arr, M, (img.shape[0], img.shape[1]))

# transfer arr to right
M = np.float32([[1, 0, 1.8*y_dist], [0, 1, 0]])

translated_right = cv2.warpAffine(translated_y, M, (img.shape[0], img.shape[1]))
translated_right_mask = cv2.warpAffine(translated_y_mask, M, (img.shape[0], img.shape[1]))
cv2.imshow('translated_right',translated_right_mask)

final = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if reversed_mask[i,j] == 255:
            for c in range(3):
                final[i,j,c] = 0
        elif translated_right_mask[i,j] == 255:
            for c in range(3):
                final[i,j,c] = translated_right[i,j,c]
        else:
            for c in range(3):
                final[i,j,c] = img[i,j,c]


cv2.imshow('right_y_mask',final)
cv2.imwrite('right_y_mask.png',final)
cv2.waitKey(0)
cv2.destroyAllWindows()