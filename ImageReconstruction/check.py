import cv2
import numpy as np

# Load the image
img = cv2.imread('newmasktest.png',0)
# revrese the image
img = cv2.bitwise_not(img)

# Define the transformation matrix
# M = np.float32([[1, 0, 0], [0, 1, 50]])

# # Apply the transformation using warpAffine
# img_down = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# # Display the results
# cv2.imshow('Original Image', img)
# cv2.imshow('Moved Down Image', img_down)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
M = cv2.moments(img)

# Calculate the center of mass
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

# cacluate the pixel distance in y direction from the center of mass to the center of the image
y_dist = cy - img.shape[0]/2

# shift the image by the distance in the y direction to the opposite direction
M = np.float32([[1, 0, 0], [0, 1, -2*y_dist]])



# Apply the transformation using warpAffine
img_down = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('Moved Down Image', img_down)
# Draw a circle at the center of mass
print(cx, cy)

# Display the result
cv2.imshow('Shape with Center of Mass', img)
cv2.waitKey(0)
cv2.destroyAllWindows()