import cv2
import numpy as np

# Step 1: Load left and right images
left_img = cv2.imread('left_image.jpg', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('right_image.jpg', cv2.IMREAD_GRAYSCALE)

# Step 2: Initialize StereoBM object
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Step 3: Compute disparity map
disparity = stereo.compute(left_img, right_img)

# Step 4: Normalize disparity for better visualization
disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)

# Step 5: Display disparity map
cv2.imshow('Disparity Map', disparity_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()