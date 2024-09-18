import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/coins.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding to convert the image to binary
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Remove noise using morphological opening
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Dilate the image to get sure background areas
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Apply distance transform to get sure foreground areas
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Find the unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Label the markers for sure foreground objects
_, markers = cv2.connectedComponents(sure_fg)

# Add 1 to all labels to distinguish the background as 1
markers = markers + 1

# Mark the unknown region with 0
markers[unknown == 255] = 0

# Apply the watershed algorithm
markers = cv2.watershed(image, markers)

# Mark the boundaries (where markers == -1) with red color
image[markers == -1] = [255, 0, 0]  # Red color for boundaries

# Display the result using matplotlib
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
plt.title('Watershed Segmentation')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(markers, cmap='gray')
plt.title('Markers')
plt.xticks([]), plt.yticks([])

plt.show()
