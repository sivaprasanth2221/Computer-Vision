import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/blurry_img.jpeg', 0)

equalized_image = cv2.equalizeHist(image)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.show()

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Histogram')
plt.hist(image.ravel(), bins=256, range=(0, 256), color='black', alpha=0.5)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.title('Equalized Histogram')
plt.hist(equalized_image.ravel(), bins=256, range=(0, 256), color='black', alpha=0.5)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.show()