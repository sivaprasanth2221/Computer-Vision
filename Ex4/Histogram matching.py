import cv2
import numpy as np
from skimage import exposure
from matplotlib import pyplot as plt

source_image = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/image1.jpg', 0)  # Load the source image in grayscale
target_image = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/blurry_img.jpeg', 0)  # Load the target image in grayscale

matched_image = exposure.match_histograms(source_image, target_image)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Source Image')
plt.imshow(source_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Target Image')
plt.imshow(target_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Matched Image')
plt.imshow(matched_image, cmap='gray')
plt.axis('off')

plt.show()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Source Histogram')
plt.hist(source_image.ravel(), bins=256, range=(0, 256), color='black', alpha=0.5)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.title('Target Histogram')
plt.hist(target_image.ravel(), bins=256, range=(0, 256), color='black', alpha=0.5)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.title('Matched Histogram')
plt.hist(matched_image.ravel(), bins=256, range=(0, 256), color='black', alpha=0.5)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.show()
