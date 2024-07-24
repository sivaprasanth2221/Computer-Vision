import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/image1.jpg', cv2.IMREAD_GRAYSCALE)

_, simple_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

adaptive_thresh_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, 11, 2)

adaptive_thresh_gaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)

_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

titles = ['Original Image', 'Simple Thresholding', 'Adaptive Mean Thresholding',
          'Adaptive Gaussian Thresholding', 'Otsu Thresholding']
images = [image, simple_thresh, adaptive_thresh_mean, adaptive_thresh_gaussian, otsu_thresh]

plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()