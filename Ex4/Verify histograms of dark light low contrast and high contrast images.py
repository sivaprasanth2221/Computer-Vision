import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(image, title, subplot_position):
    plt.subplot(2, 2, subplot_position)
    plt.hist(image.ravel(), 256, [0, 256])
    plt.title(title)

image = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/image1.jpg', cv2.IMREAD_GRAYSCALE)

dark_image = cv2.convertScaleAbs(image, alpha=1.0, beta=-115)
light_image = cv2.convertScaleAbs(image, alpha=1.0, beta=200)
low_contrast_image = cv2.convertScaleAbs(image, alpha=0.3, beta=0)
high_contrast_image = cv2.convertScaleAbs(image, alpha=2.0, beta=0)

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

sharpened_image = cv2.filter2D(high_contrast_image, -1, kernel)

_, simple_thresh = cv2.threshold(sharpened_image, 127, 255, cv2.THRESH_BINARY)

adaptive_thresh_mean = cv2.adaptiveThreshold(sharpened_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, 11, 2)

adaptive_thresh_gaussian = cv2.adaptiveThreshold(sharpened_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)

_, otsu_thresh = cv2.threshold(sharpened_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(12, 8))

plot_histogram(dark_image, 'Histogram of Dark Image', 1)
plot_histogram(light_image, 'Histogram of Light Image', 2)
plot_histogram(low_contrast_image, 'Histogram of Low Contrast Image', 3)
plot_histogram(high_contrast_image, 'Histogram of High Contrast Image', 4)

plt.tight_layout()
plt.show()

titles = ['Original Image', 'Dark Image', 'Light Image', 'Low Contrast Image', 
          'High Contrast Image', 'Sharpened Image', 'Simple Thresholding', 
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', 
          'Otsu Thresholding']
images = [image, dark_image, light_image, low_contrast_image, high_contrast_image, 
          sharpened_image, simple_thresh, adaptive_thresh_mean, adaptive_thresh_gaussian, otsu_thresh]

plt.figure(figsize=(18, 12))
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
