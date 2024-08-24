import cv2
import numpy as np
import matplotlib.pyplot as plt

def negative_transformation(image):
    return 255 - image

def log_transformation(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(image + 1))
    return np.array(log_image, dtype=np.uint8)

def gamma_transformation(image, gamma):
    normalized_image = image / 255.0
    gamma_corrected = np.power(normalized_image, gamma)
    return np.array(255 * gamma_corrected, dtype=np.uint8)


image = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/blurry_img.jpeg', cv2.IMREAD_GRAYSCALE)

negative_image = negative_transformation(image)
log_image = log_transformation(image)
gamma_image = gamma_transformation(image, 2.2)

plt.figure(figsize=(10,8))
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(2, 2, 2)
plt.title("Negative Image")
plt.imshow(negative_image, cmap='gray')

plt.subplot(2, 2, 3)
plt.title("Log Transformation Image")
plt.imshow(log_image, cmap='gray')


plt.subplot(2, 2, 4)
plt.title("Gamma Transformation Image")
plt.imshow(gamma_image, cmap='gray')
plt.show()
