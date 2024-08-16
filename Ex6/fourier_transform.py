import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/blurry_img.jpeg', cv2.IMREAD_GRAYSCALE)


f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)


magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

plt.show()