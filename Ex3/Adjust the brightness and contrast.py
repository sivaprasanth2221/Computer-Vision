import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/img/image1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

brightened_image = np.clip(image + 50, 0, 255).astype(np.uint8)

darkened_image = np.clip(image - 50, 0, 255).astype(np.uint8)

increased_contrast_image = np.clip(image * 1.5, 0, 255).astype(np.uint8)

decreased_contrast_image = np.clip(image * 0.5, 0, 255).astype(np.uint8)

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# Brightened image
plt.subplot(2, 2, 2)
plt.imshow(brightened_image)
plt.title('Brightened Image')
plt.axis('off')

# Increased contrast image
plt.subplot(2, 2, 3)
plt.imshow(increased_contrast_image)
plt.title('Increased Contrast Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(darkened_image)
plt.title('Darkened Image')
plt.axis('off')

plt.tight_layout()
plt.show()
