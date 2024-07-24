import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/blurry_img.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
alpha = 1.5
beta = 20
adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Adjusted Image')
plt.imshow(adjusted_image)
plt.axis('off')

print(f"Original Image:{image} \n Adjusted Image:{adjusted_image}")