import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/image.jpg', 0)

kernel = np.ones((5,5), np.uint8)

erosion = cv2.erode(image, kernel, iterations=1)

dilation = cv2.dilate(image, kernel, iterations=1)

opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

titles = ['Original Image', 'Erosion', 'Dilation', 'Opening', 'Closing']
images = [image, erosion, dilation, opening, closing]

for i in range(5):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()