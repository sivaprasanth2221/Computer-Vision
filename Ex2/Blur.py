import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/image1.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_gaussian = cv2.GaussianBlur(img_gray, (5, 5), 0)

img_median = cv2.medianBlur(img_gray, 5)

img_avg = cv2.blur(img_gray, (5, 5))

plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(img_gaussian, cmap='gray')
plt.title('Gaussian Blur')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(img_median, cmap='gray')
plt.title('Median Blur')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(img_avg, cmap='gray')
plt.title('Average Blur')
plt.axis('off')

plt.tight_layout()
plt.show()
