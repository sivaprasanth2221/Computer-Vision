import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/image1.jpg', cv2.IMREAD_COLOR)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges_no_blur = cv2.Canny(img_gray, threshold1=50, threshold2=150)

img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
edges_with_blur = cv2.Canny(img_blur, threshold1=50, threshold2=150)

plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(edges_no_blur, cmap='gray')
plt.title('Edges without Blurring')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(edges_with_blur, cmap='gray')
plt.title('Edges with Gaussian Blur')
plt.axis('off')

plt.tight_layout()
plt.show()
