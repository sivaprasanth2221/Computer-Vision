import cv2
import matplotlib.pyplot as plt

gray_img = cv2.imread("/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/image1.jpg",cv2.IMREAD_GRAYSCALE)
bin_img = cv2.imread("/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/image1.jpg",cv2.IMREAD_GRAYSCALE)
rgb_img = cv2.imread("/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/image1.jpg",cv2.IMREAD_COLOR)
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
_, bin_img = cv2.threshold(bin_img, 127, 255, cv2.THRESH_BINARY)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(gray_img, cmap='gray')
plt.axis('off')
plt.title("Gray Scale Image")
print(f"Grayscale Pixel Value: {gray_img}")

plt.subplot(1, 3, 2)
plt.imshow(bin_img,cmap='gray')
plt.axis('off')
plt.title("Binary Image")
print(f"Binary Pixel Value: {bin_img}")

plt.subplot(1, 3, 3)
plt.imshow(rgb_img)
plt.axis('off')
plt.title("Color Image")
print(f"RGB Pixel Value: {rgb_img}")