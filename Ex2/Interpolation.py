import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/img/image1.jpg')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

shrink_width = int(img.shape[1] * 0.5)
shrink_height = int(img.shape[0] * 0.5)

zoom_width = int(img.shape[1] * 1.5)
zoom_height = int(img.shape[0] * 1.5)

shrunk_nearest = cv2.resize(img_rgb, (shrink_width, shrink_height), interpolation=cv2.INTER_NEAREST)
shrunk_linear = cv2.resize(img_rgb, (shrink_width, shrink_height), interpolation=cv2.INTER_LINEAR)
shrunk_cubic = cv2.resize(img_rgb, (shrink_width, shrink_height), interpolation=cv2.INTER_CUBIC)
shrunk_area = cv2.resize(img_rgb, (shrink_width, shrink_height), interpolation=cv2.INTER_AREA)

zoomed_nearest = cv2.resize(img_rgb, (zoom_width, zoom_height), interpolation=cv2.INTER_NEAREST)
zoomed_linear = cv2.resize(img_rgb, (zoom_width, zoom_height), interpolation=cv2.INTER_LINEAR)
zoomed_cubic = cv2.resize(img_rgb, (zoom_width, zoom_height), interpolation=cv2.INTER_CUBIC)
zoomed_area = cv2.resize(img_rgb, (zoom_width, zoom_height), interpolation=cv2.INTER_AREA)

plt.figure(figsize=(14, 10))

plt.subplot(3, 4, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(3, 4, 2)
plt.imshow(shrunk_nearest)
plt.title('Shrunk - INTER_NEAREST')
plt.axis('off')

plt.subplot(3, 4, 3)
plt.imshow(shrunk_linear)
plt.title('Shrunk - INTER_LINEAR')
plt.axis('off')

plt.subplot(3, 4, 4)
plt.imshow(shrunk_cubic)
plt.title('Shrunk - INTER_CUBIC')
plt.axis('off')

plt.subplot(3, 4, 5)
plt.imshow(shrunk_area)
plt.title('Shrunk - INTER_AREA')
plt.axis('off')

plt.subplot(3, 4, 6)
plt.imshow(zoomed_nearest)
plt.title('Zoomed - INTER_NEAREST')
plt.axis('off')

plt.subplot(3, 4, 7)
plt.imshow(zoomed_linear)
plt.title('Zoomed - INTER_LINEAR')
plt.axis('off')

plt.subplot(3, 4, 8)
plt.imshow(zoomed_cubic)
plt.title('Zoomed - INTER_CUBIC')
plt.axis('off')

plt.subplot(3, 4, 9)
plt.imshow(zoomed_area)
plt.title('Zoomed - INTER_AREA')
plt.axis('off')

plt.tight_layout()
plt.show()
