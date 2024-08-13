import cv2
import matplotlib.pyplot as plt

rgb_img = cv2.imread("/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/image1.jpg", cv2.IMREAD_COLOR)
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

x, y, w, h = 100, 100, 300, 200

cropped_img = rgb_img[y:y+h, x:x+w]

desired_width, desired_height = 200, 150

resized_img = cv2.resize(cropped_img, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(rgb_img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cropped_img)
plt.title('Cropped Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(resized_img)
plt.title(f'Resized to {desired_width}x{desired_height}')
plt.axis('off')

plt.tight_layout()
plt.show()
