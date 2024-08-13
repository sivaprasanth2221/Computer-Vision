import cv2
import matplotlib.pyplot as plt

rgb_img = cv2.imread("/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/image1.jpg", cv2.IMREAD_COLOR)
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

R, G, B = cv2.split(rgb_img)
plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.imshow(R, cmap='Reds')
plt.title('Red Channel')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(G, cmap='Greens')
plt.title('Green Channel')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(B, cmap='Blues')
plt.title('Blue Channel')
plt.axis('off')

plt.tight_layout()
plt.show()
