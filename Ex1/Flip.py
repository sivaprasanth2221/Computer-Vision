import cv2
import matplotlib.pyplot as plt

rgb_img = cv2.imread("/Users/sivaprasanth/Documents/Computer Vision/Photograph.jpg", cv2.IMREAD_COLOR)
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

rotated_img = cv2.rotate(rgb_img, cv2.ROTATE_90_CLOCKWISE)

flipped_img_horizontal = cv2.flip(rgb_img, 1)
flipped_img_vertical = cv2.flip(rgb_img, 0)

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(rgb_img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(rotated_img)
plt.title('Rotated Image (90 degrees clockwise)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(flipped_img_horizontal)
plt.title('Horizontally Flipped Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(flipped_img_vertical)
plt.title('Vertically Flipped Image')
plt.axis('off')


plt.tight_layout()
plt.show()
