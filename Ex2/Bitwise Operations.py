import cv2
from matplotlib import pyplot as plt


img1 = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/img/image1.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/img/image2.jpg',cv2.IMREAD_GRAYSCALE)


_, img1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
_, img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

img1 = cv2.resize(img1, (512, 512))
img2 = cv2.resize(img2, (512, 512))


bitwise_and = cv2.bitwise_and(img1, img2)


bitwise_or = cv2.bitwise_or(img1, img2)


bitwise_xor = cv2.bitwise_xor(img1, img2)


images = [img1, img2, bitwise_and, bitwise_or, bitwise_xor]
titles = ['Image 1', 'Image 2', 'Bitwise AND', 'Bitwise OR', 'Bitwise XOR']

for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
