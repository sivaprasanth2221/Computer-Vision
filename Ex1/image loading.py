import cv2
import matplotlib.pyplot as plt
image = cv2.imread("/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/image1.jpg")

if image is None:
    print("Error: Unable to open image")
else:
    plt.imshow(image)
    plt.show()