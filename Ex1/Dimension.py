
import cv2


rgb_img = cv2.imread("/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/image1.jpg",cv2.IMREAD_COLOR)
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

height, width, channel = rgb_img.shape
print(f"Dimension: {height} x {width} x {channel}")