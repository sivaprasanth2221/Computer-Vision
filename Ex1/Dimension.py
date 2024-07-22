
import cv2


rgb_img = cv2.imread("/Users/sivaprasanth/Documents/Computer Vision/Photograph.jpg",cv2.IMREAD_COLOR)
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

height, width, channel = rgb_img.shape
print(f"Dimension: {height} x {width} x {channel}")