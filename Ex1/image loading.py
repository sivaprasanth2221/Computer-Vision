import cv2

image = cv2.imread("/Users/sivaprasanth/Documents/Computer Vision/Photograph.jpg")

if image is None:
    print("Error: Unable to open image")
else:
    cv2.imshow("Image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()