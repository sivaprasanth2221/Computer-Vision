import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/image.jpg', 0)  # Load the image in grayscale mode

def region_growing(img, seed, threshold=10):
    height, width = img.shape
    segmented = np.zeros_like(img)
    segmented[seed[1], seed[0]] = 255
    
    to_grow = [seed]
    
    while len(to_grow) > 0:
        x, y = to_grow.pop(0)
        current_value = img[y, x]
        
        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = x + i, y + j
                
                if 0 <= nx < width and 0 <= ny < height and segmented[ny, nx] == 0:
                    if abs(int(img[ny, nx]) - int(current_value)) < threshold:
                        segmented[ny, nx] = 255
                        to_grow.append((nx, ny))
    
    return segmented

seed_point = (100, 100)  # Example seed point
segmented_image = region_growing(image, seed_point)

plt.imshow(segmented_image, cmap='gray')
plt.title('Region Growing')
plt.show()

def split_and_merge(img, threshold=20, min_size=4):
    def split_region(img, x, y, size):
        if size <= min_size:
            return img[y:y+size, x:x+size]
        
        half = size // 2
        top_left = split_region(img, x, y, half)
        top_right = split_region(img, x + half, y, half)
        bottom_left = split_region(img, x, y + half, half)
        bottom_right = split_region(img, x + half, y + half, half)
        
        regions = [top_left, top_right, bottom_left, bottom_right]
        
        mean_values = [np.mean(region) for region in regions]
        if max(mean_values) - min(mean_values) < threshold:
            return np.ones_like(top_left) * np.mean(mean_values)
        else:
            return np.vstack((np.hstack((top_left, top_right)), np.hstack((bottom_left, bottom_right))))
    
    height, width = img.shape
    segmented = split_region(img, 0, 0, min(height, width))
    return segmented

segmented_image = split_and_merge(image)

plt.imshow(segmented_image, cmap='gray')
plt.title('Region Splitting and Merging')
plt.show()
