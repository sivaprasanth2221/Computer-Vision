import cv2
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched_image = 255 * (image - min_val) / (max_val - min_val)
    return np.array(stretched_image, dtype=np.uint8)

def intensity_level_slicing(image, lower, upper):
    sliced_image = np.zeros_like(image)
    sliced_image[(image >= lower) & (image <= upper)] = 255
    return sliced_image

def bit_plane_slicing(image, bit_plane):
    rows, cols = image.shape
    bit_plane_image = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            bit_plane_image[i, j] = 255 * ((image[i, j] >> bit_plane) & 1)
    return bit_plane_image

image = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/image1.jpg', cv2.IMREAD_GRAYSCALE)

sliced_image = intensity_level_slicing(image, 100, 200)

contrast_stretched_image = contrast_stretching(image)

# Create subplots for displaying images
plt.figure(figsize=(20, 20))

plt.subplot(3, 4, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(3, 4, 2)
plt.title("Contrast Stretched Image")
plt.imshow(contrast_stretched_image, cmap='gray')

plt.subplot(3, 4, 3)
plt.title("Intensity Level Sliced Image")
plt.imshow(sliced_image, cmap='gray')

# Display all bit-planes
for i in range(8):
    bit_plane_image = bit_plane_slicing(image, i)
    plt.subplot(3, 4, i + 4)
    plt.title(f"Bit-plane {i}")
    plt.imshow(bit_plane_image, cmap='gray')

plt.tight_layout()
plt.show()
