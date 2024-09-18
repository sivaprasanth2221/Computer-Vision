import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
img = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/coin.jpg', 0)


def region_growing(img, seed, threshold=60):
    # Initialize a blank image to store the region
    region = np.zeros_like(img)
    rows, cols = img.shape
    visited = np.zeros_like(img, dtype=bool)
    
    # Define the seed point
    seed_x, seed_y = seed
    region[seed_x, seed_y] = 255
    visited[seed_x, seed_y] = True
    
    # List to store the pixels to be processed
    stack = [(seed_x, seed_y)]
    
    while stack:
        x, y = stack.pop()
        
        # Check neighboring pixels
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]:
                    # Check the intensity difference
                    if abs(int(img[x, y]) - int(img[nx, ny])) < threshold:
                        region[nx, ny] = 255
                        stack.append((nx, ny))
                    visited[nx, ny] = True
    
    return region


# Create a copy of the input image to visualize the seed point
seed_image = img.copy()
seed_point = (100, 100)  # Example seed point
cv2.circle(seed_image, seed_point, 3, (255, 0, 0), -1)  # Draw the seed point

# Apply region growing algorithm
output = region_growing(img, seed_point)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Original Image
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Input Image')
axs[0].axis('off')

# Seed Image
axs[1].imshow(seed_image, cmap='gray')
axs[1].set_title('Seed Image')
axs[1].axis('off')

# Region Growing Output
axs[2].imshow(output, cmap='gray')
axs[2].set_title('Region Growing Output')
axs[2].axis('off')

# Display the plots
plt.tight_layout()
plt.show()


def homogeneity_criteria(region, threshold=10):
    """Check if a region is homogeneous based on intensity standard deviation."""
    return np.std(region) < threshold

def split(img, threshold=10):
    """Recursively split the image into smaller regions based on the homogeneity criterion."""
    h, w = img.shape
    if h <= 1 or w <= 1 or homogeneity_criteria(img, threshold):
        return img
    
    h_half, w_half = h // 2, w // 2
    
    # Split the image into four regions
    top_left = split(img[:h_half, :w_half], threshold)
    top_right = split(img[:h_half, w_half:], threshold)
    bottom_left = split(img[h_half:, :w_half], threshold)
    bottom_right = split(img[h_half:, w_half:], threshold)
    
    # Combine the results
    return np.block([[top_left, top_right], [bottom_left, bottom_right]])

def merge(img, threshold=10):
    """Merge regions based on homogeneity of adjacent regions."""
    h, w = img.shape
    output = img.copy()
    
    # Check adjacent regions to merge
    for i in range(1, h):
        for j in range(1, w):
            # Compare the current pixel with its neighbors and merge if the difference is within the threshold
            if abs(int(img[i, j]) - int(img[i - 1, j])) < threshold:
                output[i, j] = img[i - 1, j]
            if abs(int(img[i, j]) - int(img[i, j - 1])) < threshold:
                output[i, j] = img[i, j - 1]
    
    return output

def split_and_merge(img, threshold=10):
    """Perform split and merge segmentation."""
    split_img = split(img, threshold)
    merged_img = merge(split_img, threshold)
    return split_img, merged_img


# Perform split and merge segmentation
split_img, merged_img = split_and_merge(img, threshold=5)


# Plot the input image, region splitting image, and region merging image
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Input Image
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Input Image')
axs[0].axis('off')

# Split Image
axs[1].imshow(split_img, cmap='gray')
axs[1].set_title('Region Splitting')
axs[1].axis('off')

# Merged Image
axs[2].imshow(merged_img, cmap='gray')
axs[2].set_title('Region Merging')
axs[2].axis('off')

# Display the plots
plt.tight_layout()
plt.show()
