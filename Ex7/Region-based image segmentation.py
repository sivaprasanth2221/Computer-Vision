import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale mode
image = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/image.jpg', 0)

def split_and_merge_fixed(img, threshold=20, min_size=4):
    def split_region(img, x, y, size):
        if size <= min_size:
            return img[y:y+size, x:x+size]

        half = size // 2

        # Ensure we do not exceed the image boundary for odd sizes
        x_end = min(x + size, img.shape[1])
        y_end = min(y + size, img.shape[0])

        # Handle the case where the size is not even by adjusting the split regions
        top_left = split_region(img, x, y, half)
        top_right = split_region(img, x + half, y, x_end - (x + half))
        bottom_left = split_region(img, x, y + half, y_end - (y + half))
        bottom_right = split_region(img, x + half, y + half, min(half, x_end - (x + half), y_end - (y + half)))

        # Combine the regions
        regions = [top_left, top_right, bottom_left, bottom_right]

        # Compute the mean of each region
        mean_values = [np.mean(region) for region in regions]

        # Check if the regions are similar based on the threshold
        if max(mean_values) - min(mean_values) < threshold:
            return np.ones_like(top_left) * np.mean(mean_values)
        else:
            # Resize all regions to match the largest region before concatenation
            try:
                regions_resized = resize_all_to_match(regions)
                top = np.hstack((regions_resized[0], regions_resized[1]))
                bottom = np.hstack((regions_resized[2], regions_resized[3]))
                return np.vstack((top, bottom))
            except ValueError as e:
                print(f"Error in combining regions: {e}")
                return img[y:y+size, x:x+size]  # Return the original region if the combination fails

    # Ensure the image is square by padding
    height, width = img.shape
    new_size = max(height, width)

    # Make new size a power of two for consistent splitting
    new_size = 2 ** int(np.ceil(np.log2(new_size)))

    # Pad the image to the new size
    padded_img = np.pad(img, ((0, new_size - height), (0, new_size - width)), mode='constant', constant_values=0)

    # Perform the split and merge segmentation
    segmented = split_region(padded_img, 0, 0, min(padded_img.shape))

    # Return the segmented image, cropped back to original size
    return segmented[:height, :width]

def resize_to_match(region, target_shape):
    """Resize a single region to match a target shape."""
    return cv2.resize(region, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

def resize_all_to_match(regions):
    """Resize all regions to match the shape of the largest region for concatenation."""
    max_shape = max([r.shape for r in regions], key=lambda s: (s[0], s[1]))
    return [resize_to_match(region, max_shape) for region in regions]

# Apply the split and merge algorithm to your image
segmented_image_fixed = split_and_merge_fixed(image)

# Display the result
plt.imshow(segmented_image_fixed, cmap='gray')
plt.title('Region Splitting and Merging')
plt.show()
