import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the image
img = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/cracked_ball.webp', cv2.IMREAD_GRAYSCALE)

# Apply GaussianBlur to smooth the image
img_blur = cv2.GaussianBlur(img, (5, 5), 0)

# Step 2: Compute gradients (Intensity gradient as a proxy for surface normals)
sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=5)  # Gradient in X direction
sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=5)  # Gradient in Y direction

# Compute the magnitude and direction of gradients
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
gradient_direction = np.arctan2(sobely, sobelx)

# Step 3: Normalize the gradient magnitude for better visibility
gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

# Convert gradient magnitude to 8-bit image
gradient_magnitude = np.uint8(gradient_magnitude)

# Step 4: Threshold the gradient magnitude to detect crack regions
_, cracks = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)

# Step 5: Post-process using morphological operations to refine cracks
kernel = np.ones((3, 3), np.uint8)
cracks = cv2.morphologyEx(cracks, cv2.MORPH_CLOSE, kernel)
cracks = cv2.morphologyEx(cracks, cv2.MORPH_OPEN, kernel)

# Optional: Use gradient direction to further refine crack detection
# For instance, you might only keep regions with gradient directions that align with expected crack orientations

# Step 6: Display results
plt.figure(figsize=(10, 8))

plt.subplot(1, 2, 1)
img = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/cracked_ball.webp')
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Detected Cracks")
plt.imshow(cracks, cmap='gray')
plt.axis('off')

plt.show()
