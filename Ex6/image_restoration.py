import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

# Load the image and convert it to grayscale
image = cv2.imread('/Users/sivaprasanth/Documents/Computer Vision/Computer-Vision/img/blurry_img1.jpeg', cv2.IMREAD_GRAYSCALE)

# Apply Fourier Transform
f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)

# Define center of the frequency domain
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Step 1: Bandpass and Bandreject Filters

# Define radius for the Bandpass and Bandreject filters
radius_low = 30
radius_high = 100

# Bandpass Filter: allows frequencies within a specific range
mask_bandpass = np.zeros((rows, cols), np.uint8)
mask_bandpass[crow-radius_high:crow+radius_high, ccol-radius_high:ccol+radius_high] = 1
mask_bandpass[crow-radius_low:crow+radius_low, ccol-radius_low:ccol+radius_low] = 0

# Bandreject Filter: blocks frequencies within a specific range
mask_bandreject = np.ones((rows, cols), np.uint8)
mask_bandreject[crow-radius_high:crow+radius_high, ccol-radius_high:ccol+radius_high] = 0
mask_bandreject[crow-radius_low:crow+radius_low, ccol-radius_low:ccol+radius_low] = 1

# Apply the filters
f_transform_bandpass = f_transform_shifted * mask_bandpass
f_transform_bandreject = f_transform_shifted * mask_bandreject

# Inverse Fourier Transform to get the images back
image_bandpass = np.fft.ifft2(np.fft.ifftshift(f_transform_bandpass))
image_bandpass = np.abs(image_bandpass)

image_bandreject = np.fft.ifft2(np.fft.ifftshift(f_transform_bandreject))
image_bandreject = np.abs(image_bandreject)

# Step 2: Notch and Optimum Notch Filters

def create_notch_filter(shape, notch_centers, radius):
    rows, cols = shape
    mask = np.ones((rows, cols), np.uint8)

    for center in notch_centers:
        r, c = center
        r, c = int(r), int(c)  # Ensure indices are integers
        mask[max(0, r-radius):min(rows, r+radius), max(0, c-radius):min(cols, c+radius)] = 0

    return mask

# Define notch centers and radius for the Notch filter
notch_centers = [(crow, ccol), (crow-30, ccol-30)]
radius_notch = 10

# Define different notch centers and radius for the Optimum Notch filter
optimum_notch_centers = [(crow+30, ccol+30), (crow-60, ccol-60)]
radius_optimum_notch = 5

# Create Notch and Optimum Notch Filters
mask_notch = create_notch_filter(image.shape, notch_centers, radius_notch)
mask_optimum_notch = create_notch_filter(image.shape, optimum_notch_centers, radius_optimum_notch)

# Apply the filters
f_transform_notch = f_transform_shifted * mask_notch
f_transform_optimum_notch = f_transform_shifted * mask_optimum_notch

# Inverse Fourier Transform to get the images back
image_notch = np.fft.ifft2(np.fft.ifftshift(f_transform_notch))
image_notch = np.abs(image_notch)

image_optimum_notch = np.fft.ifft2(np.fft.ifftshift(f_transform_optimum_notch))
image_optimum_notch = np.abs(image_optimum_notch)

# Step 3: Improved Inverse Filtering

def inverse_filtering(image, degradation_function, eps=1e-3):
    # Fourier Transform of the image
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # To avoid division by zero, we add a small epsilon to the degradation function
    degradation_function = np.where(degradation_function == 0, eps, degradation_function)  # Avoid division by zero
    inverse_filter = np.conj(degradation_function) / (np.abs(degradation_function)**2 + eps)
    
    # Apply the inverse filter
    f_transform_filtered = f_transform_shifted * inverse_filter
    
    # Inverse Fourier Transform to get the restored image
    image_restored = np.fft.ifft2(np.fft.ifftshift(f_transform_filtered))
    image_restored = np.abs(image_restored)
    
    return image_restored

# Create a simple degradation function (e.g., motion blur in the frequency domain)
degradation_function = np.ones_like(image, dtype=np.float32)
degradation_function[crow-10:crow+10, ccol-10:ccol+10] = 0.1  # Simulate degradation

# Apply Improved Inverse Filtering
image_inverse_filtered = inverse_filtering(image, degradation_function)

# Step 4: Improved Wiener Filtering

def apply_wiener_filter(image, kernel_size=(5, 5), noise_power=0.01):
    # Fourier Transform of the image
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # Power spectrum of the degradation function
    degradation_function = np.ones_like(image)
    degradation_function[crow-10:crow+10, ccol-10:ccol+10] = 0.1  # Simulate degradation
    
    # Power spectrum of the image
    power_spectrum_image = np.abs(f_transform_shifted)**2
    eps = 1e-3  # Small epsilon to avoid division by zero
    
    # Wiener filter formula
    H_conjugate = np.conj(degradation_function)
    wiener_filter = H_conjugate / (np.abs(degradation_function)**2 + noise_power / (power_spectrum_image + eps))
    
    # Apply the Wiener filter
    f_transform_filtered = f_transform_shifted * wiener_filter
    
    # Inverse Fourier Transform to get the restored image
    image_wiener_filtered = np.fft.ifft2(np.fft.ifftshift(f_transform_filtered))
    image_wiener_filtered = np.abs(image_wiener_filtered)
    
    return image_wiener_filtered

# Apply Improved Wiener Filtering
image_wiener_filtered = apply_wiener_filter(image)

# Display all results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(image_bandpass, cmap='gray')
plt.title('Bandpass Filtered Image')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(image_bandreject, cmap='gray')
plt.title('Bandreject Filtered Image')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(image_notch, cmap='gray')
plt.title('Notch Filtered Image')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(image_optimum_notch, cmap='gray')
plt.title('Optimum Notch Filtered Image')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(image_inverse_filtered, cmap='gray')
plt.title('Inverse Filtered Image')
plt.axis('off')

plt.figure(figsize=(5, 5))
plt.imshow(image_wiener_filtered, cmap='gray')
plt.title('Wiener Filtered Image')
plt.axis('off')
plt.show()
