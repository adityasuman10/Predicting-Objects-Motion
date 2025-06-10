import cv2
import numpy as np
import os

# Load image in grayscale
image_path = 'cython/assests/ISAR-image-of-the-target-satellite-after-segmentation-and-normalization.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"[ERROR] Could not load image at path: {image_path}")
    exit()

# Step 1: Denoise and smooth
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Step 2: Adaptive thresholding to handle varied lighting
adaptive = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,  # You can try cv2.ADAPTIVE_THRESH_GAUSSIAN_C too
    cv2.THRESH_BINARY_INV,
    11, 2
)

# Step 3: Morphological closing to fill small gaps in object
kernel = np.ones((3, 3), np.uint8)
morphed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)

# Step 4: Canny edge detection to capture object shape, not just brightness
edges = cv2.Canny(morphed, 50, 150)

# Step 5: Contour detection
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert grayscale image to BGR for visualization
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Step 6: Draw all object contours
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 10:  # Filter very small noise
        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

# Show results
cv2.imshow("Robust Object Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("test passed", end=" ")
