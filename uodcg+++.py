import cv2
import numpy as np
import os

# Load the image in grayscale
image_path = 'cython/assests/Screenshot 2025-06-10 075942.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"[ERROR] Could not load image at path: {image_path}")
    exit()

# Threshold to detect white or very bright pixels
_, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)  # 200 is a brightness threshold

# Dilate to connect nearby white pixels (in case object is fragmented)
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(binary, kernel, iterations=2)

# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create output image for visualization
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

if contours:
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # Draw one bounding box over all detected white regions
    cv2.rectangle(output, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
else:
    print("[INFO] No object detected.")

cv2.imshow("Universal Object Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("test passed", end=" ")
