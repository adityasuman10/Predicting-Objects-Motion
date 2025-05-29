import cv2
import numpy as np

# Load the image in grayscale
img = cv2.imread("cython/assests/Screenshot 2025-05-29 101009.png", cv2.IMREAD_GRAYSCALE)

# Threshold the image
_, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert to BGR for color drawing
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Store bounding boxes of valid contours
boxes = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w * h < 20:
        continue
    boxes.append((x, y, x + w, y + h))

# Create a single bounding box around all valid boxes
if boxes:
    x_coords = [x1 for x1, y1, x2, y2 in boxes] + [x2 for x1, y1, x2, y2 in boxes]
    y_coords = [y1 for x1, y1, x2, y2 in boxes] + [y2 for x1, y1, x2, y2 in boxes]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Draw the group bounding box in blue
    cv2.rectangle(output, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
    cv2.putText(output, "cargoboat", (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Show result
cv2.imshow("Group Box Only", cv2.resize(output, (512, 512)))
cv2.waitKey(0)
cv2.destroyAllWindows()
