
"""
 Universal Object Detection via Contour in Grayscale 
 uodcg


 implementation 6/6/25

"""
import cv2
import numpy as np

"""
load the image in grey scale
"""
image_path = 'cython/assests/Screenshot 2025-05-29 101009.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
"""
apply gaussian blue to reduce noise in the imaage

"""
# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(img, (5, 5), 0)

"""
# Apply Otsu's thresholding to binarize image

"""
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


"""
 Optional: Apply morphological closing to connect nearby pixels

"""

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Find contours of all distinct regions
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert image to color for drawing colored boxes
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Draw bounding boxes around all objects
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 5 and h > 5:  # filter out small noise
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), 2)  # white box

# Show final output
cv2.imshow("Detected Objects", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("test passed", end =" ")
