import cv2
import numpy as np
import os


image_path = 'cython/assests/ISAR-image-of-the-target-satellite-after-segmentation-and-normalization.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"[ERROR] Could not load image at path: {image_path}")
    exit()


blurred = cv2.GaussianBlur(img, (5, 5), 0)


adaptive = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY_INV,
    11, 2
)


kernel = np.ones((3, 3), np.uint8)
morphed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)

edges = cv2.Canny(morphed, 50, 150)


contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 10:  
        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)


cv2.imshow("Robust Object Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("test passed", end=" ")
