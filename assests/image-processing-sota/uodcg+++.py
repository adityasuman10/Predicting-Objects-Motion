"""
 Universal Object Detection via Contour in Grayscale 
 uodcg

 implementation 6/6/25
"""
import cv2
import numpy as np
import os

"""
Load the image in grayscale
"""
image_path = 'C:/vscode/cython/assests/images (1).jpeg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


if img is None:
    print(f"[ERROR] Could not load image at path: {image_path}")
    print("Current working directory:", os.getcwd())
    print("Absolute path attempted:", os.path.abspath(image_path))
    exit()

"""
Apply Gaussian blur to reduce noise in the image
"""
blurred = cv2.GaussianBlur(img, (5, 5), 0)

"""
Apply Otsu's thresholding to binarize image
"""
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

"""
Optional: Apply morphological closing to connect nearby pixels
"""
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 5 and h > 5:  
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), 2) 


cv2.imshow("Detected Objects", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("test passed", end=" ")
