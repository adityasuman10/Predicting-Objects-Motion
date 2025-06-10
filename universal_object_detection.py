import cv2
import numpy as np
import os
image_path = 'cython/assests/Screenshot 2025-06-10 075812.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"[ERROR] Could not load image at path: {image_path}")
    exit()

_, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)  


kernel = np.ones((5, 5), np.uint8)  
dilated = cv2.dilate(binary, kernel, iterations=2)


contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

if contours:
    
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)

   
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
else:
    print("[INFO] No object detected.")

cv2.imshow("Universal Object Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("test passed", end=" ")

