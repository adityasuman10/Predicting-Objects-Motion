import cv2
import numpy as np
img = cv2.imread("cython/assests/Screenshot 2025-05-29 101009.png", cv2.IMREAD_GRAYSCALE)
_, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)      
    if w * h < 20:  
        continue
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(output, f"ID {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
# Show image
cv2.imshow("Blobs with IDs", cv2.resize(output, (512, 512)))
cv2.waitKey(0)
cv2.destroyAllWindows()
