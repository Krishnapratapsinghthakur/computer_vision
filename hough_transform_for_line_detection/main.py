import cv2

import numpy as np

image =cv2.imread("../test.jpeg")

output=image.copy()

gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blurred, 50, 150)

lines = cv2.HoughLinesP(edges,
rho=1,
theta=np.pi/180,
threshold=100,
minLineLength=50,
maxLineGap=10
)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Original", image)
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()