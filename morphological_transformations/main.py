
import cv2
import numpy as np

image = cv2.imread("../test.jpeg")

gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_ , binary= cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

dilated=cv2.dilate(binary,kernel,iterations=1)

eroded=cv2.erode(binary,kernel,iterations=1)

opened=cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)

closed=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)

#  show all the results
cv2.imshow("Original",image)
cv2.imshow("Binary",binary)
cv2.imshow("Dilated",dilated)
cv2.imshow("Eroded",eroded)
cv2.imshow("Opened",opened)
cv2.imshow("Closed",closed)

cv2.waitKey(0)
cv2.destroyAllWindows()