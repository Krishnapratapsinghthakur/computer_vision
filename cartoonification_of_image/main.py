from turtle import color
import cv2
image = cv2.imread("../test.jpeg")

gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_blurred = cv2.medianBlur(gray, 7)

edges = cv2.Canny(gray_blurred, threshold1=80, threshold2=150)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
edges_dilated= cv2.dilate(edges,kernel,iterations=1)

edges_colored= cv2.bitwise_not(edges_dilated)

color_smooth=cv2.bilateralFilter(image, d=9, sigmaColor=300, sigmaSpace=300)

edges_3channel = cv2.cvtColor(edges_colored, cv2.COLOR_GRAY2BGR)

cartoon = cv2.bitwise_and(color_smooth, edges_3channel)


cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()

