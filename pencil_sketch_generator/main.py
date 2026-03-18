import cv2

image = cv2.imread("../test.jpeg")

if image is None:
    print("Error: Could not load image. Check file path.")
    exit(1)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

inverted_gray = cv2.bitwise_not(gray)

blurred =cv2.GaussianBlur(inverted_gray,(21,21),0)

sketch = cv2.divide(gray,blurred,scale=256.0)

cv2.imshow("Original", image)
cv2.imshow("Sketch", sketch)

cv2.waitKey(0)
cv2.destroyAllWindows()

choice = input("Do you want to save the sketch? (y/n): ")
if choice == "y":
    cv2.imwrite("sketch.png", sketch)
    print("Sketch saved successfully!")
else:
    print("Sketch not saved!")
    