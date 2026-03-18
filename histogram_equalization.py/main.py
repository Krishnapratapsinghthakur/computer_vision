import cv2

image=cv2.imread("../test.jpeg",cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not load image. Check file path.")
    exit(1)

equalized=cv2.equalizeHist(image)

cv2.imshow("Original", image)
cv2.imshow("Equalized", equalized)

cv2.waitKey(0)
cv2.destroyAllWindows()

choice = input("Do you want to save the equalized image? (y/n): ")
if choice == "y":
    cv2.imwrite("equalized.jpeg", equalized)
    print("Image saved successfully!")
else:
    print("Image not saved!")  
