import cv2

image_path = "../test.jpeg"
image = cv2.imread(image_path)
# how can we extract the name of the image 
image_name = image_path.split("/")[-1]

gray_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("origninal image ",image)

cv2.imshow("Grayscale Image", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# saving the grayscale image 
# we have to ask the use for saving the image if he says 1 then we will save other wise do not save 
choice = input("Do you want to save the image? (1 for yes, 0 for no): ")
if(choice == "1"):
    cv2.imwrite(f"""grayscale-{image_name}""", gray_image)
    print("Image saved successfully")
else:
    print("Image not saved")

