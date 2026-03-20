import cv2
import numpy as np

# Step 1: Load image
image = cv2.imread('../test.jpeg')
output = image.copy()

# Step 2: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Threshold to get binary image
_, binary = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 4: Remove noise with morphological opening
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Step 5: Find definite background (dilate)
sure_bg = cv2.dilate(cleaned, kernel, iterations=3)

# Step 6: Find definite foreground (distance transform)
dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# Step 7: Find unknown region (between fg and bg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Step 8: Label markers for watershed
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1          # background = 1, not 0
markers[unknown == 255] = 0    # unknown region = 0

# Step 9: Apply watershed
markers = cv2.watershed(image, markers)

# Step 10: Color the boundaries red
output[markers == -1] = [0, 0, 255]

# Step 11: Show results
cv2.imshow('Original',         image)
cv2.imshow('Binary',           binary)
cv2.imshow('Distance Map',     dist_transform / dist_transform.max())
cv2.imshow('Segmented',        output)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('segmented_output.jpg', output)