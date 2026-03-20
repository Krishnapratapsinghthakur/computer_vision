import cv2
import numpy as np

def order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype='float32')

    # Top-left = smallest sum, bottom-right = largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right = smallest diff, bottom-left = largest diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def get_document_transform(image):
    """Find document in image and return warped (aligned) version"""

    orig = image.copy()

    # Step 1: Preprocess
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    # Step 2: Dilate edges to close gaps
    kernel        = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    # Step 3: Find contours
    contours, _ = cv2.findContours(
        edges_dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Step 4: Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    doc_contour = None

    # Step 5: Find the contour with 4 corners (the document)
    for contour in contours[:5]:  # check top 5 largest

        # Approximate contour to polygon
        perimeter = cv2.arcLength(contour, True)
        approx    = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # If it has 4 corners → it's our document!
        if len(approx) == 4:
            doc_contour = approx
            break

    if doc_contour is None:
        print('No document found!')
        return image

    # Step 6: Draw detected document outline
    cv2.drawContours(orig, [doc_contour], -1, (0, 255, 0), 3)
    cv2.imshow('Document Detected', orig)

    # Step 7: Order the 4 corner points
    pts   = doc_contour.reshape(4, 2).astype('float32')
    rect  = order_points(pts)
    tl, tr, br, bl = rect

    # Step 8: Calculate output dimensions
    # Width = max of top edge or bottom edge
    width_top    = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    width        = int(max(width_top, width_bottom))

    # Height = max of left edge or right edge
    height_left  = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    height       = int(max(height_left, height_right))

    # Step 9: Define destination points (perfect rectangle)
    dst = np.float32([
        [0,         0],           # top-left
        [width - 1, 0],           # top-right
        [width - 1, height - 1],  # bottom-right
        [0,         height - 1]   # bottom-left
    ])

    # Step 10: Get perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)

    # Step 11: Apply the transform!
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped


# ── Main ──────────────────────────────────────
image  = cv2.imread('../scene.jpeg')
result = get_document_transform(image)

# Post process — convert to clean scan look
gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
_, scan     = cv2.threshold(gray_result, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('Original',       image)
cv2.imshow('Aligned',        result)
cv2.imshow('Clean Scan',     scan)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('aligned_output.jpg', result)
cv2.imwrite('scan_output.jpg',    scan)
print('Document aligned and saved!')