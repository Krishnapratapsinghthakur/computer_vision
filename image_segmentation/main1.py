import cv2
import numpy as np

image = cv2.imread('../test.jpeg')
clone = image.copy()

# Mouse drawing variables
drawing = False
start_x, start_y = -1, -1
end_x,   end_y   = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = clone.copy()
            cv2.rectangle(img_copy, (start_x, start_y), (x, y), (0,255,0), 2)
            cv2.imshow('Draw Rectangle', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y

        # Apply GrabCut with drawn rectangle
        mask    = np.zeros(image.shape[:2], np.uint8)
        bgModel = np.zeros((1, 65), np.float64)
        fgModel = np.zeros((1, 65), np.float64)

        x1 = min(start_x, end_x)
        y1 = min(start_y, end_y)
        w  = abs(end_x - start_x)
        h  = abs(end_y - start_y)

        if w > 0 and h > 0:
            rect = (x1, y1, w, h)
            cv2.grabCut(image, mask, rect, bgModel, fgModel, 5,
                        cv2.GC_INIT_WITH_RECT)

            mask2  = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            result = image * mask2[:, :, np.newaxis]

            cv2.imshow('Segmented Result', result)

cv2.namedWindow('Draw Rectangle')
cv2.setMouseCallback('Draw Rectangle', draw_rectangle)

cv2.imshow('Draw Rectangle', image)
print('Draw a rectangle around the object you want to segment!')
print('Press Q to quit')

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()