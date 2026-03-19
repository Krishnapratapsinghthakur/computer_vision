
import cv2


# capture the vedio
cap = cv2.VideoCapture(0)

# step 2:create background subtractor
# try switching bw mog2 and knn
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=5,
    varThreshold=25,
    detectShadows=True
)

# bg_subtractor=cv2.createBackgroundSubtractorKNN(
#     history=500,
#     dist2Threshold=600,
#     detectShadows=True
# )

while True:
    ret, frame=cap.read()
    if not ret:
        break
    fg_mask=bg_subtractor.apply(frame)

    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    fg_mask=cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask=cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    

    foreground=cv2.bitwise_and(frame, frame, mask=fg_mask)
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Foreground Mask", fg_mask)
    cv2.imshow("Foreground", foreground)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()