import cv2

cap=cv2.VideoCapture(0)

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=25,
    detectShadows=True
)

kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

MIN_AREA = 500
while True:
    ret, frame = cap.read()
    if not ret:
        break
    fg_mask = bg_subtractor.apply(frame)

    fg_mask=cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask=cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    _,fg_mask=cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue
        
        x,y,w,h = cv2.boundingRect(contour)
        cv2.putText(frame, f"Motion! area: {int(area)}",
        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        motion_detected = True
    
    if motion_detected:
        cv2.putText(frame, "Motion detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", fg_mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


