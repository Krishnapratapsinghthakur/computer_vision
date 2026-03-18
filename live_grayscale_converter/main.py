import cv2

# connect to webcame 
cap=cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret,frame=cap.read()
    if not ret:
        print("error:cannot read the frame ")
        break 
    gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Original", frame)
    cv2.imshow("Grayscale", gray_frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()