import cv2

# Step 1: Load pre-trained face and eye detectors
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# Step 2: Open webcam
cap = cv2.VideoCapture(0)

print('Face detector running — press Q to quit')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 3: Convert to grayscale (Haar works on grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 4: Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Step 5: Loop through each detected face
    for (x, y, w, h) in faces:

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Label
        cv2.putText(frame, 'Face',
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        # Step 6: Detect eyes INSIDE each face region
        face_gray  = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(15, 15)
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_color,
                          (ex, ey),
                          (ex + ew, ey + eh),
                          (255, 0, 0), 2)

    # Step 7: Show face count
    cv2.putText(frame, f'Faces: {len(faces)}',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()