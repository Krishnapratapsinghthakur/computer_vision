from cProfile import label
import cv2


from pyzbar import pyzbar

def decode_barcode(frame):
    barcodes= pyzbar.decode(frame)

    for barcode in barcodes:
        x,y,w,h = barcode.rect
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type

        label= f"{barcode_type}: {barcode_data}"

        cv2.putText(frame,label,
        (x, y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2)

        print(f"found {barcode_type} barcode: {barcode_data}")

    return frame


cap=cv2.VideoCapture(0)

print('scanner ready - hold a barcode or qr code up to the camara ')


scanned=set()

while True:
    ret,frame=cap.read()
    if not ret:
        break

    barcodes=pyzbar.decode(frame)

    for barcode in barcodes:
        x,y,w,h = barcode.rect
        barcode_data = barcode.data.decode("utf8")
        barcode_type = barcode.type

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

        label= f"{barcode_type}: {barcode_data}"

        cv2.putText(frame,label,
        (x, y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2)

        if barcode_data not in scanned:
            scanned.add(barcode_data)
            print(f"found {barcode_type} barcode: {barcode_data}")


    cv2.putText(frame, f"scanned: {len(scanned)}codes", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.imshow("Barcode Scanner", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print('\n--- All Scanned Codes ---')
for code in scanned:
    print(code)