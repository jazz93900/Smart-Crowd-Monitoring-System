import cv2
import numpy as np

url = "http://192.168.1.2:4747/video"
cap = cv2.VideoCapture(url)

net = cv2.dnn.readNet(
    r"C:\Users\jasmi\OneDrive\programs\IDP Project\yolov3-tiny.weights",
    r"C:\Users\jasmi\OneDrive\programs\IDP Project\yolov3-tiny.cfg"
)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

total_people = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received.")
        break

    height, width = frame.shape[:2]

    # YOLO detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    people_count = 0
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)

            if class_id == 0 and scores[class_id] > 0.5:
                people_count += 1

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])

    # Zone detection
    zone_A = zone_B = zone_C = 0

    for (x, y, w, h) in boxes:
        if x < width/3:
            zone_A += 1
        elif x < 2*width/3:
            zone_B += 1
        else:
            zone_C += 1

    # Draw boxes
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Keyboard input
    key = cv2.waitKey(1) & 0xFF

    if key == ord('i'):
        total_people += 1

    if key == ord('o'):
        total_people -= 1

    # Density
    if total_people > 10:
        status = "HIGH"
    elif total_people > 5:
        status = "MEDIUM"
    else:
        status = "LOW"

    # Display
    cv2.putText(frame, f"Total (IR): {total_people}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(frame, f"Camera: {people_count}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(frame, f"A:{zone_A} B:{zone_B} C:{zone_C}", (10,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(frame, f"Status: {status}", (10,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Crowd Monitoring", frame)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()