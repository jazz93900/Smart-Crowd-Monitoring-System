import cv2

# DroidCam video URL
url = "http://192.168.1.6:4747/video"  

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Camera not found! Check URL or WiFi connection.")
    exit()

face_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received.")
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # If at least one face is detected, increase counter
    if len(faces) > 0:
        face_count += 1
        print(f"Face detected! Total count: {face_count}")

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display with detections
    cv2.imshow("People Detection", frame)

    # ESC key to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()