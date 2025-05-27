import cv2
import numpy as np

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize counts
in_count = 0
out_count = 0

# Store previous centroids
prev_centroids = []

# Line position (center)
line_position = 320  # Assuming 640x480 resolution

# Start camera
cap = cv2.VideoCapture(0)

def get_centroid(x, y, w, h):
    return (int(x + w/2), int(y + h/2))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    new_centroids = []

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        centroid = get_centroid(x, y, w, h)
        new_centroids.append(centroid)
        cv2.circle(frame, centroid, 4, (0, 255, 0), -1)

        # Track crossing line
        for prev in prev_centroids:
            if abs(prev[1] - centroid[1]) < 50:  # y coordinate close enough
                if prev[0] < line_position and centroid[0] > line_position:
                    in_count += 1
                elif prev[0] > line_position and centroid[0] < line_position:
                    out_count += 1

    prev_centroids = new_centroids

    # Draw center line
    cv2.line(frame, (line_position, 0), (line_position, 480), (255, 0, 255), 2)

    # Display counters
    cv2.rectangle(frame, (0, 430), (640, 480), (0, 0, 0), -1)
    cv2.putText(frame, f"IN: {in_count}", (400, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {out_count}", (30, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("People Counter", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


