import cv2
import os
from picamera2 import Picamera2

# Ask for person's name
person_name = input("Enter the name of the person: ").strip()

# Create folder if not exists
dataset_path = "dataset"
person_path = os.path.join(dataset_path, person_name)
os.makedirs(person_path, exist_ok=True)

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

saved_count = 0
max_images = 100  # limit to 100 images

print("Starting face capture. Look at the camera...")

while True:
    # Capture frame from PiCamera2
    frame = picam2.capture_array()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save only the face region
        if saved_count < max_images:
            face_img = frame[y:y+h, x:x+w]
            img_name = os.path.join(person_path, f"{saved_count}.jpg")
            cv2.imwrite(img_name, face_img)
            print(f"Saved: {img_name}")
            saved_count += 1

    cv2.imshow("Register Face - Press 'q' to quit", frame)

    if saved_count >= max_images:
        print(f"Captured {max_images} images for {person_name}. Saved in {person_path}")
        break

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Capture stopped by user.")
        break

picam2.stop()
cv2.destroyAllWindows()
