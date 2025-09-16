import cv2
from picamera2 import Picamera2

# Initialize camera
picam2 = Picamera2()

# Configure preview
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

print("Camera started. Press 'q' to quit.")

while True:
    frame = picam2.capture_array()  # Get frame
    cv2.imshow("PiCamera2 Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
