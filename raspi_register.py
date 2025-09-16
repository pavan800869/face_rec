import sys
import cv2
import os
import time

# Add system packages path for picamera2
sys.path.append('/usr/lib/python3/dist-packages')

try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: picamera2 not found. Please install with:")
    print("sudo apt install python3-picamera2")
    sys.exit(1)

# Ask for person's name
person_name = input("Enter the name of the person: ").strip()

if not person_name:
    print("Error: Please enter a valid name")
    sys.exit(1)

# Create folder if not exists
dataset_path = "dataset"
person_path = os.path.join(dataset_path, person_name)
os.makedirs(person_path, exist_ok=True)
print(f"[INFO] Dataset will be saved to: {person_path}")

# Load OpenCV's Haar Cascade for face detection
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        sys.exit(1)
    print("[INFO] Face cascade loaded successfully")
except Exception as e:
    print(f"Error loading face cascade: {e}")
    sys.exit(1)

# Initialize PiCamera2
try:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)},
        display="main"
    )
    picam2.configure(config)
    picam2.start()
    
    # Allow camera to warm up
    time.sleep(2)
    print("[INFO] Camera initialized successfully")
    
except Exception as e:
    print(f"[ERROR] Camera initialization failed: {e}")
    print("Make sure:")
    print("1. Camera is enabled: sudo raspi-config > Interface Options > Camera")
    print("2. picamera2 is installed: sudo apt install python3-picamera2")
    sys.exit(1)

saved_count = 0
max_images = 100  # limit to 100 images
frame_count = 0

print("Starting face capture. Look at the camera...")
print("Controls:")
print("- Press 'q' to quit early")
print("- Press 's' to skip current frame")
print(f"- Will automatically stop after {max_images} images")

try:
    while True:
        try:
            # Capture frame from PiCamera2 (returns RGB format)
            frame_rgb = picam2.capture_array()
            frame_count += 1
            
            # Convert RGB to BGR for OpenCV processing and display
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            # Draw info on frame
            cv2.putText(frame, f"Saved: {saved_count}/{max_images}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Person: {person_name}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add face info
                cv2.putText(frame, f"Face: {w}x{h}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Save only the face region
                if saved_count < max_images:
                    face_img = frame[y:y+h, x:x+w]
                    img_name = os.path.join(person_path, f"{saved_count:03d}.jpg")
                    
                    # Only save if face is reasonably sized
                    if w > 50 and h > 50:
                        success = cv2.imwrite(img_name, face_img)
                        if success:
                            print(f"Saved: {img_name} (Size: {w}x{h})")
                            saved_count += 1
                            
                            # Brief pause between captures
                            time.sleep(0.1)
                        else:
                            print(f"Failed to save: {img_name}")
                    else:
                        print(f"Face too small ({w}x{h}), skipping...")
            
            # Show number of faces detected
            if len(faces) == 0:
                cv2.putText(frame, "No face detected", (10, 450), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 450), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Register Face - Press 'q' to quit", frame)
            
            if saved_count >= max_images:
                print(f"Captured {max_images} images for {person_name}. Saved in {person_path}")
                break
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Capture stopped by user.")
                break
            elif key == ord('s'):
                print("Skipping current frame...")
                continue
                
        except Exception as e:
            print(f"[ERROR] Frame processing error: {e}")
            continue

except KeyboardInterrupt:
    print("\n[INFO] Capture interrupted by user")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
finally:
    # Cleanup
    try:
        picam2.stop()
        print("[INFO] Camera stopped")
    except:
        pass
    cv2.destroyAllWindows()
    print(f"[INFO] Total images saved: {saved_count}")
    print(f"[INFO] Images saved in: {person_path}")