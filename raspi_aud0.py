import sys
import cv2
import dlib
import numpy as np
import pandas as pd
from TTS.api import TTS
import threading
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

# ✅ Initialize Picamera2 with error handling
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
    print("3. You have proper permissions")
    sys.exit(1)

# ✅ Use pyttsx3 as fallback TTS (more reliable on Raspberry Pi)
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
    
    # Configure voice settings
    voices = tts_engine.getProperty('voices')
    if voices:
        # Try to use male voice if available
        for voice in voices:
            if 'male' in voice.name.lower() or 'man' in voice.name.lower():
                tts_engine.setProperty('voice', voice.id)
                break
    
    tts_engine.setProperty('rate', 150)  # Speed of speech
    tts_engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
    print("[INFO] pyttsx3 TTS initialized successfully")
    
except Exception as e:
    print(f"[ERROR] pyttsx3 TTS initialization failed: {e}")
    # Fallback to Coqui TTS if pyttsx3 fails
    try:
        from TTS.api import TTS
        tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
        SPEAKER = "p229"
        print("[INFO] Coqui TTS loaded as fallback")
        use_coqui = True
    except Exception as e2:
        print(f"[ERROR] Both TTS systems failed: {e2}")
        print("Install espeak: sudo apt install espeak espeak-ng")
        sys.exit(1)
    use_coqui = False

def speak(texts):
    """Speak a list of texts asynchronously"""
    def run_tts():
        for text in texts:
            try:
                if 'use_coqui' in globals() and use_coqui:
                    # Use Coqui TTS
                    tts.tts_to_file(
                        text=text,
                        file_path="output.wav",
                        speaker=SPEAKER,
                        speed=0.9
                    )
                    if os.name == "posix":
                        os.system("aplay output.wav")
                    else:
                        os.system("start output.wav")
                else:
                    # Use pyttsx3
                    tts_engine.say(text)
                    tts_engine.runAndWait()
            except Exception as e:
                print(f"[ERROR] TTS failed: {e}")
    threading.Thread(target=run_tts, daemon=True).start()

# 📂 Load face embeddings
try:
    features_csv = "data/features_all.csv"
    df = pd.read_csv(features_csv, index_col=0)
    print(f"[INFO] Loaded {len(df)} face embeddings")
except Exception as e:
    print(f"[ERROR] Failed to load face embeddings: {e}")
    sys.exit(1)

# 🤖 Load models
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("data_dlib/shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("data_dlib/dlib_face_recognition_resnet_model_v1.dat")
    print("[INFO] Face recognition models loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load dlib models: {e}")
    print("Make sure the model files exist in data_dlib/ directory")
    sys.exit(1)

print("[INFO] Starting real-time recognition with PiCamera2...")

welcome_texts = []
button_coords = (300, 500, 500, 580)  # Bigger button

# 🖱️ Mouse callback for Speak button
def mouse_click(event, x, y, flags, param):
    global welcome_texts
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = button_coords
        if x1 <= x <= x2 and y1 <= y <= y2:
            if welcome_texts:
                speak(welcome_texts)

cv2.namedWindow("Welcome Screen")
cv2.setMouseCallback("Welcome Screen", mouse_click)

# 🎥 Main loop
alpha = 0.0
fade_in = True
last_fade = time.time()
frame_count = 0

try:
    while True:
        try:
            # Capture frame from PiCamera2
            frame = picam2.capture_array()
            frame_count += 1
            
            # Convert RGB to BGR for OpenCV processing
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Use RGB for dlib processing (dlib expects RGB)
            frame_rgb = frame  # picamera2 already gives RGB
            
            faces = detector(frame_rgb, 0)

            welcome_texts = []

            if len(faces) == 0:
                welcome_texts.append("Avinya")

            for face in faces:
                shape = predictor(frame_rgb, face)
                face_descriptor = facerec.compute_face_descriptor(frame_rgb, shape)
                face_descriptor = np.array(face_descriptor)

                # Compare with dataset
                distances = []
                for i in range(len(df)):
                    person_name = df.index[i]
                    person_features = np.array(df.iloc[i].values)
                    dist = np.linalg.norm(person_features - face_descriptor)
                    distances.append((person_name, dist))

                best_match = min(distances, key=lambda x: x[1])
                name, min_dist = best_match

                if min_dist < 0.6:
                    text = f"{name} ({min_dist:.2f})"
                    welcome_texts.append(f"Welcome {name}")
                else:
                    text = "Unknown"
                    welcome_texts.append("Welcome Participant")

                # Draw on BGR frame for display
                cv2.rectangle(frame_bgr, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                cv2.putText(frame_bgr, text, (face.left(), face.top() - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Add frame counter
            cv2.putText(frame_bgr, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 🎯 First canvas (face recognition)
            cv2.imshow("Face Recognition", frame_bgr)

            # 🎯 Second canvas (Welcome Screen)
            welcome_canvas = np.zeros((600, 800, 3), dtype=np.uint8)

            # Handle fading effect only for "Avinya"
            if "Avinya" in welcome_texts:
                if time.time() - last_fade > 0.05:
                    if fade_in:
                        alpha += 0.05
                        if alpha >= 1.0:
                            fade_in = False
                    else:
                        alpha -= 0.05
                        if alpha <= 0.0:
                            fade_in = True
                    last_fade = time.time()

                overlay = welcome_canvas.copy()
                cv2.putText(overlay, "Avinya", (80, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, welcome_canvas, 1 - alpha, 0, welcome_canvas)
                welcome_texts.remove("Avinya")  # prevent duplicate
            else:
                alpha = 0.0  # reset

            # Draw other welcome texts
            y = 200
            for msg in welcome_texts:
                cv2.putText(welcome_canvas, msg, (80, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4, cv2.LINE_AA)
                y += 80

            # Draw Speak button
            x1, y1, x2, y2 = button_coords
            cv2.rectangle(welcome_canvas, (x1, y1), (x2, y2), (0, 255, 0), -1)
            cv2.putText(welcome_canvas, "SPEAK", (x1 + 30, y1 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)

            cv2.imshow("Welcome Screen", welcome_canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quitting...")
                break
                
        except Exception as e:
            print(f"[ERROR] Frame processing error: {e}")
            continue

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")
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
    print("[INFO] Windows closed")