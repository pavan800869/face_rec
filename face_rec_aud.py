import cv2
import dlib
import numpy as np
import pandas as pd
from TTS.api import TTS
import threading
import os
import time


# ✅ Use English-only Coqui TTS model
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)

# Pick male-like speaker (check tts.speakers for options)
SPEAKER = "p225"   # You can change based on available speakers in this model

def speak(texts):
    """Speak a list of texts asynchronously using Coqui TTS"""
    def run_tts():
        for text in texts:
            tts.tts_to_file(
                text=text,
                file_path="output.wav",
                speaker="p229",
                speed=0.7   # slower for clarity
            )
            if os.name == "posix":  # Linux / Mac
                os.system("afplay output.wav" if "darwin" in os.sys.platform else "aplay output.wav")
            else:  # Windows
                os.system("start output.wav")
    threading.Thread(target=run_tts, daemon=True).start()


# 📂 Load face embeddings
features_csv = "data/features_all.csv"
df = pd.read_csv(features_csv, index_col=0)

# 🤖 Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data_dlib/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("data_dlib/dlib_face_recognition_resnet_model_v1.dat")

cap = cv2.VideoCapture(0)
print("[INFO] Starting real-time recognition...")

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

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, text, (face.left(), face.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 🎯 First canvas
    cv2.imshow("Face Recognition", frame)

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
        break

cap.release()
cv2.destroyAllWindows()
