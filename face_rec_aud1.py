# face_recognize.py
import cv2
import dlib
import numpy as np
import pandas as pd
import pyttsx3
import threading
import queue
import time
import math

# ------------------------
# Configuration
# ------------------------
FEATURES_CSV = "data/features_all.csv"
PREDICTOR_PATH = "data_dlib/shape_predictor_68_face_landmarks.dat"
FACEREC_PATH = "data_dlib/dlib_face_recognition_resnet_model_v1.dat"

MATCH_THRESHOLD = 0.6
TRACK_DIST_THRESHOLD = 80
SPEAK_COOLDOWN = 8.0
ENTITY_STALE_TIME = 6.0
IDLE_SPEAK_ON_LEAVE = True

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ------------------------
# Load data & models
# ------------------------
df = pd.read_csv(FEATURES_CSV, index_col=0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
facerec = dlib.face_recognition_model_v1(FACEREC_PATH)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# ------------------------
# Speech thread + queue
# ------------------------
speech_q = queue.Queue()

def speech_worker(q: queue.Queue):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)
    while True:
        text = q.get()
        if text is None:
            break
        if isinstance(text, (list, tuple)):
            for t in text:
                engine.say(t)
            engine.runAndWait()
        else:
            engine.say(text)
            engine.runAndWait()
        q.task_done()
    try:
        engine.stop()
    except Exception:
        pass

speech_thread = threading.Thread(target=speech_worker, args=(speech_q,), daemon=True)
speech_thread.start()

# ------------------------
# Entity tracker
# ------------------------
next_entity_id = 1
entities = []

def center_from_rect(rect):
    return ((rect.left() + rect.right()) // 2, (rect.top() + rect.bottom()) // 2)

def find_matching_entity(center):
    best_idx, best_dist = None, None
    for i, e in enumerate(entities):
        ex, ey = e["center"]
        d = math.hypot(center[0] - ex, center[1] - ey)
        if d <= TRACK_DIST_THRESHOLD and (best_dist is None or d < best_dist):
            best_dist, best_idx = d, i
    return best_idx

# For idle detection
prev_any_entities = False
current_welcome_messages = []  # store visible messages for button click

# ------------------------
# Mouse click handler
# ------------------------
BUTTON_REGION = (50, 300, 250, 360)  # (x1, y1, x2, y2)

def on_mouse(event, x, y, flags, param):
    global current_welcome_messages
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = BUTTON_REGION
        if x1 <= x <= x2 and y1 <= y <= y2:
            if current_welcome_messages:
                # Enqueue all current messages to speak
                speech_q.put(current_welcome_messages.copy())

cv2.namedWindow("Welcome Screen")
cv2.setMouseCallback("Welcome Screen", on_mouse)

print("[INFO] Starting real-time recognition... (press 'q' to quit)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(frame_rgb, 0)

        now = time.time()
        detected_entities_in_frame = []

        for face in faces:
            shape = predictor(frame_rgb, face)
            face_desc = np.array(facerec.compute_face_descriptor(frame_rgb, shape))

            distances = []
            for i in range(len(df)):
                person_name = df.index[i]
                person_features = np.array(df.iloc[i].values, dtype=float)
                dist = np.linalg.norm(person_features - face_desc)
                distances.append((person_name, dist))

            if distances:
                best_match = min(distances, key=lambda x: x[1])
                name, min_dist = best_match
            else:
                name, min_dist = ("Unknown", 1e6)

            if min_dist < MATCH_THRESHOLD:
                label, is_known = f"Welcome {name}", True
            else:
                label, is_known = "Welcome Participant", False

            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            display_text = f"{name} ({min_dist:.2f})" if is_known else "Unknown"
            cv2.putText(frame, display_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            c = center_from_rect(face)
            match_idx = find_matching_entity(c)
            if match_idx is not None:
                e = entities[match_idx]
                e["center"], e["last_seen_ts"] = c, now
                e["label"], e["is_known"] = label, is_known
            else:
                e = {
                    "id": next_entity_id,
                    "label": label,
                    "center": c,
                    "last_seen_ts": now,
                    "last_spoken_ts": 0.0,
                    "is_known": is_known
                }
                next_entity_id += 1
                entities.append(e)
                match_idx = len(entities) - 1
            detected_entities_in_frame.append(match_idx)

        entities = [e for e in entities if now - e["last_seen_ts"] <= ENTITY_STALE_TIME]

        welcome_messages, any_entities_now = [], len(entities) > 0
        for e in entities:
            if now - e["last_seen_ts"] <= 1.0:
                welcome_messages.append(e["label"])
                if now - e.get("last_spoken_ts", 0.0) >= SPEAK_COOLDOWN:
                    speech_q.put(e["label"])
                    e["last_spoken_ts"] = now

        if IDLE_SPEAK_ON_LEAVE and (not any_entities_now) and prev_any_entities:
            speech_q.put("AVINYA")
        prev_any_entities = any_entities_now

        current_welcome_messages = welcome_messages.copy()

        # Build Welcome Screen canvas
        welcome_canvas = np.zeros((400, 800, 3), dtype=np.uint8)
        y = 120
        if welcome_messages:
            seen = set()
            for msg in welcome_messages:
                if msg in seen:
                    continue
                seen.add(msg)
                cv2.putText(welcome_canvas, msg, (50, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
                y += 60
        else:
            cv2.putText(welcome_canvas, "AVINYA", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 4, cv2.LINE_AA)

        # Draw "Speak" button
        x1, y1, x2, y2 = BUTTON_REGION
        cv2.rectangle(welcome_canvas, (x1, y1), (x2, y2), (100, 200, 250), -1)
        cv2.putText(welcome_canvas, "Speak", (x1 + 40, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)

        cv2.imshow("Face Recognition", frame)
        cv2.imshow("Welcome Screen", welcome_canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    speech_q.put(None)
    speech_thread.join(timeout=2)
