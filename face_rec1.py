# face_recognize.py
import cv2
import dlib
import numpy as np
import pandas as pd

# Load features
features_csv = "data/features_all.csv"
df = pd.read_csv(features_csv, index_col=0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data_dlib/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("data_dlib/dlib_face_recognition_resnet_model_v1.dat")

cap = cv2.VideoCapture(0)
print("[INFO] Starting real-time recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(frame_rgb, 0)

    # Default message if no one is detected
    welcome_messages = ["AVINYA"]  

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

        # Find best match
        best_match = min(distances, key=lambda x: x[1])
        name, min_dist = best_match

        # Thresholding
        if min_dist < 0.6:  # typical dlib threshold
            text = f"{name} ({min_dist:.2f})"
            welcome_messages.append(f"Welcome {name}")
        else:
            text = "Unknown"
            welcome_messages.append("Welcome Participant")

        # Draw rectangle & label on face
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, text, (face.left(), face.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # First canvas → face recognition
    cv2.imshow("Face Recognition", frame)

    # Second canvas → welcome messages
    welcome_canvas = np.zeros((400, 600, 3), dtype=np.uint8)  # black background

    y = 150
    for msg in welcome_messages:
        cv2.putText(welcome_canvas, msg, (50, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
        y += 50  # spacing between lines

    cv2.imshow("Welcome Screen", welcome_canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
