# face_train.py
import os
import dlib
import cv2
import numpy as np
import pandas as pd

# Paths
path_faces = "dataset"
features_csv = "data/features_all.csv"

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

print("[INFO] Loading dlib models...")
try:
    detector = dlib.get_frontal_face_detector()
    print("[INFO] Face detector loaded successfully")
    
    # Check if model files exist
    predictor_path = "data_dlib/shape_predictor_68_face_landmarks.dat"
    facerec_path = "data_dlib/dlib_face_recognition_resnet_model_v1.dat"
    
    if not os.path.exists(predictor_path):
        print(f"[ERROR] Model file not found: {predictor_path}")
        print("[INFO] Please download the required model files first")
        exit(1)
        
    if not os.path.exists(facerec_path):
        print(f"[ERROR] Model file not found: {facerec_path}")
        print("[INFO] Please download the required model files first")
        exit(1)
    
    predictor = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(facerec_path)
    print("[INFO] All dlib models loaded successfully")
    
except Exception as e:
    print(f"[ERROR] Failed to load dlib models: {e}")
    exit(1)

features_all = []
person_names = []

print(f"[INFO] Processing faces from {path_faces}...")

person_folders = os.listdir(path_faces)
for person_folder in person_folders:
    person_path = os.path.join(path_faces, person_folder)
    
    # Skip if not a directory (like .DS_Store files)
    if not os.path.isdir(person_path):
        continue
        
    print(f"[INFO] Processing person: {person_folder}")
    features_person = []
    
    for img_name in os.listdir(person_path):
        # Skip hidden files and non-image files
        if img_name.startswith('.') or not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
            
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        
        # Skip if image couldn't be loaded
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = detector(img_rgb, 1)
        for face in faces:
            shape = predictor(img_rgb, face)
            face_descriptor = facerec.compute_face_descriptor(img_rgb, shape)
            features_person.append(face_descriptor)

    if features_person:
        features_mean = np.mean(features_person, axis=0)
        features_all.append(features_mean)
        person_names.append(person_folder)
        print(f"[INFO] Extracted {len(features_person)} face features for {person_folder}")
    else:
        print(f"[WARNING] No faces found for {person_folder}")

if features_all:
    df = pd.DataFrame(features_all, index=person_names)
    df.to_csv(features_csv)
    print(f"[INFO] Features saved in {features_csv}")
    print(f"[INFO] Total persons processed: {len(person_names)}")
else:
    print("[ERROR] No face features extracted. Please check your dataset.")
