import sys
import cv2
import dlib
import numpy as np
import pandas as pd
from TTS.api import TTS
import threading
import os
import time
import math
import hashlib
from pathlib import Path

# Add system packages path for picamera2
sys.path.append('/usr/lib/python3/dist-packages')

try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: picamera2 not found. Please install with:")
    print("sudo apt install python3-picamera2")
    sys.exit(1)

# Screen dimensions for 7" display
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600

# TTS Cache directory
TTS_CACHE_DIR = Path("tts_cache")
TTS_CACHE_DIR.mkdir(exist_ok=True)

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

# ✅ Use Coqui TTS with Indian English accent
try:
    # Use Indian English model for clear accent
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
    print("[INFO] Coqui TTS with Indian English accent loaded successfully")
    use_coqui = True
except Exception as e:
    print(f"[ERROR] Coqui TTS initialization failed: {e}")
    # Fallback to pyttsx3
    try:
        import pyttsx3
        tts_engine = pyttsx3.init()
        
        voices = tts_engine.getProperty('voices')
        if voices:
            selected_voice = None
            # Look for Indian or clear English voices
            for voice in voices:
                if any(keyword in voice.name.lower() for keyword in ['indian', 'english', 'female']):
                    selected_voice = voice.id
                    print(f"[INFO] Selected voice: {voice.name}")
                    break
            
            if not selected_voice and voices:
                selected_voice = voices[0].id
                print(f"[INFO] Using default voice: {voices[0].name}")
            
            if selected_voice:
                tts_engine.setProperty('voice', selected_voice)
        
        tts_engine.setProperty('rate', 160)
        tts_engine.setProperty('volume', 0.95)
        print("[INFO] pyttsx3 TTS configured as fallback")
        use_coqui = False
    except Exception as e2:
        print(f"[ERROR] Both TTS systems failed: {e2}")
        sys.exit(1)

def sha1_hex(s: str) -> str:
    """Generate SHA1 hash for caching"""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def get_audio_player_command(path: str):
    """Get appropriate audio player command for the system"""
    if os.system("which aplay > /dev/null 2>&1") == 0:
        return f"aplay '{path}'"
    elif os.system("which ffplay > /dev/null 2>&1") == 0:
        return f"ffplay -nodisp -autoexit -loglevel quiet '{path}'"
    elif os.system("which cvlc > /dev/null 2>&1") == 0:
        return f"cvlc --play-and-exit --intf dummy '{path}'"
    return None

class TTSCache:
    """TTS Cache manager for recursive speaking with audio caching"""
    
    def __init__(self):
        self.current_texts = []
        self.is_speaking = False
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.speaker_thread = None
        
    def update_texts(self, texts):
        """Update texts to be spoken recursively"""
        with self.lock:
            self.current_texts = texts.copy()
            if texts and not self.is_speaking:
                self._start_speaking()
    
    def _ensure_audio_file(self, text):
        """Generate or retrieve cached audio file"""
        text_hash = sha1_hex(text)
        audio_path = TTS_CACHE_DIR / f"{text_hash}.wav"
        
        if audio_path.exists():
            return str(audio_path)
        
        if use_coqui:
            try:
                tts.tts_to_file(text=text, file_path=str(audio_path))
                return str(audio_path)
            except Exception as e:
                print(f"[ERROR] TTS generation failed: {e}")
                return None
        return None
    
    def _play_audio(self, file_path):
        """Play audio file using system player"""
        cmd = get_audio_player_command(file_path)
        if cmd:
            try:
                os.system(cmd)
            except Exception as e:
                print(f"[ERROR] Audio playback failed: {e}")
    
    def _start_speaking(self):
        """Start the recursive speaking thread"""
        if self.speaker_thread and self.speaker_thread.is_alive():
            return
        
        self.is_speaking = True
        self.stop_event.clear()
        self.speaker_thread = threading.Thread(target=self._speak_loop, daemon=True)
        self.speaker_thread.start()
    
    def _speak_loop(self):
        """Main speaking loop - speaks texts recursively"""
        while not self.stop_event.is_set():
            with self.lock:
                texts_to_speak = self.current_texts.copy()
            
            if not texts_to_speak:
                time.sleep(0.5)
                continue
            
            # Speak each text in sequence
            for text in texts_to_speak:
                if self.stop_event.is_set():
                    break
                
                if use_coqui:
                    audio_file = self._ensure_audio_file(text)
                    if audio_file:
                        self._play_audio(audio_file)
                    else:
                        print(f"[TTS] Speaking: {text}")
                else:
                    # Use pyttsx3 fallback
                    try:
                        tts_engine.say(text)
                        tts_engine.runAndWait()
                    except Exception as e:
                        print(f"[ERROR] pyttsx3 speak failed: {e}")
                
                # Check if texts changed during speaking
                with self.lock:
                    if self.current_texts != texts_to_speak:
                        break
                
                # Brief pause between messages
                time.sleep(0.3)
            
            # Pause before repeating
            time.sleep(2.0)
    
    def stop(self):
        """Stop speaking and cleanup"""
        with self.lock:
            self.current_texts = []
        self.stop_event.set()
        if self.speaker_thread and self.speaker_thread.is_alive():
            self.speaker_thread.join(timeout=2.0)
        self.is_speaking = False

# Initialize TTS Cache
tts_cache = TTSCache()

def calculate_text_scale(text, max_width, max_height, base_font_scale=2.5, min_font_scale=1.0):
    """Calculate appropriate font scale to fit text within bounds"""
    font_scale = base_font_scale
    thickness = max(2, int(font_scale * 2))
    
    while font_scale > min_font_scale:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        if text_size[0] <= max_width and text_size[1] <= max_height:
            return font_scale, thickness
        font_scale -= 0.1
        thickness = max(2, int(font_scale * 2))
    
    return min_font_scale, max(2, int(min_font_scale * 2))

def draw_gradient_background(canvas, color1, color2):
    """Draw a vertical gradient background"""
    height, width = canvas.shape[:2]
    for i in range(height):
        alpha = i / height
        blended_color = [
            int(color1[j] * (1 - alpha) + color2[j] * alpha) 
            for j in range(3)
        ]
        cv2.line(canvas, (0, i), (width, i), blended_color, 1)

def draw_pulsing_circle(canvas, center, radius, color, pulse_factor):
    """Draw a pulsing circle effect"""
    pulsing_radius = int(radius * pulse_factor)
    alpha = max(0.3, 1.0 - pulse_factor)
    
    # Create overlay for transparency
    overlay = canvas.copy()
    cv2.circle(overlay, center, pulsing_radius, color, -1)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

def draw_animated_text(canvas, text, position, font_scale, color, thickness, pulse_factor=1.0):
    """Draw text with animation effects"""
    # Add glow effect
    glow_color = (100, 100, 255) if text == "AVINYA" else (200, 200, 200)
    
    # Draw glow (multiple layers)
    for i in range(3, 0, -1):
        glow_thickness = thickness + i * 2
        glow_alpha = 0.3 * (4 - i) * pulse_factor
        overlay = canvas.copy()
        cv2.putText(overlay, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, glow_color, glow_thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, glow_alpha, canvas, 1 - glow_alpha, 0, canvas)
    
    # Draw main text
    cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
               font_scale, color, thickness, cv2.LINE_AA)

def draw_auto_fitted_text(canvas, text, y_position, max_width, color=(255, 255, 255)):
    """Draw text with automatic size adjustment to fit within width"""
    font_scale, thickness = calculate_text_scale(text, max_width, 80)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (SCREEN_WIDTH - text_size[0]) // 2
    
    # Background with animation
    pulse_alpha = 0.6 + 0.2 * math.sin(time.time() * 2)
    bg_x1 = text_x - 20
    bg_x2 = text_x + text_size[0] + 20
    
    overlay = canvas.copy()
    cv2.rectangle(overlay, (bg_x1, y_position - 40), (bg_x2, y_position + 20), 
                 (50, 100, 50), -1)
    cv2.addWeighted(overlay, pulse_alpha, canvas, 1 - pulse_alpha, 0, canvas)
    
    # Draw text with glow
    draw_animated_text(canvas, text, (text_x, y_position), font_scale, color, thickness)

def compute_face_descriptor_with_validation(frame_rgb, shape):
    """Compute face descriptor with additional validation"""
    try:
        # Check if we have enough landmarks
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        
        # Basic validation - check if landmarks are reasonable
        face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
        
        # Skip very small or very large detections
        if face_width < 30 or face_height < 30 or face_width > 500 or face_height > 500:
            return None
            
        # Check landmark quality - eyes should be reasonably aligned
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
        eye_distance = np.linalg.norm(right_eye - left_eye)
        
        # Skip if eyes are too close or too far (poor detection)
        if eye_distance < 20 or eye_distance > 200:
            return None
            
        descriptor = facerec.compute_face_descriptor(frame_rgb, shape)
        return np.array(descriptor)
    except Exception as e:
        print(f"[DEBUG] Face descriptor computation failed: {e}")
        return None

def recognize_face_improved(face_descriptor, df, confidence_threshold=0.45, top_k_validation=3):
    """Improved face recognition with multiple validation steps"""
    if face_descriptor is None:
        return "Unknown", 1.0
    
    # Compute distances to all known faces
    distances = []
    for i in range(len(df)):
        person_name = df.index[i]
        person_features = np.array(df.iloc[i].values)
        
        # Use Euclidean distance
        euclidean_dist = np.linalg.norm(person_features - face_descriptor)
        
        # Also compute cosine similarity for additional validation
        cosine_sim = np.dot(person_features, face_descriptor) / (
            np.linalg.norm(person_features) * np.linalg.norm(face_descriptor)
        )
        cosine_dist = 1 - cosine_sim
        
        # Combined score (weighted average)
        combined_dist = 0.7 * euclidean_dist + 0.3 * cosine_dist
        
        distances.append((person_name, euclidean_dist, cosine_dist, combined_dist))
    
    # Sort by combined distance
    distances.sort(key=lambda x: x[3])
    
    best_match = distances[0]
    name, euclidean_dist, cosine_dist, combined_dist = best_match
    
    # Multi-stage validation
    
    # Stage 1: Primary threshold check
    if combined_dist > confidence_threshold:
        return "Unknown", combined_dist
    
    # Stage 2: Check if the best match is significantly better than the second best
    if len(distances) > 1:
        second_best_dist = distances[1][3]
        margin = second_best_dist - combined_dist
        
        # Require a minimum margin to avoid false positives
        min_margin = 0.1
        if margin < min_margin:
            return "Unknown", combined_dist
    
    # Stage 3: Check consistency across top matches
    # If multiple people are very close, it's likely unknown
    top_k_matches = distances[:min(top_k_validation, len(distances))]
    top_k_distances = [match[3] for match in top_k_matches]
    
    # Check if there's a clear winner
    std_dev = np.std(top_k_distances)
    if std_dev < 0.05:  # Too many similar matches - probably unknown
        return "Unknown", combined_dist
    
    # Stage 4: Additional euclidean distance check
    if euclidean_dist > 0.5:  # Conservative euclidean threshold
        return "Unknown", combined_dist
    
    # Stage 5: Cosine similarity check
    if cosine_dist > 0.4:  # Conservative cosine threshold
        return "Unknown", combined_dist
    
    return name, combined_dist

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

# Create windows
cv2.namedWindow("AVINYA - Welcome System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AVINYA - Welcome System", SCREEN_WIDTH, SCREEN_HEIGHT)

cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Recognition", 640, 480)
cv2.moveWindow("Face Recognition", 0, 0)

# 🎥 Main loop
frame_count = 0
pulse_time = 0
idle_animation_time = 0
face_stable_count = {}  # Track face stability over multiple frames
min_stable_frames = 5   # Minimum frames for stable recognition

try:
    while True:
        try:
            # Capture frame from PiCamera2
            frame = picam2.capture_array()
            frame_count += 1
            pulse_time += 0.02
            idle_animation_time += 0.03
            
            # Convert RGB to BGR for OpenCV processing
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Use RGB for dlib processing (dlib expects RGB)
            frame_rgb = frame  # picamera2 already gives RGB
            
            # Detect faces with improved parameters
            faces = detector(frame_rgb, 1)  # Increased upsampling for better detection
            
            welcome_texts = []
            is_idle = len(faces) == 0
            
            if is_idle:
                # Stop TTS when idle
                tts_cache.stop()
                # Reset face stability tracking
                face_stable_count.clear()
            else:
                current_frame_faces = {}
                
                for i, face in enumerate(faces):
                    # Validate face size
                    face_width = face.right() - face.left()
                    face_height = face.bottom() - face.top()
                    
                    # Skip very small faces (likely false positives)
                    if face_width < 80 or face_height < 80:
                        continue
                    
                    shape = predictor(frame_rgb, face)
                    face_descriptor = compute_face_descriptor_with_validation(frame_rgb, shape)
                    
                    if face_descriptor is None:
                        continue
                    
                    # Improved face recognition
                    name, confidence = recognize_face_improved(face_descriptor, df)
                    
                    # Track face stability
                    face_key = f"face_{i}"
                    if face_key not in face_stable_count:
                        face_stable_count[face_key] = {"name": name, "count": 1, "confidence": confidence}
                    else:
                        if face_stable_count[face_key]["name"] == name:
                            face_stable_count[face_key]["count"] += 1
                        else:
                            # Name changed, reset counter
                            face_stable_count[face_key] = {"name": name, "count": 1, "confidence": confidence}
                    
                    current_frame_faces[face_key] = True
                    
                    # Only use stable recognitions
                    stable_name = name
                    if face_stable_count[face_key]["count"] >= min_stable_frames:
                        stable_name = face_stable_count[face_key]["name"]
                    else:
                        stable_name = "Verifying..."
                    
                    # Display logic
                    if stable_name != "Verifying..." and stable_name != "Unknown":
                        text = f"{stable_name} ({confidence:.2f})"
                        color = (0, 255, 0)  # Green for recognized
                        # Updated welcome message for recognized people
                        if face_stable_count[face_key]["count"] >= min_stable_frames:
                            welcome_texts.append(f"Thank you {stable_name}, Welcome to AVINYA 2025")
                    elif stable_name == "Unknown":
                        text = f"Unknown ({confidence:.2f})"
                        color = (0, 165, 255)  # Orange for unknown
                        # Updated message for unknown people
                        if face_stable_count[face_key]["count"] >= min_stable_frames:
                            welcome_texts.append("Welcome Participant")
                    else:
                        text = stable_name
                        color = (255, 255, 0)  # Yellow for verifying
                    
                    # Draw enhanced face rectangle with color coding
                    cv2.rectangle(frame_bgr, (face.left()-5, face.top()-5), 
                                 (face.right()+5, face.bottom()+5), color, 3)
                    cv2.rectangle(frame_bgr, (face.left()-2, face.top()-2), 
                                 (face.right()+2, face.bottom()+2), (255, 255, 255), 1)
                    
                    # Enhanced text with background
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame_bgr, (face.left(), face.top() - 35), 
                                 (face.left() + text_size[0] + 10, face.top() - 5), 
                                 (0, 0, 0), -1)
                    cv2.putText(frame_bgr, text, (face.left() + 5, face.top() - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Show stability indicator
                    stability_text = f"Stable: {face_stable_count[face_key]['count']}/{min_stable_frames}"
                    cv2.putText(frame_bgr, stability_text, (face.left() + 5, face.bottom() + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Clean up face tracking for faces that are no longer detected
                face_stable_count = {k: v for k, v in face_stable_count.items() if k in current_frame_faces}
                
                # Update TTS with welcome messages only for stable recognitions
                if welcome_texts:
                    tts_cache.update_texts(welcome_texts)
            
            # Add enhanced frame counter with recognition stats
            cv2.rectangle(frame_bgr, (5, 5), (250, 60), (0, 0, 0), -1)
            cv2.putText(frame_bgr, f"Frame: {frame_count}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Faces: {len(faces)}, Stable: {len([v for v in face_stable_count.values() if v['count'] >= min_stable_frames])}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Show face recognition window
            cv2.imshow("Face Recognition", frame_bgr)
            
            # 🎯 Main Welcome Screen with enhanced UI (NO BUTTON)
            welcome_canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
            
            if is_idle:
                # Animated gradient background for idle state
                bg_color1 = (20, 30, 60)
                bg_color2 = (60, 40, 80)
                draw_gradient_background(welcome_canvas, bg_color1, bg_color2)
                
                # Pulsing circles around AVINYA
                center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
                
                # Multiple pulsing circles
                for i in range(3):
                    circle_pulse = 0.3 + 0.7 * math.sin(pulse_time * 2 + i * 0.5)
                    radius = 150 + i * 50
                    color = (50 + i * 30, 100 + i * 20, 200 - i * 30)
                    draw_pulsing_circle(welcome_canvas, center, radius, color, circle_pulse)
                
                # Animated AVINYA text in center
                text_pulse = 0.8 + 0.2 * math.sin(idle_animation_time * 3)
                font_scale = 4.0 * text_pulse
                
                # Center the text
                text_size = cv2.getTextSize("AVINYA", cv2.FONT_HERSHEY_SIMPLEX, font_scale, 6)[0]
                text_x = (SCREEN_WIDTH - text_size[0]) // 2
                text_y = (SCREEN_HEIGHT + text_size[1]) // 2
                
                draw_animated_text(welcome_canvas, "AVINYA", (text_x, text_y), 
                                 font_scale, (255, 255, 255), 6, text_pulse)
                
                # Subtitle with fade effect
                subtitle_alpha = 0.7 + 0.3 * math.sin(idle_animation_time * 1.5)
                subtitle = "Intelligent Welcome System"
                subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                subtitle_x = (SCREEN_WIDTH - subtitle_size[0]) // 2
                subtitle_y = text_y + 80
                
                overlay = welcome_canvas.copy()
                cv2.putText(overlay, subtitle, (subtitle_x, subtitle_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 200, 255), 2, cv2.LINE_AA)
                cv2.addWeighted(overlay, subtitle_alpha, welcome_canvas, 1 - subtitle_alpha, 0, welcome_canvas)
                
            else:
                # Active state - different background
                bg_color1 = (10, 50, 20)
                bg_color2 = (30, 80, 40)
                draw_gradient_background(welcome_canvas, bg_color1, bg_color2)
                
                # Welcome messages with auto-fitting text
                max_text_width = SCREEN_WIDTH - 80  # Leave margins
                y_start = (SCREEN_HEIGHT - len(welcome_texts) * 100) // 2
                
                for i, msg in enumerate(welcome_texts):
                    y = y_start + i * 100
                    draw_auto_fitted_text(welcome_canvas, msg, y, max_text_width)
            
            # Add corner decorations
            corner_size = 50
            cv2.circle(welcome_canvas, (corner_size, corner_size), 30, (100, 150, 255), 2)
            cv2.circle(welcome_canvas, (SCREEN_WIDTH - corner_size, corner_size), 30, (100, 150, 255), 2)
            cv2.circle(welcome_canvas, (corner_size, SCREEN_HEIGHT - corner_size), 30, (100, 150, 255), 2)
            cv2.circle(welcome_canvas, (SCREEN_WIDTH - corner_size, SCREEN_HEIGHT - corner_size), 30, (100, 150, 255), 2)

            cv2.imshow("AVINYA - Welcome System", welcome_canvas)

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
        tts_cache.stop()
        print("[INFO] TTS stopped")
    except:
        pass
    try:
        picam2.stop()
        print("[INFO] Camera stopped")
    except:
        pass
    cv2.destroyAllWindows()
    print("[INFO] Windows closed")