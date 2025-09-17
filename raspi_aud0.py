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
    
    # Configure voice settings for formal, clear speech
    voices = tts_engine.getProperty('voices')
    if voices:
        # Prefer female voices for clarity, then male voices
        selected_voice = None
        
        # First pass: look for clear female voices
        for voice in voices:
            if 'female' in voice.name.lower() or 'woman' in voice.name.lower():
                selected_voice = voice.id
                print(f"[INFO] Selected female voice: {voice.name}")
                break
        
        # Second pass: if no female voice, look for clear male voices
        if not selected_voice:
            for voice in voices:
                if 'male' in voice.name.lower() or 'man' in voice.name.lower():
                    selected_voice = voice.id
                    print(f"[INFO] Selected male voice: {voice.name}")
                    break
        
        # Use the first available voice if no gender-specific voice found
        if not selected_voice and voices:
            selected_voice = voices[0].id
            print(f"[INFO] Using default voice: {voices[0].name}")
        
        if selected_voice:
            tts_engine.setProperty('voice', selected_voice)
    
    # Formal speech settings: medium pace, clear pronunciation
    tts_engine.setProperty('rate', 180)      # Medium pace (words per minute)
    tts_engine.setProperty('volume', 0.95)   # Clear, audible volume
    
    # Additional properties for better clarity (if supported)
    try:
        tts_engine.setProperty('pitch', 50)      # Neutral pitch
        tts_engine.setProperty('gap', 50)        # Small pause between words
    except:
        pass  # Some TTS engines don't support these properties
    
    print("[INFO] pyttsx3 TTS configured for formal speech")
    use_coqui = False
    
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

def speak(texts):
    """Speak a list of texts asynchronously"""
    def run_tts():
        for text in texts:
            try:
                if use_coqui:
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

def draw_modern_button(canvas, rect, text, is_hovered=False):
    """Draw a modern gradient button with shadow"""
    x1, y1, x2, y2 = rect
    
    # Button shadow
    shadow_offset = 5
    cv2.rectangle(canvas, (x1 + shadow_offset, y1 + shadow_offset), 
                 (x2 + shadow_offset, y2 + shadow_offset), (30, 30, 30), -1)
    
    # Button gradient
    button_height = y2 - y1
    if is_hovered:
        color1, color2 = (0, 220, 0), (0, 150, 0)
    else:
        color1, color2 = (0, 180, 0), (0, 120, 0)
    
    for i in range(button_height):
        alpha = i / button_height
        blended_color = [
            int(color1[j] * (1 - alpha) + color2[j] * alpha) 
            for j in range(3)
        ]
        cv2.line(canvas, (x1, y1 + i), (x2, y1 + i), blended_color, 1)
    
    # Button border
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    # Button text with shadow
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    text_y = y1 + (y2 - y1 + text_size[1]) // 2
    
    # Text shadow
    cv2.putText(canvas, text, (text_x + 2, text_y + 2), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
    # Main text
    cv2.putText(canvas, text, (text_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

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
button_coords = (SCREEN_WIDTH//2 - 120, SCREEN_HEIGHT - 100, SCREEN_WIDTH//2 + 120, SCREEN_HEIGHT - 40)
mouse_pos = (0, 0)

# 🖱️ Mouse callback for Speak button and hover effects
def mouse_callback(event, x, y, flags, param):
    global welcome_texts, mouse_pos
    mouse_pos = (x, y)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = button_coords
        if x1 <= x <= x2 and y1 <= y <= y2:
            if welcome_texts:
                speak(welcome_texts)

cv2.namedWindow("AVINYA - Welcome System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AVINYA - Welcome System", SCREEN_WIDTH, SCREEN_HEIGHT)
cv2.setMouseCallback("AVINYA - Welcome System", mouse_callback)

# Create face recognition window
cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Recognition", 640, 480)
cv2.moveWindow("Face Recognition", 0, 0)

# 🎥 Main loop
alpha = 0.0
fade_in = True
last_fade = time.time()
frame_count = 0
pulse_time = 0
idle_animation_time = 0

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
            
            faces = detector(frame_rgb, 0)

            welcome_texts = []
            is_idle = len(faces) == 0

            if is_idle:
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

                # Draw enhanced face rectangle
                cv2.rectangle(frame_bgr, (face.left()-5, face.top()-5), 
                             (face.right()+5, face.bottom()+5), (0, 255, 0), 3)
                cv2.rectangle(frame_bgr, (face.left()-2, face.top()-2), 
                             (face.right()+2, face.bottom()+2), (255, 255, 255), 1)
                
                # Enhanced text with background
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(frame_bgr, (face.left(), face.top() - 35), 
                             (face.left() + text_size[0] + 10, face.top() - 5), 
                             (0, 0, 0), -1)
                cv2.putText(frame_bgr, text, (face.left() + 5, face.top() - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Add frame counter with modern styling
            cv2.rectangle(frame_bgr, (5, 5), (150, 35), (0, 0, 0), -1)
            cv2.putText(frame_bgr, f"Frame: {frame_count}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show face recognition window
            cv2.imshow("Face Recognition", frame_bgr)

            # 🎯 Main Welcome Screen with enhanced UI
            welcome_canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

            if is_idle:
                # Animated gradient background for idle state
                bg_color1 = (20, 30, 60)
                bg_color2 = (60, 40, 80)
                draw_gradient_background(welcome_canvas, bg_color1, bg_color2)
                
                # Pulsing circles around AVINYA
                pulse_factor = 0.5 + 0.5 * math.sin(pulse_time * 2)
                center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
                
                # Multiple pulsing circles
                for i in range(3):
                    circle_pulse = 0.3 + 0.7 * math.sin(pulse_time * 2 + i * 0.5)
                    radius = 150 + i * 50
                    alpha = 0.1 + 0.2 * circle_pulse
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
                
                # Welcome messages with enhanced styling
                y_start = (SCREEN_HEIGHT - len(welcome_texts) * 80) // 2
                for i, msg in enumerate(welcome_texts):
                    y = y_start + i * 80
                    
                    # Message background
                    msg_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)[0]
                    bg_x1 = (SCREEN_WIDTH - msg_size[0]) // 2 - 20
                    bg_x2 = (SCREEN_WIDTH + msg_size[0]) // 2 + 20
                    
                    # Animated background
                    bg_alpha = 0.6 + 0.2 * math.sin(pulse_time * 2 + i)
                    overlay = welcome_canvas.copy()
                    cv2.rectangle(overlay, (bg_x1, y - 40), (bg_x2, y + 20), 
                                 (50, 100, 50), -1)
                    cv2.addWeighted(overlay, bg_alpha, welcome_canvas, 1 - bg_alpha, 0, welcome_canvas)
                    
                    # Main text
                    text_x = (SCREEN_WIDTH - msg_size[0]) // 2
                    draw_animated_text(welcome_canvas, msg, (text_x, y), 
                                     2.5, (255, 255, 255), 4)

            # Check if mouse is hovering over button
            x1, y1, x2, y2 = button_coords
            is_hovered = x1 <= mouse_pos[0] <= x2 and y1 <= mouse_pos[1] <= y2
            
            # Draw modern speak button
            draw_modern_button(welcome_canvas, button_coords, "SPEAK", is_hovered)
            
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
        picam2.stop()
        print("[INFO] Camera stopped")
    except:
        pass
    cv2.destroyAllWindows()
    print("[INFO] Windows closed")