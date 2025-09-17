#!/usr/bin/env python3
"""
Avinya Professional Welcome UI (Raspberry Pi + PiCamera)
- Uses picamera2 (PiCamera v2/v3)
- Face recognition with dlib or Haar cascade fallback
- Coqui TTS (cached) or pyttsx3 fallback
- Displays main welcome UI and live camera feed
- Skips "avinya" for TTS
"""

import os
import sys
import time
import threading
import hashlib
from pathlib import Path
from typing import List

import cv2
import numpy as np

# --- optional libraries ---
try:
    import dlib
    _HAS_DLIB = True
except Exception:
    dlib = None
    _HAS_DLIB = False

try:
    from TTS.api import TTS
    _COQUI_AVAILABLE = True
except Exception:
    _COQUI_AVAILABLE = False

try:
    import pyttsx3
    _PYTTX3_AVAILABLE = True
except Exception:
    _PYTTX3_AVAILABLE = False

# PiCamera2 import
try:
    from picamera2 import Picamera2, Preview
    PICAMERA_AVAILABLE = True
except Exception:
    PICAMERA_AVAILABLE = False

# -----------------------
# Config
# -----------------------
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 600
LOGO_SIZE = 280
MATCH_THRESHOLD = 0.6
USE_COQUI = True and _COQUI_AVAILABLE
COQUI_MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"
COQUI_SPEED = 0.95
TTS_CACHE_DIR = Path("tts_cache")
TTS_CACHE_DIR.mkdir(exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def platform_player_command(path: str):
    if sys.platform.startswith("linux"):
        if os.system("which aplay > /dev/null 2>&1") == 0:
            return f"aplay {path}"
        if os.system("which ffplay > /dev/null 2>&1") == 0:
            return f"ffplay -nodisp -autoexit -loglevel quiet {path}"
    elif sys.platform.startswith("darwin"):
        return f"afplay {path}"
    return None

# -----------------------
# Logo & UI
# -----------------------
def make_logo_image(size=LOGO_SIZE):
    logo = np.full((size, size, 3), 255, np.uint8)
    cx, cy = size // 2, size // 2
    cv2.circle(logo, (cx, cy), size//2 - 4, (230, 230, 235), -1)
    cv2.circle(logo, (cx, cy), size//2 - 18, (245, 247, 250), -1)
    cv2.circle(logo, (cx, cy), size//8, (20, 40, 80), -1)
    # label
    text = "AVINYA"
    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size/300.0, 2)[0]
    cv2.putText(logo, text, (cx - ts[0]//2, cy + size//3),
                cv2.FONT_HERSHEY_SIMPLEX, size/300.0, (20,40,80), 2, cv2.LINE_AA)
    return logo

LOGO_IMG = make_logo_image(LOGO_SIZE)

def draw_background(canvas):
    h, w = canvas.shape[:2]
    for i in range(h):
        alpha = i/h
        col = tuple([int(245*(1-alpha) + 255*alpha),
                     int(247*(1-alpha) + 255*alpha),
                     int(250*(1-alpha) + 255*alpha)])
        canvas[i,:,:] = col

def draw_logo_center(canvas):
    h, w = canvas.shape[:2]
    lh, lw = LOGO_IMG.shape[:2]
    canvas[(h-lh)//2:(h+lh)//2, (w-lw)//2:(w+lw)//2] = LOGO_IMG

# -----------------------
# Recognition placeholder (same as before)
# -----------------------
# Here you would keep the compute_recognition() function exactly as before
# using dlib or Haar cascade. Omitted for brevity.
if FEATURES_CSV.exists():
    try:
        import pandas as pd
        _df = pd.read_csv(str(FEATURES_CSV), index_col=0)
        FEATURE_NAMES = list(_df.index)
        FEATURES_MATRIX = _df.to_numpy(dtype=float)
        print(f"[INFO] Loaded {len(FEATURE_NAMES)} embeddings")
    except Exception as e:
        print("[WARN] Failed to load features CSV:", e)
        FEATURE_NAMES = []
        FEATURES_MATRIX = np.empty((0, 128))
else:
    FEATURE_NAMES = []
    FEATURES_MATRIX = np.empty((0, 128))

if _HAS_DLIB and DLIB_PREDICTOR.exists() and DLIB_FACEREC.exists():
    try:
        _detector = dlib.get_frontal_face_detector()
        _predictor = dlib.shape_predictor(str(DLIB_PREDICTOR))
        _facerec = dlib.face_recognition_model_v1(str(DLIB_FACEREC))
        USING_DLIB = True
        print("[INFO] dlib models loaded")
    except Exception as e:
        print("[WARN] dlib load error:", e)
        USING_DLIB = False
        _detector = None
        _predictor = None
        _facerec = None
else:
    USING_DLIB = False
    _detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    _predictor = None
    _facerec = None

def compute_recognition(frame_rgb: np.ndarray):
    """
    Return list of dicts {'rect': (dlib rect or (x,y,w,h)), 'name': str|None, 'dist': float}
    """
    results = []
    if USING_DLIB:
        rects = _detector(frame_rgb, 0)
        for r in rects:
            try:
                shape = _predictor(frame_rgb, r)
                desc = np.array(_facerec.compute_face_descriptor(frame_rgb, shape), dtype=float)
                if FEATURES_MATRIX.size > 0:
                    dists = np.linalg.norm(FEATURES_MATRIX - desc, axis=1)
                    idx = int(dists.argmin())
                    min_dist = float(dists[idx])
                    name = FEATURE_NAMES[idx] if min_dist < MATCH_THRESHOLD else None
                else:
                    name = None
                    min_dist = 1.0
                results.append({'rect': r, 'name': name, 'dist': min_dist})
            except Exception as e:
                print("[WARN] descriptor error:", e)
    else:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
        dets = _detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        for (x,y,w,h) in dets:
            results.append({'rect': (x,y,w,h), 'name': None, 'dist': 1.0})
    return results

# -----------------------
# TTS Manager (same as before)
# -----------------------
# Keep TTSManager class exactly as in your original code
class TTSManager:
    """
    Caches WAV files for Coqui TTS, plays them in a loop while enabled.
    Falls back to pyttsx3 live speak if Coqui not available.
    Skips message "avinya" (case-insensitive).
    This manager is ALWAYS enabled by default.
    """

    def __init__(self, use_coqui: bool = True):
        self.use_coqui = use_coqui and _COQUI_AVAILABLE
        self.coqui = None
        if self.use_coqui:
            try:
                self.coqui = TTS(model_name=COQUI_MODEL_NAME, progress_bar=False, gpu=False)
                print("[TTS] Coqui ready")
            except Exception as e:
                print("[TTS] Coqui init failed:", e)
                self.use_coqui = False

        self.use_pyttsx3 = (not self.use_coqui) and _PYTTX3_AVAILABLE
        self.py_engine = None
        if self.use_pyttsx3:
            try:
                self.py_engine = pyttsx3.init()
                print("[TTS] pyttsx3 ready")
            except Exception as e:
                print("[TTS] pyttsx3 init failed:", e)
                self.use_pyttsx3 = False

        self.current_texts: List[str] = []
        self.lock = threading.Lock()
        # ALWAYS enabled by default
        self.enabled = True
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def update_texts(self, texts: List[str]):
        # filter out 'avinya' and empty strings
        filtered = [t for t in texts if t and t.strip() and t.strip().lower() != "avinya"]
        with self.lock:
            self.current_texts = filtered

    def stop(self):
        self._stop.set()
        self._worker.join(timeout=1.0)

    def _ensure_audio(self, text: str) -> Optional[str]:
        key = sha1_hex(text)
        target = TTS_CACHE_DIR / f"{key}.wav"
        if target.exists():
            return str(target)
        if self.use_coqui and self.coqui:
            try:
                try:
                    # modern API may accept speed
                    self.coqui.tts_to_file(text=text, file_path=str(target), speed=COQUI_SPEED)
                except TypeError:
                    self.coqui.tts_to_file(text=text, file_path=str(target))
                return str(target)
            except Exception as e:
                print("[TTS] Coqui generating failed:", e)
                return None
        # pyttsx3: no caching to file here
        return None

    def _play_file(self, path: str):
        cmd = platform_player_command(path)
        if cmd:
            try:
                os.system(cmd)
            except Exception as e:
                print("[TTS] playback error:", e)
        else:
            # no system player - fallback print
            print("[TTS] (no player) ->", path)

    def _loop(self):
        prev_texts = None
        prepared = []
        while not self._stop.is_set():
            with self.lock:
                enabled = self.enabled
                texts = list(self.current_texts)
            if not enabled or not texts:
                prev_texts = None
                time.sleep(0.15)
                continue

            if texts != prev_texts:
                prepared = []
                for t in texts:
                    fp = None
                    if self.use_coqui:
                        fp = self._ensure_audio(t)
                    prepared.append((t, fp))
                prev_texts = list(texts)

            # play prepared sequence
            for t, fp in prepared:
                if self._stop.is_set():
                    break
                with self.lock:
                    # If texts changed while in playback, break to rebuild
                    if self.current_texts != prev_texts:
                        break
                if fp and os.path.exists(fp):
                    self._play_file(fp)
                else:
                    # direct speak fallback
                    if self.use_pyttsx3 and self.py_engine:
                        try:
                            self.py_engine.say(t)
                            self.py_engine.runAndWait()
                        except Exception as e:
                            print("[TTS] pyttsx3 speak error:", e)
                    else:
                        print("[TTS] ->", t)
                # small inter-message pause
                for _ in range(8):
                    if self._stop.is_set():
                        break
                    time.sleep(0.05)
            # pause before repeating
            for _ in range(20):
                if self._stop.is_set():
                    break
                time.sleep(0.05)

# -----------------------
# Main loop (PiCamera version)
# -----------------------
def main():
    if not PICAMERA_AVAILABLE:
        print("[ERROR] picamera2 not available")
        return

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    cv2.namedWindow("AVINYA - Main", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AVINYA - Main", SCREEN_WIDTH, SCREEN_HEIGHT)
    cv2.namedWindow("AVINYA - Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AVINYA - Camera", 800, 600)

    tts = TTSManager(use_coqui=USE_COQUI)

    req_lines = [
        "Please stand in front of the camera",
        "Ensure your face is well-lit"
    ]

    try:
        while True:
            frame = picam2.capture_array()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            recs = compute_recognition(frame_rgb)
            faces_present = len(recs) > 0

            canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), np.uint8)
            draw_background(canvas)
            draw_logo_center(canvas)

            welcome_msgs = []
            if faces_present:
                for r in recs:
                    name = r['name'] if r['name'] else "Participant"
                    welcome_msgs.append(f"Welcome {name}")
                uniq = list(dict.fromkeys(welcome_msgs))
                tts.update_texts(uniq)
            else:
                tts.update_texts([])
            cv2.imshow("AVINYA - Main", canvas)

            cam_display = frame.copy()
            for rec in recs:
                if hasattr(rec['rect'], 'left'):
                    r = rec['rect']
                    x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()
                else:
                    x, y, w, h = rec['rect']
                    x1, y1, x2, y2 = x, y, x + w, y + h
                cv2.rectangle(cam_display, (x1,y1), (x2,y2), (0,140,200), 2)
            cv2.imshow("AVINYA - Camera", cam_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        tts.stop()
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
