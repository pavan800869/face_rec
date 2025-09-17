#!/usr/bin/env python3
"""
Avinya Professional Welcome UI (mac / local webcam)

This edited version:
- Main welcoming UI: centered Avinya logo on a minimal background.
- Separate camera window: live webcam feed with detection boxes and labels.
- Face recognition (dlib preferred; Haar cascade fallback).
- Coqui TTS primary (cached WAV files) with pyttsx3 fallback.
- TTS will NOT speak messages equal to "avinya" (case-insensitive).
- If no faces -> main UI shows only logo + muted requirements; no TTS.
- If faces present -> show "Welcome <name>" (if recognized) or "Welcome Participant".
- TTS is ALWAYS ON by default and will recursively speak the current welcome messages
  (Coqui audio cached to tts_cache/*.wav). No Speak button or toggles in the UI.
- Clean shutdown.

Run:
    python avinya_ui_camera_no_fade_no_button.py
"""
import os
import sys
import time
import threading
import hashlib
import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# --- optional libraries ---
try:
    import dlib
    _HAS_DLIB = True
except Exception:
    dlib = None
    _HAS_DLIB = False

# Coqui TTS optional
_COQUI_AVAILABLE = False
try:
    from TTS.api import TTS
    _COQUI_AVAILABLE = True
except Exception:
    _COQUI_AVAILABLE = False

# pyttsx3 fallback
_PYTTX3_AVAILABLE = False
try:
    import pyttsx3
    _PYTTX3_AVAILABLE = True
except Exception:
    pyttsx3 = None
    _PYTTX3_AVAILABLE = False

# -----------------------
# Configuration
# -----------------------
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600

LOGO_SIZE = 280
LOGO_COLOR = (20, 40, 80)  # BGR
BG_COLOR_TOP = (245, 247, 250)
BG_COLOR_BOTTOM = (255, 255, 255)

MATCH_THRESHOLD = 0.60

USE_COQUI = True and _COQUI_AVAILABLE
COQUI_MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"
COQUI_SPEED = 0.95
TTS_CACHE_DIR = Path("tts_cache")
TTS_CACHE_DIR.mkdir(exist_ok=True)

FEATURES_CSV = Path("data/features_all.csv")
DLIB_PREDICTOR = Path("data_dlib/shape_predictor_68_face_landmarks.dat")
DLIB_FACEREC = Path("data_dlib/dlib_face_recognition_resnet_model_v1.dat")

WINDOW_MAIN = "AVINYA - Welcome"
WINDOW_CAMERA = "AVINYA - Camera"

# -----------------------
# Helpers
# -----------------------
def platform_player_command(path: str) -> Optional[str]:
    """Return a shell command to play a wav file for the current platform, or None."""
    if sys.platform.startswith("darwin"):
        if shutil.which("afplay"):
            return f'afplay "{path}"'
    else:
        if shutil.which("aplay"):
            return f'aplay "{path}"'
        if shutil.which("ffplay"):
            return f'ffplay -nodisp -autoexit -loglevel quiet "{path}"'
        if shutil.which("cvlc"):
            return f'cvlc --play-and-exit --intf dummy "{path}"'
    return None

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# -----------------------
# Logo & drawing
# -----------------------
def make_logo_image(size: int = LOGO_SIZE) -> np.ndarray:
    logo = np.full((size, size, 3), 255, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    cv2.circle(logo, (cx, cy), size // 2 - 4, (230, 230, 235), -1)
    cv2.circle(logo, (cx, cy), size // 2 - 18, (245, 247, 250), -1)
    cv2.circle(logo, (cx, cy), size // 8, LOGO_COLOR, -1)
    # small AVINYA label beneath center symbol
    text = "AVINYA"
    font_scale = size / 300.0
    thickness = max(1, int(font_scale * 2.5))
    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    tx = cx - ts[0] // 2
    ty = cy + size // 3
    cv2.putText(logo, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, LOGO_COLOR, thickness, cv2.LINE_AA)
    return logo

LOGO_IMG = make_logo_image(LOGO_SIZE)

def draw_background(canvas: np.ndarray):
    h, w = canvas.shape[:2]
    for i in range(h):
        alpha = i / float(h)
        col = (
            int(BG_COLOR_TOP[0] * (1 - alpha) + BG_COLOR_BOTTOM[0] * alpha),
            int(BG_COLOR_TOP[1] * (1 - alpha) + BG_COLOR_BOTTOM[1] * alpha),
            int(BG_COLOR_TOP[2] * (1 - alpha) + BG_COLOR_BOTTOM[2] * alpha),
        )
        canvas[i, :, :] = col

def draw_logo_center(canvas: np.ndarray, logo_img: np.ndarray):
    h, w = canvas.shape[:2]
    lh, lw = logo_img.shape[:2]
    x = (w - lw) // 2
    y = (h - lh) // 2 - 40
    # subtle drop shadow
    sh_offset = 8
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x + sh_offset + 10, y + sh_offset + 10),
                  (x + lw - 10, y + lh - 10), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.06, canvas, 0.94, 0, canvas)
    canvas[y:y+lh, x:x+lw] = logo_img

def draw_requirements(canvas: np.ndarray, lines: List[str]):
    h, w = canvas.shape[:2]
    base_y = (h + LOGO_SIZE) // 2 + 10
    for i, ln in enumerate(lines):
        ts = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        x = (w - ts[0]) // 2
        y = base_y + i * 28
        cv2.putText(canvas, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1, cv2.LINE_AA)

# -----------------------
# Recognition helpers
# -----------------------
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
# TTS manager
# -----------------------
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
# Main loop
# -----------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] cannot open webcam")
        return

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_MAIN, SCREEN_WIDTH, SCREEN_HEIGHT)
    cv2.namedWindow(WINDOW_CAMERA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CAMERA, 800, 600)

    # TTS always ON by default
    tts = TTSManager(use_coqui=USE_COQUI)

    # requirements text shown on idle
    req_lines = [
        "Please stand in front of the camera",
        "Ensure your face is well-lit"
    ]

    last_time = time.time()
    fps = 0.0
    frame_count = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("[WARN] frame grab failed")
                time.sleep(0.1)
                continue

            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                fps = frame_count / (now - last_time)
                frame_count = 0
                last_time = now

            # recognition uses RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            recs = compute_recognition(frame_rgb)
            faces_present = len(recs) > 0

            # MAIN CANVAS (no camera preview)
            canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
            draw_background(canvas)
            draw_logo_center(canvas, LOGO_IMG)

            # detection & messages
            welcome_msgs = []
            if faces_present:
                # prepare messages
                for rec in recs:
                    if rec['name']:
                        welcome_msgs.append(f"Welcome {rec['name']}")
                    else:
                        welcome_msgs.append("Welcome Participant")
                # display unique messages beneath the logo
                uniq = list(dict.fromkeys(welcome_msgs))
                base_y = (SCREEN_HEIGHT + LOGO_SIZE) // 2
                for i, msg in enumerate(uniq):
                    y = base_y + 40 + i * 46
                    ts = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                    x = (SCREEN_WIDTH - ts[0]) // 2
                    cv2.putText(canvas, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, LOGO_COLOR, 2, cv2.LINE_AA)
                # update TTS with texts (TTSManager filters "avinya")
                tts.update_texts(uniq)
            else:
                # idle: show small requirements
                draw_requirements(canvas, req_lines)
                tts.update_texts([])

            # fps small
            cv2.putText(canvas, f"FPS: {fps:.1f}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90,90,90), 1)

            cv2.imshow(WINDOW_MAIN, canvas)

            # CAMERA WINDOW: show live feed with annotations (separate)
            cam_display = frame_bgr.copy()
            # draw header
            cv2.rectangle(cam_display, (0,0), (cam_display.shape[1], 34), (230,230,230), -1)
            cv2.putText(cam_display, "AVINYA Camera", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30,30,30), 2)
            # annotate faces
            for rec in recs:
                if USING_DLIB:
                    r = rec['rect']
                    x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()
                else:
                    x, y, w, h = rec['rect']
                    x1, y1, x2, y2 = x, y, x + w, y + h
                # clamp
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(cam_display.shape[1]-1, x2), min(cam_display.shape[0]-1, y2)
                color = (0, 140, 200)
                cv2.rectangle(cam_display, (x1, y1), (x2, y2), color, 2)
                label = rec['name'] if rec['name'] else "Participant"
                ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(cam_display, (x1, y1 - 28), (x1 + ts[0] + 10, y1 - 6), color, -1)
                cv2.putText(cam_display, label, (x1 + 6, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            cv2.imshow(WINDOW_CAMERA, cam_display)

            # input handling: 'q' or ESC to quit (no toggles for speak)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        print("[INFO] Shutting down...")
        try:
            tts.stop()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[INFO] Exit cleanly.")

if __name__ == "__main__":
    main()
