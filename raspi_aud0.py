#!/usr/bin/env python3
"""
Avinya Professional Welcome UI (Pi / local)

- Keeps the original polished UI (centered Avinya symbol, pulsing rings, animated subtitle).
- Separates camera window (live feed + face boxes).
- Continuously speaks detected welcome messages (Coqui TTS primary, pyttsx3 fallback).
- SPEAK button removed; speech is ON by default and runs iteratively until no messages.
- Messages equal to "avinya" (case-insensitive) are NOT spoken.
- Long messages are wrapped to fit inside the UI and displayed line-by-line.
- Audio for Coqui is cached to tts_cache/<sha1>.wav to avoid re-generation each loop.
- Gentle debouncing: speech repeats while the same messages remain, but not excessively.
- Clean shutdown on exit.

Notes:
- This file is intentionally verbose and modular for clarity and to provide a rich,
  professional single-file demo.
- If you want to run on mac/local webcam instead of PiCamera2: replace the Picamera2 capture
  with cv2.VideoCapture(0) read logic (kept as-is per the UI you're running).
"""

import os
import sys
import time
import math
import threading
import hashlib
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# -------------------------
# Optional libraries
# -------------------------
try:
    import dlib
    _HAS_DLIB = True
except Exception:
    dlib = None
    _HAS_DLIB = False

# Coqui TTS
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

# picamera2 (for Raspberry Pi) — keep for environments that use it
try:
    # allow nonstandard package path if needed (your environment may not need this)
    sys.path.append('/usr/lib/python3/dist-packages')
    from picamera2 import Picamera2
    _HAS_PICAMERA2 = True
except Exception:
    Picamera2 = None
    _HAS_PICAMERA2 = False

# -------------------------
# Configuration
# -------------------------
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600

LOGO_SIZE = 280
LOGO_COLOR = (20, 40, 80)  # BGR for logo text/core
BG_COLOR_TOP = (245, 247, 250)
BG_COLOR_BOTTOM = (255, 255, 255)

MATCH_THRESHOLD = 0.60

# TTS settings
COQUI_MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"
COQUI_SPEED = 0.95
TTS_CACHE_DIR = Path("tts_cache")
TTS_CACHE_DIR.mkdir(exist_ok=True)

# Data/model paths
FEATURES_CSV = Path("data/features_all.csv")
DLIB_PREDICTOR = Path("data_dlib/shape_predictor_68_face_landmarks.dat")
DLIB_FACEREC = Path("data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Windows
WINDOW_MAIN = "AVINYA - Welcome System"
WINDOW_CAMERA = "AVINYA - Face Recognition"

# UI positions (keep original centre behavior)
# Helper for spacing text under the logo
TEXT_BASE_OFFSET = 40

# Audio playback helper: platform player
def platform_player_command(path: str) -> Optional[str]:
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


# -------------------------
# Logo & UI drawing helpers
# -------------------------
def make_logo_image(size: int = LOGO_SIZE) -> np.ndarray:
    logo = np.full((size, size, 3), 255, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    # soft ring layers
    cv2.circle(logo, (cx, cy), size // 2 - 4, (230, 230, 235), -1)
    cv2.circle(logo, (cx, cy), size // 2 - 18, (245, 247, 250), -1)
    cv2.circle(logo, (cx, cy), size // 8, LOGO_COLOR, -1)
    # label "AVINYA" below core
    text = "AVINYA"
    font_scale = size / 300.0
    thickness = max(1, int(font_scale * 2.5))
    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    tx = cx - ts[0] // 2
    ty = cy + size // 3
    cv2.putText(logo, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, LOGO_COLOR, thickness, cv2.LINE_AA)
    return logo


LOGO_IMG = make_logo_image(LOGO_SIZE)


def draw_gradient_background(canvas: np.ndarray, color_top: Tuple[int, int, int], color_bottom: Tuple[int, int, int]):
    h, w = canvas.shape[:2]
    # vertical gradient
    for i in range(h):
        alpha = i / float(h)
        col = (
            int(color_top[0] * (1 - alpha) + color_bottom[0] * alpha),
            int(color_top[1] * (1 - alpha) + color_bottom[1] * alpha),
            int(color_top[2] * (1 - alpha) + color_bottom[2] * alpha),
        )
        canvas[i, :, :] = col


def draw_logo_center(canvas: np.ndarray, logo_img: np.ndarray):
    h, w = canvas.shape[:2]
    lh, lw = logo_img.shape[:2]
    x = (w - lw) // 2
    y = (h - lh) // 2 - 40
    # subtle shadow
    sh_offset = 8
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x + sh_offset + 10, y + sh_offset + 10),
                  (x + lw - 10, y + lh - 10), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.06, canvas, 0.94, 0, canvas)
    canvas[y:y + lh, x:x + lw] = logo_img


def draw_pulsing_circles(canvas: np.ndarray, center: Tuple[int, int], base_radius: int, pulse_phase: float):
    # three pulsing faint circles
    for i in range(3):
        pf = 0.6 + 0.4 * math.sin(pulse_phase * (1.0 + i * 0.3) + i)
        r = int(base_radius * (0.9 + i * 0.25) * (0.9 + 0.12 * pf))
        color = (40 + i * 30, 100 + i * 20, 200 - i * 30)
        alpha = 0.08 + 0.08 * pf
        overlay = canvas.copy()
        cv2.circle(overlay, center, r, color, -1)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)


def draw_animated_text_glow(canvas: np.ndarray, text: str, center_x: int, y: int,
                            font_scale: float, color: Tuple[int, int, int], thickness: int, glow: float):
    # glow by drawing multiple thicker translucent outlines
    for i in range(4, 0, -1):
        overlay = canvas.copy()
        weight = 0.03 * i * glow
        cv2.putText(overlay, text, (center_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (200, 200, 255), thickness + i * 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, weight, canvas, 1 - weight, 0, canvas)
    # main text
    cv2.putText(canvas, text, (center_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def wrap_text_to_width(text: str, font, scale: float, thickness: int, max_width: int) -> List[str]:
    """
    Wrap text into lines so that each line's pixel width <= max_width.
    Splits on spaces; if a single word > max_width, it will be split character-wise.
    """
    words = text.split()
    if not words:
        return []
    lines = []
    cur = words[0]
    for w in words[1:]:
        test = cur + " " + w
        tw = cv2.getTextSize(test, font, scale, thickness)[0][0]
        if tw <= max_width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    # append last
    lines.append(cur)

    # ensure no single line is too long; if so, break character-wise
    final_lines = []
    for ln in lines:
        width = cv2.getTextSize(ln, font, scale, thickness)[0][0]
        if width <= max_width:
            final_lines.append(ln)
        else:
            # break long word/line into chunks
            cur_s = ""
            for ch in ln:
                test = cur_s + ch
                tw = cv2.getTextSize(test, font, scale, thickness)[0][0]
                if tw <= max_width:
                    cur_s = test
                else:
                    final_lines.append(cur_s)
                    cur_s = ch
            if cur_s:
                final_lines.append(cur_s)
    return final_lines


# -------------------------
# Recognition helpers
# -------------------------
# Load embeddings if present
if FEATURES_CSV.exists():
    try:
        import pandas as pd
        _df = pd.read_csv(str(FEATURES_CSV), index_col=0)
        FEATURE_NAMES = list(_df.index)
        FEATURES_MATRIX = _df.to_numpy(dtype=float)
        print(f"[INFO] Loaded {len(FEATURE_NAMES)} embeddings from {FEATURES_CSV}")
    except Exception as e:
        print("[WARN] Failed to load features:", e)
        FEATURE_NAMES = []
        FEATURES_MATRIX = np.empty((0, 128))
else:
    FEATURE_NAMES = []
    FEATURES_MATRIX = np.empty((0, 128))

# dlib or Haar
if _HAS_DLIB and DLIB_PREDICTOR.exists() and DLIB_FACEREC.exists():
    try:
        _detector = dlib.get_frontal_face_detector()
        _predictor = dlib.shape_predictor(str(DLIB_PREDICTOR))
        _facerec = dlib.face_recognition_model_v1(str(DLIB_FACEREC))
        USING_DLIB = True
        print("[INFO] dlib recognition available.")
    except Exception as e:
        print("[WARN] dlib init error:", e)
        USING_DLIB = False
        _detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        _predictor = None
        _facerec = None
else:
    USING_DLIB = False
    _detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    _predictor = None
    _facerec = None
    print("[INFO] Using Haar cascade fallback for face detection.")


def compute_recognition(frame_rgb: np.ndarray):
    """
    Return a list of recognition dicts: {'rect': r, 'name': name_or_None, 'dist': float}
    If USING_DLIB: r is a dlib.rect, else r is (x,y,w,h).
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
        for (x, y, w, h) in dets:
            results.append({'rect': (x, y, w, h), 'name': None, 'dist': 1.0})
    return results


# -------------------------
# TTS manager (Coqui primary, pyttsx3 fallback)
# -------------------------
class TTSManager:
    """
    Continuously loop through provided messages while enabled.
    Uses Coqui to produce WAV files cached under tts_cache/<sha1>.wav.
    Falls back to pyttsx3 running live (no caching).
    Skips messages equal to 'avinya' (case-insensitive).
    """

    def __init__(self, use_coqui: bool = True):
        self.use_coqui = use_coqui and _COQUI_AVAILABLE
        self.coqui = None
        if self.use_coqui:
            try:
                self.coqui = TTS(model_name=COQUI_MODEL_NAME, progress_bar=False, gpu=False)
                print("[TTS] Coqui initialized")
            except Exception as e:
                print("[TTS] Coqui init failed:", e)
                self.use_coqui = False

        self.use_pyttsx3 = (not self.use_coqui) and _PYTTX3_AVAILABLE
        self.py_engine = None
        if self.use_pyttsx3:
            try:
                self.py_engine = pyttsx3.init()
                print("[TTS] pyttsx3 init OK")
            except Exception as e:
                print("[TTS] pyttsx3 init error:", e)
                self.use_pyttsx3 = False

        self.lock = threading.Lock()
        self.current_texts: List[str] = []   # ordered unique messages to speak
        self.enabled = True                  # speech ON by default as requested
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        # throttle: minimal time between re-speaking the same message batch
        self._last_spoken_batch_ts = 0.0
        self._min_repeat_interval = 4.0  # seconds before repeating same batch

    def update_texts(self, texts: List[str]):
        # filter 'avinya' and blanks, keep order and uniqueness
        filtered = []
        seen = set()
        for t in texts:
            if not t or not t.strip():
                continue
            if t.strip().lower() == "avinya":
                # never speak the AVINYA text
                continue
            if t not in seen:
                seen.add(t)
                filtered.append(t)
        with self.lock:
            self.current_texts = filtered

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _ensure_cached_wav(self, text: str) -> Optional[str]:
        # returns path to wav file or None
        key = sha1_hex(text)
        target = TTS_CACHE_DIR / f"{key}.wav"
        if target.exists():
            return str(target)
        if self.use_coqui and self.coqui:
            try:
                # try API with speed param, fallback if signature mismatch
                try:
                    self.coqui.tts_to_file(text=text, file_path=str(target), speed=COQUI_SPEED)
                except TypeError:
                    self.coqui.tts_to_file(text=text, file_path=str(target))
                return str(target)
            except Exception as e:
                print("[TTS] Coqui generation error:", e)
                return None
        # no caching for pyttsx3
        return None

    def _play_wav(self, path: str):
        cmd = platform_player_command(path)
        if cmd:
            try:
                os.system(cmd)
            except Exception as e:
                print("[TTS] playback failed:", e)
        else:
            # no system player: attempt to fallback to pyttsx3 speaking the text (approx)
            if self.use_pyttsx3 and self.py_engine:
                try:
                    self.py_engine.runAndWait()
                except Exception:
                    pass

    def _worker(self):
        prev_batch = None
        prepared = []
        while not self._stop.is_set():
            with self.lock:
                enabled = self.enabled
                texts = list(self.current_texts)
            if not enabled or not texts:
                prev_batch = None
                time.sleep(0.12)
                continue

            # throttle repetitive immediate regen
            now = time.time()
            if texts != prev_batch or (now - self._last_spoken_batch_ts) >= self._min_repeat_interval:
                # prepare (text, wavpath or None)
                prepared = []
                for t in texts:
                    wav = None
                    if self.use_coqui:
                        wav = self._ensure_cached_wav(t)
                    prepared.append((t, wav))
                prev_batch = list(texts)
                self._last_spoken_batch_ts = 0.0  # will set after playing
            # play sequence once
            for t, wav in prepared:
                if self._stop.is_set():
                    break
                with self.lock:
                    # if texts changed mid-play, break to prepare fresh
                    if self.current_texts != prev_batch:
                        break
                if wav and os.path.exists(wav):
                    self._play_wav(wav)
                else:
                    # fallback to pyttsx3 if available
                    if self.use_pyttsx3 and self.py_engine:
                        try:
                            self.py_engine.say(t)
                            self.py_engine.runAndWait()
                        except Exception as e:
                            print("[TTS] pyttsx3 error:", e)
                    else:
                        # last resort: print
                        print("[TTS] ->", t)
                # small inter-message pause
                for _ in range(8):
                    if self._stop.is_set():
                        break
                    time.sleep(0.06)

            # record last spoken time to avoid immediate repeat
            self._last_spoken_batch_ts = time.time()
            # pause slightly before repeating entire sequence
            for _ in range(30):
                if self._stop.is_set():
                    break
                time.sleep(0.05)


# -------------------------
# Main runtime
# -------------------------
def main():
    # Initialize camera (prefer Picamera2 if present, else error per your environment)
    cap = None
    picam = None
    if _HAS_PICAMERA2 and Picamera2 is not None:
        try:
            picam = Picamera2()
            cfg = picam.create_preview_configuration({"main": {"format": "RGB888", "size": (640, 480)}})
            picam.configure(cfg)
            picam.start()
            print("[CAM] Picamera2 initialized.")
        except Exception as e:
            print("[WARN] Picamera2 init failed:", e)
            picam = None
    else:
        # Try local webcam as fallback (helpful for development on Mac)
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap = None
                print("[WARN] Local webcam not available")
            else:
                print("[CAM] Local webcam initialized.")
        except Exception as e:
            print("[WARN] Webcam init failed:", e)
            cap = None

    if picam is None and cap is None:
        print("[ERROR] No camera available (Picamera2 or local webcam). Exiting.")
        return

    # init windows
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_MAIN, SCREEN_WIDTH, SCREEN_HEIGHT)
    cv2.namedWindow(WINDOW_CAMERA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CAMERA, 800, 600)

    # TTS manager (Coqui primary)
    ttsm = TTSManager(use_coqui=_COQUI_AVAILABLE)

    # keep a short history of last displayed messages to avoid verbal spam
    last_displayed_msgs: List[str] = []
    last_msg_update_ts = 0.0

    # idle requirement small lines
    req_lines = [
        "Please stand in front of the camera",
        "Ensure your face is well-lit",
        "System will greet you automatically"
    ]

    frame_count = 0
    last_fps_time = time.time()
    fps = 0.0

    pulse_phase = 0.0
    idle_phase = 0.0

    try:
        while True:
            # Capture frame either from picam or cv2
            if picam is not None:
                try:
                    frame = picam.capture_array()
                    if frame is None:
                        print("[WARN] Picamera frame None")
                        time.sleep(0.05)
                        continue
                except Exception as e:
                    print("[WARN] Picamera read failed:", e)
                    time.sleep(0.05)
                    continue
                # frame is RGB from picamera
                frame_rgb = frame
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                ret, frame_bgr = cap.read()
                if not ret:
                    print("[WARN] webcam frame fail")
                    time.sleep(0.05)
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - last_fps_time)
                frame_count = 0
                last_fps_time = now

            # recognition
            recs = compute_recognition(frame_rgb)
            face_count = len(recs)
            # prepare messages
            welcome_msgs: List[str] = []
            if face_count == 0:
                # idle: no visible faces -> no speaking of "Avinya" required by user
                welcome_msgs = []
            else:
                for r in recs:
                    if r.get('name'):
                        welcome_msgs.append(f"Welcome {r['name']}")
                    else:
                        welcome_msgs.append("Welcome Participant")

            # deduplicate preserve order
            uniq_msgs = list(dict.fromkeys(welcome_msgs))

            # update TTS manager: only if msgs non-empty (speaking ON by default)
            ttsm.update_texts(uniq_msgs)

            # --- build main UI canvas (no camera preview in this canvas) ---
            canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
            # choose gradient based on idle/active
            if face_count == 0:
                draw_gradient_background(canvas, (20, 30, 60), (60, 40, 80))
            else:
                draw_gradient_background(canvas, (10, 50, 20), (30, 80, 40))

            # pulsing rings behind logo (use a center)
            center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            pulse_phase += 0.03
            draw_pulsing_circles(canvas, center, 140, pulse_phase)

            # logo
            draw_logo_center(canvas, LOGO_IMG)

            # animated AVINYA text (kept subtle per UI)
            idle_phase += 0.04
            glow = 0.6 + 0.4 * math.sin(idle_phase * 1.5)
            # center pos for text (place under logo)
            label_font_scale = 3.0
            label_thickness = 6
            # compute baseline x so text is centered when drawn with putText left coordinate
            txt = "AVINYA"
            tsz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness)[0]
            tx = (SCREEN_WIDTH - tsz[0]) // 2
            ty = (SCREEN_HEIGHT // 2) + LOGO_SIZE // 2
            draw_animated_text_glow(canvas, txt, tx, ty, label_font_scale, (255, 255, 255), label_thickness, glow)

            # display messages under logo, wrapped to fit width if necessary
            # choose a maximum text width (80% of screen)
            max_text_width = int(SCREEN_WIDTH * 0.85)
            msg_y_start = ty + TEXT_BASE_OFFSET
            if uniq_msgs:
                # For each message, wrap and draw multiple lines neatly
                line_gap = 42
                y = msg_y_start
                # cap total vertical usage to avoid running off-screen; if too many lines, shrink font
                base_font_scale = 1.2
                base_thickness = 2
                all_lines = []
                for m in uniq_msgs:
                    wrapped = wrap_text_to_width(m, cv2.FONT_HERSHEY_SIMPLEX, base_font_scale, base_thickness, max_text_width)
                    all_lines.extend(wrapped)
                # if too many lines to fit below logo, reduce font scale iteratively
                max_lines_fit = (SCREEN_HEIGHT - msg_y_start - 80) // line_gap
                font_scale = base_font_scale
                thickness = base_thickness
                while len(all_lines) > max_lines_fit and font_scale > 0.5:
                    font_scale -= 0.1
                    thickness = max(1, int(font_scale * 2))
                    all_lines = []
                    for m in uniq_msgs:
                        all_lines.extend(wrap_text_to_width(m, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness, max_text_width))
                    # recalc
                    max_lines_fit = (SCREEN_HEIGHT - msg_y_start - 80) // line_gap
                # draw each line centered
                for ln in all_lines:
                    tw = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
                    x = (SCREEN_WIDTH - tw) // 2
                    cv2.putText(canvas, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    y += line_gap
            else:
                # idle: show muted requirements under logo
                small_fs = 0.7
                small_th = 1
                base_y = msg_y_start + 10
                for i, ln in enumerate(["Please stand in front of the camera", "Ensure your face is well-lit", "System will greet you automatically"]):
                    tw = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, small_fs, small_th)[0][0]
                    x = (SCREEN_WIDTH - tw) // 2
                    cv2.putText(canvas, ln, (x, base_y + i * 28), cv2.FONT_HERSHEY_SIMPLEX, small_fs, (200, 200, 200), small_th, cv2.LINE_AA)

            # top-left small FPS counter and face count
            overlay_text = f"FPS: {fps:.1f}  Faces: {face_count}"
            cv2.putText(canvas, overlay_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

            # show main window
            cv2.imshow(WINDOW_MAIN, canvas)

            # annotate and show camera window separately (live feed)
            cam_disp = frame_bgr.copy()
            # header
            cv2.rectangle(cam_disp, (0, 0), (cam_disp.shape[1], 34), (230, 230, 230), -1)
            cv2.putText(cam_disp, "AVINYA Camera", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
            # annotate faces
            for r in recs:
                if USING_DLIB:
                    rect = r['rect']
                    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
                else:
                    x, y, w, h = r['rect']
                    x1, y1, x2, y2 = x, y, x + w, y + h
                # clamp
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(cam_disp.shape[1] - 1, x2), min(cam_disp.shape[0] - 1, y2)
                cv2.rectangle(cam_disp, (x1, y1), (x2, y2), (0, 140, 200), 2)
                label = r['name'] if r['name'] else "Participant"
                txt = label if r['name'] else "Participant"
                ts = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(cam_disp, (x1, y1 - 28), (x1 + ts[0] + 10, y1 - 6), (0, 140, 200), -1)
                cv2.putText(cam_disp, txt, (x1 + 6, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # show camera window
            cv2.imshow(WINDOW_CAMERA, cam_disp)

            # handle inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            # optional: space to toggle speech on/off (useful in live testing)
            if key == ord(' '):
                with ttsm.lock:
                    ttsm.enabled = not ttsm.enabled
                print("[UI] Speech enabled:", ttsm.enabled)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        # shutdown
        print("[INFO] Shutting down...")
        try:
            ttsm.stop()
        except Exception:
            pass
        try:
            if picam is not None:
                picam.stop()
        except Exception:
            pass
        try:
            if cap is not None:
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
