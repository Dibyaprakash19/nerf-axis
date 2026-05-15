"""
nerf_scan/config.py — All constants in one place.
Edit only here to tune the pipeline.
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
PI_SCAN_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../pi_scan
PACKAGE_DIR  = os.path.dirname(os.path.abspath(__file__))                   # .../pi_scan/nerf_scan
DATA_DIR     = os.path.join(PACKAGE_DIR, "data")
MODEL_PATH   = os.path.join(PI_SCAN_DIR, "midas_small.tflite")
GLB_PATH     = os.path.join(DATA_DIR, "current_mesh.glb")

os.makedirs(DATA_DIR, exist_ok=True)

# ── Camera ────────────────────────────────────────────────────────────────────
# 256×256 = square = zero resize overhead before MiDaS (model input = 256)
CAM_W        = 256
CAM_H        = 256

# Gesture detection — low-res for speed
GESTURE_W    = 320
GESTURE_H    = 240

# ── TFT ───────────────────────────────────────────────────────────────────────
TFT_W        = 128    # portrait width
TFT_H        = 160    # portrait height
TFT_DC       = 24
TFT_RST      = 25

# ── Stepper (BCM pins: IN1..IN4 on ULN2003) ──────────────────────────────────
STEP_PINS    = [17, 18, 27, 22]
STEPS_90     = 128    # 128 × 8-state sequence ≈ 90°

# ── Mesh ──────────────────────────────────────────────────────────────────────
# step=2 on 256×256 → 128×128 grid = exactly TFT_W × TFT_H vertices
MESH_STEP    = 2      # depth grid subsampling for GLB (quality)
TFT_PTS      = 2000  # max points rendered on TFT (speed)
ROTATION_SPEED = 0.10  # radians per frame
FRAME_DELAY    = 0.01  # seconds between frames (~25 fps ceiling)

# ── Gesture ───────────────────────────────────────────────────────────────────
GESTURE_HOLD_S  = 1.0   # seconds gesture must be held to trigger
# Skin HSV bounds (widened for varying light)
SKIN_LO      = (0, 5, 40)
SKIN_HI      = (40, 255, 255)

# ── Web viewer ────────────────────────────────────────────────────────────────
WEB_PORT       = 8080
LOOP_DISPLAY_S = 15   # seconds to display model before auto-rescan
