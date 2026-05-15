"""
full_capture_scan.py — Complete 4-view 3D scanning pipeline.

Dependencies (all lightweight / piwheels):
    tflite-runtime, numpy, opencv-python-headless, Pillow, RPi.GPIO
    pygltflib  (optional — GLB export)

Zero trimesh. Zero scikit-image. Zero rembg. Zero onnxruntime.
"""

import os
import sys
import time
import signal
import subprocess

import numpy as np
from PIL import Image, ImageDraw

import RPi.GPIO as GPIO

# ── Holy Bible TFT driver (DirectST7735 only) ─────────────────
from st7735_direct_hello import DirectST7735, WIDTH, HEIGHT

import cv2
# ── Lightweight scanner (trimesh-free, rembg-free) ────────────
from lightweight_scanner import LightScanner, fuse_views, save_glb, remove_background_opencv

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "midas_small.tflite")
DATA_DIR   = os.path.join(BASE_DIR, "data", "capture_scan")
os.makedirs(DATA_DIR, exist_ok=True)

# Stepper — BCM pins (IN1..IN4 on ULN2003)
STEP_PINS      = [17, 18, 27, 22]
STEPS_FOR_90   = 128       # 128 × 8-state sequence ≈ 1 024 half-steps ≈ 90°

# Renderer
MESH_STEP      = 24        # subsample depth grid (larger = fewer triangles = faster)
FOCAL          = 200       # perspective projection focal length (pixels)
ROTATION_SPEED = 0.12      # radians per frame
FRAME_DELAY    = 0.01      # seconds between frames

# ══════════════════════════════════════════════════════════════
#  TFT HELPERS
# ══════════════════════════════════════════════════════════════
def make_display():
    return DirectST7735(dc_pin=24, rst_pin=25)

def show_text(disp, line1: str, line2: str = "", color="white"):
    img  = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(img)
    draw.text((4, HEIGHT // 2 - 14), line1, fill=color)
    if line2:
        draw.text((4, HEIGHT // 2 + 4), line2, fill="gray")
    disp.display(img)

# ══════════════════════════════════════════════════════════════
#  STEPPER
# ══════════════════════════════════════════════════════════════
_STEP_SEQ = [
    [1,0,0,0], [1,1,0,0], [0,1,0,0], [0,1,1,0],
    [0,0,1,0], [0,0,1,1], [0,0,0,1], [1,0,0,1],
]

def _init_gpio():
    GPIO.setmode(GPIO.BCM)
    for p in STEP_PINS:
        GPIO.setup(p, GPIO.OUT)
        GPIO.output(p, False)

def _release_gpio():
    for p in STEP_PINS:
        GPIO.output(p, False)

def rotate_90(direction: int = -1):
    """Rotate turntable 90°. direction=-1 → anticlockwise."""
    seq = _STEP_SEQ if direction == 1 else list(reversed(_STEP_SEQ))
    for _ in range(STEPS_FOR_90):
        for state in seq:
            for i, pin in enumerate(STEP_PINS):
                GPIO.output(pin, state[i])
            time.sleep(0.001)
    _release_gpio()

# ══════════════════════════════════════════════════════════════
#  CAPTURE
# ══════════════════════════════════════════════════════════════
def capture_image(path: str):
    cmd = [
        "libcamera-still", "-n", "--immediate",
        "--width", "1024", "--height", "1024", "-o", path
    ]
    subprocess.run(cmd, check=True)

def remove_background(img_path: str) -> np.ndarray:
    """
    Load image from disk and remove background using pure OpenCV GrabCut.
    Returns an RGB numpy array with background zeroed out.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {img_path}")
    return remove_background_opencv(img_bgr)

# ══════════════════════════════════════════════════════════════
#  3-D RENDERER  (pure NumPy + Pillow — no trimesh)
# ══════════════════════════════════════════════════════════════
def project(verts: np.ndarray, focal: float = FOCAL):
    """Perspective-project (N,3) → (N,2) pixel coords."""
    z = verts[:, 2] + 150          # push object in front of camera
    z = np.where(z < 1, 1, z)      # avoid divide-by-zero
    f = focal / z
    px = (WIDTH  / 2 + verts[:, 0] * f).astype(np.int32)
    py = (HEIGHT / 2 - verts[:, 1] * f).astype(np.int32)
    return np.stack([px, py], axis=1)

def rotate_y(verts: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    out  = verts.copy()
    out[:, 0] =  verts[:, 0] * c + verts[:, 2] * s
    out[:, 2] = -verts[:, 0] * s + verts[:, 2] * c
    return out

def render_frame(disp, verts: np.ndarray, faces: np.ndarray,
                 colors: np.ndarray, angle: float):
    rot   = rotate_y(verts, angle)
    proj  = project(rot)

    img  = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(img)

    if len(faces) == 0:
        # Fallback: point cloud
        for i in range(len(verts)):
            x, y = proj[i]
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                c = tuple(int(ch) for ch in colors[i])
                draw.point((x, y), fill=c)
    else:
        # Painter's algorithm — sort triangles back-to-front
        z_avg    = rot[faces, 2].mean(axis=1)
        order    = np.argsort(z_avg)[::-1]

        for idx in order:
            tri  = faces[idx]
            pts  = [tuple(proj[v]) for v in tri]
            # Average vertex color for this face
            fc   = colors[tri].mean(axis=0)
            # Depth shading
            z    = float(z_avg[idx])
            shade = np.clip((z + 50) / 100.0, 0.3, 1.0)
            fill  = tuple(int(ch * shade) for ch in fc)
            draw.polygon(pts, fill=fill, outline=None)

    disp.display(img)

def display_rotating_3d(disp, verts: np.ndarray, faces: np.ndarray,
                        colors: np.ndarray):
    """Spin forever until Ctrl-C."""
    angle = 0.0
    while True:
        render_frame(disp, verts, faces, colors, angle)
        angle += ROTATION_SPEED
        time.sleep(FRAME_DELAY)

# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
def run_scan():
    disp    = make_display()
    scanner = LightScanner(MODEL_PATH)

    show_text(disp, "4-View 3D Scan", "starting...", color="cyan")
    time.sleep(1)

    _init_gpio()
    views = []

    for i in range(4):
        show_text(disp, f"Capture {i+1}/4", "shoot!", color="white")

        img_path = os.path.join(DATA_DIR, f"view_{i}.jpg")
        capture_image(img_path)

        show_text(disp, f"View {i+1}/4", "removing BG...", color="cyan")
        clean_rgb = remove_background(img_path)   # returns RGB ndarray

        show_text(disp, f"View {i+1}/4", "depth map...", color="cyan")
        depth, rgb = scanner.get_depth(clean_rgb)  # accepts ndarray directly

        # Build per-view mesh (trimesh-free)
        v, f, c = scanner.depth_to_mesh(
            depth, rgb,
            view_angle = i * 90,
            step       = MESH_STEP
        )
        views.append((v, f, c))
        print(f"[view {i}] verts={len(v)} faces={len(f)}")

        if i < 3:
            show_text(disp, f"Rotating 90°...", f"view {i+2} next", color="yellow")
            rotate_90(direction=-1)
            time.sleep(0.5)

    # ── Fuse ────────────────────────────────────────────────
    show_text(disp, "Fusing 4 views...", color="green")
    final_v, final_f, final_c = fuse_views(views)
    print(f"[fused] verts={len(final_v)} faces={len(final_f)}")

    # ── Save GLB (optional) ──────────────────────────────────
    glb_path = os.path.join(DATA_DIR, "final_mesh.glb")
    save_glb(final_v, final_f, glb_path)

    show_text(disp, "Scan Complete!", "rotating...", color="green")
    time.sleep(1.5)

    # ── Visualise ────────────────────────────────────────────
    display_rotating_3d(disp, final_v, final_f, final_c)


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
def _cleanup(sig=None, frame=None):
    GPIO.cleanup()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT,  _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    try:
        run_scan()
    except Exception as e:
        print(f"Fatal: {e}")
    finally:
        GPIO.cleanup()
