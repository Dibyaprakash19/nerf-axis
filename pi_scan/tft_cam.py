import time
import st7735 as ST7735
from PIL import Image, ImageDraw
import sys
import numpy as np

# ── Configuration ───────────────────────────────────────────────────────────
# TFT Pin Mapping (as provided by user)
# PORT=0, CS=0 (CE0), DC=24, RST=25
# SDA=MOSI (GPIO10), SCK=SCLK (GPIO11)
# LED=3.3V (Hardware controlled)

TFT_WIDTH = 160
TFT_HEIGHT = 128

# ── Initialize Display ──────────────────────────────────────────────────────
print("Initializing ST7735 Display...")
disp = ST7735.ST7735(
    port=0,
    cs=0,
    dc=24,
    rst=25,
    rotation=90,      # 90 or 270 for landscape
    width=TFT_WIDTH,
    height=TFT_HEIGHT
)

disp.begin()

# Initial Splash Screen
img = Image.new("RGB", (TFT_WIDTH, TFT_HEIGHT), "black")
draw = ImageDraw.Draw(img)
draw.text((20, 20), "3D Scanner", fill="white")
draw.text((20, 60), "Camera Init...", fill="yellow")
disp.display(img)

# ── Initialize Camera ──────────────────────────────────────────────────────
print("Initializing Libcamera...")

try:
    from picamera2 import Picamera2
    cam = Picamera2()
    # Configure for a small resolution to keep preview snappy
    config = cam.create_preview_configuration(main={"size": (640, 480)})
    cam.configure(config)
    cam.start()
    HAS_CAM = True
    print("Using Picamera2 (Libcamera)")
except ImportError:
    try:
        import cv2
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            raise Exception("OpenCV could not open camera")
        HAS_CAM = "cv2"
        print("Using OpenCV (Fallback)")
    except Exception as e:
        print(f"Error: Could not initialize camera. {e}")
        draw.text((20, 90), "Camera Error!", fill="red")
        disp.display(img)
        sys.exit(1)

# ── Main Loop ──────────────────────────────────────────────────────────────
print("Starting Camera Preview on TFT. Press Ctrl+C to stop.")
try:
    while True:
        if HAS_CAM == True:
            # Picamera2 capture
            frame = cam.capture_array()
            # frame is RGB numpy array
            pil_img = Image.fromarray(frame)
        elif HAS_CAM == "cv2":
            # OpenCV capture
            ret, frame = cam.read()
            if not ret:
                continue
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

        # Process image for TFT
        # 1. Aspect ratio crop/resize
        # We want to fill the 160x128 screen
        pil_img = pil_img.resize((TFT_WIDTH, TFT_HEIGHT), Image.Resampling.LANCZOS)
        
        # 2. Update display
        disp.display(pil_img)
        
        # Slight delay to yield
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping preview...")
finally:
    if HAS_CAM == True:
        cam.stop()
    elif HAS_CAM == "cv2":
        cam.release()
    
    # Clear screen on exit
    img = Image.new("RGB", (TFT_WIDTH, TFT_HEIGHT), "black")
    disp.display(img)
    print("Done.")
