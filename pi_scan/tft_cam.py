import time
import signal
import sys
from PIL import Image, ImageDraw
import numpy as np

# Import Holy Bible TFT driver
try:
    from st7735_direct_hello import DirectST7735, WIDTH, HEIGHT
except ImportError:
    print("Error: Could not import DirectST7735 from st7735_direct_hello.py")
    sys.exit(1)

# ── Initialize Display ──────────────────────────────────────────────────────
print("Initializing ST7735 Direct Display...")
try:
    disp = DirectST7735(dc_pin=24, rst_pin=25)
except Exception as e:
    print(f"Error initializing display: {e}")
    sys.exit(1)

# Initial Splash Screen
img = Image.new("RGB", (WIDTH, HEIGHT), "black")
draw = ImageDraw.Draw(img)
draw.rectangle((0, 0, WIDTH - 1, HEIGHT - 1), outline="cyan", width=1)
draw.text((10, 40), "NeRF-Axis", fill="cyan")
draw.text((10, 60), "Camera Preview", fill="white")
draw.text((10, 95), "Initializing...", fill="yellow")
disp.display(img)

# ── Initialize Camera ──────────────────────────────────────────────────────
print("Initializing Picamera2...")
try:
    from picamera2 import Picamera2
    cam = Picamera2()
    # Configure for a small resolution to keep preview snappy (640x480 is standard)
    config = cam.create_preview_configuration(main={"size": (640, 480)})
    cam.configure(config)
    cam.start()
    print("Picamera2 (Libcamera) started successfully.")
except Exception as e:
    print(f"Error: Could not initialize Picamera2. {e}")
    draw.text((10, 120), "Camera Error!", fill="red")
    disp.display(img)
    sys.exit(1)

# ── Cleanup Handler ────────────────────────────────────────────────────────
def cleanup(sig=None, frame_sig=None):
    print("\nStopping preview...")
    try:
        cam.stop()
        cam.close()
    except Exception:
        pass
    
    # Clear screen on exit
    try:
        black_img = Image.new("RGB", (WIDTH, HEIGHT), "black")
        disp.display(black_img)
        disp.close()
    except Exception:
        pass
    print("Done.")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# ── Main Loop ──────────────────────────────────────────────────────────────
print("Starting Camera Preview on TFT. Press Ctrl+C to stop.")

frame_count = 0
start_time = time.time()

try:
    while True:
        # Capture frame as numpy array (RGB)
        frame = cam.capture_array()
        
        # Convert to PIL Image
        pil_img = Image.fromarray(frame)
        
        # Crop the center to preserve aspect ratio of 128:160 (portrait)
        # Width=640, Height=480. 
        # Target aspect ratio is 128 / 160 = 0.8
        # Target width = 480 * 0.8 = 384
        left = (640 - 384) // 2
        right = left + 384
        box = (left, 0, right, 480)
        cropped_img = pil_img.crop(box)
        
        # Resize to TFT resolution (128x160)
        # We use Resampling.BILINEAR for snappy preview
        resized_img = cropped_img.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
        
        # Update display
        disp.display(resized_img)
        
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = 30.0 / elapsed
            print(f"[Preview] Active | {fps:.1f} FPS", flush=True)
            start_time = time.time()
            
except KeyboardInterrupt:
    cleanup()
except Exception as e:
    print(f"Exception during preview: {e}")
    cleanup()
