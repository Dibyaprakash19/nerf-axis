import time
import os
import cv2
import sys
from PIL import Image, ImageDraw
from nerf_scan.camera import CameraManager
from nerf_scan.tft_ui import TFTDisplay
from nerf_scan.config import TFT_W, TFT_H

CAM_W_HIRES = 1920
CAM_H_HIRES = 1080
NUM_IMAGES = 16

def show_preview_and_countdown(cam, disp, count, total):
    for tick in [3, 2, 1]:
        frame = cam.frame()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Crop to portrait for TFT
        h, w = img.size[1], img.size[0]
        scale = max(TFT_W / w, TFT_H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h))
        
        # Center crop
        left = (new_w - TFT_W) // 2
        top = (new_h - TFT_H) // 2
        img = img.crop((left, top, left + TFT_W, top + TFT_H))
        
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), f"IMG: {count}/{total}", fill="cyan")
        draw.text((50, 60), f"{tick}...", fill="red")
        
        disp._fast_display(img)
        time.sleep(1.0)
        
def capture_calib_images():
    disp = TFTDisplay()
    cam = CameraManager()
    
    os.makedirs("calib", exist_ok=True)
    
    try:
        cam.start()
        disp.text("Calibration", "Get Ready!", color="green")
        time.sleep(2)
        
        for i in range(1, NUM_IMAGES + 1):
            show_preview_and_countdown(cam, disp, i, NUM_IMAGES)
            
            disp.text("Capturing...", "", color="yellow")
            frame = cam.capture_still_bgr(CAM_W_HIRES, CAM_H_HIRES)
            
            filename = f"calib/calib_{i:02d}.jpg"
            cv2.imwrite(filename, frame)
            
            disp.text(f"Saved {i}/{NUM_IMAGES}", "Move checkerboard", color="cyan")
            time.sleep(2.5)
            
        disp.text("Done!", "Running calib script", color="green")
        time.sleep(2)
    finally:
        cam.stop()
        disp.clear()

if __name__ == "__main__":
    capture_calib_images()
