import sys
import os
import cv2
import numpy as np

# Add the current directory to sys.path so we can import nerf_scan
sys.path.append(os.getcwd())

from nerf_scan.camera import CameraManager
from nerf_scan.scanner_lite import remove_bg

def main():
    print("Initializing camera...")
    cam = CameraManager()
    try:
        cam.start()
        print("Capturing image...")
        # Capture at a reasonable resolution for viewing
        img_bgr = cam.capture_still_bgr(1024, 768)
        
        print("Saving original image...")
        cv2.imwrite("original.jpg", img_bgr)
        
        print("Removing background...")
        # remove_bg returns RGB
        img_rgb_no_bg = remove_bg(img_bgr)
        
        # Convert back to BGR for saving with OpenCV
        img_bgr_no_bg = cv2.cvtColor(img_rgb_no_bg, cv2.COLOR_RGB2BGR)
        
        print("Saving background removed image...")
        cv2.imwrite("no_bg.jpg", img_bgr_no_bg)
        
        print("Done! Files saved as original.jpg and no_bg.jpg")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cam.stop()

if __name__ == "__main__":
    main()
