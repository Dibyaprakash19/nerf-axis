import sys
import os
import time
import shutil
import subprocess

# get project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
pi_scan_dir = os.path.join(project_root, 'pi_scan')
sys.path.insert(0, pi_scan_dir)

from nerf_scan.camera import CameraManager
from nerf_scan.stepper_ctrl import setup as gpio_setup, cleanup as gpio_cleanup, _SEQ, STEP_PINS, STEPS_90
import RPi.GPIO as GPIO
from nerf_scan.tft_ui import TFTDisplay

NUM_VIEWS = 36
CAM_W_HIRES = 1920
CAM_H_HIRES = 1080

def rotate_angle(degrees, ccw=True):
    steps = int(STEPS_90 * (degrees / 90.0))
    seq = list(reversed(_SEQ)) if ccw else _SEQ
    
    for _ in range(steps):
        for state in seq:
            for i, pin in enumerate(STEP_PINS):
                GPIO.output(pin, state[i])
            time.sleep(0.001)
            
    for p in STEP_PINS:
        GPIO.output(p, False)

def capture_images(cam, disp):
    img_dir = "scan/images"
    os.makedirs(img_dir, exist_ok=True)
    
    # wipe old run data
    for f in os.listdir(img_dir):
        os.remove(os.path.join(img_dir, f))
        
    import cv2
    
    # Setup CLAHE for preprocessing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    for i in range(NUM_VIEWS):
        disp.text(f"Capture {i+1}/{NUM_VIEWS}", "COLMAP Pipeline", color="white")
        frame = cam.capture_still_bgr(CAM_W_HIRES, CAM_H_HIRES)
        
        # Preprocessing: Enhance contrast in LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        frame_preprocessed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        cv2.imwrite(f"{img_dir}/{i:03d}.jpg", frame_preprocessed)
        
        disp.text("Rotating...", "10 degrees", color="yellow")
        rotate_angle(10, ccw=True)
        time.sleep(0.4)

def run_colmap_pipeline(disp):
    os.makedirs("scan/sparse", exist_ok=True)
    os.makedirs("scan/dense", exist_ok=True)
    db_path = "scan/database.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # 4GB Pi needs low memory settings!
    disp.text("COLMAP", "Extracting...", color="cyan")
    subprocess.run([
        "colmap", "feature_extractor", 
        "--database_path", db_path, 
        "--image_path", "scan/images",
        "--SiftExtraction.max_image_size", "1000",
        "--SiftExtraction.max_num_features", "2048"
    ], check=False)
    
    disp.text("COLMAP", "Matching...", color="cyan")
    subprocess.run([
        "colmap", "exhaustive_matcher", 
        "--database_path", db_path,
        "--SiftMatching.max_num_matches", "8192"
    ], check=False)
    
    disp.text("COLMAP", "Mapping...", color="cyan")
    subprocess.run([
        "colmap", "mapper", 
        "--database_path", db_path, 
        "--image_path", "scan/images", 
        "--output_path", "scan/sparse"
    ], check=False)
    
    disp.text("COLMAP", "Undistort...", color="cyan")
    subprocess.run([
        "colmap", "image_undistorter", 
        "--image_path", "scan/images", 
        "--input_path", "scan/sparse/0", 
        "--output_path", "scan/dense", 
        "--output_type", "COLMAP"
    ], check=False)
    
    disp.text("COLMAP", "Stereo...", color="yellow")
    subprocess.run([
        "colmap", "patch_match_stereo", 
        "--workspace_path", "scan/dense",
        "--PatchMatchStereo.max_image_size", "1000"
    ], check=False)
    
    disp.text("COLMAP", "Fusion...", color="cyan")
    subprocess.run([
        "colmap", "stereo_fusion", 
        "--workspace_path", "scan/dense", 
        "--output_path", "scan/dense/fused.ply"
    ], check=False)
    
    disp.text("COLMAP", "Meshing...", color="cyan")
    subprocess.run([
        "colmap", "poisson_mesher", 
        "--input_path", "scan/dense/fused.ply", 
        "--output_path", "scan/mesh.ply"
    ], check=False)

def main():
    gpio_setup()
    disp = TFTDisplay()
    cam = CameraManager()
    
    try:
        cam.start()
        capture_images(cam, disp)
        cam.stop()
        
        run_colmap_pipeline(disp)
        
        # ditch the images, only keep mesh and 2 samples
        import glob
        for f in glob.glob("scan/images/*.jpg"):
            if not f.endswith("000.jpg") and not f.endswith("018.jpg"):
                os.remove(f)
                
        if os.path.exists("scan/dense/stereo"):
            shutil.rmtree("scan/dense/stereo", ignore_errors=True)
            
        disp.text("Done!", "Check scan dir", color="green")
        
    finally:
        cam.stop()
        gpio_cleanup()
        
if __name__ == "__main__":
    main()
