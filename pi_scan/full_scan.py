import os
import sys
import time
import math
import signal
import argparse
import subprocess
import numpy as np
import RPi.GPIO as GPIO
import trimesh
from PIL import Image, ImageDraw, ImageOps

# Driver logic (Holy Bible - DO NOT MODIFY st7735_direct_hello.py)
from st7735_direct_hello import DirectST7735, WIDTH, HEIGHT
from scanner import DepthEngine

class ScannerController:
    """
    Coordinates the 3D scanning hardware and software pipeline.
    """
    # ULN2003 Driver Pins
    MOTOR_PINS = [17, 18, 27, 22]
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.output_dir = "data/live_scan"
        self.mesh_path = os.path.join(self.output_dir, "mesh_4view.glb")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "processed"), exist_ok=True)
        
        self.display = DirectST7735(dc_pin=24, rst_pin=25)
        self.engine = DepthEngine(model_path)
        
        self._init_stepper()

    def _init_stepper(self):
        GPIO.setmode(GPIO.BCM)
        for pin in self.MOTOR_PINS:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, False)

    def _rotate_90(self, direction=-1):
        """Rotates the turntable 90 degrees (128 steps). direction -1 is anticlockwise."""
        seq = [
            [1,0,0,0], [1,1,0,0], [0,1,0,0], [0,1,1,0],
            [0,0,1,0], [0,0,1,1], [0,0,0,1], [1,0,0,1]
        ]
        if direction == -1:
            seq = list(reversed(seq))
            
        steps_for_90_deg = 128
        for _ in range(steps_for_90_deg):
            for state in seq:
                for i, pin in enumerate(self.MOTOR_PINS):
                    GPIO.output(pin, state[i])
                time.sleep(0.001)
        self._release_stepper()

    def _release_stepper(self):
        for pin in self.MOTOR_PINS:
            GPIO.output(pin, False)

    def _capture_view(self, view_idx: int):
        """Captures a high-quality still image using libcamera."""
        path = os.path.join(self.output_dir, "processed", f"view_{view_idx:02d}.jpg")
        # Still images only as requested
        cmd = ["libcamera-still", "-n", "--immediate", "--width", "1024", "--height", "1024", "-o", path]
        subprocess.run(cmd, check=True)
        
        # Preprocess: enhance contrast
        img = Image.open(path)
        img = ImageOps.autocontrast(img)
        img.save(path)
        return path

    def run_scan_sequence(self):
        """Orchestrates 4 captures and turntable rotations. Returns to 0 at the end."""
        images = []
        for i in range(4):
            # Capture first, then rotate (except after the last capture which also rotates back)
            self.display_message(f"CAPTURING {i+1}/4")
            images.append(self._capture_view(i))
            
            # Rotate 90 degrees anticlockwise
            self.display_message(f"ROTATING 90°...")
            self._rotate_90(direction=-1)
            time.sleep(0.5) # Settle time
            
        self.display_message("BUILDING MESH...")
        # Pass remove_bg=True to scanner.py DepthEngine
        self.engine.generate_fused_mesh(images, self.mesh_path, step=4, remove_bg=True)
        return self.mesh_path

    def display_message(self, text: str):
        img = Image.new("RGB", (WIDTH, HEIGHT), "black")
        draw = ImageDraw.Draw(img)
        # Use a simpler way to center text or just fixed position
        draw.text((10, 75), text, fill="white")
        self.display.display(img)

    def visualize_result(self):
        """Renders the rotating 3D object on the TFT screen without axes."""
        mesh = trimesh.load(self.mesh_path, process=False)
        verts = np.asarray(mesh.vertices)
        verts -= verts.mean(axis=0)
        
        # Scale to fit TFT nicely
        max_dim = np.max(np.abs(verts))
        if max_dim > 0:
            verts *= (60 / max_dim)
        
        colors = np.asarray(mesh.visual.vertex_colors)[:, :3] if hasattr(mesh.visual, 'vertex_colors') else None
        
        # Subsample for smooth rotation on Pi
        if len(verts) > 3000:
            idx = np.random.choice(len(verts), 3000, replace=False)
            verts, colors = verts[idx], (colors[idx] if colors is not None else None)

        angle = 0.0
        while True:
            # Rotation around Y axis
            c, s = math.cos(angle), math.sin(angle)
            # Tilt for better perspective
            tilt_angle = -0.4
            cx, sx = math.cos(tilt_angle), math.sin(tilt_angle)
            
            rx = verts[:, 0] * c + verts[:, 2] * s
            rz_tmp = -verts[:, 0] * s + verts[:, 2] * c
            ry = verts[:, 1] * cx - rz_tmp * sx
            rz = verts[:, 1] * sx + rz_tmp * cx
            
            # Simple perspective projection
            f = 200 / (200 + rz)
            px = WIDTH / 2 + rx * f
            py = HEIGHT / 2 - ry * f
            
            img = Image.new("RGB", (WIDTH, HEIGHT), "black")
            draw = ImageDraw.Draw(img)
            
            # Painter's algorithm: sort by depth
            for i in np.argsort(rz)[::-1]:
                x, y, z = px[i], py[i], rz[i]
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    # Basic shading based on Z depth
                    v = np.clip((z + 60) / 120, 0.4, 1.0)
                    if colors is not None:
                        fill = tuple(int(ch * v) for ch in colors[i])
                    else:
                        fill = (int(100*v), int(150*v), int(255*v))
                    
                    # Draw point as small rectangle
                    draw.point((x, y), fill=fill)
            
            self.display.display(img)
            angle += 0.15
            time.sleep(0.01)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-capture", action="store_true")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "midas_small.tflite")
    
    ctrl = ScannerController(model_path)
    # Ensure GPIO is cleaned up on exit
    def cleanup_and_exit(s, f):
        GPIO.cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    if not args.no_capture:
        ctrl.run_scan_sequence()
    
    print("Starting visualization. Press Ctrl+C to stop.")
    ctrl.visualize_result()

if __name__ == "__main__":
    main()
