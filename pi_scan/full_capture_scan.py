"""
full_capture_scan.py - Automated 4-view 3D scanning pipeline.

Captures still images using a camera, rotates a turntable using a stepper motor,
estimates depth maps with a MiDaS model, segments the background using contours,
fuses views into a single 3D mesh, and displays the rotating 3D result on a TFT screen.
"""

import os
import sys
import time
import signal
import subprocess
import numpy as np
from PIL import Image, ImageDraw
import RPi.GPIO as GPIO
import cv2

from st7735_direct_hello import DirectST7735, WIDTH, HEIGHT
from lightweight_scanner import MiDaSScanner, fuse_views, save_glb

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "midas_small.tflite")
DATA_DIR = os.path.join(BASE_DIR, "data", "capture_scan")
os.makedirs(DATA_DIR, exist_ok=True)

# Stepper turntable control pins (BCM numbers for ULN2003 driver)
MOTOR_PINS = [17, 18, 27, 22]
STEPS_PER_90_DEG = 128

# Rendering settings for TFT screen visualization
MESH_DOWNSAMPLE_STEP = 24
FOCAL_LENGTH = 200
ROTATION_SPEED = 0.12
FRAME_DELAY = 0.01

# Stepper control sequences (half-stepping)
STEP_SEQUENCE = [
    [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0],
    [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 1],
]


def init_tft_display() -> DirectST7735:
    """Initialize the ST7735 TFT display driver."""
    return DirectST7735(dc_pin=24, rst_pin=25)


def display_status(display: DirectST7735, title: str, subtitle: str = "", text_color: str = "white"):
    """Render a text status block on the TFT screen and print to console."""
    print(f"[Scanner Status] {title} - {subtitle}")
    image = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(image)
    
    draw.text((4, HEIGHT // 2 - 14), title, fill=text_color)
    if subtitle:
        draw.text((4, HEIGHT // 2 + 4), subtitle, fill="gray")
        
    display.display(image)


def display_preview(display: DirectST7735, image_path: str, caption: str = ""):
    """Show a resized thumbnail preview of the captured view on the TFT."""
    try:
        image = Image.open(image_path)
        image = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        if caption:
            draw = ImageDraw.Draw(image)
            draw.text((4, 4), caption, fill="white", stroke_width=1, stroke_fill="black")
        display.display(image)
    except Exception as e:
        print(f"Warning: Failed to render TFT preview: {e}")


def init_gpio():
    """Configure Raspberry Pi GPIO pins for stepper motor control."""
    GPIO.setmode(GPIO.BCM)
    for pin in MOTOR_PINS:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, False)


def release_gpio():
    """De-energize motor coils to save power and prevent overheating."""
    for pin in MOTOR_PINS:
        GPIO.output(pin, False)


def rotate_turntable_90(direction: int = -1):
    """Rotate the turntable 90 degrees. direction=-1 for CCW, 1 for CW."""
    sequence = STEP_SEQUENCE if direction == 1 else list(reversed(STEP_SEQUENCE))
    for _ in range(STEPS_PER_90_DEG):
        for state in sequence:
            for i, pin in enumerate(MOTOR_PINS):
                GPIO.output(pin, state[i])
            time.sleep(0.001)
    release_gpio()


def capture_still_image(output_path: str):
    """Capture a high-resolution square photo using libcamera."""
    cmd = [
        "libcamera-still", "-n", "--immediate",
        "--width", "1024", "--height", "1024", "-o", output_path
    ]
    subprocess.run(cmd, check=True)


def project_vertices(vertices: np.ndarray, focal: float = FOCAL_LENGTH) -> np.ndarray:
    """Project 3D vertices using simple perspective projection to 2D pixel space."""
    z_coords = vertices[:, 2] + 150
    z_coords = np.where(z_coords < 1, 1, z_coords)
    factor = focal / z_coords
    
    px = (WIDTH / 2 + vertices[:, 0] * factor).astype(np.int32)
    py = (HEIGHT / 2 - vertices[:, 1] * factor).astype(np.int32)
    return np.stack([px, py], axis=1)


def rotate_y_axis(vertices: np.ndarray, angle: float) -> np.ndarray:
    """Rotate 3D coordinates around the Y (vertical) axis by a given angle in radians."""
    cos_val, sin_val = np.cos(angle), np.sin(angle)
    rotated = vertices.copy()
    rotated[:, 0] = vertices[:, 0] * cos_val + vertices[:, 2] * sin_val
    rotated[:, 2] = -vertices[:, 0] * sin_val + vertices[:, 2] * cos_val
    return rotated


def render_mesh_frame(display: DirectST7735, vertices: np.ndarray, faces: np.ndarray,
                      colors: np.ndarray, angle: float):
    """Rotate, project, and render the 3D mesh triangles using the painter's algorithm."""
    rotated = rotate_y_axis(vertices, angle)
    projected = project_vertices(rotated)

    image = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(image)

    if len(faces) == 0:
        # Fallback: point cloud rendering if no faces are available
        for i in range(len(vertices)):
            x, y = projected[i]
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                pixel_color = tuple(int(ch) for ch in colors[i])
                draw.point((x, y), fill=pixel_color)
    else:
        # Painter's Algorithm: Sort faces from back to front (descending Z depth)
        average_depths = rotated[faces, 2].mean(axis=1)
        draw_order = np.argsort(average_depths)[::-1]

        for idx in draw_order:
            triangle = faces[idx]
            points = [tuple(projected[vertex_idx]) for vertex_idx in triangle]
            
            # Shading factor based on depth to create a sense of distance
            face_color = colors[triangle].mean(axis=0)
            depth_val = float(average_depths[idx])
            shading = np.clip((depth_val + 50) / 100.0, 0.3, 1.0)
            fill_color = tuple(int(channel * shading) for channel in face_color)
            
            draw.polygon(points, fill=fill_color, outline=None)

    display.display(image)


def render_3d_loop(display: DirectST7735, vertices: np.ndarray, faces: np.ndarray,
                   colors: np.ndarray):
    """Infinite visualization loop spinning the reconstructed 3D model on the TFT."""
    # Decimate faces if the mesh is too dense for fast rendering on the Raspberry Pi
    max_renderable_faces = 2500
    if len(faces) > max_renderable_faces:
        print(f"Decimating display faces from {len(faces)} to {max_renderable_faces} for smooth FPS.")
        decimation_indices = np.linspace(0, len(faces) - 1, max_renderable_faces, dtype=int)
        faces = faces[decimation_indices]

    angle = 0.0
    frame_count = 0
    start_time = time.time()
    
    while True:
        render_mesh_frame(display, vertices, faces, colors, angle)
        angle += ROTATION_SPEED
        frame_count += 1
        
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        if frame_count % 30 == 0:
            print(f"[3D Render] Active spinning frame {frame_count} - {fps:.2f} FPS")
        start_time = time.time()
            
        time.sleep(FRAME_DELAY)


def run_scan_pipeline():
    """Main automated scanner operation orchestration."""
    display = init_tft_display()
    scanner = MiDaSScanner(MODEL_PATH)

    display_status(display, "3D Scan System", "initializing...", text_color="cyan")
    time.sleep(1)

    init_gpio()
    captured_views = []

    for i in range(4):
        display_status(display, f"Capture {i+1}/4", "shooting image...", text_color="white")

        image_path = os.path.join(DATA_DIR, f"view_{i}.jpg")
        capture_still_image(image_path)

        display_status(display, f"View {i+1}/4", "removing background...", text_color="cyan")
        clean_path = os.path.join(DATA_DIR, f"view_{i}_clean.png")
        scanner.remove_background(image_path, clean_path)

        display_status(display, f"View {i+1}/4", "extracting depth...", text_color="cyan")
        depth, rgb = scanner.process_image(clean_path)

        # Build 3D mesh representation for this view
        step = MESH_DOWNSAMPLE_STEP
        depth_subsampled = depth[::step, ::step]
        rgb_subsampled = rgb[::step, ::step]
        
        view_vertices, view_colors = scanner.depth_to_vertices(
            depth_subsampled, rgb_subsampled,
            angle_degrees=i * 90
        )
        
        h_sub, w_sub = depth_subsampled.shape
        view_faces = scanner.create_faces(h_sub, w_sub)
        
        # Rescale normalized color channel outputs (0.0-1.0) back to uint8 (0-255)
        view_colors = (view_colors * 255).astype(np.uint8)

        captured_views.append((view_vertices, view_faces, view_colors))
        print(f"[View {i+1}] Processed {len(view_vertices)} vertices, {len(view_faces)} faces.")
        
        display_preview(display, clean_path, caption=f"View {i+1} Ready")
        time.sleep(1)

        if i < 3:
            display_status(display, "Rotating 90°...", f"preparing view {i+2}", text_color="yellow")
            rotate_turntable_90(direction=-1)
            time.sleep(0.5)

    display_status(display, "Fusing Views", "building 3D mesh...", text_color="green")
    fused_vertices, fused_faces, fused_colors = fuse_views(captured_views)
    print(f"[Fused Mesh] Total vertices: {len(fused_vertices)}, total faces: {len(fused_faces)}")

    glb_output_path = os.path.join(DATA_DIR, "final_mesh.glb")
    save_glb(fused_vertices, fused_faces, glb_output_path)

    display_status(display, "Scan Complete!", "rendering 3D loop...", text_color="green")
    time.sleep(1.5)

    render_3d_loop(display, fused_vertices, fused_faces, fused_colors)


def shutdown_handler(signal_received=None, frame_received=None):
    """Safely release hardware pins and exit execution."""
    GPIO.cleanup()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    try:
        run_scan_pipeline()
    except Exception as fatal_error:
        print(f"Fatal scanning pipeline crash: {fatal_error}")
    finally:
        GPIO.cleanup()
