"""
nerf_scan/render_video.py - Render the current mesh to an MP4 video.
"""
import os, sys, cv2, math
import numpy as np
from PIL import Image

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nerf_scan.config import GLB_PATH, DATA_DIR

def render_to_mp4(output_path="scan_video.mp4"):
    # Load verts/colors (we'll look for the most recent npz if GLB is harder to parse)
    # Actually, let's just use the latest data from DATA_DIR
    npz_path = os.path.join(DATA_DIR, "current_mesh.npz")
    if not os.path.exists(npz_path):
        print("No scan data found to render.")
        return

    data = np.load(npz_path)
    v, c = data['verts'], data['colors']

    # Rendering constants
    W, H = 640, 640
    FOCAL = 800.0
    FPS = 30
    DURATION = 6 # seconds
    TOTAL_FRAMES = FPS * DURATION

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (W, H))

    tilt = np.float32(-0.3)
    cx, sx = np.cos(tilt), np.sin(tilt)

    print(f"Rendering {TOTAL_FRAMES} frames...")
    for i in range(TOTAL_FRAMES):
        angle = (i / TOTAL_FRAMES) * 2 * math.pi
        ca, sa = np.cos(angle), np.sin(angle)

        # Rotate
        rx = v[:, 0] * ca + v[:, 2] * sa
        rz_ = -v[:, 0] * sa + v[:, 2] * ca
        ry = v[:, 1] * cx - rz_ * sx
        rz = v[:, 1] * sx + rz_ * cx

        # Project
        z_cam = rz + 180.0
        np.clip(z_cam, 1.0, None, out=z_cam)
        f = FOCAL / z_cam
        px = (W / 2 + rx * f).astype(np.int32)
        py = (H / 2 - ry * f).astype(np.int32)

        # Shading
        shade = np.clip((rz + 60) / 120.0, 0.4, 1.0).reshape(-1, 1)
        cs = (c * shade).astype(np.uint8)

        # Draw
        buf = np.zeros((H, W, 3), np.uint8)
        order = np.argsort(rz)[::-1]
        xs, ys, colors_s = px[order], py[order], cs[order]
        
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        # Draw 2x2 points for "neater" look
        for dx in range(2):
            for dy in range(2):
                x_ok, y_ok = xs[ok]+dx, ys[ok]+dy
                mask = (x_ok < W) & (y_ok < H)
                buf[y_ok[mask], x_ok[mask]] = colors_s[ok][mask]

        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        out.write(frame)
        if i % 30 == 0: print(f"Progress: {i}/{TOTAL_FRAMES}")

    out.release()
    print(f"Done! Video saved to {output_path}")

if __name__ == "__main__":
    render_to_mp4()
