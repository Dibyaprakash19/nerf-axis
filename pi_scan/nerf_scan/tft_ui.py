"""
nerf_scan/tft_ui.py — TFT display helpers using DirectST7735 (Holy Bible driver).

  Fast renderer uses NumPy pixel-buffer — no Python loops, ~5ms per frame.
  Depth-sorted point cloud spins on TFT at the native 128×160 resolution.

Usage:
    disp = TFTDisplay()
    disp.text("Scanning...", "view 1/4", color="cyan")
    disp.spin(verts_Nx3, colors_Nx3, stop_event)   # blocks until stop_event set
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math, time, threading, cv2
import numpy as np
from PIL import Image, ImageDraw

from st7735_direct_hello import DirectST7735
from .config import TFT_W, TFT_H, TFT_DC, TFT_RST, TFT_PTS, ROTATION_SPEED, FRAME_DELAY

_FOCAL = 220.0   # perspective focal length (px)


class TFTDisplay:
    """Thread-safe wrapper around DirectST7735 with text + fast 3D renderer."""

    def __init__(self):
        self._d = DirectST7735(dc_pin=TFT_DC, rst_pin=TFT_RST)
        self._lock = threading.Lock()
        with self._lock:
            self._d.reset()
            self._d.off()
            time.sleep(0.5)
            self._d.cmd(0x29) # DISPON
            time.sleep(0.1)
            self._d.init()
            time.sleep(0.2)

    def _fast_display(self, img):
        """Ultra-fast NumPy-based display driver. Fixes bit-shift overflow."""
        # Ensure we use uint32 for shifting to avoid overflow
        arr = np.array(img.convert("RGB"), dtype=np.uint32)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        
        # RGB565: (r5 << 11) | (g6 << 5) | b5
        # Bit manipulation in bulk
        rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        
        # Split into two bytes (Big Endian)
        buf = np.empty((TFT_H, TFT_W, 2), dtype=np.uint8)
        buf[:,:,0] = (rgb565 >> 8) & 0xFF
        buf[:,:,1] = rgb565 & 0xFF
        
        with self._lock:
            self._d.set_window(0, 0, TFT_W - 1, TFT_H - 1)
            self._d.write(buf.tobytes(), True)

    # ── Text overlay ──────────────────────────────────────────────────────────

    def text(self, line1: str, line2: str = "WAITING FOR GESTURE", color: str = "white"):
        """Show two lines of status text centred on a black background."""
        img  = Image.new("RGB", (TFT_W, TFT_H), "black")
        draw = ImageDraw.Draw(img)
        draw.text((4, TFT_H // 2 - 14), line1, fill=color)
        if line2:
            draw.text((4, TFT_H // 2 + 4), line2, fill="#777")
        self._fast_display(img)

    def show_preview(self, frame_bgr: np.ndarray, label: str = ""):
        """Efficiently blit camera frame to TFT with optional gesture label."""
        # Convert BGR to RGB and resize
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((TFT_W, TFT_H), Image.BILINEAR)
        
        if label:
            draw = ImageDraw.Draw(img)
            # Semi-transparent overlay for label
            draw.rectangle([4, 4, 120, 24], fill=(0, 0, 0, 150))
            draw.text((8, 8), label, fill="cyan")
            
        self._fast_display(img)

    def clear(self, color: str = "black"):
        self._fast_display(Image.new("RGB", (TFT_W, TFT_H), color))

    # ── 3D point-cloud renderer ───────────────────────────────────────────────

    def spin(self, verts: np.ndarray, colors: np.ndarray,
             stop_event: threading.Event):
        """
        Rotate and project `verts` (N×3 float32) coloured by `colors` (N×3 uint8)
        onto the TFT until `stop_event` is set.

        Renderer pipeline (all NumPy, no Python loops):
          1. Random subsample to TFT_PTS points
          2. Y-axis rotation
          3. Perspective projection
          4. Write pixels into a (H,W,3) buffer → PIL → TFT
        """
        v = verts
        c = colors

        # Subsample for speed — deterministic random seed per call
        if len(v) > TFT_PTS:
            rng  = np.random.default_rng(42)
            idx  = rng.choice(len(v), TFT_PTS, replace=False)
            v, c = v[idx], c[idx]

        angle = 0.0
        tilt  = np.float32(-0.35)   # fixed X-tilt for better perspective
        cx, sx = np.cos(tilt), np.sin(tilt)

        while not stop_event.is_set():
            ca, sa = np.cos(angle), np.sin(angle)

            # Y-rotation then X-tilt
            rx  =  v[:, 0] * ca + v[:, 2] * sa
            rz_ = -v[:, 0] * sa + v[:, 2] * ca
            ry  =  v[:, 1] * cx - rz_ * sx
            rz  =  v[:, 1] * sx + rz_ * cx

            # Perspective
            z_cam = rz + 150.0
            np.clip(z_cam, 1.0, None, out=z_cam)
            f  = _FOCAL / z_cam
            px = (TFT_W / 2 + rx * f).astype(np.int32)
            py = (TFT_H / 2 - ry * f).astype(np.int32)

            # Depth shading (simple ambient + depth)
            shade = np.clip((rz + 55) / 110.0, 0.35, 1.0).reshape(-1, 1)
            shaded_c = (c * shade).astype(np.uint8)

            # Paint into pixel buffer (back-to-front via argsort)
            buf   = np.zeros((TFT_H, TFT_W, 3), np.uint8)
            order = np.argsort(rz)[::-1]        # back-to-front

            # Vectorised mask + paint
            xs, ys = px[order], py[order]
            cs     = shaded_c[order]
            ok     = (xs >= 0) & (xs < TFT_W) & (ys >= 0) & (ys < TFT_H)
            buf[ys[ok], xs[ok]] = cs[ok]

            self._fast_display(Image.fromarray(buf))
            angle += ROTATION_SPEED
            time.sleep(FRAME_DELAY)
