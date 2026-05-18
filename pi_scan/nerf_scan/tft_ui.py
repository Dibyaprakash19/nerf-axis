"""
nerf_scan/tft_ui.py - TFT display management.
"""
import math, time, threading, cv2
import numpy as np
from PIL import Image, ImageDraw
from st7735_direct_hello import DirectST7735
from .config import TFT_W, TFT_H, TFT_DC, TFT_RST, TFT_PTS, ROTATION_SPEED

_FOCAL = 220.0 # Perspective focal length

class TFTDisplay:
    """Thread-safe TFT wrapper with 3D point cloud renderer."""

    def __init__(self):
        self._d = DirectST7735(dc_pin=TFT_DC, rst_pin=TFT_RST)
        self._lock = threading.Lock()
        with self._lock:
            self._d.reset()
            self._d.off()
            time.sleep(0.5)
            self._d.cmd(0x29) # DISPON
            self._d.init()

    def _fast_display(self, img):
        """High-speed NumPy blitter (RGB565)."""
        arr = np.array(img.convert("RGB"), dtype=np.uint32)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        
        buf = np.empty((TFT_H, TFT_W, 2), dtype=np.uint8)
        buf[:,:,0] = (rgb565 >> 8) & 0xFF
        buf[:,:,1] = rgb565 & 0xFF
        
        with self._lock:
            self._d.set_window(0, 0, TFT_W - 1, TFT_H - 1)
            self._d.write(buf.tobytes(), True)

    def text(self, line1: str, line2: str = "", color: str = "white"):
        """Status text overlay."""
        img = Image.new("RGB", (TFT_W, TFT_H), "black")
        draw = ImageDraw.Draw(img)
        draw.text((8, 60), line1, fill=color)
        if line2:
            draw.text((8, 80), line2, fill="#888")
        self._fast_display(img)

    def clear(self):
        self._fast_display(Image.new("RGB", (TFT_W, TFT_H), "black"))

    def spin(self, verts: np.ndarray, colors: np.ndarray, stop_event: threading.Event):
        """Render spinning point cloud."""
        v, c = verts, colors
        if len(v) > TFT_PTS:
            idx = np.random.default_rng(42).choice(len(v), TFT_PTS, replace=False)
            v, c = v[idx], c[idx]

        angle = 0.0
        tilt = np.float32(-0.3)
        cx, sx = np.cos(tilt), np.sin(tilt)

        while not stop_event.is_set():
            ca, sa = np.cos(angle), np.sin(angle)
            
            # Rotation
            rx = v[:, 0] * ca + v[:, 2] * sa
            rz_ = -v[:, 0] * sa + v[:, 2] * ca
            ry = v[:, 1] * cx - rz_ * sx
            rz = v[:, 1] * sx + rz_ * cx

            # Projection
            z_cam = rz + 150.0
            np.clip(z_cam, 1.0, None, out=z_cam)
            f = _FOCAL / z_cam
            px = (TFT_W / 2 + rx * f).astype(np.int32)
            py = (TFT_H / 2 - ry * f).astype(np.int32)

            # Depth shading
            shade = np.clip((rz + 50) / 100.0, 0.4, 1.0).reshape(-1, 1)
            cs = (c * shade).astype(np.uint8)

            # Draw buffer
            buf = np.zeros((TFT_H, TFT_W, 3), np.uint8)
            order = np.argsort(rz)[::-1]
            xs, ys, colors_s = px[order], py[order], cs[order]
            
            ok = (xs >= 0) & (xs < TFT_W - 1) & (ys >= 0) & (ys < TFT_H - 1)
            
            # Draw 2x2 blocks for "solid" appearance
            for dx in range(2):
                for dy in range(2):
                    buf[ys[ok]+dy, xs[ok]+dx] = colors_s[ok]
            
            self._fast_display(Image.fromarray(buf))
            angle += ROTATION_SPEED
