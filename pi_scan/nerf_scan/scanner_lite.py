"""
nerf_scan/scanner_lite.py — Ultra-lightweight 3D depth engine.

  Capture:  256×256  (square → zero pre-resize for MiDaS-256)
  Step=2  → 128×128 vertex grid  = TFT_W × TFT_H exactly
  No trimesh. No rembg. No scikit-image. No onnxruntime.
"""

import os
import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

from .config import MODEL_PATH, MESH_STEP


# ── Background removal (GrabCut + threshold fallback) ─────────────────────────

def remove_bg(img_bgr: np.ndarray) -> np.ndarray:
    """
    Input:  BGR (H,W,3) uint8
    Output: RGB (H,W,3) uint8  — background set to black (0,0,0)
    """
    h, w = img_bgr.shape[:2]
    mask    = np.zeros((h, w), np.uint8)
    bgd     = np.zeros((1, 65), np.float64)
    fgd     = np.zeros((1, 65), np.float64)
    x0, y0  = int(w * .08), int(h * .08)
    rw, rh  = int(w * .84), int(h * .84)

    try:
        cv2.grabCut(img_bgr, mask, (x0, y0, rw, rh), bgd, fgd,
                    5, cv2.GC_INIT_WITH_RECT)
        fg = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
        ).astype(np.uint8)
        k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k)
    except Exception:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, fg = cv2.threshold(gray, 15, 1, cv2.THRESH_BINARY)
        fg = fg.astype(np.uint8)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.bitwise_and(rgb, rgb, mask=fg)


# ── MiDaS engine ─────────────────────────────────────────────────────────────

class ScannerLite:
    """Thin wrapper around MiDaS tflite model."""

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self._interp = tflite.Interpreter(model_path=model_path)
        self._interp.allocate_tensors()
        self._in     = self._interp.get_input_details()
        self._out    = self._interp.get_output_details()
        self._msize  = self._in[0]['shape'][1]   # 256 for midas_small

    def depth_from_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """
        rgb: (H, W, 3) uint8 — already background-removed
        Returns depth (H, W) float32, normalised 0..1
        """
        h, w = rgb.shape[:2]
        # Resize to model input (256×256) if not already
        if h != self._msize or w != self._msize:
            inp = cv2.resize(rgb, (self._msize, self._msize))
        else:
            inp = rgb.copy()

        inp = inp.astype(np.float32) / 255.0
        inp = ((inp - self._MEAN) / self._STD)[np.newaxis]     # (1,256,256,3)

        self._interp.set_tensor(self._in[0]['index'],  inp)
        self._interp.invoke()
        d = self._interp.get_tensor(self._out[0]['index'])[0]  # (256,256)

        # Resize depth back to source image size
        if h != self._msize or w != self._msize:
            d = cv2.resize(d.astype(np.float32), (w, h))

        # Normalise to 0..1
        dmin, dmax = d.min(), d.max()
        return (d - dmin) / (dmax - dmin + 1e-6)

    def back_project(self, depth_norm: np.ndarray, rgb: np.ndarray,
                     view_angle: float = 0.0,
                     step: int = MESH_STEP) -> tuple:
        """
        Convert depth map → (vertices, faces, colors).

        step=2 on 256×256 → 128×128 grid = TFT_W × TFT_H vertices.

        Returns
        -------
        verts  : (N, 3) float32
        faces  : (M, 3) int32   (empty array if N is small)
        colors : (N, 3) uint8
        """
        d  = depth_norm[::step, ::step]
        c  = rgb[::step, ::step]
        hs, ws = d.shape

        # Vectorised back-projection
        ci, ri = np.meshgrid(np.arange(ws), np.arange(hs))
        X = (ci - ws / 2.0).ravel().astype(np.float32)
        Y = -(ri - hs / 2.0).ravel().astype(np.float32)
        Z = (d.ravel() * (ws * 0.5)).astype(np.float32)

        # Background mask (skip black pixels — GrabCut set them to 0)
        cf   = c.reshape(-1, 3)
        ok   = cf.sum(axis=1) >= 15
        g2l  = np.full(hs * ws, -1, np.int32)
        lidx = np.where(ok)[0]
        g2l[lidx] = np.arange(len(lidx), dtype=np.int32)

        verts  = np.stack([X, Y, Z], axis=1)[ok]
        colors = cf[ok].astype(np.uint8)

        # Turntable Y-axis rotation
        rad = np.radians(view_angle)
        ca, sa = np.cos(rad), np.sin(rad)
        Xr =  verts[:, 0] * ca + verts[:, 2] * sa
        Zr = -verts[:, 0] * sa + verts[:, 2] * ca
        verts[:, 0] = Xr
        verts[:, 2] = Zr

        # Grid faces (only where all 4 quad corners are valid foreground)
        faces = []
        for r in range(hs - 1):
            for col in range(ws - 1):
                v00 = r * ws + col;  v01 = v00 + 1
                v10 = (r + 1) * ws + col;  v11 = v10 + 1
                l00, l01 = g2l[v00], g2l[v01]
                l10, l11 = g2l[v10], g2l[v11]
                if l00 >= 0 and l01 >= 0 and l10 >= 0:
                    faces.append([l00, l10, l01])
                if l01 >= 0 and l10 >= 0 and l11 >= 0:
                    faces.append([l01, l10, l11])

        faces_arr = (np.array(faces, np.int32)
                     if faces else np.zeros((0, 3), np.int32))
        return verts.astype(np.float32), faces_arr, colors


# ── Mesh fusion ───────────────────────────────────────────────────────────────

def fuse(views: list) -> tuple:
    """
    Merge list of (verts, faces, colors) into one centred, scaled mesh.
    Returns (verts, faces, colors).
    """
    all_v, all_f, all_c = [], [], []
    offset = 0
    for v, f, c in views:
        if len(v) == 0:
            continue
        all_v.append(v)
        all_c.append(c)
        if len(f):
            all_f.append(f + offset)
        offset += len(v)

    if not all_v:
        raise RuntimeError("No object found in any view — check lighting/background.")

    fv = np.vstack(all_v)
    ff = np.vstack(all_f) if all_f else np.zeros((0, 3), np.int32)
    fc = np.vstack(all_c)

    fv -= fv.mean(axis=0)
    m   = np.abs(fv).max()
    if m > 0:
        fv *= 50.0 / m      # scale so max extent = 50 units (fits renderer)

    return fv.astype(np.float32), ff, fc.astype(np.uint8)
