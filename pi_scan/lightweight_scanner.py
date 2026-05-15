"""
lightweight_scanner.py — Trimesh-FREE depth engine for Pi 4B.

Dependencies (all lightweight / pre-built on piwheels):
    tflite-runtime           — MiDaS inference
    numpy                    — math
    opencv-python-headless   — image I/O + GrabCut BG removal
    Pillow                   — image ops

No trimesh. No scikit-image. No rembg. No onnxruntime.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

# ── TFLite runtime ─────────────────────────────────────────────
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from tensorflow import lite as tflite
    except ImportError:
        print("Error: install tflite-runtime  →  pip install tflite-runtime")
        sys.exit(1)

# ── No optional imports needed — pure OpenCV BG removal below ──


class LightScanner:
    """
    Lightweight MiDaS depth engine.
    No trimesh. No scikit-image. Pure NumPy + OpenCV.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.interp = tflite.Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self._in  = self.interp.get_input_details()
        self._out = self.interp.get_output_details()
        self.input_size = self._in[0]['shape'][1]   # e.g. 256 for midas_small

    # ── Image preprocessing ────────────────────────────────────
    def _preprocess_array(self, img_rgb: np.ndarray):
        """Prepare an already-loaded RGB array for MiDaS inference."""
        h0, w0 = img_rgb.shape[:2]
        resized = cv2.resize(img_rgb, (self.input_size, self.input_size))
        inp = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp  = ((inp - mean) / std)[np.newaxis]
        return inp, img_rgb, (h0, w0)

    def _preprocess(self, image_path: str):
        """Load image from disk and prepare for inference."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._preprocess_array(img_rgb)

    # ── MiDaS inference ────────────────────────────────────────
    def get_depth(self, source):
        """
        Run MiDaS on a file path OR a pre-processed numpy RGB array.
        Returns (depth_map H×W float32, rgb_image H×W×3 uint8).
        """
        if isinstance(source, str):
            inp, img_rgb, (h0, w0) = self._preprocess(source)
        elif isinstance(source, np.ndarray):
            inp, img_rgb, (h0, w0) = self._preprocess_array(source)
        else:
            raise TypeError(f"source must be a path or numpy array, got {type(source)}")

        self.interp.set_tensor(self._in[0]['index'], inp)
        self.interp.invoke()
        depth = self.interp.get_tensor(self._out[0]['index'])[0]
        depth = cv2.resize(depth.astype(np.float32), (w0, h0))
        return depth, img_rgb

    # alias kept for compatibility
    def process_image(self, source):
        return self.get_depth(source)

    # ── Back-projection → vertices + faces ────────────────────
    def depth_to_mesh(self, depth: np.ndarray, rgb: np.ndarray,
                      view_angle: float = 0.0, step: int = 16,
                      focal: float = 180.0, scale: float = 0.5):
        """
        Convert a depth map to a NumPy vertex/face arrays.
        Returns:
            vertices  — (N,3) float32
            faces     — (M,3) int32
            colors    — (N,3) uint8
        """
        h, w = depth.shape

        # Subsample
        d = depth[::step, ::step]
        c = rgb[::step, ::step]
        hs, ws = d.shape

        # Normalise depth
        dmin, dmax = d.min(), d.max()
        d_norm = (d - dmin) / (dmax - dmin + 1e-6)

        # Vectorised back-projection
        col_idx, row_idx = np.meshgrid(np.arange(ws), np.arange(hs))
        X = (col_idx - ws / 2.0).ravel()
        Y = -(row_idx - hs / 2.0).ravel()
        Z = d_norm.ravel() * (ws * scale)

        # Background mask: skip very dark pixels
        colors_flat = c.reshape(-1, 3)
        mask = colors_flat.sum(axis=1) >= 15      # not black background

        # Remap masked vertices to compact indices
        global_to_local = np.full(hs * ws, -1, dtype=np.int32)
        local_idx = np.where(mask)[0]
        global_to_local[local_idx] = np.arange(len(local_idx), dtype=np.int32)

        vertices = np.stack([X, Y, Z], axis=1)[mask]
        colors   = colors_flat[mask]

        # Turntable Y-rotation
        rad = np.radians(view_angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        Xr =  vertices[:, 0] * cos_a + vertices[:, 2] * sin_a
        Zr = -vertices[:, 0] * sin_a + vertices[:, 2] * cos_a
        vertices[:, 0] = Xr
        vertices[:, 2] = Zr

        # Grid faces (only if both triangle corners are valid)
        faces = []
        for r in range(hs - 1):
            for col in range(ws - 1):
                v00 = r * ws + col
                v01 = v00 + 1
                v10 = (r + 1) * ws + col
                v11 = v10 + 1
                l00 = global_to_local[v00]
                l01 = global_to_local[v01]
                l10 = global_to_local[v10]
                l11 = global_to_local[v11]
                if l00 >= 0 and l01 >= 0 and l10 >= 0:
                    faces.append([l00, l10, l01])
                if l01 >= 0 and l10 >= 0 and l11 >= 0:
                    faces.append([l01, l10, l11])

        faces_arr = np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), np.int32)
        return vertices.astype(np.float32), faces_arr, colors


def remove_background_opencv(img_bgr: np.ndarray) -> np.ndarray:
    """
    Pure OpenCV GrabCut background removal.
    Input:  BGR image (as returned by cv2.imread)
    Output: RGB image with background set to black (0,0,0)

    Assumes the object is roughly centred (uses an 80% ROI rect).
    Falls back to simple threshold mask if GrabCut fails.
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        mask     = np.zeros((h, w), np.uint8)
        bgd_mdl  = np.zeros((1, 65), np.float64)
        fgd_mdl  = np.zeros((1, 65), np.float64)

        # Centre rectangle — 80 % of image area
        x0 = int(w * 0.10);  y0 = int(h * 0.10)
        rw = int(w * 0.80);  rh = int(h * 0.80)
        rect = (x0, y0, rw, rh)

        cv2.grabCut(img_bgr, mask, rect, bgd_mdl, fgd_mdl,
                    5, cv2.GC_INIT_WITH_RECT)

        # Pixels marked probable/definite foreground
        fg_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
        ).astype(np.uint8)

        # Optional: clean up small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)

        result = cv2.bitwise_and(img_rgb, img_rgb, mask=fg_mask)
        print("[BG] GrabCut removal OK")
        return result

    except Exception as e:
        print(f"[BG] GrabCut failed ({e}), using threshold fallback")
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_and(img_rgb, img_rgb, mask=mask)


def save_glb(vertices: np.ndarray, faces: np.ndarray, path: str) -> bool:
    """
    Write a minimal binary GLB file.
    No trimesh. Pure pygltflib (small pure-Python wheel).
    Falls back silently if pygltflib is not installed.
    """
    try:
        from pygltflib import (GLTF2, Asset, Scene, Node, Mesh, Primitive,
                               Accessor, BufferView, Buffer, Attributes)
        import base64

        v = vertices.astype(np.float32)
        f = faces.astype(np.uint32)

        v_bytes = v.tobytes()
        f_bytes = f.tobytes()
        blob    = f_bytes + v_bytes

        gltf = GLTF2(
            asset=Asset(version="2.0"),
            scenes=[Scene(nodes=[0])],
            nodes=[Node(mesh=0)],
            meshes=[Mesh(primitives=[
                Primitive(attributes=Attributes(POSITION=1), indices=0)
            ])],
            accessors=[
                # index accessor
                Accessor(bufferView=0, componentType=5125,   # UNSIGNED_INT
                         count=len(f) * 3, type="SCALAR"),
                # position accessor
                Accessor(bufferView=1, componentType=5126,   # FLOAT
                         count=len(v), type="VEC3",
                         max=v.max(axis=0).tolist(),
                         min=v.min(axis=0).tolist()),
            ],
            bufferViews=[
                BufferView(buffer=0, byteOffset=0,
                           byteLength=len(f_bytes), target=34963),        # ELEMENT_ARRAY_BUFFER
                BufferView(buffer=0, byteOffset=len(f_bytes),
                           byteLength=len(v_bytes), target=34962),        # ARRAY_BUFFER
            ],
            buffers=[Buffer(
                uri="data:application/octet-stream;base64," + base64.b64encode(blob).decode(),
                byteLength=len(blob)
            )],
        )
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        gltf.save(path)
        print(f"[GLB] saved → {path}")
        return True
    except ImportError:
        print("[GLB] pygltflib not installed – skipping GLB export")
        return False
    except Exception as e:
        print(f"[GLB] export error: {e}")
        return False


def fuse_views(views):
    """
    Merge a list of (vertices, faces, colors) tuples into one mesh.
    Returns (vertices, faces, colors) — no trimesh.
    """
    all_v, all_f, all_c = [], [], []
    offset = 0
    for v, f, c in views:
        all_v.append(v)
        all_c.append(c)
        if len(f):
            all_f.append(f + offset)
        offset += len(v)

    final_v = np.vstack(all_v)
    final_f = np.vstack(all_f) if all_f else np.zeros((0, 3), np.int32)
    final_c = np.vstack(all_c)

    # Centre
    final_v -= final_v.mean(axis=0)
    # Scale to fit renderer (~50 unit radius)
    m = np.abs(final_v).max()
    if m > 0:
        final_v *= (50.0 / m)

    return final_v.astype(np.float32), final_f, final_c.astype(np.uint8)
