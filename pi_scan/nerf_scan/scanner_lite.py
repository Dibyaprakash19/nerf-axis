"""
nerf_scan/scanner_lite.py — TSDF + Marching Cubes 3D reconstruction engine.

Pipeline per scan:
  1. 4x: capture → remove_bg → MiDaS depth
  2. Vectorized TSDF fusion of the 4 depth maps
  3. Gaussian smooth → Marching cubes surface
  4. Trimesh mesh cleaning + color assignment
  5. Scale/center → return (verts, faces, colors)
"""

import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite

from .config import MODEL_PATH, MESH_STEP

# ── DeepLabV3 Background Removal ─────────────────────────────────────────────

class BackgroundRemover:
    """Runs DeepLabV3 TFLite for clean object masking."""

    def __init__(self, model_path=None):
        if model_path is None:
            # Look in parent dir (pi_scan/)
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deeplab_model.tflite")
        
        try:
            from tflite_runtime.interpreter import Interpreter
            self._interp = Interpreter(model_path=model_path)
            self._interp.allocate_tensors()
            self._in  = self._interp.get_input_details()
            self._out = self._interp.get_output_details()
            self._msize = self._in[0]['shape'][1] # usually 257
            print(f"[segmenter] DeepLabV3 loaded OK from {model_path}")
        except Exception as e:
            print(f"[segmenter] init failed: {e}")
            self._interp = None

    def mask(self, img_bgr: np.ndarray) -> np.ndarray:
        if self._interp is None:
            return None
        h, w = img_bgr.shape[:2]
        
        # Preprocess
        small = cv2.resize(img_bgr, (self._msize, self._msize))
        inp   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
        inp   = (inp / 127.5) - 1.0
        inp   = np.expand_dims(inp, axis=0)
        
        self._interp.set_tensor(self._in[0]['index'], inp)
        self._interp.invoke()
        
        m = self._interp.get_tensor(self._out[0]['index'])
        m = np.squeeze(m)
        
        # If output is (H, W, Classes), argmax it
        if len(m.shape) == 3:
            m = np.argmax(m, axis=-1)
            
        # Mask: anything not background (class 0 in PASCAL VOC)
        mask = (m != 0).astype(np.uint8)
        
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

_remover = BackgroundRemover()

def remove_bg(img_bgr: np.ndarray) -> np.ndarray:
    """Remove background. Returns RGB image with background zeroed."""
    mask = _remover.mask(img_bgr)
    if mask is None:
        # GrabCut fallback
        h, w  = img_bgr.shape[:2]
        m     = np.zeros((h, w), np.uint8)
        bgd   = np.zeros((1, 65), np.float64)
        fgd   = np.zeros((1, 65), np.float64)
        cv2.grabCut(img_bgr, m, (10, 10, w - 20, h - 20), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((m == 2) | (m == 0), 0, 1).astype(np.uint8)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.bitwise_and(rgb, rgb, mask=mask)


# ── MiDaS Depth Engine ───────────────────────────────────────────────────────

class ScannerLite:
    """MiDaS TFLite depth inference + TSDF 3D fusion."""

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self._interp = tflite.Interpreter(model_path=model_path)
        self._interp.allocate_tensors()
        self._in    = self._interp.get_input_details()
        self._out   = self._interp.get_output_details()
        self._msize = self._in[0]['shape'][1]

    def depth_from_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Run MiDaS. Returns depth (H,W) float32 normalised 0..1."""
        h, w = rgb.shape[:2]
        inp  = cv2.resize(rgb, (self._msize, self._msize)) if (h != self._msize or w != self._msize) else rgb.copy()
        inp  = inp.astype(np.float32) / 255.0
        inp  = ((inp - self._MEAN) / self._STD)[np.newaxis]
        self._interp.set_tensor(self._in[0]['index'], inp)
        self._interp.invoke()
        d = self._interp.get_tensor(self._out[0]['index']).squeeze()
        if h != self._msize or w != self._msize:
            d = cv2.resize(d.astype(np.float32), (w, h))
        dmin, dmax = d.min(), d.max()
        return (d - dmin) / (dmax - dmin + 1e-6)

    def back_project(self, depth_norm: np.ndarray, rgb: np.ndarray,
                     view_angle: float = 0.0, step: int = MESH_STEP) -> tuple:
        """Legacy single-view projection — still used for TFT preview."""
        d  = depth_norm[::step, ::step]
        c  = rgb[::step, ::step]
        hs, ws = d.shape[:2]
        ci, ri = np.meshgrid(np.arange(ws), np.arange(hs))
        X = (ci - ws / 2.0).ravel().astype(np.float32)
        Y = -(ri - hs / 2.0).ravel().astype(np.float32)
        Z = (d.ravel() * (ws * 0.5)).astype(np.float32)
        cf  = c.reshape(-1, 3)
        ok  = cf.sum(axis=1) >= 15
        g2l = np.full(hs * ws, -1, np.int32)
        g2l[np.where(ok)[0]] = np.arange(ok.sum(), dtype=np.int32)
        verts  = np.stack([X, Y, Z], axis=1)[ok]
        colors = cf[ok].astype(np.uint8)
        rad = np.radians(view_angle)
        ca, sa = np.cos(rad), np.sin(rad)
        Xr =  verts[:, 0] * ca + verts[:, 2] * sa
        Zr = -verts[:, 0] * sa + verts[:, 2] * ca
        verts[:, 0], verts[:, 2] = Xr, Zr
        faces = []
        for r in range(hs - 1):
            for col in range(ws - 1):
                v00 = r * ws + col; v01 = v00 + 1
                v10 = (r + 1) * ws + col; v11 = v10 + 1
                l00, l01, l10, l11 = g2l[v00], g2l[v01], g2l[v10], g2l[v11]
                if l00 >= 0 and l01 >= 0 and l10 >= 0:
                    faces.append([l00, l10, l01])
                if l01 >= 0 and l10 >= 0 and l11 >= 0:
                    faces.append([l01, l10, l11])
        faces_arr = np.array(faces, np.int32) if faces else np.zeros((0, 3), np.int32)
        return verts.astype(np.float32), faces_arr, colors


# ── TSDF Fusion Engine ───────────────────────────────────────────────────────

def _depth_to_world_pts(depth: np.ndarray, rgb: np.ndarray,
                         angle_deg: float) -> tuple:
    """
    Back-project a depth map to world-space 3D points.
    Normalises depth per-view for consistent cross-view alignment.
    Returns (pts_xyz, colors) both shape (N,3) after masking background.
    """
    h, w = depth.shape
    # Per-view depth normalisation for consistent scale
    d = depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-6) * 1.2

    # Focal-style back-projection
    y, x  = np.mgrid[0:h, 0:w]
    X = (x - w / 2.0) * d / 140.0
    Y = (y - h / 2.0) * d / 140.0
    Z = d

    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # Turntable Y-axis rotation
    rad     = np.radians(angle_deg)
    ca, sa  = np.cos(rad), np.sin(rad)
    Xr =  pts[:, 0] * ca + pts[:, 2] * sa
    Zr = -pts[:, 0] * sa + pts[:, 2] * ca
    pts[:, 0] = Xr
    pts[:, 2] = Zr

    # Mask out black (background) pixels
    colors = rgb.reshape(-1, 3).astype(np.float32)
    fg     = colors.sum(axis=1) >= 15
    return pts[fg].astype(np.float32), colors[fg].astype(np.uint8)


def fuse(views: list) -> tuple:
    """
    TSDF-based fusion of 4 (depth, rgb, angle) views.

    views — list of (depth_norm, rgb, angle_deg)
            (main.py now passes these instead of pre-projected verts)

    Returns (verts, faces, colors) centred and scaled for the renderer.
    """
    # ── Step 1: collect all world-space points ───────────────────────────────
    all_pts, all_colors = [], []
    for depth, rgb, angle in views:
        pts, cols = _depth_to_world_pts(depth, rgb, angle)
        all_pts.append(pts)
        all_colors.append(cols)

    if not all_pts:
        raise RuntimeError("No foreground points found — check lighting/background.")

    pts_all = np.vstack(all_pts)
    col_all = np.vstack(all_colors)

    # ── Step 2: build occupancy voxel grid ──────────────────────────────────
    VOL = 96      # 96^3 voxels — ~3.5M floats = ~14 MB, safe on Pi 4
    pmin = pts_all.min(axis=0)
    pmax = pts_all.max(axis=0)
    span = pmax - pmin + 1e-6

    # Map world pts → voxel indices
    idx = ((pts_all - pmin) / span * (VOL - 1)).clip(0, VOL - 1).astype(np.int32)
    ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]

    # Occupancy: use np.add.at for accumulation (fully vectorized)
    density = np.zeros((VOL, VOL, VOL), np.float32)
    np.add.at(density, (ix, iy, iz), 1.0)

    # ── Step 3: Gaussian smooth + marching cubes ─────────────────────────────
    density = gaussian_filter(density, sigma=1.5)

    # Adaptive level — use a percentile of non-zero voxels as isovalue
    nonzero = density[density > 0]
    level   = float(np.percentile(nonzero, 40)) if len(nonzero) > 100 else 0.5

    try:
        from skimage.measure import marching_cubes
        verts_v, faces, _, _ = marching_cubes(density, level=level)
    except Exception as e:
        print(f"[fuse] marching cubes failed ({e}), using point cloud fallback")
        return _point_cloud_fallback(pts_all, col_all)

    print(f"[fuse] marching cubes → {len(verts_v):,} verts, {len(faces):,} faces")

    # ── Step 4: map voxel verts back to world space ──────────────────────────
    verts_w = (verts_v / (VOL - 1)) * span + pmin

    # ── Step 5: assign colors via KD-tree nearest neighbour ─────────────────
    try:
        from scipy.spatial import KDTree
        tree   = KDTree(pts_all)
        _, idx_near = tree.query(verts_w, workers=-1)
        colors = col_all[idx_near].astype(np.uint8)
    except Exception:
        colors = np.full((len(verts_w), 3), 180, np.uint8)

    # ── Step 6: trimesh clean (remove degenerate faces, fix normals) ─────────
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=verts_w, faces=faces, process=True)
        # Keep only the biggest connected component
        comps = mesh.split(only_watertight=False)
        if comps:
            mesh = max(comps, key=lambda m: len(m.vertices))
        # Re-map colors to cleaned vertex set
        # (trimesh may reorder verts, so just use uniform color here)
        verts_w = np.array(mesh.vertices, np.float32)
        faces   = np.array(mesh.faces,    np.int32)
        colors  = np.full((len(verts_w), 3), 180, np.uint8)
        print(f"[fuse] trimesh clean → {len(verts_w):,} verts")
    except Exception as e:
        print(f"[fuse] trimesh skip ({e})")

    # ── Step 7: centre and scale to renderer units ───────────────────────────
    verts_w -= verts_w.mean(axis=0)
    m = np.abs(verts_w).max()
    if m > 0:
        verts_w *= 50.0 / m

    return verts_w.astype(np.float32), faces.astype(np.int32), colors


def _point_cloud_fallback(pts: np.ndarray, cols: np.ndarray) -> tuple:
    """Simple point cloud when marching cubes fails."""
    pts -= pts.mean(axis=0)
    m = np.abs(pts).max()
    if m > 0:
        pts *= 50.0 / m
    faces = np.zeros((0, 3), np.int32)
    return pts.astype(np.float32), faces, cols.astype(np.uint8)
