"""
Single-view relief mesh fallback.

This does not try to fuse inconsistent monocular depths from multiple angles.
It captures one image, runs MiDaS on the full image, masks only for geometry,
and exports a front-facing textured depth relief mesh.
"""
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import store
from .camera import CameraManager
from .config import CAM_H, CAM_W
from .scanner_lite import ScannerLite, _remover, remove_bg
from .tft_ui import TFTDisplay


OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "single_view_relief")
MESH_PATH = os.path.join(OUT_DIR, "single_view_relief.glb")


def _mask_after_depth(frame_bgr: np.ndarray) -> np.ndarray:
    mask = _remover.mask(frame_bgr)
    if mask is None or int(mask.sum()) < 100:
        masked_rgb = remove_bg(frame_bgr)
        return (masked_rgb.sum(axis=2) > 15).astype(np.uint8)
    return mask.astype(np.uint8)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(np.uint8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = (labels == largest).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=2)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
    return out


def _relief_mesh(depth: np.ndarray, rgb: np.ndarray, mask: np.ndarray, step: int = 2):
    mask = _largest_component(mask)
    depth_s = depth[::step, ::step].astype(np.float32)
    rgb_s = rgb[::step, ::step].astype(np.uint8)
    mask_s = mask[::step, ::step].astype(bool)
    h, w = depth_s.shape

    if mask_s.sum() < 50:
        raise RuntimeError("Mask too small; object was not isolated.")

    fg_depth = depth_s[mask_s]
    lo, hi = np.percentile(fg_depth, [3, 97])
    d = np.clip((depth_s - lo) / (hi - lo + 1e-6), 0, 1)
    d = cv2.GaussianBlur(d, (0, 0), 1.0)

    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx - w / 2.0).astype(np.float32)
    y = -(yy - h / 2.0).astype(np.float32)
    z = (d - 0.5).astype(np.float32) * (w * 0.45)

    flat_index = np.full(h * w, -1, np.int32)
    valid = mask_s.reshape(-1)
    flat_index[np.where(valid)[0]] = np.arange(valid.sum(), dtype=np.int32)

    verts = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=1)[valid]
    colors = rgb_s.reshape(-1, 3)[valid]

    faces = []
    for r in range(h - 1):
        for c in range(w - 1):
            ids = [
                flat_index[r * w + c],
                flat_index[r * w + c + 1],
                flat_index[(r + 1) * w + c],
                flat_index[(r + 1) * w + c + 1],
            ]
            if min(ids[0], ids[1], ids[2]) >= 0:
                faces.append([ids[0], ids[2], ids[1]])
            if min(ids[1], ids[2], ids[3]) >= 0:
                faces.append([ids[1], ids[2], ids[3]])

    verts -= verts.mean(axis=0)
    m = np.abs(verts).max()
    if m > 0:
        verts *= 50.0 / m
    return verts.astype(np.float32), np.asarray(faces, np.int32), colors.astype(np.uint8), mask


def _save_debug(frame_bgr: np.ndarray, depth: np.ndarray, rgb: np.ndarray, mask: np.ndarray):
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUT_DIR, "raw.jpg"), frame_bgr)
    cv2.imwrite(os.path.join(OUT_DIR, "mask.jpg"), mask.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(OUT_DIR, "masked.jpg"), cv2.cvtColor(cv2.bitwise_and(rgb, rgb, mask=mask), cv2.COLOR_RGB2BGR))
    depth_u8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT_DIR, "depth.jpg"), cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO))


def main():
    down = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: down.set())
    signal.signal(signal.SIGTERM, lambda *_: down.set())

    disp = TFTDisplay()
    scanner = ScannerLite()
    cam = CameraManager()

    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        disp.text("Single View", "capturing...", color="cyan")
        cam.start()
        frame_bgr = cam.capture_still_bgr(CAM_W, CAM_H)
        cam.stop()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        disp.text("Single View", "MiDaS full image", color="cyan")
        depth = scanner.depth_from_rgb(rgb)
        mask = _mask_after_depth(frame_bgr)
        verts, faces, colors, mask = _relief_mesh(depth, rgb, mask)
        _save_debug(frame_bgr, depth, rgb, mask)

        store.save(verts, faces, colors, path=MESH_PATH)
        disp.text("Relief Mesh", "front view ready", color="green")
        print(f"[single_view_relief] mesh -> {MESH_PATH}")

        spin_stop = threading.Event()
        threading.Thread(target=disp.spin, args=(verts, colors, spin_stop), daemon=True).start()
        while not down.is_set():
            time.sleep(0.2)
        spin_stop.set()
    finally:
        cam.stop()
        disp.clear()
        print("[single_view_relief] shutdown")


if __name__ == "__main__":
    main()
