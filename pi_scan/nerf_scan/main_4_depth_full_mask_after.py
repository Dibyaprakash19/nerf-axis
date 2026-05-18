"""
4-view experiment: full image into MiDaS, mask only after depth.

This is separate from nerf_scan/main.py so the current automated setup stays
unchanged. It writes debug images beside the output mesh.
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
from .scanner_lite import ScannerLite, _remover, fuse, remove_bg
from .stepper_ctrl import cleanup as gpio_cleanup, rotate_90, setup as gpio_setup
from .tft_ui import TFTDisplay


OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "depth_full_mask_after")
MESH_PATH = os.path.join(OUT_DIR, "mesh_4view_depth_full_mask_after.glb")


def _mask_after_depth(frame_bgr: np.ndarray) -> np.ndarray:
    """Return RGB image with non-object pixels zeroed, without affecting depth inference."""
    mask = _remover.mask(frame_bgr)
    if mask is None or int(mask.sum()) < 100:
        masked_rgb = remove_bg(frame_bgr)
    else:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        masked_rgb = cv2.bitwise_and(rgb, rgb, mask=mask.astype(np.uint8))
    return masked_rgb


def _save_debug(i: int, frame_bgr: np.ndarray, depth: np.ndarray, masked_rgb: np.ndarray):
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUT_DIR, f"view_{i:02d}_raw.jpg"), frame_bgr)

    depth_u8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(OUT_DIR, f"view_{i:02d}_depth.jpg"), depth_color)

    mask = (masked_rgb.sum(axis=2) > 15).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(OUT_DIR, f"view_{i:02d}_mask.jpg"), mask)
    cv2.imwrite(os.path.join(OUT_DIR, f"view_{i:02d}_masked.jpg"), cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR))


def main():
    down = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: down.set())
    signal.signal(signal.SIGTERM, lambda *_: down.set())

    gpio_setup()
    disp = TFTDisplay()
    scanner = ScannerLite()
    cam = CameraManager()
    views = []

    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        disp.text("4 View Test", "Full depth + mask", color="cyan")
        cam.start()

        for i in range(4):
            disp.text(f"View {i + 1}/4", "MiDaS full image", color="cyan")
            frame_bgr = cam.capture_still_bgr(CAM_W, CAM_H)

            rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            depth = scanner.depth_from_rgb(rgb_full)
            masked_rgb = _mask_after_depth(frame_bgr)
            _save_debug(i, frame_bgr, depth, masked_rgb)

            views.append((depth, masked_rgb, i * 90))
            print(f"[depth_full_mask_after] view {i + 1}/4 ok")

            if i < 3:
                disp.text("Rotating 90", f"Next {i + 2}/4", color="yellow")
                rotate_90(ccw=True)
                time.sleep(0.4)

        cam.stop()
        disp.text("Fusing", "debug mesh", color="green")
        verts, faces, colors = fuse(views)
        store.save(verts, faces, colors, path=MESH_PATH)
        disp.text("Done", "spinning preview", color="green")
        print(f"[depth_full_mask_after] mesh -> {MESH_PATH}")

        spin_stop = threading.Event()
        threading.Thread(target=disp.spin, args=(verts, colors, spin_stop), daemon=True).start()
        while not down.is_set():
            time.sleep(0.2)
        spin_stop.set()
    finally:
        cam.stop()
        gpio_cleanup()
        disp.clear()
        print("[depth_full_mask_after] shutdown")


if __name__ == "__main__":
    main()
