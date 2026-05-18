"""
4-view MiDaS surface-shell experiment.

Each camera view becomes its own depth-relief surface. The four surfaces are
then placed around the turntable at 0/90/180/270 degrees, instead of forcing
independent MiDaS depths into a single TSDF volume.
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
from .stepper_ctrl import cleanup as gpio_cleanup, rotate_90, setup as gpio_setup
from .tft_ui import TFTDisplay


OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "surface_shell")
MESH_PATH = os.path.join(OUT_DIR, "surface_shell_4view.glb")


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


def _surface_from_view(depth: np.ndarray, rgb: np.ndarray, mask: np.ndarray, angle_deg: float, step: int = 2):
    mask = _largest_component(mask)
    depth_s = depth[::step, ::step].astype(np.float32)
    rgb_s = rgb[::step, ::step].astype(np.uint8)
    mask_s = mask[::step, ::step].astype(bool)
    h, w = depth_s.shape
    if mask_s.sum() < 50:
        raise RuntimeError(f"Mask too small for view angle {angle_deg}")

    fg_depth = depth_s[mask_s]
    lo, hi = np.percentile(fg_depth, [3, 97])
    d = np.clip((depth_s - lo) / (hi - lo + 1e-6), 0, 1)
    d = cv2.GaussianBlur(d, (0, 0), 1.0)

    yy, xx = np.mgrid[0:h, 0:w]
    x_local = (xx - w / 2.0).astype(np.float32)
    y_local = -(yy - h / 2.0).astype(np.float32)

    # Put the relief surface near the outside of a rough object shell.
    shell_radius = w * 0.24
    relief_depth = (d - 0.5).astype(np.float32) * (w * 0.22)
    z_local = shell_radius + relief_depth

    flat_index = np.full(h * w, -1, np.int32)
    valid = mask_s.reshape(-1)
    flat_index[np.where(valid)[0]] = np.arange(valid.sum(), dtype=np.int32)

    local = np.stack([x_local.reshape(-1), y_local.reshape(-1), z_local.reshape(-1)], axis=1)[valid]
    colors = rgb_s.reshape(-1, 3)[valid]

    rad = np.radians(angle_deg)
    ca, sa = np.cos(rad), np.sin(rad)
    x = local[:, 0] * ca + local[:, 2] * sa
    z = -local[:, 0] * sa + local[:, 2] * ca
    verts = np.stack([x, local[:, 1], z], axis=1)

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

    return verts.astype(np.float32), np.asarray(faces, np.int32), colors.astype(np.uint8), mask


def _save_debug(i: int, frame_bgr: np.ndarray, depth: np.ndarray, rgb: np.ndarray, mask: np.ndarray):
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUT_DIR, f"view_{i:02d}_raw.jpg"), frame_bgr)
    cv2.imwrite(os.path.join(OUT_DIR, f"view_{i:02d}_mask.jpg"), mask.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(OUT_DIR, f"view_{i:02d}_masked.jpg"), cv2.cvtColor(cv2.bitwise_and(rgb, rgb, mask=mask), cv2.COLOR_RGB2BGR))
    depth_u8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT_DIR, f"view_{i:02d}_depth.jpg"), cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO))


def main():
    down = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: down.set())
    signal.signal(signal.SIGTERM, lambda *_: down.set())

    gpio_setup()
    disp = TFTDisplay()
    scanner = ScannerLite()
    cam = CameraManager()
    all_verts, all_faces, all_colors = [], [], []
    offset = 0

    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        cam.start()
        for i, angle in enumerate((0, 90, 180, 270)):
            disp.text(f"Surface {i + 1}/4", "MiDaS full image", color="cyan")
            frame_bgr = cam.capture_still_bgr(CAM_W, CAM_H)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            depth = scanner.depth_from_rgb(rgb)
            mask = _mask_after_depth(frame_bgr)
            verts, faces, colors, mask = _surface_from_view(depth, rgb, mask, angle)
            _save_debug(i, frame_bgr, depth, rgb, mask)

            all_verts.append(verts)
            all_faces.append(faces + offset)
            all_colors.append(colors)
            offset += len(verts)
            print(f"[surface_shell] view {i + 1}/4 angle={angle} verts={len(verts)}")

            if i < 3:
                disp.text("Rotating 90", f"Next {i + 2}/4", color="yellow")
                rotate_90(ccw=True)
                time.sleep(0.4)

        cam.stop()
        verts = np.vstack(all_verts).astype(np.float32)
        faces = np.vstack(all_faces).astype(np.int32)
        colors = np.vstack(all_colors).astype(np.uint8)
        verts -= verts.mean(axis=0)
        m = np.abs(verts).max()
        if m > 0:
            verts *= 50.0 / m

        disp.text("Saving", "surface shell", color="green")
        store.save(verts, faces, colors, path=MESH_PATH)
        print(f"[surface_shell] mesh -> {MESH_PATH}")

        spin_stop = threading.Event()
        threading.Thread(target=disp.spin, args=(verts, colors, spin_stop), daemon=True).start()
        while not down.is_set():
            time.sleep(0.2)
        spin_stop.set()
    finally:
        cam.stop()
        gpio_cleanup()
        disp.clear()
        print("[surface_shell] shutdown")


if __name__ == "__main__":
    main()
