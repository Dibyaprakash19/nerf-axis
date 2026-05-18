"""
4-view aligned MiDaS surface shell.

Motor assumption: the turntable/object rotates clockwise when viewed from
above. Capture order is therefore placed at 0, -90, -180, -270 degrees.

Instead of TSDF fusing inconsistent MiDaS maps, each view becomes a normalized
surface patch. Neighboring patches are joined with seam strips along their
silhouette edges so the object reads as one shell instead of four loose planes.
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


OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "surface_shell_aligned")
MESH_PATH = os.path.join(OUT_DIR, "surface_shell_aligned_4view.glb")
TARGET_W = 96
TARGET_H = 128


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


def _crop_to_mask(depth: np.ndarray, rgb: np.ndarray, mask: np.ndarray):
    mask = _largest_component(mask)
    ys, xs = np.where(mask > 0)
    if len(xs) < 50:
        raise RuntimeError("Mask too small to crop")

    pad = 8
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)

    depth_c = depth[y0:y1, x0:x1].astype(np.float32)
    rgb_c = rgb[y0:y1, x0:x1].astype(np.uint8)
    mask_c = mask[y0:y1, x0:x1].astype(np.uint8)

    depth_r = cv2.resize(depth_c, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    rgb_r = cv2.resize(rgb_c, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask_c, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
    mask_r = _largest_component((mask_r > 0).astype(np.uint8))
    return depth_r, rgb_r, mask_r


def _edge_columns(mask: np.ndarray):
    left = np.full(mask.shape[0], -1, np.int32)
    right = np.full(mask.shape[0], -1, np.int32)
    for r in range(mask.shape[0]):
        cols = np.where(mask[r] > 0)[0]
        if len(cols):
            left[r] = cols[0]
            right[r] = cols[-1]
    return left, right


def _surface_from_view(depth: np.ndarray, rgb: np.ndarray, mask: np.ndarray, angle_deg: float):
    depth, rgb, mask = _crop_to_mask(depth, rgb, mask)
    mask_bool = mask.astype(bool)

    fg_depth = depth[mask_bool]
    lo, hi = np.percentile(fg_depth, [4, 96])
    d = np.clip((depth - lo) / (hi - lo + 1e-6), 0, 1)
    d = cv2.GaussianBlur(d, (0, 0), 1.0)

    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]

    # Normalize all views to the same front-facing silhouette size.
    x_local = ((xx / max(1, w - 1)) - 0.5).astype(np.float32) * 72.0
    y_local = (0.5 - (yy / max(1, h - 1))).astype(np.float32) * 96.0

    # Put each view on a shell; MiDaS relief changes local radius only.
    shell_radius = 38.0
    relief = (d - 0.5).astype(np.float32) * 24.0
    z_local = shell_radius + relief

    flat_index = np.full(h * w, -1, np.int32)
    valid = mask_bool.reshape(-1)
    flat_index[np.where(valid)[0]] = np.arange(valid.sum(), dtype=np.int32)

    local = np.stack([x_local.reshape(-1), y_local.reshape(-1), z_local.reshape(-1)], axis=1)[valid]
    colors = rgb.reshape(-1, 3)[valid].astype(np.uint8)

    rad = np.radians(angle_deg)
    ca, sa = np.cos(rad), np.sin(rad)
    world_x = local[:, 0] * ca + local[:, 2] * sa
    world_z = -local[:, 0] * sa + local[:, 2] * ca
    verts = np.stack([world_x, local[:, 1], world_z], axis=1).astype(np.float32)

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

    left_cols, right_cols = _edge_columns(mask)
    return {
        "verts": verts,
        "faces": np.asarray(faces, np.int32),
        "colors": colors,
        "mask": mask,
        "rgb": rgb,
        "depth": depth,
        "flat_index": flat_index.reshape(h, w),
        "left_cols": left_cols,
        "right_cols": right_cols,
    }


def _add_seams(patches: list):
    verts = [p["verts"] for p in patches]
    colors = [p["colors"] for p in patches]
    faces = []
    offsets = []
    offset = 0
    for p in patches:
        offsets.append(offset)
        faces.append(p["faces"] + offset)
        offset += len(p["verts"])

    # Clockwise capture order: right edge of current view meets left edge of next view.
    for i, p in enumerate(patches):
        q = patches[(i + 1) % len(patches)]
        po = offsets[i]
        qo = offsets[(i + 1) % len(patches)]
        h = min(p["flat_index"].shape[0], q["flat_index"].shape[0])
        for r in range(h - 1):
            pr0 = p["right_cols"][r]
            pr1 = p["right_cols"][r + 1]
            ql0 = q["left_cols"][r]
            ql1 = q["left_cols"][r + 1]
            if min(pr0, pr1, ql0, ql1) < 0:
                continue
            a = p["flat_index"][r, pr0]
            b = p["flat_index"][r + 1, pr1]
            c = q["flat_index"][r, ql0]
            d = q["flat_index"][r + 1, ql1]
            if min(a, b, c, d) < 0:
                continue
            faces.append(np.asarray([[a + po, b + po, c + qo], [c + qo, b + po, d + qo]], np.int32))

    return np.vstack(verts), np.vstack(faces), np.vstack(colors)


def _save_debug(i: int, frame_bgr: np.ndarray, depth: np.ndarray, rgb: np.ndarray, mask: np.ndarray, patch: dict):
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUT_DIR, f"view_{i:02d}_raw.jpg"), frame_bgr)
    cv2.imwrite(os.path.join(OUT_DIR, f"view_{i:02d}_mask.jpg"), mask.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(OUT_DIR, f"view_{i:02d}_cropped_mask.jpg"), patch["mask"].astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(OUT_DIR, f"view_{i:02d}_cropped_rgb.jpg"), cv2.cvtColor(patch["rgb"], cv2.COLOR_RGB2BGR))
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
    patches = []

    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        cam.start()
        # Clockwise object rotation from top view means view angles go negative.
        for i, angle in enumerate((0, -90, -180, -270)):
            disp.text(f"Aligned {i + 1}/4", f"angle {angle}", color="cyan")
            frame_bgr = cam.capture_still_bgr(CAM_W, CAM_H)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            depth = scanner.depth_from_rgb(rgb)
            mask = _mask_after_depth(frame_bgr)
            patch = _surface_from_view(depth, rgb, mask, angle)
            _save_debug(i, frame_bgr, depth, rgb, mask, patch)
            patches.append(patch)
            print(f"[aligned_shell] view {i + 1}/4 angle={angle} verts={len(patch['verts'])}")

            if i < 3:
                disp.text("Rotating CW", f"Next {i + 2}/4", color="yellow")
                rotate_90(ccw=True)
                time.sleep(0.4)

        cam.stop()
        verts, faces, colors = _add_seams(patches)
        verts -= verts.mean(axis=0)
        m = np.abs(verts).max()
        if m > 0:
            verts *= 50.0 / m

        disp.text("Saving", "aligned shell", color="green")
        store.save(verts.astype(np.float32), faces.astype(np.int32), colors.astype(np.uint8), path=MESH_PATH)
        print(f"[aligned_shell] mesh -> {MESH_PATH}")

        spin_stop = threading.Event()
        threading.Thread(target=disp.spin, args=(verts, colors, spin_stop), daemon=True).start()
        while not down.is_set():
            time.sleep(0.2)
        spin_stop.set()
    finally:
        cam.stop()
        gpio_cleanup()
        disp.clear()
        print("[aligned_shell] shutdown")


if __name__ == "__main__":
    main()
