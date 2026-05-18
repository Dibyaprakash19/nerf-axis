"""
nerf_scan/main_16.py - Separate 16-view scanner experiment.

This intentionally leaves nerf_scan/main.py and the legacy 4-view fallback alone.
"""
import argparse
import os
import signal
import sys
import threading
import time

import cv2
import RPi.GPIO as GPIO
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from . import store, web
from .camera import CameraManager
from .config import CAM_H, CAM_W, STEP_PINS, STEPS_90, TFT_H, TFT_W, WEB_PORT
from .scanner_lite import ScannerLite, fuse, remove_bg
from .stepper_ctrl import _SEQ, cleanup as gpio_cleanup, setup as gpio_setup
from .tft_ui import TFTDisplay


def rotate_angle(degrees: float, ccw: bool = True):
    """Local arbitrary-angle rotate for this 16-view experiment."""
    steps = max(1, round((degrees / 90.0) * STEPS_90))
    seq = list(reversed(_SEQ)) if ccw else _SEQ
    for _ in range(steps):
        for state in seq:
            for i, pin in enumerate(STEP_PINS):
                GPIO.output(pin, state[i])
            time.sleep(0.001)
    for pin in STEP_PINS:
        GPIO.output(pin, False)


def _show_preview_countdown(cam: CameraManager, disp: TFTDisplay, view_idx: int, total: int):
    for tick in (2, 1):
        frame = cam.frame()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = img.size
        scale = max(TFT_W / w, TFT_H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h))
        left, top = (new_w - TFT_W) // 2, (new_h - TFT_H) // 2
        img = img.crop((left, top, left + TFT_W, top + TFT_H))
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), f"View {view_idx + 1}/{total}", fill="cyan")
        draw.text((52, 62), f"{tick}", fill="red")
        disp._fast_display(img)
        time.sleep(1.0)


def scan_cycle(cam: CameraManager, scanner: ScannerLite, disp: TFTDisplay, views_count: int):
    views = []
    step_degrees = 360.0 / views_count

    for i in range(views_count):
        _show_preview_countdown(cam, disp, i, views_count)
        disp.text(f"View {i + 1}/{views_count}", "Depth + mask", color="cyan")

        frame_bgr = cam.capture_still_bgr(CAM_W, CAM_H)
        rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        depth = scanner.depth_from_rgb(rgb_full)
        masked_rgb = remove_bg(frame_bgr)

        views.append((depth, masked_rgb, i * step_degrees))
        print(f"[main_16] view {i + 1}/{views_count}: depth ok, angle={i * step_degrees:.1f}")

        if i < views_count - 1:
            disp.text(f"Rotating {step_degrees:.1f}", f"Next {i + 2}/{views_count}", color="yellow")
            rotate_angle(step_degrees, ccw=True)
            time.sleep(0.4)

    disp.text("Fusing...", "16-view TSDF", color="green")
    return fuse(views)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", type=int, default=16)
    parser.add_argument("--keep-web-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.views < 4:
        raise SystemExit("--views must be at least 4")

    down = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: down.set())
    signal.signal(signal.SIGTERM, lambda *_: down.set())

    gpio_setup()
    disp = TFTDisplay()
    scanner = ScannerLite()
    cam = CameraManager()
    web.start()

    try:
        store.cleanup()
        disp.text("NeRF-Axis 16", "Starting once", color="cyan")
        cam.start()
        try:
            verts, faces, colors = scan_cycle(cam, scanner, disp, args.views)
        finally:
            cam.stop()

        store.save(verts, faces, colors)
        disp.text("Done", f"Web UI: {WEB_PORT}", color="green")
        print(f"[main_16] Mesh ready -> http://gp5.local:{WEB_PORT}/")

        spin_stop = threading.Event()
        threading.Thread(target=disp.spin, args=(verts, colors, spin_stop), daemon=True).start()
        timeout = time.time() + max(0, args.keep_web_seconds)
        while not down.is_set() and time.time() < timeout:
            time.sleep(0.2)
        spin_stop.set()
    finally:
        cam.stop()
        web.stop()
        gpio_cleanup()
        disp.clear()
        print("[main_16] shutdown")


if __name__ == "__main__":
    main()
