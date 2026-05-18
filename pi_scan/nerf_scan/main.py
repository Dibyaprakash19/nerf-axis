"""
nerf_scan/main.py - Autonomous scan orchestrator.
"""
import sys, os, signal, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config   import WEB_PORT, CAM_W, CAM_H
from .camera   import CameraManager
from .scanner_lite import ScannerLite, remove_bg, fuse
from .stepper_ctrl import setup as gpio_setup, rotate_90, cleanup as gpio_cleanup
from .tft_ui   import TFTDisplay
from . import store, web

def _scan_cycle(cam: CameraManager, scanner: ScannerLite, disp: TFTDisplay) -> tuple:
    """Capture 4 views and fuse into a TSDF mesh."""
    views = []
    preview_verts, preview_colors = None, None

    for i in range(4):
        disp.text(f"View {i+1} / 4", "Stay still...", color="white")

        frame_bgr = cam.capture_still_bgr(CAM_W, CAM_H)

        disp.text(f"View {i+1} / 4", "Processing...", color="cyan")
        rgb   = remove_bg(frame_bgr)
        depth = scanner.depth_from_rgb(rgb)

        # Collect raw data for TSDF fusion
        views.append((depth, rgb, i * 90))

        # Keep one view for the TFT preview while fusing
        if i == 0:
            v, _, c = scanner.back_project(depth, rgb, view_angle=0)
            preview_verts, preview_colors = v, c

        print(f"[scan] view {i}: depth ok, {rgb.shape}")

        if i < 3:
            disp.text("Rotating 90°", f"Next: view {i+2}", color="yellow")
            rotate_90(ccw=True)
            time.sleep(0.4)

    disp.text("Fusing...", "TSDF + Marching Cubes", color="green")
    return fuse(views)

def main():
    _down = threading.Event()
    signal.signal(signal.SIGINT,  lambda *_: _down.set())
    signal.signal(signal.SIGTERM, lambda *_: _down.set())

    gpio_setup()
    disp    = TFTDisplay()
    scanner = ScannerLite()
    cam     = CameraManager()

    web.start()

    try:
        while not _down.is_set():
            store.cleanup()
            disp.text("NeRF-Axis", "Starting Auto-Scan", color="cyan")
            print("[main] Starting scan...")
            time.sleep(1)

            try:
                cam.start()
                verts, faces, colors = _scan_cycle(cam, scanner, disp)
                cam.stop()
            except Exception as e:
                cam.stop()
                disp.text("Scan error", str(e)[:24], color="red")
                print(f"[main] error: {e}")
                time.sleep(3)
                continue

            store.save(verts, faces, colors)
            disp.text("Done", f"Web UI: {WEB_PORT}", color="green")
            print(f"[main] Mesh ready -> http://gp5.local:{WEB_PORT}/")
            time.sleep(1.5)

            # Spin preview on TFT
            spin_stop = threading.Event()
            threading.Thread(target=disp.spin, args=(verts, colors, spin_stop), daemon=True).start()

            # Wait for web UI 'New Scan' trigger or 60s timeout
            timeout = time.time() + 60
            while not web.rescan_event.is_set() and not _down.is_set() and time.time() < timeout:
                time.sleep(0.2)

            web.rescan_event.clear()
            spin_stop.set()
            time.sleep(0.2)

    finally:
        cam.stop()
        web.stop()
        gpio_cleanup()
        disp.clear()
        print("[main] shutdown")

if __name__ == "__main__":
    main()
