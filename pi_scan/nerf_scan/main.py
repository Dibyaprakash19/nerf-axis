"""
nerf_scan/main.py  —  State-machine orchestrator.

State diagram:

    IDLE ──[👍 hold 1.5s → ✌️ hold 1.5s]──► LOOP
    LOOP: scan → display(15s) → scan → display → …
    LOOP ──[👍 → ✌️ during display]──────────► IDLE
    Any state ──[SIGINT / SIGTERM]───────────► shutdown

Camera is shared between gesture watching and scan capture.
During each view capture, GestureWatcher reconfigures picamera2 to
256×256 still mode, grabs one frame, then restores 320×240 video mode.
No libcamera-still subprocess — single camera instance throughout.

Run directly:
    cd ~/pi_scan
    source venv/bin/activate
    python -m nerf_scan.main
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import signal, threading, time

from .config   import LOOP_DISPLAY_S, WEB_PORT, CAM_W, CAM_H
from .gesture  import GestureWatcher, GESTURE_THUMBS, GESTURE_VSIGN
from .scanner_lite import ScannerLite, remove_bg, fuse
from .stepper_ctrl import setup as gpio_setup, rotate_90, cleanup as gpio_cleanup
from .tft_ui   import TFTDisplay
from . import store, web


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trigger(watcher: GestureWatcher, disp: TFTDisplay,
             timeout: float = None, is_deactivation: bool = False) -> bool:
    """
    Wait for any two valid gestures. Shows instructions or current loop status.
    """
    base = "SCANNING" if is_deactivation else "SHOW 2 SIGNS"
    ok = watcher.wait_for_sequence(count=2, timeout_per=timeout, disp=disp, base_label=base)
    
    if ok and not is_deactivation:
        disp.text("STARTING", "OPERATION LOOP", color="green")
        time.sleep(1.5)
    return ok


def _scan_cycle(watcher: GestureWatcher, scanner: ScannerLite,
                disp: TFTDisplay) -> tuple:
    """
    4-view scan.  Camera reconfigures to 256×256 per shot, returns to
    320×240 video between shots so gesture detection stays live.

    Back-projection step=2 on 256×256 → 128×128 vertices per view,
    exactly matching TFT_W × TFT_H — no interpolation needed for display.
    """
    views = []

    for i in range(4):
        disp.text(f"View {i+1} / 4", "hold still", color="white")

        # Reconfigures camera internally: 320×240 video → 256×256 still → back
        frame_bgr = watcher.capture_still_bgr(CAM_W, CAM_H)

        disp.text(f"View {i+1} / 4", "GrabCut...", color="cyan")
        rgb = remove_bg(frame_bgr)

        disp.text(f"View {i+1} / 4", "MiDaS...", color="cyan")
        depth = scanner.depth_from_rgb(rgb)

        v, f, c = scanner.back_project(depth, rgb, view_angle=i * 90)
        views.append((v, f, c))
        print(f"[scan] view {i}: {len(v):,} verts  {len(f):,} faces")

        if i < 3:
            disp.text("Rotating 90deg", f"-> view {i+2}", color="yellow")
            rotate_90(ccw=True)
            time.sleep(0.4)   # vibration settle

    disp.text("Fusing...", "4 views", color="green")
    return fuse(views)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    _down = threading.Event()
    def _sig(*_): _down.set()
    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    gpio_setup()
    disp    = TFTDisplay()
    scanner = ScannerLite()
    watcher = GestureWatcher()

    web.start()
    watcher.start()
    disp.text("NeRF-Axis", "WAITING FOR ACTIVATION", color="cyan")
    time.sleep(2)

    try:
        while not _down.is_set():

            # ── IDLE: wait for activation ─────────────────────────────────────
            if not _trigger(watcher, disp, timeout=None):
                break  # only exits on _down via wait_for's stop check

            # ── LOOP: scan → display → rescan until deactivation ──────────────
            in_loop = True
            while in_loop and not _down.is_set():

                store.cleanup()

                try:
                    verts, faces, colors = _scan_cycle(watcher, scanner, disp)
                except Exception as exc:
                    disp.text("Scan error", str(exc)[:24], color="red")
                    print(f"[main] scan error: {exc}")
                    time.sleep(3)
                    break   # back to IDLE

                store.save(verts, faces, colors)
                disp.text("Done ✓", f":8080 to view", color="green")
                print(f"[main] mesh ready → http://gp5.local:{WEB_PORT}/")
                time.sleep(0.8)

                # Spin TFT in background; watch for deactivation gesture
                spin_stop = threading.Event()
                threading.Thread(
                    target=disp.spin,
                    args=(verts, colors, spin_stop),
                    daemon=True,
                ).start()

                deactivate = _trigger(watcher, disp, timeout=LOOP_DISPLAY_S, is_deactivation=True)

                spin_stop.set()
                time.sleep(0.15)   # let spinner thread flush

                if deactivate:
                    in_loop = False   # → IDLE
                # else: timeout → rescan automatically

    finally:
        watcher.stop()
        web.stop()
        gpio_cleanup()
        disp.clear()
        print("[main] shutdown")


if __name__ == "__main__":
    main()
