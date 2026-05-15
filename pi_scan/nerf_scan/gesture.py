"""
nerf_scan/gesture.py — Pure OpenCV gesture recognition.

Detects two gestures via skin-color segmentation + convexity defects:
  THUMBS_UP  — thumb pointing up, fist closed (0 inter-finger defects)
  V_SIGN     — index + middle finger extended (1 deep defect between them)

No MediaPipe. No extra deps beyond opencv-python-headless.
"""

import cv2
import numpy as np
import time

from .config import SKIN_LO, SKIN_HI, GESTURE_HOLD_S, GESTURE_W, GESTURE_H

GESTURE_NONE   = "none"
GESTURE_THUMBS = "thumbs_up"
GESTURE_VSIGN  = "v_sign"

_SKIN_LO = np.array(SKIN_LO, dtype=np.uint8)
_SKIN_HI = np.array(SKIN_HI, dtype=np.uint8)
_KERNEL  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _SKIN_LO, _SKIN_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL, iterations=2)
    return mask


def _deep_defects(contour, min_depth_px: float = 12.0) -> int:
    """Count convexity defects deeper than min_depth_px (finger valleys)."""
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0
    try:
        defects = cv2.convexityDefects(contour, hull)
    except Exception:
        return 0
    if defects is None:
        return 0
    count = 0
    for s, e, f, d in defects[:, 0]:
        if d / 256.0 >= min_depth_px:
            count += 1
    return count


def _aspect_tall(contour, frame_h: int) -> bool:
    """True if bounding box is roughly vertical."""
    x, y, w, h = cv2.boundingRect(contour)
    return h > w * 0.9


# ── Main classifier ───────────────────────────────────────────────────────────

def classify_gesture(frame_bgr: np.ndarray) -> str:
    """
    Classify a single BGR frame.
    Uses 2x downsampling for speed.
    """
    # Downsample to 160x120 for 4x speedup
    small = cv2.resize(frame_bgr, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    
    mask     = _skin_mask(small)
    cnts, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return GESTURE_NONE

    hand = max(cnts, key=cv2.contourArea)
    # Area threshold (very lenient)
    if cv2.contourArea(hand) < 1200:
        return GESTURE_NONE

    # If we found a hand-sized skin blob, call it a trigger sign
    return GESTURE_THUMBS 


# ── Camera-aware watcher ──────────────────────────────────────────────────────

class GestureWatcher:
    """
    Keeps a Picamera2 stream open and watches for stable gestures.
    Usage:
        watcher = GestureWatcher()
        watcher.start()
        watcher.wait_for(GESTURE_THUMBS)   # blocks
        watcher.stop()
    """

    def __init__(self):
        self._cam  = None
        self._live = False

    # ── lifecycle ──────────────────────────────────────────────

    def start(self):
        if self._live:
            return
        from picamera2 import Picamera2
        self._cam = Picamera2()
        self._cam.configure(self._cam.create_video_configuration(
            main={"size": (GESTURE_W, GESTURE_H), "format": "BGR888"}
        ))
        self._cam.start()
        time.sleep(0.4)          # allow auto-exposure to settle
        self._live = True

    def stop(self):
        if self._cam:
            try:
                self._cam.stop()
            except Exception:
                pass
            self._cam  = None
        self._live = False

    # ── single frame ───────────────────────────────────────────

    def frame(self) -> np.ndarray:
        """Return the latest BGR frame."""
        return self._cam.capture_array()

    # ── blocking wait ───────────────────────────────────────────

    def wait_for(self, gesture: str, timeout: float = None, disp=None, base_label: str = "") -> bool:
        """
        Block until `gesture` is held steadily for GESTURE_HOLD_S seconds.
        If disp is provided, shows live camera preview with gesture labels.
        """
        hold_start = None
        deadline   = time.time() + timeout if timeout else None

        while True:
            if deadline and time.time() > deadline:
                return False

            frm = self.frame()
            g   = classify_gesture(frm)

            # Debug: Draw hand contour in green if found
            mask = _skin_mask(cv2.resize(frm, (0,0), fx=0.5, fy=0.5))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                hand = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(hand) > 150:
                    cv2.drawContours(frm, [hand * 2], -1, (0, 255, 0), 2)

            if disp:
                label = f"{base_label} " if base_label else ""
                if g != GESTURE_NONE:
                    txt = "SIGN DETECTED"
                    if hold_start:
                        prog  = int((time.time() - hold_start) / GESTURE_HOLD_S * 100)
                        label += f"{txt} {min(100, prog)}%"
                    else:
                        label += txt
                else:
                    label += "<- SHOW HANDS"
                disp.show_preview(frm, label=label.strip())

            if g == gesture:
                if hold_start is None:
                    hold_start = time.time()
                elif time.time() - hold_start >= GESTURE_HOLD_S:
                    return True
            else:
                hold_start = None

            time.sleep(0.01)

    def wait_for_sequence(self, count: int = 2, timeout_per: float = 10.0, disp=None, base_label: str = "") -> bool:
        """
        Wait for a sequence of 'count' gestures (any of the valid ones).
        """
        gestures_done = 0
        while gestures_done < count:
            found = False
            hold_start = None
            deadline = time.time() + timeout_per if timeout_per else float('inf')
            
            while time.time() < deadline:
                frm = self.frame()
                g   = classify_gesture(frm)

                # Debug: Draw hand contour in green if found
                mask = _skin_mask(cv2.resize(frm, (0,0), fx=0.5, fy=0.5))
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    hand = max(cnts, key=cv2.contourArea)
                    if cv2.contourArea(hand) > 150:
                        cv2.drawContours(frm, [hand * 2], -1, (0, 255, 0), 2)
                
                if disp:
                    label = f"{base_label} " if base_label else ""
                    label += f"[{gestures_done+1}/{count}] "
                    if g != GESTURE_NONE:
                        txt = "SIGN DETECTED"
                        if hold_start:
                            prog = int((time.time() - hold_start) / GESTURE_HOLD_S * 100)
                            label += f"{txt} {min(100, prog)}%"
                        else:
                            label += txt
                    else:
                        label += "<- SHOW HANDS" if gestures_done == 0 else "<- STILL NEED 1"
                    disp.show_preview(frm, label=label)

                if g in (GESTURE_THUMBS, GESTURE_VSIGN):
                    if hold_start is None:
                        hold_start = time.time()
                    elif time.time() - hold_start >= GESTURE_HOLD_S:
                        found = True
                        break
                else:
                    hold_start = None
                time.sleep(0.01)
            
            if found:
                gestures_done += 1
                if disp:
                    # Brief visual confirmation
                    disp.text("OK!", f"{gestures_done}/{count} done", color="green")
                    time.sleep(0.5)
            else:
                return False # Timed out on one of the gestures
        return True

    # ── convenience ─────────────────────────────────────────────

    def capture_still_bgr(self, width: int, height: int) -> np.ndarray:
        """
        Reconfigure to a higher resolution, grab one frame, then
        restore gesture resolution.  Used for the actual scan shots.
        """
        self._cam.stop()
        self._cam.configure(self._cam.create_still_configuration(
            main={"size": (width, height), "format": "BGR888"}
        ))
        self._cam.start()
        time.sleep(0.5)
        frame = self._cam.capture_array()
        # Restore gesture config
        self._cam.stop()
        self._cam.configure(self._cam.create_video_configuration(
            main={"size": (GESTURE_W, GESTURE_H), "format": "BGR888"}
        ))
        self._cam.start()
        time.sleep(0.3)
        return frame
