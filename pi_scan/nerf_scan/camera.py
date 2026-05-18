"""
nerf_scan/camera.py - Hardware management for Picamera2.
"""
import time
import numpy as np
from .config import GESTURE_W, GESTURE_H

class CameraManager:
    """Manages the camera lifecycle and frame capture."""

    def __init__(self):
        self._cam = None
        self._live = False

    def start(self):
        """Init picamera2 in video mode."""
        if self._live: return
        from picamera2 import Picamera2
        self._cam = Picamera2()
        self._cam.configure(self._cam.create_video_configuration(
            main={"size": (GESTURE_W, GESTURE_H), "format": "BGR888"}
        ))
        self._cam.start()
        time.sleep(0.4) # settle AE/AWB
        self._live = True

    def stop(self):
        """Close camera."""
        if self._cam:
            try:
                self._cam.stop()
                self._cam.close()
            except: pass
            self._cam = None
        self._live = False

    def frame(self) -> np.ndarray:
        """Grab latest video frame."""
        return self._cam.capture_array()

    def capture_still_bgr(self, width: int, height: int) -> np.ndarray:
        """Switch to still mode, grab frame, and return to video mode."""
        self._cam.stop()
        self._cam.configure(self._cam.create_still_configuration(
            main={"size": (width, height), "format": "BGR888"}
        ))
        self._cam.start()
        time.sleep(0.5)
        frame = self._cam.capture_array()
        
        # Back to video mode
        self._cam.stop()
        self._cam.configure(self._cam.create_video_configuration(
            main={"size": (GESTURE_W, GESTURE_H), "format": "BGR888"}
        ))
        self._cam.start()
        time.sleep(0.3)
        return frame
