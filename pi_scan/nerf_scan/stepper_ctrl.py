"""
nerf_scan/stepper_ctrl.py — 28BYJ-48 stepper via ULN2003.

Usage:
    from nerf_scan.stepper_ctrl import setup, rotate_90, cleanup
    setup()
    rotate_90(ccw=True)    # 90° anticlockwise
    cleanup()
"""

import time
import RPi.GPIO as GPIO
from .config import STEP_PINS, STEPS_90

_SEQ = [
    [1,0,0,0], [1,1,0,0], [0,1,0,0], [0,1,1,0],
    [0,0,1,0], [0,0,1,1], [0,0,0,1], [1,0,0,1],
]

def setup():
    GPIO.setmode(GPIO.BCM)
    for p in STEP_PINS:
        GPIO.setup(p, GPIO.OUT)
        GPIO.output(p, False)

def _release():
    for p in STEP_PINS:
        GPIO.output(p, False)

def rotate_90(ccw: bool = True):
    """Rotate turntable 90°.  ccw=True → anticlockwise (default scan dir)."""
    seq = list(reversed(_SEQ)) if ccw else _SEQ
    for _ in range(STEPS_90):
        for state in seq:
            for i, pin in enumerate(STEP_PINS):
                GPIO.output(pin, state[i])
            time.sleep(0.001)
    _release()

def cleanup():
    _release()
    GPIO.cleanup()
